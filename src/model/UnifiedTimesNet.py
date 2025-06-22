import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def FFT_for_Period(x, k=2):
    """
    Find top k periods from the time series using FFT
    """
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        # Inception block parameters
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=d_model, out_channels=d_ff, kernel_size=[1, 1], bias=False),
            nn.BatchNorm2d(d_ff),
            nn.ReLU(),
            nn.Conv2d(in_channels=d_ff, out_channels=d_model, kernel_size=[1, 1], bias=False)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            # Avoid division by zero
            if (T + self.pred_len) % period != 0:
                length = (((T + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (T + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (T + self.pred_len)
                out = x
            
            # 2D transformation
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # Apply convolution
            out = self.conv(out)
            
            # Reshape back and trim
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = out[:, :(T + self.pred_len), :]
            res.append(out)
        
        # Adaptive aggregation
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T + self.pred_len, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Residual connection
        res = res + x
        return res


class UnifiedTimesNet(nn.Module):
    def __init__(self, num_assets, asset_features, market_features, seq_len, hidden_dim, 
                 num_layers=2, top_k=3, d_ff=None, dropout=0.1):
        super(UnifiedTimesNet, self).__init__()
        
        self.num_assets = num_assets
        self.asset_features = asset_features
        self.market_features = market_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if d_ff is None:
            d_ff = hidden_dim * 4
            
        # Asset processing branch
        self.asset_embedding = nn.Linear(asset_features, hidden_dim)
        self.asset_blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff)
            for _ in range(num_layers)
        ])
        
        # Market processing branch  
        self.market_embedding = nn.Linear(market_features, hidden_dim)
        self.market_blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff)
            for _ in range(num_layers)
        ])
        
        # Asset selection head
        self.asset_norm = nn.LayerNorm(hidden_dim)
        self.asset_projection = nn.Linear(hidden_dim, 1)
        
        # Market state head
        self.market_norm = nn.LayerNorm(hidden_dim)
        self.market_projection = nn.Linear(hidden_dim, 2)  # mu, sigma for market exposure
        
        self.dropout = nn.Dropout(dropout)
        
    def forward_asset_selection(self, x_asset, mask=None):
        """
        Forward pass for asset selection (replaces ASU)
        x_asset: [batch, num_assets, seq_len, asset_features]
        mask: [batch, num_assets] - True for invalid assets
        """
        B, N, T, F = x_asset.shape
        
        # Reshape for processing
        x = x_asset.permute(0, 1, 3, 2).reshape(B * N, F, T).permute(0, 2, 1)  # [B*N, T, F]
        
        # Embedding
        x = self.asset_embedding(x)  # [B*N, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply TimesBlocks
        for block in self.asset_blocks:
            x = block(x)
            
        # Global pooling and projection
        x = self.asset_norm(x.mean(dim=1))  # [B*N, hidden_dim]
        scores = self.asset_projection(x).reshape(B, N)  # [B, N]
        
        # Apply mask and return probabilities
        if mask is not None:
            scores[mask] = -math.inf
            
        return torch.sigmoid(scores)
    
    def forward_market_state(self, x_market):
        """
        Forward pass for market state prediction (replaces MSU)
        x_market: [batch, seq_len, market_features]
        """
        B, T, F = x_market.shape
        
        # Embedding
        x = self.market_embedding(x_market)  # [B, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply TimesBlocks
        for block in self.market_blocks:
            x = block(x)
            
        # Global pooling and projection
        x = self.market_norm(x.mean(dim=1))  # [B, hidden_dim]
        params = self.market_projection(x)  # [B, 2]
        
        return params
    
    def forward(self, x_asset, x_market=None, mask=None):
        """
        Unified forward pass
        """
        # Asset selection scores
        asset_scores = self.forward_asset_selection(x_asset, mask)
        
        # Market state parameters (if market data provided)
        market_params = None
        if x_market is not None:
            market_params = self.forward_market_state(x_market)
            
        return asset_scores, market_params
