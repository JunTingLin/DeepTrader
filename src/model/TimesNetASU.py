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


class TimesNetASU(nn.Module):
    """
    TimesNet-based Asset Selection Unit (replaces original GCN-based ASU)
    """
    def __init__(self, num_assets, in_features, hidden_dim, seq_len, 
                 num_layers=4, top_k=3, d_ff=None, dropout=0.1):
        super(TimesNetASU, self).__init__()
        
        self.num_assets = num_assets
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if d_ff is None:
            d_ff = hidden_dim * 4
            
        # Input embedding
        self.asset_embedding = nn.Linear(in_features, hidden_dim)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff)
            for _ in range(num_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for final scores
        self.bn_output = nn.BatchNorm1d(num_features=num_assets)
        
    def forward(self, inputs, mask):
        """
        inputs: [batch, num_assets, seq_len, in_features]
        mask: [batch, num_assets] - True for invalid assets
        outputs: [batch, num_assets] - asset selection scores
        """
        B, N, T, F = inputs.shape
        
        # Reshape for processing: [B*N, T, F]
        x = inputs.reshape(B * N, T, F)
        
        # Embedding
        x = self.asset_embedding(x)  # [B*N, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply TimesNet blocks
        for block in self.blocks:
            x = block(x)
            
        # Global pooling and projection
        x = self.norm(x.mean(dim=1))  # [B*N, hidden_dim]
        x = self.projection(x).squeeze(-1)  # [B*N]
        
        # Reshape back to [B, N]
        scores = x.reshape(B, N)
        
        # Apply batch normalization
        scores = self.bn_output(scores)
        
        # Convert to probabilities using sigmoid
        scores = torch.sigmoid(scores)
        
        # Apply mask: set invalid assets to -inf
        if mask is not None:
            scores = scores.clone()  # Avoid in-place operation
            scores[mask] = -math.inf
            
        return scores