import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .Inception_Block import Inception_Block_V1


def FFT_for_Period(x, k=2):
    """
    Find top k periods from the time series using FFT.
    This identifies the most significant periodic components in the data.
    
    Args:
        x: Input time series [batch, time_len, features]
        k: Number of top periods to extract
    
    Returns:
        period: Array of detected periods
        period_weight: Weights corresponding to each period's importance
    """
    # Apply FFT to find frequency components
    xf = torch.fft.rfft(x, dim=1)
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0  # Remove DC component
    
    # Find top k most significant frequencies
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list
    return period, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    """
    TimesBlock: Core component of TimesNet for temporal 2D-variation modeling.
    Converts 1D time series to 2D tensors based on detected periods and applies
    multi-scale convolutions for feature extraction.
    """
    def __init__(self, seq_len, pred_len, top_k, d_model, d_ff, num_kernels=6):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        
        # Use original TimesNet Inception blocks for multi-scale processing
        self.conv = nn.Sequential(
            Inception_Block_V1(d_model, d_ff, num_kernels=num_kernels),
            nn.GELU(),  # Use GELU as in original TimesNet
            Inception_Block_V1(d_ff, d_model, num_kernels=num_kernels)
        )

    def forward(self, x):
        """
        Forward pass of TimesBlock.
        
        Args:
            x: Input tensor [batch, time_len, features]
        
        Returns:
            Output tensor with same shape as input
        """
        B, T, N = x.size()
        period_list, period_weight = FFT_for_Period(x, self.k)
        
        res = []
        for i in range(self.k):
            period = period_list[i]
            
            # Handle padding for reshaping to 2D
            if (T + self.pred_len) % period != 0:
                length = (((T + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (T + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (T + self.pred_len)
                out = x
            
            # Core innovation: 1D time series -> 2D tensor transformation
            # Reshape based on detected period to capture intra/inter-period variations
            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous()
            
            # Apply multi-scale Inception convolutions in 2D space
            out = self.conv(out)
            
            # Transform back to 1D and trim to original length
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            out = out[:, :(T + self.pred_len), :]
            res.append(out)
        
        # Adaptive aggregation weighted by period importance
        res = torch.stack(res, dim=-1)
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T + self.pred_len, N, 1)
        res = torch.sum(res * period_weight, -1)
        
        # Residual connection for stable training
        res = res + x
        return res


class TimesNetASU(nn.Module):
    """
    TimesNet-based Asset Selection Unit (replaces original GCN-based ASU).
    Uses temporal 2D-variation modeling for asset selection in portfolio management.
    
    Architecture:
    - Input embedding to map asset features to hidden dimension
    - Multiple TimesBlocks for multi-scale temporal feature extraction
    - Output projection and normalization for asset scoring
    """
    def __init__(self, num_assets, in_features, hidden_dim, seq_len, 
                 num_layers=4, top_k=3, d_ff=None, num_kernels=6, dropout=0.1):
        super(TimesNetASU, self).__init__()
        
        self.num_assets = num_assets
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if d_ff is None:
            d_ff = hidden_dim * 4
            
        # Input embedding layer to map asset features to model dimension
        self.asset_embedding = nn.Linear(in_features, hidden_dim)
        
        # Stack of TimesNet blocks for hierarchical feature extraction
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff, num_kernels)
            for _ in range(num_layers)
        ])
        
        # Output layers for asset score generation
        self.norm = nn.LayerNorm(hidden_dim)
        self.projection = nn.Linear(hidden_dim, 1)
        self.dropout = nn.Dropout(dropout)
        
        # Batch normalization for stable asset scoring
        self.bn_output = nn.BatchNorm1d(num_features=num_assets)
        
    def forward(self, inputs, mask):
        """
        Forward pass for asset selection scoring.
        
        Args:
            inputs: Asset time series [batch, num_assets, seq_len, in_features]
            mask: Invalid asset mask [batch, num_assets] - True for invalid assets
        
        Returns:
            Asset selection scores [batch, num_assets]
        """
        B, N, T, F = inputs.shape
        
        # Reshape for parallel processing of all assets: [B*N, T, F]
        x = inputs.reshape(B * N, T, F)
        
        # Embedding layer to map to model dimension
        x = self.asset_embedding(x)  # [B*N, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply stack of TimesNet blocks for hierarchical feature extraction
        for block in self.blocks:
            x = block(x)
            
        # Global temporal pooling and projection to scalar score
        x = self.norm(x.mean(dim=1))  # [B*N, hidden_dim]
        x = self.projection(x).squeeze(-1)  # [B*N]
        
        # Reshape back to asset dimension: [B, N]
        scores = x.reshape(B, N)
        
        # Apply batch normalization for stable training
        scores = self.bn_output(scores)
        
        # Convert to probabilities using sigmoid activation
        scores = torch.sigmoid(scores)
        
        # Apply mask to exclude invalid assets (delisted, suspended, etc.)
        if mask is not None:
            scores = scores.clone()  # Avoid in-place operation for gradient safety
            scores[mask] = -math.inf
            
        return scores