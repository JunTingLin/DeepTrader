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


class TimesNetMSU(nn.Module):
    """
    TimesNet-based Market State Unit (replaces original LSTM-based MSU).
    Uses temporal 2D-variation modeling for market state prediction.
    
    Architecture:
    - Input embedding to map market features to hidden dimension
    - Multiple TimesBlocks for multi-scale temporal feature extraction
    - Attention mechanism for temporal aggregation (similar to original MSU)
    - Output projection for market exposure distribution parameters
    """
    def __init__(self, in_features, seq_len, hidden_dim, num_layers=2, 
                 top_k=3, d_ff=None, num_kernels=6, dropout=0.1):
        super(TimesNetMSU, self).__init__()
        
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if d_ff is None:
            d_ff = hidden_dim * 4
            
        # Input embedding layer to map market features to model dimension
        self.embedding = nn.Linear(in_features, hidden_dim)
        
        # Stack of TimesNet blocks for hierarchical feature extraction
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff, num_kernels)
            for _ in range(num_layers)
        ])
        
        # Attention mechanism for temporal aggregation (preserving original MSU design)
        self.attn1 = nn.Linear(hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)
        
        # Output layers for market exposure parameter generation
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)  # mu, sigma for market exposure distribution
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        """
        Forward pass for market state parameter prediction.
        
        Args:
            X: Market time series [batch_size, seq_len, in_features]
        
        Returns:
            Market exposure parameters [batch_size, 2] - (mu, sigma) for distribution
        """
        B, T, F = X.shape
        
        # Embedding layer to map to model dimension
        x = self.embedding(X)  # [B, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply stack of TimesNet blocks for hierarchical feature extraction
        for block in self.blocks:
            x = block(x)
            
        # Attention mechanism for temporal aggregation (preserving original MSU design)
        attn_scores = self.attn2(torch.tanh(self.attn1(x)))  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)  # [B, T]
        
        # Weighted aggregation across time dimension
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, hidden_dim]
        
        # Output projection for market exposure distribution parameters
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        # Note: dropout commented out to match original MSU implementation
        # embed = self.dropout(embed)  
        parameters = self.linear2(embed)  # [B, 2] -> (mu, sigma)
        
        return parameters


if __name__ == '__main__':
    # Test the model with default parameters
    batch_size = 16
    seq_len = 13
    in_features = 4
    hidden_dim = 128
    
    model = TimesNetMSU(in_features, seq_len, hidden_dim, num_kernels=6)
    x = torch.randn(batch_size, seq_len, in_features)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("TimesNetMSU test passed!")