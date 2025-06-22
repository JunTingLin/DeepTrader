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


class TimesNetMSU(nn.Module):
    """
    TimesNet-based Market State Unit (replaces original LSTM-based MSU)
    """
    def __init__(self, in_features, seq_len, hidden_dim, num_layers=2, 
                 top_k=3, d_ff=None, dropout=0.1):
        super(TimesNetMSU, self).__init__()
        
        self.in_features = in_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        if d_ff is None:
            d_ff = hidden_dim * 4
            
        # Input embedding
        self.embedding = nn.Linear(in_features, hidden_dim)
        
        # TimesNet blocks
        self.blocks = nn.ModuleList([
            TimesBlock(seq_len, 0, top_k, hidden_dim, d_ff)
            for _ in range(num_layers)
        ])
        
        # Attention mechanism (similar to original MSU)
        self.attn1 = nn.Linear(hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)  # mu, sigma for market exposure
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X):
        """
        X: [batch_size, seq_len, in_features]
        return: [batch_size, 2] - parameters for market exposure distribution
        """
        B, T, F = X.shape
        
        # Embedding
        x = self.embedding(X)  # [B, T, hidden_dim]
        x = self.dropout(x)
        
        # Apply TimesNet blocks
        for block in self.blocks:
            x = block(x)
            
        # Attention mechanism (similar to original MSU)
        attn_scores = self.attn2(torch.tanh(self.attn1(x)))  # [B, T, 1]
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)  # [B, T]
        
        # Weighted aggregation
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), x).squeeze(1)  # [B, hidden_dim]
        
        # Output projection
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))
        # embed = self.dropout(embed)  # Commented out like in original
        parameters = self.linear2(embed)  # [B, 2]
        
        return parameters


if __name__ == '__main__':
    # Test the model
    batch_size = 16
    seq_len = 13
    in_features = 4
    hidden_dim = 128
    
    model = TimesNetMSU(in_features, seq_len, hidden_dim)
    x = torch.randn(batch_size, seq_len, in_features)
    output = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print("TimesNetMSU test passed!")