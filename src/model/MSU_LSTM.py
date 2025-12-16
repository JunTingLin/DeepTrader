"""
MSU_LSTM (Market State Unit with LSTM + Temporal Attention)

Original DeepTrader MSU architecture for market trend (rho) prediction.

Architecture:
- LSTM encoder for temporal feature extraction
- Temporal attention mechanism to focus on important time steps
- Prediction head for rho (market trend parameter)

This is designed for smaller datasets compared to Transformer-based models.
"""

import torch
import torch.nn as nn


class MSU_LSTM(nn.Module):
    """
    MSU with LSTM + Temporal Attention

    Architecture:
        - LSTM: Temporal encoder
        - Temporal Attention: Focus on important time steps
        - Prediction Head: Output rho (market trend)

    Args:
        in_features (int): Number of input features (e.g., 27 for market data)
        window_len (int): Sequence length in weeks (e.g., 13)
        hidden_dim (int): LSTM hidden dimension (default: 128)
        dropout (float): Dropout rate for regularization (default: 0.5)

    Input:
        X: [batch_size, window_len, in_features]

    Output:
        rho: [batch_size] - Predicted market trend parameter
    """

    def __init__(self, in_features, window_len, hidden_dim=128, dropout=0.5):
        super(MSU_LSTM, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout

        # LSTM encoder
        self.lstm = nn.LSTM(
            input_size=in_features,
            hidden_size=hidden_dim,
            batch_first=False  # Expects [seq_len, batch, features]
        )

        # Temporal attention
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        # Prediction head
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, 1)  # Output: rho (single value)

    def forward(self, X):
        """
        Forward pass

        Args:
            X: [batch_size, window_len, in_features]

        Returns:
            rho: [batch_size] - Predicted market trend parameter
        """
        batch_size = X.size(0)

        # LSTM expects [seq_len, batch, features]
        X = X.permute(1, 0, 2)  # [window_len, batch_size, in_features]

        # LSTM encoding
        outputs, (h_n, c_n) = self.lstm(X)
        # outputs: [window_len, batch_size, hidden_dim]
        # h_n: [1, batch_size, hidden_dim]

        # Temporal attention
        # Repeat h_n for all time steps
        H_n = h_n.repeat(self.window_len, 1, 1)  # [window_len, batch_size, hidden_dim]

        # Compute attention scores
        concat_features = torch.cat([outputs, H_n], dim=2)  # [window_len, batch_size, 2*hidden_dim]
        scores = self.attn2(torch.tanh(self.attn1(concat_features)))  # [window_len, batch_size, 1]
        scores = scores.squeeze(2).transpose(1, 0)  # [batch_size, window_len]

        # Apply softmax to get attention weights
        attn_weights = torch.softmax(scores, dim=1)  # [batch_size, window_len]

        # Apply attention weights to outputs
        outputs = outputs.permute(1, 0, 2)  # [batch_size, window_len, hidden_dim]
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # [batch_size, hidden_dim]

        # Prediction head
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))  # [batch_size, hidden_dim]
        embed = self.dropout(embed)
        rho = self.linear2(embed).squeeze(-1)  # [batch_size]

        return rho

    def get_attention_weights(self, X):
        """
        Get attention weights for visualization

        Args:
            X: [batch_size, window_len, in_features]

        Returns:
            attn_weights: [batch_size, window_len]
        """
        with torch.no_grad():
            X = X.permute(1, 0, 2)
            outputs, (h_n, c_n) = self.lstm(X)
            H_n = h_n.repeat(self.window_len, 1, 1)
            concat_features = torch.cat([outputs, H_n], dim=2)
            scores = self.attn2(torch.tanh(self.attn1(concat_features)))
            scores = scores.squeeze(2).transpose(1, 0)
            attn_weights = torch.softmax(scores, dim=1)
        return attn_weights


if __name__ == '__main__':
    # Test the model
    print("Testing MSU_LSTM...")

    batch_size = 16
    window_len = 13
    in_features = 27

    # Create dummy input
    X = torch.randn(batch_size, window_len, in_features)

    # Create model
    model = MSU_LSTM(in_features=in_features, window_len=window_len, hidden_dim=128, dropout=0.5)

    # Forward pass
    rho = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {rho.shape}")
    print(f"Output (rho): {rho[:5]}")

    # Test attention weights
    attn_weights = model.get_attention_weights(X)
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"Attention weights sum: {attn_weights.sum(dim=1)[:5]}")  # Should be close to 1.0

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
