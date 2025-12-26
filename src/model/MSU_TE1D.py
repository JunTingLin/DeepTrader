"""
MSU_TE1D (Market State Unit with Transformer Encoder 1D)

Uses pre-trained Transformer Encoder (TE_1D) from Stage 1 + custom MLP head.

Architecture:
- TE_1D encoder: Transformer-based temporal feature extraction
- MLP prediction head: Output rho (market trend parameter)

This model can load pre-trained TE_1D weights from Stage 1 PMSU training.
"""

import torch
import torch.nn as nn
from model.TE import TE_1D


class MSU_TE1D(nn.Module):
    """
    MSU with Transformer Encoder (TE_1D) + MLP Head

    Architecture:
        - TE_1D: Transformer encoder (can be pretrained from Stage 1)
        - MLP Head: Predict rho from temporal embeddings

    Args:
        in_features (int): Number of input features (e.g., 27 for market data)
        window_len (int): Sequence length in weeks (e.g., 13)
        hidden_dim (int): Transformer hidden dimension (default: 128)
        depth (int): Number of transformer layers (default: 2)
        heads (int): Number of attention heads (default: 4)
        mlp_dim (int): MLP dimension in transformer (default: 32)
        dim_head (int): Dimension per attention head (default: 4)
        dropout (float): Dropout rate (default: 0.1)
        emb_dropout (float): Embedding dropout rate (default: 0.1)
        mlp_hidden_dim (int): Hidden dimension for prediction MLP (default: 128)

    Input:
        X: [batch_size, window_len, in_features]

    Output:
        rho: [batch_size] - Predicted market trend parameter
    """

    def __init__(
        self,
        in_features,
        window_len=13,
        hidden_dim=128,
        depth=2,
        heads=4,
        mlp_dim=32,
        dim_head=4,
        dropout=0.1,
        emb_dropout=0.1,
        mlp_hidden_dim=128
    ):
        super(MSU_TE1D, self).__init__()
        self.in_features = in_features
        self.window_len = window_len
        self.hidden_dim = hidden_dim

        # ===== Transformer Encoder (TE_1D) =====
        self.TE_1D = TE_1D(
            window_len=window_len,
            dim=hidden_dim,
            depth=depth,
            heads=heads,
            mlp_dim=mlp_dim,
            channels=in_features,
            dim_head=dim_head,
            dropout=dropout,
            emb_dropout=emb_dropout
        )

        # ===== Temporal Attention Pooling =====
        # Use the EXACT SAME 2-layer attention as MSU_LSTM
        # This uses both sequence embeddings and a global context vector
        self.attn1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.attn2 = nn.Linear(hidden_dim, 1)

        # ===== MLP Prediction Head =====
        # After attention pooling, predict rho
        # Use BatchNorm like MSU_LSTM for better training stability
        self.linear1 = nn.Linear(hidden_dim, mlp_hidden_dim)
        self.bn1 = nn.BatchNorm1d(mlp_hidden_dim)
        self.dropout_layer = nn.Dropout(dropout)
        self.linear2 = nn.Linear(mlp_hidden_dim, 1)  # Output: rho (single value)

    def forward(self, X):
        """
        Forward pass

        Args:
            X: [batch_size, window_len, in_features]

        Returns:
            rho: [batch_size] - Predicted market trend parameter
        """
        # ===== Encoder =====
        encoded = self.TE_1D(X)  # [batch_size, window_len, hidden_dim]

        # ===== Temporal Attention Pooling =====
        # Use SAME 2-layer attention as MSU_LSTM
        # Create global context (mean pooling as context)
        global_context = encoded.mean(dim=1, keepdim=True).expand(-1, self.window_len, -1)  # [batch, window_len, hidden_dim]
        concat_features = torch.cat([encoded, global_context], dim=2)  # [batch, window_len, 2*hidden_dim]

        # 2-layer attention network
        attn_scores = self.attn2(torch.tanh(self.attn1(concat_features)))  # [batch, window_len, 1]
        attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)  # [batch, window_len]

        # Apply attention weights
        pooled = torch.bmm(attn_weights.unsqueeze(1), encoded).squeeze(1)  # [batch_size, hidden_dim]

        # ===== Prediction head =====
        # Use BatchNorm like MSU_LSTM
        embed = torch.relu(self.bn1(self.linear1(pooled)))  # [batch_size, mlp_hidden_dim]
        embed = self.dropout_layer(embed)
        rho = self.linear2(embed).squeeze(-1)  # [batch_size]

        return rho

    def get_attention_weights(self, X):
        """
        Get temporal attention weights for visualization

        Args:
            X: [batch_size, window_len, in_features]

        Returns:
            attn_weights: [batch_size, window_len] - Attention weights over time
        """
        with torch.no_grad():
            encoded = self.TE_1D(X)  # [batch_size, window_len, hidden_dim]
            global_context = encoded.mean(dim=1, keepdim=True).expand(-1, self.window_len, -1)
            concat_features = torch.cat([encoded, global_context], dim=2)
            attn_scores = self.attn2(torch.tanh(self.attn1(concat_features)))
            attn_weights = torch.softmax(attn_scores.squeeze(-1), dim=1)
        return attn_weights

    def load_pretrained_encoder(self, checkpoint_path):
        """
        Load pre-trained TE_1D encoder from Stage 1 PMSU checkpoint

        Args:
            checkpoint_path (str): Path to Stage 1 PMSU checkpoint (.pth file)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Extract TE_1D state dict from Stage 1 PMSU
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint

        # Filter TE_1D weights
        te1d_state_dict = {k.replace('TE_1D.', ''): v for k, v in state_dict.items() if k.startswith('TE_1D.')}

        # Load into TE_1D encoder
        self.TE_1D.load_state_dict(te1d_state_dict, strict=True)
        print(f"✅ Loaded pre-trained TE_1D encoder from: {checkpoint_path}")
        print(f"   Loaded {len(te1d_state_dict)} parameter tensors")


if __name__ == '__main__':
    # Test the model
    print("Testing MSU_TE1D...")

    batch_size = 16
    window_len = 13
    in_features = 27

    # Create dummy input
    X = torch.randn(batch_size, window_len, in_features)

    # Create model
    model = MSU_TE1D(
        in_features=in_features,
        window_len=window_len,
        hidden_dim=128,
        depth=2,
        heads=4,
        mlp_dim=32,
        dim_head=4,
        dropout=0.1,
        emb_dropout=0.1,
        mlp_hidden_dim=128
    )

    # Forward pass
    rho = model(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {rho.shape}")
    print(f"Output (rho): {rho[:5]}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    te1d_params = sum(p.numel() for p in model.TE_1D.parameters())
    mlp_params = sum(p.numel() for p in model.mlp_head.parameters())

    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"TE_1D parameters: {te1d_params:,}")
    print(f"MLP head parameters: {mlp_params:,}")
