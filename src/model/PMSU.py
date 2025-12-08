"""
PMSU (Pretrain MSU) - Masked Reconstruction

Pretrain version of MSU for Stage 1: Self-supervised masked reconstruction.

Architecture:
- Encoder (TE_1D): Will be transferred to Stage 2 MSU
- Decoder: Will be discarded in Stage 2

Task: Reconstruct masked market features
"""

import torch
import torch.nn as nn
from model.TE import TE_1D


class PMSU(nn.Module):
    """
    PMSU: Pretrain MSU for Stage 1 (Masked Reconstruction)

    Architecture:
        - Encoder: TE_1D transformer (will be transferred to Stage 2)
        - Decoder: Reconstruction head (will be discarded in Stage 2)

    Task: Given masked market data [13, 27], reconstruct original values

    Only requires in_features to be specified (e.g., 27 for market data).
    """
    def __init__(self, in_features):
        super().__init__()

        # Fixed hyperparameters
        self.in_features = in_features
        self.window_len = 13
        self.hidden_dim = 128

        # ===== ENCODER (will be transferred to Stage 2) =====
        self.TE_1D = TE_1D(
            window_len=13,
            dim=128,
            depth=2,
            heads=4,
            mlp_dim=32,
            channels=in_features,
            dim_head=4,
            dropout=0.1,
            emb_dropout=0.1
        )

        # ===== DECODER (will be discarded in Stage 2) =====
        # Reconstruct each timestep independently
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, in_features)  # Output: 27 features
        )

    def forward(self, X):
        """
        Forward pass

        Args:
            X: [batch, 13, in_features] - Masked input sequence

        Returns:
            reconstructed: [batch, 13, in_features] - Reconstructed sequence
        """
        # ===== Encoder =====
        encoded = self.TE_1D(X)  # [batch, 13, 128]

        # ===== Decoder =====
        # Decode each timestep independently
        batch_size, seq_len, hidden_dim = encoded.shape

        # Reshape to process all timesteps at once
        encoded_flat = encoded.reshape(batch_size * seq_len, hidden_dim)  # [batch*13, 128]
        decoded_flat = self.decoder(encoded_flat)  # [batch*13, in_features]

        # Reshape back
        reconstructed = decoded_flat.reshape(batch_size, seq_len, self.in_features)  # [batch, 13, in_features]

        return reconstructed

    def get_encoder_state_dict(self):
        """
        Get encoder state dict for transfer to Stage 2 MSU

        Returns:
            dict: State dict containing only encoder (TE_1D)
        """
        return {
            'TE_1D': self.TE_1D.state_dict()
        }
