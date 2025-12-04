"""
PMSU (Pretrain MSU)

Pretrain version of MSU for Stage 1: Binary trend prediction (0 or 1).
Same architecture as MSU, but with a binary classification head.
"""

import torch
import torch.nn as nn
from model.TE import TE_1D


class PMSU(nn.Module):
    """
    PMSU: Pretrain MSU for Stage 1 (binary trend prediction)

    Fixed architecture:
        - window_len: 13
        - hidden_dim: 128
        - Transformer encoder
        - Temporal attention
        - Binary classification head (output: 0 or 1)

    Only requires in_features to be specified (e.g., 1 for fake data).
    """
    def __init__(self, in_features):
        super().__init__()

        # Fixed hyperparameters
        self.in_features = in_features
        self.window_len = 13
        self.hidden_dim = 128

        # ===== Transformer Encoder =====
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

        # ===== Temporal Attention =====
        self.attn1 = nn.Linear(128, 128)
        self.attn2 = nn.Linear(128, 1)

        # ===== Middle Layer =====
        self.linear1 = nn.Linear(128, 128)
        self.bn1 = nn.BatchNorm1d(128)

        # ===== Binary Classification Head =====
        self.head = nn.Linear(128, 1)

    def forward(self, X):
        """
        Forward pass

        Args:
            X: [batch, 13, in_features] - Input sequence

        Returns:
            output: [batch, 1] - Binary logits (before sigmoid)
        """
        # X is already [batch, 13, in_features] - correct format for TE_1D

        # ===== Encoder =====
        outputs = self.TE_1D(X)  # [batch, 13, 128]

        # ===== Temporal Attention =====
        scores = self.attn2(torch.tanh(self.attn1(outputs)))  # [batch, 13, 1]
        scores = scores.squeeze(2)  # [batch, 13]
        attn_weights = torch.softmax(scores, dim=1)  # [batch, 13]

        # Weighted sum
        attn_embed = torch.bmm(attn_weights.unsqueeze(1), outputs).squeeze(1)  # [batch, 128]

        # ===== Middle Layer =====
        embed = torch.relu(self.bn1(self.linear1(attn_embed)))  # [batch, 128]

        # ===== Binary Classification Head =====
        output = self.head(embed)  # [batch, 1]

        return output
