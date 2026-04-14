"""
News Encoder and Fusion modules for integrating news sentiment into DeepTrader.

This module provides:
1. NewsEncoder: Projects CLS embeddings to hidden dimension
2. Four fusion methods: concat, add, gate, cross_attn
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class NewsEncoder(nn.Module):
    """
    Encodes news CLS embeddings by projecting to the model's hidden dimension.

    Input: (batch, num_stocks, window_len, embedding_dim) - e.g., (32, 10, 13, 768)
    Output: (batch, num_stocks, hidden_dim) - e.g., (32, 10, 128)
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        window_len: int = 13,
        dropout: float = 0.1,
        aggregation: str = 'mean'  # 'mean', 'last', 'attention'
    ):
        super(NewsEncoder, self).__init__()

        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.window_len = window_len
        self.aggregation = aggregation

        # Project from embedding_dim to hidden_dim
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # For attention aggregation
        if aggregation == 'attention':
            self.time_attn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.Tanh(),
                nn.Linear(hidden_dim // 4, 1)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, num_stocks, window_len, embedding_dim)
        Returns:
            (batch, num_stocks, hidden_dim)
        """
        batch, num_stocks, window_len, emb_dim = x.shape

        # Reshape for projection
        x = x.view(batch * num_stocks * window_len, emb_dim)
        x = self.projection(x)
        x = x.view(batch, num_stocks, window_len, self.hidden_dim)

        # Aggregate over time dimension
        if self.aggregation == 'mean':
            x = x.mean(dim=2)  # (batch, num_stocks, hidden_dim)
        elif self.aggregation == 'last':
            x = x[:, :, -1, :]  # (batch, num_stocks, hidden_dim)
        elif self.aggregation == 'attention':
            # Compute attention weights over time
            attn_scores = self.time_attn(x).squeeze(-1)  # (batch, num_stocks, window_len)
            attn_weights = F.softmax(attn_scores, dim=-1).unsqueeze(-1)  # (batch, num_stocks, window_len, 1)
            x = (x * attn_weights).sum(dim=2)  # (batch, num_stocks, hidden_dim)

        return x


class FusionConcat(nn.Module):
    """
    Concatenation fusion: concat ASU and News features, then project back.

    ASU: (batch, num_stocks, hidden_dim)
    News: (batch, num_stocks, hidden_dim)
    Output: (batch, num_stocks, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super(FusionConcat, self).__init__()

        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, asu_out: torch.Tensor, news_out: torch.Tensor, return_gate: bool = False):
        """
        Args:
            asu_out: (batch, num_stocks, hidden_dim)
            news_out: (batch, num_stocks, hidden_dim)
            return_gate: ignored for concat fusion (no gate)
        Returns:
            if return_gate: (fused, None)
            else: fused only
        """
        combined = torch.cat([asu_out, news_out], dim=-1)  # (batch, num_stocks, hidden_dim*2)
        fused = self.fusion(combined)
        if return_gate:
            return fused, None
        return fused


class FusionAdd(nn.Module):
    """
    Addition fusion: simply add ASU and News features.

    ASU: (batch, num_stocks, hidden_dim)
    News: (batch, num_stocks, hidden_dim)
    Output: (batch, num_stocks, hidden_dim)
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super(FusionAdd, self).__init__()

        # Optional: layer norm after addition
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, asu_out: torch.Tensor, news_out: torch.Tensor, return_gate: bool = False):
        """
        Args:
            asu_out: (batch, num_stocks, hidden_dim)
            news_out: (batch, num_stocks, hidden_dim)
            return_gate: ignored for add fusion (no gate)
        Returns:
            if return_gate: (fused, None)
            else: fused only
        """
        fused = asu_out + news_out
        fused = self.norm(fused)
        fused = self.dropout(fused)
        if return_gate:
            return fused, None
        return fused


class FusionGate(nn.Module):
    """
    Gating fusion: learn a gate to weight ASU vs News contribution.

    gate = sigmoid(W * [asu, news])
    output = gate * asu + (1 - gate) * news

    Gate interpretation:
    - gate > 0.5 → more weight on ASU (price/volume features)
    - gate < 0.5 → more weight on News features
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super(FusionGate, self).__init__()

        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, asu_out: torch.Tensor, news_out: torch.Tensor, return_gate: bool = False):
        """
        Args:
            asu_out: (batch, num_stocks, hidden_dim)
            news_out: (batch, num_stocks, hidden_dim)
            return_gate: if True, also return gate values averaged over hidden_dim
        Returns:
            if return_gate:
                (fused, gate_values) where gate_values is (batch, num_stocks)
            else:
                fused only
        """
        combined = torch.cat([asu_out, news_out], dim=-1)  # (batch, num_stocks, hidden_dim*2)
        gate = self.gate_net(combined)  # (batch, num_stocks, hidden_dim)

        fused = gate * asu_out + (1 - gate) * news_out
        fused = self.norm(fused)

        if return_gate:
            # Average over hidden_dim to get (batch, num_stocks)
            gate_values = gate.mean(dim=-1)
            return fused, gate_values
        return fused


class FusionGMU(nn.Module):
    """
    Original Gated Multimodal Unit (GMU) from Arevalo et al. (2017).
    "Gated Multimodal Units for Information Fusion", arXiv:1702.01992.

    Gate:   z = σ(W_z [v; n])               — single linear layer
    Output: h = z ⊙ tanh(W_v v) + (1-z) ⊙ tanh(W_n n)

    Difference from FusionGate:
    - Gate is a single Linear (no hidden layer, no LN/GELU)
    - Both modalities are passed through tanh projections before gating
    """

    def __init__(self, hidden_dim: int = 128, dropout: float = 0.1):
        super(FusionGMU, self).__init__()

        # Single linear gate: [v; n] → z  (original GMU, no hidden layer)
        self.gate = nn.Linear(hidden_dim * 2, hidden_dim)

        # Tanh projections for each modality (original GMU)
        self.proj_v = nn.Linear(hidden_dim, hidden_dim)
        self.proj_n = nn.Linear(hidden_dim, hidden_dim)

        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, asu_out: torch.Tensor, news_out: torch.Tensor, return_gate: bool = False):
        """
        Args:
            asu_out:  (batch, num_stocks, hidden_dim) — price/volume features (v)
            news_out: (batch, num_stocks, hidden_dim) — news features (n)
            return_gate: if True, also return gate values averaged over hidden_dim
        Returns:
            if return_gate: (fused, gate_values) where gate_values is (batch, num_stocks)
            else: fused only
        """
        combined = torch.cat([asu_out, news_out], dim=-1)  # (batch, num_stocks, hidden_dim*2)
        z = torch.sigmoid(self.gate(combined))              # (batch, num_stocks, hidden_dim)

        h = z * torch.tanh(self.proj_v(asu_out)) + (1 - z) * torch.tanh(self.proj_n(news_out))
        h = self.norm(h)

        if return_gate:
            gate_values = z.mean(dim=-1)  # (batch, num_stocks)
            return h, gate_values
        return h


class FusionCrossAttention(nn.Module):
    """
    Cross-attention fusion: News attends to ASU features.

    Q from News, K/V from ASU.
    Output is News + Attention(News -> ASU)
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super(FusionCrossAttention, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        # Q, K, V projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # FFN for post-attention
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.Dropout(dropout)
        )

    def forward(self, asu_out: torch.Tensor, news_out: torch.Tensor, return_gate: bool = False):
        """
        Args:
            asu_out: (batch, num_stocks, hidden_dim) - K, V
            news_out: (batch, num_stocks, hidden_dim) - Q
            return_gate: ignored for cross_attn fusion (no gate)
        Returns:
            if return_gate: (fused, None)
            else: fused only
        """
        batch, num_stocks, _ = news_out.shape

        # Q from news, K/V from ASU
        Q = self.q_proj(news_out)  # (batch, num_stocks, hidden_dim)
        K = self.k_proj(asu_out)
        V = self.v_proj(asu_out)

        # Reshape for multi-head attention
        Q = Q.view(batch, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch, num_stocks, self.num_heads, self.head_dim).transpose(1, 2)
        # Q, K, V: (batch, num_heads, num_stocks, head_dim)

        # Attention
        scale = math.sqrt(self.head_dim)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / scale  # (batch, num_heads, num_stocks, num_stocks)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)  # (batch, num_heads, num_stocks, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch, num_stocks, self.hidden_dim)
        attn_output = self.out_proj(attn_output)

        # Residual connection
        x = self.norm1(news_out + attn_output)

        # FFN with residual
        x = self.norm2(x + self.ffn(x))

        if return_gate:
            return x, None
        return x


def get_fusion_module(
    fusion_method: str,
    hidden_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.1
) -> nn.Module:
    """
    Factory function to get the appropriate fusion module.

    Args:
        fusion_method: One of 'concat', 'add', 'gate', 'cross_attn'
        hidden_dim: Hidden dimension
        num_heads: Number of heads for cross-attention
        dropout: Dropout rate

    Returns:
        Fusion module
    """
    fusion_methods = {
        'concat': FusionConcat,
        'add': FusionAdd,
        'gmu': FusionGMU,       # original GMU (Arevalo et al., 2017)
        'gate': FusionGate,     # enhanced GMU (2-layer MLP + LN + GELU)
        'cross_attn': FusionCrossAttention,
    }

    if fusion_method not in fusion_methods:
        raise ValueError(f"Unknown fusion method: {fusion_method}. "
                         f"Choose from {list(fusion_methods.keys())}")

    if fusion_method == 'cross_attn':
        return FusionCrossAttention(hidden_dim=hidden_dim, num_heads=num_heads, dropout=dropout)
    else:
        return fusion_methods[fusion_method](hidden_dim=hidden_dim, dropout=dropout)



class NewsIntegrationModule(nn.Module):
    """
    Complete news integration module that combines NewsEncoder and Fusion.

    This module:
    1. Encodes CLS embeddings using NewsEncoder
    2. Fuses with ASU output using the specified fusion method
    """

    def __init__(
        self,
        embedding_dim: int = 768,
        hidden_dim: int = 128,
        window_len: int = 13,
        fusion_method: str = 'concat',
        num_heads: int = 4,
        dropout: float = 0.1,
        aggregation: str = 'mean'
    ):
        super(NewsIntegrationModule, self).__init__()

        self.news_encoder = NewsEncoder(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            window_len=window_len,
            dropout=dropout,
            aggregation=aggregation
        )

        self.fusion = get_fusion_module(
            fusion_method=fusion_method,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )

        self.fusion_method = fusion_method

    def forward(
        self,
        asu_out: torch.Tensor,
        news_embeddings: torch.Tensor,
        return_gate: bool = False
    ):
        """
        Args:
            asu_out: (batch, num_stocks, hidden_dim) - Output from ASU's SAGCN
            news_embeddings: (batch, num_stocks, window_len, embedding_dim) - CLS embeddings
            return_gate: if True and fusion_method='gate', also return gate values

        Returns:
            if return_gate:
                (fused, gate_values) where gate_values is (batch, num_stocks) or None
            else:
                fused only
        """
        # Encode news embeddings
        news_encoded = self.news_encoder(news_embeddings)  # (batch, num_stocks, hidden_dim)

        # Fuse with ASU output
        if return_gate:
            fused, gate_values = self.fusion(asu_out, news_encoded, return_gate=True)
            return fused, gate_values
        else:
            fused = self.fusion(asu_out, news_encoded)
            return fused


if __name__ == "__main__":
    # Test the modules
    batch_size = 32
    num_stocks = 10
    window_len = 13
    embedding_dim = 768
    hidden_dim = 128

    # Create dummy inputs
    asu_out = torch.randn(batch_size, num_stocks, hidden_dim)
    news_embeddings = torch.randn(batch_size, num_stocks, window_len, embedding_dim)

    print("Testing fusion methods:")
    for fusion_method in ['concat', 'add', 'gmu', 'gate', 'cross_attn']:
        module = NewsIntegrationModule(
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            window_len=window_len,
            fusion_method=fusion_method
        )

        output = module(asu_out, news_embeddings)
        print(f"  {fusion_method}: input shapes = ASU {asu_out.shape}, News {news_embeddings.shape}")
        print(f"           output shape = {output.shape}")

        # Count parameters
        num_params = sum(p.numel() for p in module.parameters())
        print(f"           parameters = {num_params:,}")
        print()
