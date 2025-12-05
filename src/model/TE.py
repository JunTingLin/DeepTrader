import math
import torch
from torch import nn, einsum
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def get_sinusoidal_encoding(seq_len, d_model, device='cpu'):
    """
    Generate sinusoidal positional encoding (fixed, not learnable).

    Uses sin/cos functions as in "Attention Is All You Need" (Vaswani et al., 2017).
    This allows the model to extrapolate to sequence lengths longer than those seen during training.

    Args:
        seq_len: Length of the sequence
        d_model: Dimension of the model (must be even)
        device: Device to create the tensor on

    Returns:
        Positional encoding tensor of shape [seq_len, d_model]
    """
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32, device=device) *
                         -(math.log(10000.0) / d_model))

    pe = torch.zeros(seq_len, d_model, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)

    return pe

# classes


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = self.attend(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads,
                                       dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * \
            (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {
            'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, img):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class TE(nn.Module):
    """
    Temporal Encoder for 2D data (assets × time).

    Key design:
    - Time dimension (w): Uses sinusoidal positional encoding (has order)
    - Asset dimension (h): No positional encoding (permutation invariant set)

    This preserves permutation invariance over assets while maintaining
    temporal order information.

    Input: [batch, channels, num_assets, time_len]
    Output: [batch, channels, num_assets, time_len]
    """
    def __init__(self, *, image_size, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # image_size = (num_assets, time_len)
        # h = num_assets (no positional order)
        # w = time_len (has temporal order)

        self.h = image_size[0]  # num_assets
        self.w = image_size[1]  # time_len
        self.dim = dim

        patch_dim = channels
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c h w -> b h w c'),  # [batch, assets, time, channels]
            nn.Linear(patch_dim, dim),
        )

        # Sinusoidal PE for time dimension only (not assets!)
        # Shape: [1, 1, time_len, dim] for broadcasting over [batch, assets, time, dim]
        time_pe = get_sinusoidal_encoding(self.w, dim)  # [time_len, dim]
        self.register_buffer('time_pos_embedding', time_pe.unsqueeze(0).unsqueeze(0))  # [1, 1, time_len, dim]

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        # img: [batch, channels, num_assets, time_len]
        x = self.to_patch_embedding(img)  # [batch, assets, time, dim]
        b, h, w, d = x.shape

        # Add positional encoding ONLY for time dimension
        # Assets have no positional encoding (permutation invariant)
        if w > self.w:
            # If time length exceeds training length, generate new PE on the fly
            time_pe = get_sinusoidal_encoding(w, self.dim, device=x.device).unsqueeze(0).unsqueeze(0)
            x = x + time_pe  # [batch, assets, time, dim] + [1, 1, time, dim]
        else:
            x = x + self.time_pos_embedding[:, :, :w, :]  # Broadcast over batch and assets

        x = self.dropout(x)

        # Flatten (assets, time) for transformer
        x = rearrange(x, 'b h w c -> b (h w) c')  # [batch, assets*time, dim]
        x = self.transformer(x)

        # Reshape back to 2D
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)  # [batch, dim, assets, time]

        return x


class TE_1D(nn.Module):
    """
    Temporal Encoder for 1D sequences (time series).

    Uses sinusoidal positional encoding for time dimension (not learnable).
    This allows extrapolation to longer sequences than seen during training.

    Input: [batch, sequence_length, channels]
    Output: [batch, sequence_length, dim]
    """
    def __init__(self, *, window_len=13, dim, depth, heads, mlp_dim, channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        # Expected input : (b, l, c) where b=batch, l=sequence_length, c=channels
        self.dim = dim
        self.window_len = window_len

        patch_dim = channels
        self.to_patch_embedding = nn.Sequential(
            nn.Linear(patch_dim, dim),
        )

        # Use sinusoidal PE instead of learnable PE for time dimension
        # Registered as buffer (not a parameter) so it won't be trained
        self.register_buffer('pos_embedding',
                            get_sinusoidal_encoding(window_len, dim).unsqueeze(0))  # [1, L, dim]

        self.dropout = nn.Dropout(emb_dropout)
        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

    def forward(self, img):
        # img: [B, L, C] where B=batch, L=sequence_length, C=channels
        x = self.to_patch_embedding(img)  # [B, L, dim]
        b, n, _ = x.shape

        # Add sinusoidal positional encoding
        # Support variable length sequences (up to window_len)
        if n > self.window_len:
            # If sequence is longer than training length, generate new PE on the fly
            pos_enc = get_sinusoidal_encoding(n, self.dim, device=x.device).unsqueeze(0)
            x = x + pos_enc
        else:
            x = x + self.pos_embedding[:, :n, :]  # [B, L, dim] + [1, L, dim] -> [B, L, dim]

        x = self.dropout(x)
        x = self.transformer(x)  # Self-attention over L (sequence dimension)

        return x
