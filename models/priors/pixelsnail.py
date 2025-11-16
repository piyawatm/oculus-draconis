# models/priors/pixelsnail.py
# Faithful-ish PixelSNAIL prior for VQ-VAE token sequences (2D masked conv + attention)
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior
from typing import Optional

# -------------------------
# Utility layers
# -------------------------
class MaskedConv2d(nn.Conv2d):
    """
    2D masked convolution enforcing autoregressive raster-scan causality.
    mask_type: 'A' (no access to current pixel channels) or 'B' (allow current pixel conditioning)
    """
    def __init__(self, in_ch, out_ch, kernel_size, mask_type='B', stride=1, padding=0, bias=True):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        assert mask_type in ('A', 'B')
        self.register_buffer('mask', self.weight.data.clone())
        self.mask.fill_(1)
        kh, kw = self.kernel_size
        cy, cx = kh // 2, kw // 2

        # mask out positions that are not allowed (future in raster order)
        for i in range(kh):
            for j in range(kw):
                if i > cy or (i == cy and j > cx):
                    self.mask[:, :, i, j] = 0
        if mask_type == 'A':
            # in the center pixel, we must also mask out all input channels in 'A' type
            self.mask[:, :, cy, cx] = 0

    def forward(self, x):
        # apply mask to weights
        self.weight.data *= self.mask
        return super().forward(x)

# -------------------------
# Gated residual block (PixelSNAIL style)
# -------------------------
class GatedResidual2D(nn.Module):
    """
    Gated residual block composed of:
      - 1x1 conv (bottleneck)
      - masked conv (k x k)
      - 1x1 conv to produce 2 * channels (for gated activation)
    gated activation: tanh(a) * sigmoid(b)
    """
    def __init__(self, channels, kernel_size=3, mask_type='B', dropout=0.0):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.masked = MaskedConv2d(channels, channels, kernel_size=(kernel_size, kernel_size), mask_type=mask_type, padding=pad)
        self.conv2 = nn.Conv2d(channels, channels * 2, kernel_size=1)
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.res_scale = nn.Parameter(torch.tensor(1.0))  # optional learned scaling

    def forward(self, x):
        out = self.conv1(x)
        out = F.relu(out)
        out = self.masked(out)
        out = F.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)  # [B, 2C, H, W]
        a, b = out.chunk(2, dim=1)
        gated = torch.tanh(a) * torch.sigmoid(b)
        return x + self.res_scale * gated

# -------------------------
# Attention block (causal over raster order)
# -------------------------
class AttentionBlock2D(nn.Module):
    """
    Self-attention over flattened spatial positions with causal mask (raster order).
    Uses 1x1 convs to compute q/k/v then flattened attention.
    Works well for small H*W (e.g., 8x8).
    """
    def __init__(self, channels, key_dim=64, val_dim=128, n_heads=4):
        super().__init__()
        self.channels = channels
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.n_heads = n_heads

        # project to queries, keys, values
        self.q_proj = nn.Conv2d(channels, n_heads * key_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(channels, n_heads * key_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(channels, n_heads * val_dim, kernel_size=1)
        self.out = nn.Conv2d(n_heads * val_dim, channels, kernel_size=1)
        self.scale = key_dim ** -0.5

    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        N = H * W
        q = self.q_proj(x).reshape(B, self.n_heads, self.key_dim, N)  # [B, heads, kd, N]
        k = self.k_proj(x).reshape(B, self.n_heads, self.key_dim, N)
        v = self.v_proj(x).reshape(B, self.n_heads, self.val_dim, N)

        # transpose to [B, heads, N, dim]
        q = q.permute(0, 1, 3, 2)  # [B, h, N, kd]
        k = k.permute(0, 1, 3, 2)  # [B, h, N, kd]
        v = v.permute(0, 1, 3, 2)  # [B, h, N, vd]

        # scaled dot-product with causal mask
        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B,h,N,N]
        # causal mask: allow j <= i (so position i can attend to positions <= i)
        mask = torch.tril(torch.ones(N, N, device=x.device)).unsqueeze(0).unsqueeze(0)  # [1,1,N,N]
        attn_logits = attn_logits.masked_fill(mask == 0, float("-inf"))
        attn = torch.softmax(attn_logits, dim=-1)  # [B,h,N,N]

        out = torch.matmul(attn, v)  # [B,h,N,vd]
        out = out.permute(0, 1, 3, 2).reshape(B, self.n_heads * self.val_dim, H, W)
        out = self.out(out)
        return x + out  # residual

# -------------------------
# PixelSNAIL block: stack of gated residuals and optional attention
# -------------------------
class PixelSNAILBlock2D(nn.Module):
    def __init__(self, channels, n_residual=2, kernel_size=3, dropout=0.0, attn_every: int = 4,
                 attn_kwargs: Optional[dict] = None, mask_type='B'):
        """
        attn_every: insert attention after every `attn_every` residuals inside this block (0 = none)
        attn_kwargs: dict passed to AttentionBlock2D
        """
        super().__init__()
        self.res_blocks = nn.ModuleList()
        for i in range(n_residual):
            self.res_blocks.append(GatedResidual2D(channels, kernel_size=kernel_size, mask_type=mask_type, dropout=dropout))
        self.attn_every = attn_every
        self.attn_kwargs = attn_kwargs or {}
        # optional attention (only created if attn_every > 0)
        self.attn = AttentionBlock2D(channels, **self.attn_kwargs) if self.attn_every and self.attn_every > 0 else None

    def forward(self, x):
        for i, r in enumerate(self.res_blocks):
            x = r(x)
            # if attention present and this is the designated position, apply it
            if self.attn is not None and ((i + 1) % self.attn_every == 0):
                x = self.attn(x)
        return x

# -------------------------
# PixelSNAIL Prior (2D)
# -------------------------
class PixelSNAILPrior(ARPrior):
    """
    PixelSNAIL-like prior for VQ-VAE codes (2D masked conv + attention).
    - vocab_size: number of discrete codes + BOS
    - height/width: spatial layout of codes (e.g., 8x8)
    - n_layer: number of PixelSNAIL blocks
    """
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 12,
        height: int = 8,
        width: int = 8,
        block_size: int = 64, ## Not using this.
        n_residual: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        attn_every: int = 4,
        attn_key_dim: int = 32,
        attn_val_dim: int = 64,
        attn_heads: int = 4,
    ):
        block_size = height * width + 1  # +1 for BOS
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        self.height = height
        self.width = width
        self.d_model = d_model

        # embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # positional embedding: separate row & col embeddings summed (learned)
        self.row_emb = nn.Embedding(height, d_model)
        self.col_emb = nn.Embedding(width, d_model)
        self.drop = nn.Dropout(dropout)

        # initial conv (type B mask) to mix embeddings into 2D
        self.input_conv = MaskedConv2d(d_model, d_model, kernel_size=(kernel_size, kernel_size), mask_type='B', padding=kernel_size//2)

        # stack of PixelSNAIL 2D blocks
        self.blocks = nn.ModuleList()
        for _ in range(n_layer):
            block = PixelSNAILBlock2D(
                d_model,
                n_residual=n_residual,
                kernel_size=kernel_size,
                dropout=dropout,
                attn_every=attn_every,
                attn_kwargs={'key_dim': attn_key_dim, 'val_dim': attn_val_dim, 'n_heads': attn_heads},
                mask_type='B'
            )
            self.blocks.append(block)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            # tie output logit weights to token embedding
            self.head.weight = self.token_emb.weight

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        idx: [B, T] where T <= block_size (T=H*W typically)
        Returns logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        # expect T <= H*W (excluding BOS). If BOS included in idx, user should pass tokens excluding BOS for forward
        assert T <= self.height * self.width, f"T={T} exceeds H*W={self.height * self.width}"

        # positional indices (row/col) for T positions in raster order
        # compute row,col arrays of length T
        device = idx.device
        rows = torch.arange(self.height, device=device).unsqueeze(1).repeat(1, self.width).view(-1)[:T]  # [T]
        cols = torch.arange(self.width, device=device).repeat(self.height)[:T]  # [T]

        # token embedding => [B, T, D]
        x = self.token_emb(idx)  # [B, T, D]
        # add row/col embeddings
        x = x + self.row_emb(rows).unsqueeze(0) + self.col_emb(cols).unsqueeze(0)
        x = self.drop(x)

        # reshape to 2D: [B, D, H, W], but T may be < H*W (we assume full grid T==H*W for sampling/eval)
        # fill remaining positions with zeros if T < H*W (not typical here)
        if T == self.height * self.width:
            x2d = x.transpose(1, 2).reshape(B, self.d_model, self.height, self.width)
        else:
            # pad
            pad = self.height * self.width - T
            pad_tensor = torch.zeros(B, pad, self.d_model, device=device, dtype=x.dtype)
            xpad = torch.cat([x, pad_tensor], dim=1)
            x2d = xpad.transpose(1, 2).reshape(B, self.d_model, self.height, self.width)

        # initial masked conv (B)
        x2d = self.input_conv(x2d)
        # pass through blocks
        for blk in self.blocks:
            x2d = blk(x2d)

        # back to [B, T, D]
        x_flat = x2d.reshape(B, self.d_model, -1)[:, :, :T]  # [B, D, T]
        x_flat = x_flat.transpose(1, 2)  # [B, T, D]

        x_flat = self.ln_f(x_flat)
        logits = self.head(x_flat)  # [B, T, V]
        return logits

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, eos_id: int | None = None):
        # Keep the ARPrior.generate() behavior (handles iterative sampling)
        return super().generate(idx, max_new_tokens, temperature, top_k, eos_id)
