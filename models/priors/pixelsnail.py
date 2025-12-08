# models/priors/pixelsnail.py
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior
from typing import Optional

# Utility layers
class MaskedConv2d(nn.Conv2d):
    """
    2D masked convolution enforcing autoregressive raster-scan causality.
    mask_type: 'A' (no access to current pixel channels) or 'B' (allow current pixel conditioning)
    """
    def __init__(self, in_ch, out_ch, kernel_size, mask_type='B', stride=1, padding=0, bias=True):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=padding, bias=bias)
        assert mask_type in ('A', 'B')
        # Extract kernel dimensions (handle both int and tuple)
        kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        
        # create mask buffer of same shape as weight
        mask = torch.ones(out_ch, in_ch, kh, kw)
        cy, cx = kh // 2, kw // 2

        for i in range(kh):
            for j in range(kw):
                if i > cy or (i == cy and j > cx):
                    mask[:, :, i, j] = 0
        if mask_type == 'A':
            mask[:, :, cy, cx] = 0
        
        self.register_buffer('mask', mask)

    def forward(self, x):
        # Apply mask to conv weights without in-place mutation (compatible with torch.compile)
        w = self.weight * self.mask
        return F.conv2d(x, w, bias=self.bias, stride=self.stride, padding=self.padding)

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
        B, C, H, W = x.shape
        N = H * W
        q = self.q_proj(x).reshape(B, self.n_heads, self.key_dim, N)
        k = self.k_proj(x).reshape(B, self.n_heads, self.key_dim, N)
        v = self.v_proj(x).reshape(B, self.n_heads, self.val_dim, N)

        # transpose to [B, heads, N, dim]
        q = q.permute(0, 1, 3, 2)
        k = k.permute(0, 1, 3, 2)
        v = v.permute(0, 1, 3, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale 
        mask = torch.tril(torch.ones(N, N, device=x.device, dtype=torch.bool))  # Use bool for better performance
        attn_logits = attn_logits.masked_fill(~mask.unsqueeze(0).unsqueeze(0), float("-inf"))
        
        # Clamp to prevent overflow in softmax
        attn_logits = torch.clamp(attn_logits, min=-50, max=50)
        attn = torch.softmax(attn_logits, dim=-1)

        out = torch.matmul(attn, v)
        out = out.permute(0, 1, 3, 2).reshape(B, self.n_heads * self.val_dim, H, W)
        out = self.out(out)
        return x + out

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
        vocab_size: int = None,
        d_model: int = 256,
        n_layer: int = 12,
        height: int = 8,
        width: int = 8,
        block_size: int = 64,  # accepted for compatibility
        n_residual: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.0,
        tie_embeddings: bool = True,
        attn_every: int = 4,
        attn_key_dim: int = 32,
        attn_val_dim: int = 64,
        attn_heads: int = 4,
        config=None,  # Add config parameter
    ):
        # Support both config object and individual parameters
        if config is not None:
            vocab_size = getattr(config, 'vocab_size', vocab_size)
            d_model = getattr(config, 'd_model', d_model)
            n_layer = getattr(config, 'n_layer', n_layer)
            height = getattr(config, 'height', height)
            width = getattr(config, 'width', width)
            block_size = getattr(config, 'block_size', block_size)
            n_residual = getattr(config, 'n_residual', n_residual)
            kernel_size = getattr(config, 'kernel_size', kernel_size)
            dropout = getattr(config, 'dropout', dropout)
            tie_embeddings = getattr(config, 'tie_embeddings', tie_embeddings)
            attn_every = getattr(config, 'attn_every', attn_every)
            attn_key_dim = getattr(config, 'attn_key_dim', attn_key_dim)
            attn_val_dim = getattr(config, 'attn_val_dim', attn_val_dim)
            attn_heads = getattr(config, 'attn_heads', attn_heads)
        
        # Infer height/width from block_size if not explicitly set
        if height == 8 and width == 8 and block_size != 64:
            # Calculate from block_size (assuming square grid)
            import math
            side = int(math.isqrt(block_size))
            if side * side == block_size:
                height = width = side
        
        # block_size should be height * width (without BOS for the actual sequence)
        # But we store it as height * width for compatibility
        actual_block_size = height * width
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=actual_block_size)

        self.height = height
        self.width = width
        self.d_model = d_model
        self.n_layer = n_layer

        self.token_emb = nn.Embedding(vocab_size, d_model)
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
        # PixelSNAIL accumulates skip outputs from each block
        self.skip_projs = nn.ModuleList([
            nn.Conv2d(d_model, d_model, kernel_size=1)
            for _ in range(n_layer)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.token_emb.weight
        
        # Initialize weights properly
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights to prevent NaN during training."""
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Parameter):
            # For res_scale parameter
            nn.init.ones_(m)

    def forward(self, idx: torch.LongTensor, targets: torch.LongTensor = None) -> tuple:
        """
        idx: [B, T] where T <= block_size (T=H*W typically)
        targets: [B, T] optional targets for loss computation
        Returns: (logits, loss) or (logits, None) if targets is None
        """
        B, T = idx.shape
        
        # Handle partial sequences during generation by padding to full grid
        if T < self.height * self.width:
            # Pad with first token (should be BOS during generation)
            pad_token = idx[0, 0].item() if T > 0 else 0
            pad = self.height * self.width - T
            padding = torch.full((B, pad), pad_token, dtype=idx.dtype, device=idx.device)
            idx_padded = torch.cat([idx, padding], dim=1)
            T_padded = self.height * self.width
        elif T == self.height * self.width:
            idx_padded = idx
            T_padded = T
        else:
            # If T > block_size, truncate to last block_size tokens
            idx_padded = idx[:, -self.height * self.width:]
            T_padded = self.height * self.width

        # positional indices (row/col) for T_padded positions in raster order
        device = idx_padded.device
        rows = torch.arange(self.height, device=device).unsqueeze(1).repeat(1, self.width).view(-1)[:T_padded]  # [T_padded]
        cols = torch.arange(self.width, device=device).repeat(self.height)[:T_padded]  # [T_padded]

        # token embedding => [B, T_padded, D]
        x = self.token_emb(idx_padded)
        # add row/col embeddings
        x = x + self.row_emb(rows).unsqueeze(0) + self.col_emb(cols).unsqueeze(0)
        x = self.drop(x)

        # reshape to 2D: [B, D, H, W]
        x2d = x.transpose(1, 2).reshape(B, self.d_model, self.height, self.width)

        # initial masked conv (B)
        x2d = self.input_conv(x2d)

        # Scale down skip connections to prevent explosion
        accum = None
        for blk, skip_proj in zip(self.blocks, self.skip_projs):
            out = blk(x2d)
            skip = skip_proj(out)

            if accum is None:
                accum = skip
            else:
                # Scale accumulated skip connections to prevent gradient explosion
                accum = accum + 0.1 * skip  # Scale factor to prevent explosion

            # Residual connection with scaling
            x2d = x2d + 0.1 * accum  # Scale to prevent explosion

        # Extract only the logits for actual input length (not padding)
        x_flat = x2d.reshape(B, self.d_model, -1)[:, :, :T]  # [B, D, T]
        x_flat = x_flat.transpose(1, 2)  # [B, T, D]

        x_flat = self.ln_f(x_flat)
        logits = self.head(x_flat)  # [B, T, V]
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.LongTensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None, eos_id: int | None = None):
        
        return super().generate(idx, max_new_tokens, temperature, top_k, eos_id)