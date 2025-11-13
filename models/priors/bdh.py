# models/priors/bdh.py
# Faithful BDH (Dragon Hatchling) implementation adapted to ARPrior
# for use as an autoregressive prior over VQ-VAE tokens.
#
# Differences from original repo:
# - Uses ARPrior base class (provides generate()).
# - Takes config via __init__ args instead of BDHConfig dataclass.
# - Asserts T <= block_size for compatibility with your pipeline.
#
# Otherwise: RoPE, latent encoder/encoder_v/decoder, multiplicative gating,
# strict lower-triangular (no self) causal mask, parameter-based embeddings.

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.priors.base import ARPrior


# ----------------------------
# RoPE utilities
# ----------------------------

def get_freqs(dim: int, theta: float = 2.0 ** 16) -> torch.Tensor:
    """
    Build RoPE base frequencies for a 1D sequence.
    dim is the latent dimension N per head; we use dim//2 frequencies.
    Returns: [1, 1, 1, dim//2]
    """
    half_dim = dim // 2
    # positions scaled in [0,1)
    idx = torch.arange(0, half_dim, dtype=torch.float32)
    freq = theta ** (-idx / half_dim)  # [half_dim]
    return freq.view(1, 1, 1, half_dim)  # broadcastable


def apply_rope(x: torch.Tensor, freqs: torch.Tensor, seq_len: int) -> torch.Tensor:
    """
    x: [B, H, T, N]
    freqs: [1, 1, 1, N/2]
    Returns: same shape as x with RoPE applied to last dimension.
    """
    B, H, T, N = x.shape
    half = N // 2

    # positions [T] -> [1,1,T,1]
    pos = torch.arange(T, device=x.device, dtype=torch.float32).view(1, 1, T, 1)
    phases = pos * freqs.to(x.device)  # [1,1,T,half]

    cos = phases.cos()
    sin = phases.sin()

    x_even = x[..., 0:half]
    x_odd = x[..., half:2 * half]

    # rotate (even, odd)
    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    x_out = torch.cat([x_rot_even, x_rot_odd], dim=-1)
    return x_out


# ----------------------------
# Attention module
# ----------------------------

class BDHAttention(nn.Module):
    """
    Latent-space attention with RoPE and strictly causal mask.
    Q, K: [B, H, T, N]
    V:    [B, 1, T, D]  (broadcasted along head dim)
    """

    def __init__(self, n_head: int, latent_dim: int):
        super().__init__()
        self.n_head = n_head
        self.latent_dim = latent_dim
        # RoPE frequencies
        freqs = get_freqs(latent_dim)
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns:
            y: [B, H, T, D]
        """
        assert K is Q, "Original BDH uses self-attention with K == Q"
        B, H, T, N = Q.shape
        _, _, Tv, D = V.shape
        assert T == Tv, "Q and V must have same sequence length"

        # RoPE on Q and K
        Qr = apply_rope(Q, self.freqs, T)
        Kr = apply_rope(K, self.freqs, T)

        # scores: [B,H,T,T]
        scale = 1.0 / math.sqrt(N)
        scores = torch.einsum("bhtn,bhsn->bhts", Qr, Kr) * scale

        # strictly causal: no attending to self, only to earlier tokens
        mask = torch.tril(torch.ones(T, T, device=Q.device, dtype=torch.bool), diagonal=-1)
        neg = torch.finfo(scores.dtype).min
        scores = scores.masked_fill(~mask, neg)

        attn = F.softmax(scores, dim=-1)  # [B,H,T,T]

        # broadcast V across heads: [B,1,T,D] -> [B,H,T,D]
        Vh = V.expand(B, H, T, D)

        # attn @ V
        y = torch.einsum("bhts,bhsd->bhtd", attn, Vh)  # [B,H,T,D]
        return y


# ----------------------------
# Original BDH as ARPrior
# ----------------------------

class BDHPrior(ARPrior):
    """
    Faithful BDH from the paper repo, adapted to ARPrior.

    Args:
        vocab_size: number of discrete codes (for you: K_code or K_code+1 with BOS)
        d_model:    embedding / model dimension (D)
        n_layer:    number of BDH layers
        n_head:     number of heads
        block_size: max sequence length (T); for BOS-aware, T+1
        mlp_internal_dim_multiplier: controls latent size N per head
        dropout:    dropout on the multiplicative latent gate
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 4,
        block_size: int = 64,
        mlp_internal_dim_multiplier: int = 128,
        dropout: float = 0.1,
    ):
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model

        # latent dim N per head (matches original repository logic)
        self.latent_dim = (mlp_internal_dim_multiplier * d_model) // n_head

        # Parameter-based embeddings (like original code)
        self.embed = nn.Parameter(torch.empty(vocab_size, d_model))
        self.lm_head = nn.Parameter(torch.empty(d_model, vocab_size))

        # encoder/encoder_v: project model dim -> per-head latent
        # shapes: [H, D, N]
        self.encoder = nn.Parameter(
            torch.empty(n_head, d_model, self.latent_dim)
        )
        self.encoder_v = nn.Parameter(
            torch.empty(n_head, d_model, self.latent_dim)
        )

        # decoder: concatenate all heads latents back to D
        # shape: [(H*N), D]
        self.decoder = nn.Parameter(
            torch.empty(n_head * self.latent_dim, d_model)
        )

        # single LayerNorm reused, no affine params (original: elementwise_affine=False)
        self.ln = nn.LayerNorm(d_model, elementwise_affine=False)

        # attention module with RoPE and strict causal masking
        self.attn = BDHAttention(n_head=n_head, latent_dim=self.latent_dim)

        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Simple init; original repo uses default initializations; this is safe
        nn.init.normal_(self.embed, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.encoder)
        nn.init.xavier_uniform_(self.encoder_v)
        nn.init.xavier_uniform_(self.decoder)

    def forward(self, idx: torch.LongTensor, targets: Optional[torch.LongTensor] = None):
        """
        idx:     [B, T] int tokens
        targets: [B, T] int tokens or None

        Returns:
            logits: [B, T, vocab_size] if targets is None
            (logits, loss) if targets is not None
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Embed tokens: F.embedding with Parameter weight to mimic original
        x = F.embedding(idx, self.embed)          # [B, T, D]
        x = x.unsqueeze(1)                        # [B, 1, T, D]
        x = self.ln(x)                            # initial LN

        # BDH layers
        for _ in range(self.n_layer):
            # 1) project to latent per head
            # x: [B,1,T,D], encoder: [H,D,N] -> [B,H,T,N]
            x_latent = torch.einsum("bhtd,hdn->bhtn", x, self.encoder)
            x_sparse = F.relu(x_latent)

            # 2) attention in latent with V from x
            yKV = self.attn(x_sparse, x_sparse, x)    # [B,H,T,D]
            yKV = self.ln(yKV)                        # LN in D space

            # 3) project back to latent and apply multiplicative gate
            y_latent = torch.einsum("bhtd,hdn->bhtn", yKV, self.encoder_v)
            y_sparse = F.relu(y_latent)
            gated = self.drop(x_sparse * y_sparse)    # [B,H,T,N]

            # 4) concatenate heads and project back to model dim
            B_, H_, T_, N_ = gated.shape
            gated_flat = gated.reshape(B_, T_, H_ * N_)   # [B,T,H*N]
            y = torch.einsum("btd,df->btf", gated_flat, self.decoder)  # [B,T,D]
            y = self.ln(y)
            y = y.unsqueeze(1)                        # [B,1,T,D]

            # 5) residual + LN (reuse ln like original)
            x = self.ln(x + y)

        # final logits
        x_final = x.view(B, T, self.d_model)          # [B,T,D]
        logits = torch.einsum("btd,df->btf", x_final, self.lm_head)  # [B,T,V]

        if targets is None:
            return logits

        loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            targets.view(-1),
        )
        return logits, loss
