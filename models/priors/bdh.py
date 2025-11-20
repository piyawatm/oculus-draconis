# models/priors/bdh.py
# 2D-aware BDH (Dragon Hatchling) with optional 2D locality sparsity,
# adapted to ARPrior for VQ-VAE image codes.
#
# Keeps BDH novelties:
# - Latent encoder / encoder_v / decoder (N >> D)
# - Attention in latent space
# - Multiplicative latent gating
# - Parameter-based embeddings and head
#
# Adds:
# - 2D RoPE over (row, col) positions
# - 2D-aware grid mapping (H x W) with optional BOS at index 0
# - Optional 2D locality window: each token attends only to nearby spatial neighbors
#   (plus BOS visible to all positions)

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.priors.base import ARPrior


# ----------------------------
# RoPE utilities (1D)
# ----------------------------

def get_freqs(dim: int, theta: float = 2.0 ** 16) -> torch.Tensor:
    """
    Build RoPE base frequencies for a 1D sequence.

    dim: the sub-dimension M of the vector to which RoPE is applied.
         We use dim//2 frequencies.
    Returns: [1, 1, 1, dim//2]
    """
    half_dim = dim // 2
    idx = torch.arange(0, half_dim, dtype=torch.float32)
    freq = theta ** (-idx / half_dim)  # [half_dim]
    return freq.view(1, 1, 1, half_dim)


def rope_1d(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Apply 1D RoPE to a sub-vector.

    x:    [B, H, T, M]
    pos:  [1, 1, T, 1] (scalar positions per token)
    freqs:[1, 1, 1, M/2]

    Returns: [B, H, T, M] with RoPE applied along last dimension.
    """
    B, H, T, M = x.shape
    assert M % 2 == 0, "RoPE sub-dimension must be even"
    half = M // 2

    phases = pos * freqs.to(x.device)  # [1,1,T,half]
    cos = phases.cos()
    sin = phases.sin()

    x_even = x[..., :half]
    x_odd = x[..., half:2 * half]

    x_rot_even = x_even * cos - x_odd * sin
    x_rot_odd = x_even * sin + x_odd * cos

    return torch.cat([x_rot_even, x_rot_odd], dim=-1)


def apply_rope_2d(
    x: torch.Tensor,
    row_pos: torch.Tensor,
    col_pos: torch.Tensor,
    row_freqs: torch.Tensor,
    col_freqs: torch.Tensor,
) -> torch.Tensor:
    """
    Apply 2D RoPE to latent x using separate row and column positions.

    x:        [B, H, T, N]
    row_pos:  [1, 1, T, 1] (row index per token, float)
    col_pos:  [1, 1, T, 1] (col index per token, float)
    row_freqs / col_freqs: frequencies for row/col RoPE

    Strategy:
    - Split latent dim N into two halves: first half encodes row, second half encodes col.
    - Apply independent 1D RoPE to each half using row_pos / col_pos.
    """
    B, H, T, N = x.shape
    assert N % 2 == 0, "latent_dim N must be even to split for 2D RoPE"
    half = N // 2

    x_row = x[..., :half]   # [B,H,T,half]
    x_col = x[..., half:]   # [B,H,T,half]

    x_row = rope_1d(x_row, row_pos, row_freqs)
    x_col = rope_1d(x_col, col_pos, col_freqs)

    return torch.cat([x_row, x_col], dim=-1)


# ----------------------------
# 2D-aware Attention module with optional locality sparsity
# ----------------------------

class BDH2DAttention(nn.Module):
    """
    Latent-space attention with 2D RoPE, strictly causal mask, and optional 2D locality.

    Q, K: [B, H, T, N]  (latent)
    V:    [B, 1, T, D]  (model space, broadcast across heads)

    2D-awareness:
    - For a block_size T, attempts to interpret tokens (except optional BOS)
      as an HxW grid of VQ codes.
    - Uses 2D RoPE over (row, col) positions.
    - Causality is strictly lower-triangular (no attending to self).
    - Optional locality window: each token only attends to spatial neighbors
      within a radius (plus BOS visible to all).
    """

    def __init__(
        self,
        n_head: int,
        latent_dim: int,
        block_size: int,
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        locality_radius: Optional[int] = None,
    ):
        super().__init__()
        self.n_head = n_head
        self.latent_dim = latent_dim
        self.block_size = block_size
        self.locality_radius = locality_radius

        # Infer H and W if not provided:
        # - If block_size is a square: use that (no BOS).
        # - Else if (block_size - 1) is a square: treat as BOS + H*W.
        H, W, has_bos = self._infer_grid(block_size, image_height, image_width)
        self.image_height = H
        self.image_width = W
        self.has_bos = has_bos

        # Precompute row/col positions for each token index 0..block_size-1
        row_pos, col_pos = self._build_2d_positions(block_size, H, W, has_bos)
        # [1,1,T,1] for RoPE
        self.register_buffer("row_pos", row_pos, persistent=False)
        self.register_buffer("col_pos", col_pos, persistent=False)

        # RoPE frequencies for each half of the latent dim
        assert latent_dim % 2 == 0, "latent_dim must be even for 2D RoPE"
        half = latent_dim // 2
        row_freqs = get_freqs(half)
        col_freqs = get_freqs(half)
        self.register_buffer("row_freqs", row_freqs, persistent=False)
        self.register_buffer("col_freqs", col_freqs, persistent=False)

        # Precompute 2D locality mask [T, T] if requested and grid known
        if locality_radius is not None and H is not None and W is not None:
            locality_mask = self._build_locality_mask(
                block_size, H, W, has_bos, locality_radius
            )
            # [T, T] bool
            self.register_buffer("locality_mask", locality_mask, persistent=False)
        else:
            self.locality_mask = None

    @staticmethod
    def _infer_grid(
        block_size: int,
        H_cfg: Optional[int],
        W_cfg: Optional[int],
    ) -> Tuple[Optional[int], Optional[int], bool]:
        """
        Infer (H,W) and whether there's a BOS token.

        Priority:
        1. If H_cfg and W_cfg provided, trust them; detect BOS from block_size.
        2. Else, if block_size is a perfect square, assume no BOS, H=W=sqrt(block_size).
        3. Else, if block_size-1 is a perfect square, assume BOS + H*W.
        4. Else, return (None, None, False) -> fall back to 1D behavior in practice.
        """
        if H_cfg is not None and W_cfg is not None:
            H, W = H_cfg, W_cfg
            has_bos = (block_size == H * W + 1)
            return H, W, has_bos

        s = int(math.isqrt(block_size))
        if s * s == block_size:
            return s, s, False

        s = int(math.isqrt(block_size - 1))
        if s * s == (block_size - 1):
            return s, s, True

        return None, None, False

    @staticmethod
    def _build_2d_positions(
        block_size: int,
        H: Optional[int],
        W: Optional[int],
        has_bos: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build row/col positions for each token index 0..block_size-1.

        Returns:
            row_pos: [1,1,T,1] float
            col_pos: [1,1,T,1] float
        """
        row = torch.zeros(block_size, dtype=torch.float32)
        col = torch.zeros(block_size, dtype=torch.float32)

        if H is not None and W is not None:
            grid_T = H * W
            for t in range(block_size):
                if has_bos:
                    if t == 0:
                        # BOS: outside grid, but we will handle locality separately
                        row[t] = -1.0
                        col[t] = -1.0
                    else:
                        idx = t - 1  # codes are t=1..grid_T
                        if idx < grid_T:
                            r = idx // W
                            c = idx % W
                            row[t] = float(r)
                            col[t] = float(c)
                        else:
                            # Extra tokens beyond the grid; map after grid
                            r = grid_T // W
                            c = grid_T % W
                            row[t] = float(r)
                            col[t] = float(c)
                else:
                    if t < grid_T:
                        r = t // W
                        c = t % W
                        row[t] = float(r)
                        col[t] = float(c)
                    else:
                        r = grid_T // W
                        c = grid_T % W
                        row[t] = float(r)
                        col[t] = float(c)

        row_pos = row.view(1, 1, block_size, 1)
        col_pos = col.view(1, 1, block_size, 1)
        return row_pos, col_pos

    @staticmethod
    def _build_locality_mask(
        block_size: int,
        H: int,
        W: int,
        has_bos: bool,
        radius: int,
    ) -> torch.Tensor:
        """
        Build a [T,T] boolean locality mask where entry (i,j) is True if j is within
        a 2D spatial radius of i, in the (row,col) grid. BOS gets special treatment:
        it can be attended to by any position.

        radius: locality window radius in grid steps (Manhattan or Euclidean).
        """
        T = block_size
        row = torch.zeros(T, dtype=torch.float32)
        col = torch.zeros(T, dtype=torch.float32)
        grid_T = H * W

        for t in range(T):
            if has_bos:
                if t == 0:
                    # BOS position, treat separately
                    row[t] = -1.0
                    col[t] = -1.0
                else:
                    idx = t - 1
                    if idx < grid_T:
                        r = idx // W
                        c = idx % W
                        row[t] = float(r)
                        col[t] = float(c)
                    else:
                        r = grid_T // W
                        c = grid_T % W
                        row[t] = float(r)
                        col[t] = float(c)
            else:
                if t < grid_T:
                    r = t // W
                    c = t % W
                    row[t] = float(r)
                    col[t] = float(c)
                else:
                    r = grid_T // W
                    c = grid_T % W
                    row[t] = float(r)
                    col[t] = float(c)

        # Pairwise squared Euclidean distances [T,T]
        dr = row.view(T, 1) - row.view(1, T)
        dc = col.view(T, 1) - col.view(1, T)
        dist_sq = dr * dr + dc * dc

        # Local if within radius^2
        local = dist_sq <= (radius * radius)

        if has_bos:
            # Allow all tokens to attend to BOS (column 0)
            local[:, 0] = True

        return local.bool()

    def forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
    ) -> torch.Tensor:
        """
        Q, K: [B, H, T, N]  (latent)
        V:    [B, 1, T, D]  (model space)

        Returns:
            y: [B, H, T, D]
        """
        assert K is Q, "BDH uses self-attention with K == Q"
        B, H, T, N = Q.shape
        _, _, Tv, D = V.shape
        assert T == Tv, "Q and V must have same sequence length"

        # Slice row/col positions and freqs for current T
        row_pos = self.row_pos[:, :, :T, :]  # [1,1,T,1]
        col_pos = self.col_pos[:, :, :T, :]  # [1,1,T,1]

        # 2D RoPE on Q and K
        Qr = apply_rope_2d(Q, row_pos, col_pos, self.row_freqs, self.col_freqs)
        Kr = apply_rope_2d(K, row_pos, col_pos, self.row_freqs, self.col_freqs)

        # scores: [B, H, T, T]
        scale = 1.0 / math.sqrt(N)
        scores = torch.einsum("bhtn,bhsn->bhts", Qr, Kr) * scale

        # Strictly causal: j < i (no self-attend, no future)
        causal = torch.tril(
            torch.ones(T, T, device=Q.device, dtype=torch.bool),
            diagonal=-1,
        )

        if self.locality_mask is not None:
            # locality_mask is [T,T] on CPU; move to device & slice
            local = self.locality_mask[:T, :T].to(Q.device)
            mask = causal & local
        else:
            mask = causal

        neg = torch.finfo(scores.dtype).min  # AMP-safe large negative
        scores = scores.masked_fill(~mask, neg)

        attn = F.softmax(scores, dim=-1)  # [B,H,T,T]

        # Broadcast V across heads: [B,1,T,D] -> [B,H,T,D]
        Vh = V.expand(B, H, T, D)

        # attn @ V
        y = torch.einsum("bhts,bhsd->bhtd", attn, Vh)  # [B,H,T,D]
        return y


# ----------------------------
# 2D-aware BDH as ARPrior
# ----------------------------

class BDHPrior(ARPrior):
    """
    2D-aware BDH prior with optional locality.

    Args:
        vocab_size: number of discrete codes (e.g., K_code or K_code+1 with BOS)
        d_model:    embedding / model dimension (D)
        n_layer:    number of BDH layers
        n_head:     number of heads
        block_size: max sequence length (T); for BOS-aware, T = H*W+1
        mlp_internal_dim_multiplier: controls latent size N per head
        dropout:    dropout on the multiplicative latent gate
        image_height, image_width: optional; if not provided, we'll infer H=W
                                   from block_size or block_size-1 if square.
        locality_radius: optional int; if set, applies 2D locality window in attention.
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
        image_height: Optional[int] = None,
        image_width: Optional[int] = None,
        locality_radius: Optional[int] = None,
    ):
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        self.n_layer = n_layer
        self.n_head = n_head
        self.d_model = d_model

        # latent dim N per head (BDH-style)
        self.latent_dim = (mlp_internal_dim_multiplier * d_model) // n_head

        # Parameter-based embeddings (as in original BDH)
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

        # 2D-aware attention module with RoPE, strict causality, and optional locality
        self.attn = BDH2DAttention(
            n_head=n_head,
            latent_dim=self.latent_dim,
            block_size=block_size,
            image_height=image_height,
            image_width=image_width,
            locality_radius=locality_radius,
        )

        self.drop = nn.Dropout(dropout)

        self._reset_parameters()

    def _reset_parameters(self):
        # Simple init; original repo uses default initializations; this is safe
        nn.init.normal_(self.embed, mean=0.0, std=0.02)
        nn.init.normal_(self.lm_head, mean=0.0, std=0.02)
        nn.init.xavier_uniform_(self.encoder)
        nn.init.xavier_uniform_(self.encoder_v)
        nn.init.xavier_uniform_(self.decoder)

    def forward(
        self,
        idx: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None,
    ):
        """
        idx:     [B, T] int tokens
        targets: [B, T] int tokens or None

        Returns:
            logits: [B, T, vocab_size] if targets is None
            (logits, loss) if targets is not None
        """
        B, T = idx.shape
        assert T <= self.block_size, (
            f"Sequence length {T} exceeds block_size {self.block_size}"
        )

        # Embed tokens: F.embedding with Parameter weight to mimic original BDH
        x = F.embedding(idx, self.embed)          # [B, T, D]
        x = x.unsqueeze(1)                        # [B, 1, T, D]
        x = self.ln(x)                            # initial LN

        # BDH layers
        for _ in range(self.n_layer):
            # 1) project to latent per head
            # x: [B,1,T,D], encoder: [H,D,N] -> [B,H,T,N]
            x_latent = torch.einsum("bhtd,hdn->bhtn", x, self.encoder)
            x_sparse = F.relu(x_latent)

            # 2) attention in latent with V from x (2D RoPE + strict causality + locality)
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
