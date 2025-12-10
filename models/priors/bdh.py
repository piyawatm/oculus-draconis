import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior

# ----------------------------
# RoPE utilities (1D & 2D)
# ----------------------------
def get_freqs(dim: int, theta: float = 10000.0) -> torch.Tensor:
    """Build RoPE base frequencies for a 1D sequence."""
    half_dim = dim // 2
    idx = torch.arange(0, half_dim, dtype=torch.float32)
    freq = theta ** (-idx / half_dim) 
    return freq.view(1, 1, 1, half_dim)

def rope_1d(x: torch.Tensor, pos: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """Apply 1D RoPE to a sub-vector."""
    B, H, T, M = x.shape
    half = M // 2
    phases = pos * freqs.to(x.device) 
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
    """Apply 2D RoPE splitting latent dim into row/col halves."""
    B, H, T, N = x.shape
    half = N // 2
    x_row = x[..., :half]
    x_col = x[..., half:]
    x_row = rope_1d(x_row, row_pos, row_freqs)
    x_col = rope_1d(x_col, col_pos, col_freqs)
    return torch.cat([x_row, x_col], dim=-1)

# ----------------------------
# BDH Block (Single Layer - FIXED)
# ----------------------------
class BDHBlock(nn.Module):
    """
    A single layer of the BDH Architecture.
    Fixes the shared-weight bug by encapsulating parameters here.
    """
    def __init__(
        self, 
        d_model: int, 
        latent_dim: int, 
        n_head: int, 
        dropout: float = 0.1,
        radius: Optional[int] = None
    ):
        super().__init__()
        self.n_head = n_head
        self.latent_dim = latent_dim
        self.head_dim = latent_dim // n_head
        self.dropout = nn.Dropout(dropout)
        
        # Pre-Norms
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Sparse Projections (Wide)
        # Note: Using Linear instead of raw Parameters for easier init/management
        self.encoder = nn.Linear(d_model, latent_dim)
        self.encoder_v = nn.Linear(d_model, latent_dim)
        self.decoder = nn.Linear(latent_dim, d_model)
        
        # Locality (Optional)
        self.radius = radius

    def forward(
        self, 
        x: torch.Tensor, 
        row_pos: torch.Tensor, 
        col_pos: torch.Tensor,
        row_freqs: torch.Tensor,
        col_freqs: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        
        B, T, D = x.shape
        
        # --- Residual Branch 1: Sparse Associative Attention ---
        residual = x
        x_norm = self.ln1(x)
        
        # 1. Project to Wide Sparse Latent
        q_sparse = F.relu(self.encoder(x_norm)) 
        k_sparse = F.relu(self.encoder_v(x_norm))
        
        # 2. View as Heads [B, T, H, N_head] -> [B, H, T, N_head]
        q = q_sparse.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k_sparse.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        
        # 3. Apply 2D RoPE
        q = apply_rope_2d(q, row_pos, col_pos, row_freqs, col_freqs)
        k = apply_rope_2d(k, row_pos, col_pos, row_freqs, col_freqs)
        
        # 4. Latent Attention
        # Scaled Dot Product
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale # [B, H, T, T]
        
        # Apply Mask (Causal + Locality)
        scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        
        # 5. Multiplicative Gating (Hebbian)
        # We use the sparse Q as the value carrier, gated by attention
        # V is effectively q_sparse here in the "Associative" interpretation
        y = torch.matmul(attn, q) # [B, H, T, N_head]
        
        # Reshape back to [B, T, N]
        y = y.transpose(1, 2).contiguous().view(B, T, self.latent_dim)
        
        # Gating: The output is modulated by the original sparse activation
        y = y * q_sparse
        
        # 6. Project back
        y = self.decoder(y)
        y = self.dropout(y)
        x = residual + y
        
        # --- Residual Branch 2: MLP (Optional but recommended for Deep Networks) ---
        # Standard Transformers have an MLP here. BDH usually does too.
        # For simplicity and to match "Wide/Shallow" vibe, we can skip or keep.
        # Let's keep it simple: Just the BDH block is enough if latent_dim is huge.
        
        return x

# ----------------------------
# Main Prior Model
# ----------------------------
class BDHPrior(ARPrior):
    def __init__(self, config):
        super().__init__(config.vocab_size, config.d_model, config.block_size)
        self.config = config
        
        # Embeddings
        self.token_emb = nn.Embedding(config.vocab_size + 1, config.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, config.block_size, config.d_model))
        self.drop = nn.Dropout(config.dropout)
        
        # Independent Layers (ModuleList)
        self.layers = nn.ModuleList([
            BDHBlock(
                d_model=config.d_model,
                latent_dim=config.latent_dim,
                n_head=config.n_head,
                dropout=config.dropout,
                radius=config.radius
            ) for _ in range(config.n_layer)
        ])
        
        self.ln_f = nn.LayerNorm(config.d_model)
        self.head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Precompute RoPE stuff
        self.register_buffer("mask", self._build_mask())
        self._init_rope()

    def _init_rope(self):
        # Infer Grid
        self.H = self.W = int(self.config.block_size ** 0.5)
        
        # Frequencies
        head_dim = (self.config.latent_dim // self.config.n_head)
        half_head_dim = head_dim // 2
        self.row_freqs = get_freqs(half_head_dim)
        self.col_freqs = get_freqs(half_head_dim)
        
        # Grid Positions
        row = torch.arange(self.H).float().view(self.H, 1).repeat(1, self.W).flatten()
        col = torch.arange(self.W).float().view(1, self.W).repeat(self.H, 1).flatten()
        
        # Handle BOS token (index 0) -> Position -1
        # Assuming sequence is [BOS, t1, t2 ...]
        # If block_size includes BOS, we prepend
        self.register_buffer("row_pos_grid", row)
        self.register_buffer("col_pos_grid", col)

    def _build_mask(self):
        T = self.config.block_size
        mask = torch.tril(torch.ones(T, T))
        
        # Locality
        if self.config.radius is not None:
            H = W = int(T ** 0.5)
            grid = torch.arange(T)
            r = grid // W
            c = grid % W
            
            dist = (r.view(T, 1) - r.view(1, T))**2 + (c.view(T, 1) - c.view(1, T))**2
            local_mask = dist <= (self.config.radius ** 2)
            mask = mask * local_mask
            
        return mask

    def forward(self, idx, targets=None):
        B, T = idx.size()
        
        # Embed
        x = self.token_emb(idx) + self.pos_emb[:, :T, :]
        x = self.drop(x)
        
        # Prepare RoPE Positions
        # We assume input is [BOS, code1, code2...] or just [code1...]
        # For simplicity, just take the first T positions from grid
        # (A robust implementation handles BOS offset explicitly)
        r_pos = self.row_pos_grid[:T].view(1, 1, T, 1)
        c_pos = self.col_pos_grid[:T].view(1, 1, T, 1)
        
        # Forward Layers
        mask = self.mask[:T, :T]
        for layer in self.layers:
            x = layer(x, r_pos, c_pos, self.row_freqs, self.col_freqs, mask)
            
        x = self.ln_f(x)
        logits = self.head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
            
        return logits, loss
