# models/priors/bdh.py
# BDH (Dragon Hatchling) Prior for VQ-VAE tokens
# - Implements sparse graph attention with top-k neighbors
# - Inherits ARPrior (so interchangeable with GPTPrior etc.)
# - Optimized for token sequences from VQ-VAE (e.g., 64 tokens, vocab=512)

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior

# -------------------------------------------------------------
# Basic Building Blocks
# -------------------------------------------------------------
class SparseAttention(nn.Module):
    """
    Top-k sparse self-attention block (approximation of BDH connectivity)
    Uses cosine similarity to select neighbors.
    """
    def __init__(self, d_model, n_head, k_neighbors):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.k = k_neighbors
        self.head_dim = d_model // n_head
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x):
        B, T, D = x.shape
        H = self.n_head

        q = self.q_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)  # [B,H,T,hd]
        k = self.k_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, H, self.head_dim).transpose(1, 2)

        # L2 normalize for cosine similarity
        qn = F.normalize(q, dim=-1)
        kn = F.normalize(k, dim=-1)

        # compute similarities [B,H,T,T]
        sim = torch.einsum("bhid,bhjd->bhij", qn, kn) * self.scale

        # mask future tokens (causal)
        mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
        sim.masked_fill_(~mask, -float("inf"))

        # pick top-k per query position
        if self.k < T:
            topk = torch.topk(sim, self.k, dim=-1)
            mask_sparse = torch.zeros_like(sim, dtype=torch.bool)
            mask_sparse.scatter_(-1, topk.indices, True)
            sim.masked_fill_(~mask_sparse, -float("inf"))

        attn = F.softmax(sim, dim=-1)
        y = torch.einsum("bhij,bhjd->bhid", attn, v)
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.out_proj(y)


class BDHBlock(nn.Module):
    def __init__(self, d_model, n_head, k_neighbors, mlp_mult=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SparseAttention(d_model, n_head, k_neighbors)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d_model, d_model),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


# -------------------------------------------------------------
# BDH Prior
# -------------------------------------------------------------
class BDHPrior(ARPrior):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 4,
        block_size: int = 64,
        k_neighbors: int = 8,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [BDHBlock(d_model, n_head, k_neighbors, mlp_mult, dropout) for _ in range(n_layer)]
        )

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.head.weight = self.token_emb.weight

    def forward(self, idx):
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        for _ in range(max_new_tokens):
            logits = self(idx)[:, -1, :] / max(temperature, 1e-8)
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = -float("inf")
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_id), dim=1)
            if idx.size(1) >= self.block_size:
                break
        return idx
