# models/priors/gpt.py
# Tiny GPT-style decoder-only Transformer prior with causal self-attention.
# Drop-in replacement for BDH: identical forward() and generate() signatures.

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.head_dim = d_model // n_head

        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # register a buffer for a big causal mask (we'll slice to [T,T] as needed)
        self.register_buffer("mask", torch.tril(torch.ones(4096, 4096)).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        B, T, D = x.shape
        qkv = self.qkv(x)  # [B, T, 3D]
        q, k, v = qkv.chunk(3, dim=-1)

        # [B, T, H, Hd] -> [B, H, T, Hd]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)  # [B, H, T, T]

        causal = self.mask[:, :, :T, :T]  # [1,1,T,T]
        att = att.masked_fill(causal == 0, float("-inf"))

        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # [B, H, T, Hd]
        y = y.transpose(1, 2).contiguous().view(B, T, D)  # [B, T, D]
        y = self.resid_drop(self.proj(y))
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, mlp_mult * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_mult * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class GPTBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, mlp_mult: int, dropout: float):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, mlp_mult, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPTPrior(ARPrior):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_layer: int = 6,
        n_head: int = 4,
        block_size: int = 64,
        mlp_mult: int = 4,
        dropout: float = 0.1,
        tie_embeddings: bool = True,
    ):
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        self.blocks = nn.ModuleList(
            [GPTBlock(d_model, n_head, mlp_mult, dropout) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        if tie_embeddings:
            self.head.weight = self.token_emb.weight

        # init (slight) to stabilize tiny models
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [1,T]
        x = self.token_emb(idx) + self.pos_emb(pos)            # [B,T,D]
        x = self.drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.ln_f(x)
        logits = self.head(x)                                  # [B,T,V]
        return logits
