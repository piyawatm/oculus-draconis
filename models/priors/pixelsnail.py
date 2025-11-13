# models/priors/pixelsnail.py
# PixelSNAIL Prior for VQ-VAE token sequences

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior

# Basic building blocks

class ResidualBlock(nn.Module):
    """Residual block for PixelSNAIL (simplified)"""
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()

    def forward(self, x):
        # x: [B, C, T]
        out = self.activation(self.conv1(x))
        out = self.dropout(self.conv2(out))
        return x + out  # residual connection

class PixelSNAILBlock(nn.Module):
    """Single PixelSNAIL block with residual layers"""
    def __init__(self, channels, n_residual=2, dropout=0.1):
        super().__init__()
        self.res_blocks = nn.ModuleList(
            [ResidualBlock(channels, dropout) for _ in range(n_residual)]
        )

    def forward(self, x):
        for blk in self.res_blocks:
            x = blk(x)
        return x

# PixelSNAIL Prior
# -------------------------------------------------------------

class PixelSNAILPrior(ARPrior):
    """Autoregressive PixelSNAIL prior for VQ-VAE tokens"""
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,       # token embedding dimension
        n_layer: int = 6,         # number of PixelSNAIL blocks
        block_size: int = 64,     # number of tokens from VQ-VAE
        n_residual: int = 2,      # residual layers per block
        dropout: float = 0.1,     # dropout in residual blocks
        tie_embeddings: bool = True,
    ):
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        # Token embeddings + positional embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.drop = nn.Dropout(dropout)

        # PixelSNAIL blocks
        self.blocks = nn.ModuleList(
            [PixelSNAILBlock(d_model, n_residual=n_residual, dropout=dropout) for _ in range(n_layer)]
        )

        # Output layer mapping to vocab size
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie embeddings if requested
        if tie_embeddings:
            self.head.weight = self.token_emb.weight

    # -----------------------------
    # Forward pass
    # -----------------------------
    def forward(self, idx: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            idx: [B, T] token IDs (T <= block_size)
        Returns:
            logits: [B, T, vocab_size]
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"

        # Token + positional embeddings
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [1, T]
        x = self.token_emb(idx) + self.pos_emb(pos)           # [B, T, D]
        x = self.drop(x)

        # Convert to [B, D, T] for conv1d residual blocks
        x = x.transpose(1, 2)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(1, 2)  # back to [B, T, D]

        x = self.ln_f(x)
        logits = self.head(x)  # [B, T, vocab_size]
        return logits

    # -----------------------------
    # Autoregressive generation
    # -----------------------------
    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        eos_id: int | None = None,
    ) -> torch.LongTensor:
        """
        Simple autoregressive sampling.
        Uses ARPrior.generate() if not overridden.
        """
        return super().generate(idx, max_new_tokens, temperature, top_k, eos_id)
