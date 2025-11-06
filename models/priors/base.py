# models/priors/base.py
# Minimal interface for all autoregressive priors (BDH, GPT, Performer, RWKV, ...)

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class ARPrior(nn.Module):
    """Common interface so training/sampling code is identical across priors."""

    def __init__(self, vocab_size: int, d_model: int, block_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.block_size = block_size

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Args:
            x: LongTensor [B, T] with token IDs (T <= block_size)
        Returns:
            logits: FloatTensor [B, T, vocab_size]
        """
        raise NotImplementedError

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
        Simple autoregressive sampling loop.
        Args:
            idx: [B, T0] prefix tokens
            max_new_tokens: how many tokens to append
            temperature: softmax temperature
            top_k: keep only top-k logits (nucleus/top-p not included here)
            eos_id: optional early stop ID
        Returns:
            [B, T0 + max_new_tokens] sampled tokens
        """
        self.eval()
        B = idx.size(0)
        for _ in range(max_new_tokens):
            # respect block_size: only feed the last block_size tokens
            idx_cond = idx[:, -self.block_size :]

            logits = self(idx_cond)[:, -1, :] / max(1e-8, temperature)  # [B, V]

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)  # [B, 1]
            idx = torch.cat([idx, next_id], dim=1)

            if eos_id is not None:
                done = (next_id.squeeze(1) == eos_id)
                if torch.all(done):
                    break
            if idx.size(1) >= self.block_size:
                # stop if we reached the model's max length
                break
        return idx
