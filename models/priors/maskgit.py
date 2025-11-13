# models/priors/maskgit.py
# Implementation of MaskGIT (Masked Generative Image Transformer)
# Adapted to fit the ARPrior interface.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Union

from .base import ARPrior

class MaskGITPrior(ARPrior):
    """
    MaskGIT: Non-autoregressive (parallel) prior.
    
    This model is trained using a "Masked Language Model" (MLM) objective,
    similar to BERT, not an autoregressive (causal) one.

    The `generate` method is completely overridden to implement the
    iterative parallel decoding (refinement) schedule.
    """
    def __init__(
        self, 
        vocab_size: int, 
        d_model: int, 
        block_size: int, 
        n_head: int = 8, 
        n_layer: int = 12, 
        dropout: float = 0.1
    ):
        # Note: vocab_size is the VQ codebook size (e.g., 512)
        super().__init__(vocab_size, d_model, block_size)
        
        # The mask token is a new, special token *in addition* to VQ codes
        self.mask_token_id = vocab_size 
        
        # Token embedding needs to include the mask token
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model) # size = V + 1
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)

        # Standard Bidirectional Transformer Encoder
        encoder_layer = TransformerEncoderLayer(
            d_model, 
            n_head, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            activation='gelu', 
            batch_first=True,
            norm_first=True # Pre-norm
        )
        self.transformer = TransformerEncoder(
            encoder_layer, 
            num_layers=n_layer, 
            norm=nn.LayerNorm(d_model)
        )
        
        # Output head predicts one of the *original* VQ codes (size V)
        self.to_logits = nn.Linear(d_model, vocab_size)

        self.init_weights()

    def init_weights(self):
        # Standard init
        nn.init.normal_(self.tok_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.to_logits.weight)
        nn.init.zeros_(self.to_logits.bias)

    def _get_mask_ratio(self, step: int, total_steps: int, schedule: str = 'cosine') -> float:
        """
        Calculates the ratio of tokens to *keep masked* at a given step.
        Goes from 1.0 (all masked) to 0.0 (all unmasked).
        """
        if schedule == 'cosine':
            # Cosine schedule from the paper
            ratio = math.cos(0.5 * math.pi * (step / total_steps))
        else:
            # Linear schedule
            ratio = 1.0 - (step / total_steps)
        return ratio

    def forward(self, x: torch.LongTensor) -> torch.Tensor:
        """
        Training pass: Bidirectional transformer forward.
        
        Args:
            x: LongTensor [B, T] of token IDs.
               *During training, this x should be the *masked* input.*
        
        Returns:
            logits: FloatTensor [B, T, vocab_size]
        """
        B, T = x.shape
        assert T <= self.block_size, "Input sequence is longer than model's block_size"
        
        # Get embeddings
        tok_emb = self.tok_emb(x) # [B, T, D]
        pos = torch.arange(0, T, device=x.device).unsqueeze(0) # [1, T]
        pos_emb = self.pos_emb(pos) # [1, T, D]
        
        x_emb = self.dropout(tok_emb + pos_emb)
        
        # Forward through bidirectional transformer
        # Note: No causal mask is applied!
        x_out = self.transformer(x_emb) # [B, T, D]
        
        # Project to logits
        logits = self.to_logits(x_out) # [B, T, vocab_size]
        
        return logits

    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        sampling_top_k: Union[int, None] = None, # Renamed from top_k
        eos_id: Union[int, None] = None, # Ignored by MaskGIT
        n_steps: int = 12, # Number of refinement steps
        schedule: str = 'cosine',
    ) -> torch.LongTensor:
        """
        Iterative parallel generation (MaskGIT's sampling process).
        This *overrides* the base ARPrior.generate method.
        
        Args:
            idx: [B, T0] prefix tokens. Only used to get batch_size and device.
            max_new_tokens: Should be set to self.block_size (total tokens).
        """
        self.eval()
        B = idx.size(0)
        T = self.block_size
        device = idx.device
        
        if max_new_tokens != T:
            print(f"Warning: MaskGIT generates all {T} tokens at once. "
                  f"Ignoring max_new_tokens={max_new_tokens}.")

        # 1. Start with all tokens masked
        tokens = torch.full((B, T), self.mask_token_id, dtype=torch.long, device=device)
        
        # 2. Mask state: True = masked, False = unmasked (known)
        mask = torch.ones_like(tokens, dtype=torch.bool)

        for t in range(n_steps):
            # 3. Get logits for all positions
            logits = self(tokens) # [B, T, V]
            
            # 4. Apply temperature and optional top-k *sampling*
            logits = logits / max(1e-8, temperature)
            if sampling_top_k is not None:
                v, _ = torch.topk(logits, min(sampling_top_k, logits.size(-1)))
                logits[logits < v[..., -1].unsqueeze(-1)] = float("-inf")
            
            probs = F.softmax(logits, dim=-1) # [B, T, V]
            
            # 5. Sample new tokens for all positions
            sampled_ids = torch.multinomial(probs.view(-1, self.vocab_size), num_samples=1).view(B, T)
            
            # 6. Get confidence scores (prob of the token we just sampled)
            sampled_probs = torch.gather(probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1) # [B, T]
            
            # 7. Don't re-sample known tokens. Set their confidence to -inf
            # so they are never chosen as "most confident".
            sampled_probs[~mask] = -1e9
            
            # 8. Determine how many tokens to unmask in this step
            n_masked_current = mask.sum(dim=-1).float() # [B]
            
            # Last step: unmask all remaining
            if t == n_steps - 1:
                n_to_unmask = n_masked_current.int()
            else:
                # Get target mask ratio for *next* step
                mask_ratio_next = self._get_mask_ratio(t + 1, n_steps, schedule)
                n_masked_target = math.floor(T * mask_ratio_next) # [B]
                
                n_to_unmask = (n_masked_current - n_masked_target).int()
                # Ensure we always unmask at least one token (to make progress)
                n_to_unmask = n_to_unmask.clamp(min=1)

            # Handle edge case where n_to_unmask > n_masked_current
            n_to_unmask = torch.min(n_to_unmask, n_masked_current.int())

            if torch.all(n_to_unmask == 0):
                break # All tokens are unmasked

            # 9. Find the most confident *newly predicted* tokens
            # We need to do this per-batch item
            for i in range(B):
                k = n_to_unmask[i].item()
                if k == 0:
                    continue
                
                # Find the top-k most confident predictions *for this batch item*
                indices_to_unmask = torch.topk(sampled_probs[i], k=k).indices
                
                # 10. Update tokens and mask
                tokens[i, indices_to_unmask] = sampled_ids[i, indices_to_unmask]
                mask[i, indices_to_unmask] = False
        
        return tokens