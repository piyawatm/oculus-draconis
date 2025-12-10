# models/priors/maskgit.py
# MaskGIT: Masked Generative Image Transformer
# Non-autoregressive prior using iterative parallel decoding

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from typing import Optional, Union

from .base import ARPrior


class MaskGITPrior(ARPrior):
    """
    MaskGIT: Non-autoregressive prior using masked language modeling.
    
    Trained with MLM objective (like BERT), generates via iterative parallel decoding.
    """
    
    def __init__(
        self,
        vocab_size: int = None,
        d_model: int = 512,
        block_size: int = 256,
        n_head: int = 8,
        n_layer: int = 12,
        dropout: float = 0.1,
        mlm_probability: float = 0.15,  # Probability of masking tokens during training
        config: Optional[object] = None,
    ):
        # Support both config object and individual parameters
        if config is not None:
            vocab_size = getattr(config, 'vocab_size', vocab_size)
            d_model = getattr(config, 'd_model', d_model)
            block_size = getattr(config, 'block_size', block_size)
            n_head = getattr(config, 'n_head', n_head)
            n_layer = getattr(config, 'n_layer', n_layer)
            dropout = getattr(config, 'dropout', dropout)
            mlm_probability = getattr(config, 'mlm_probability', mlm_probability)
        
        # Mask token is added as vocab_size (so total vocab is vocab_size + 1)
        # For VQ-VAE: vocab_size = 1024 codes, mask_token = 1024, total = 1025
        super().__init__(vocab_size, d_model, block_size)
        
        self.mask_token_id = vocab_size  # Mask token ID
        self.mlm_probability = mlm_probability
        
        # Embeddings: vocab_size + 1 to include mask token
        self.tok_emb = nn.Embedding(vocab_size + 1, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Bidirectional Transformer Encoder (no causal mask)
        encoder_layer = TransformerEncoderLayer(
            d_model,
            n_head,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.transformer = TransformerEncoder(
            encoder_layer,
            num_layers=n_layer,
            norm=nn.LayerNorm(d_model)
        )
        
        # Output head predicts original VQ codes (vocab_size, not vocab_size+1)
        self.to_logits = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    @staticmethod
    def _init_weights(m: nn.Module):
        """Initialize weights following standard transformer practices."""
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
    
    def _random_masking(
        self,
        tokens: torch.LongTensor,
        mask_prob: float
    ) -> tuple[torch.LongTensor, torch.BoolTensor]:
        """
        Randomly mask tokens for MLM training.
        
        Args:
            tokens: [B, T] input tokens
            mask_prob: probability of masking each token
            
        Returns:
            masked_tokens: [B, T] tokens with some replaced by mask_token_id
            mask: [B, T] boolean mask (True = masked position)
        """
        B, T = tokens.shape
        device = tokens.device
        
        # Create random mask
        mask = torch.rand(B, T, device=device) < mask_prob
        
        # Don't mask all tokens (at least keep one)
        mask = mask & (~mask.all(dim=1, keepdim=True))
        
        # Replace masked tokens with mask_token_id
        masked_tokens = tokens.clone()
        masked_tokens[mask] = self.mask_token_id
        
        return masked_tokens, mask
    
    def forward(
        self,
        idx: torch.LongTensor,
        targets: Optional[torch.LongTensor] = None
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass for training (MLM) or inference.
        
        Args:
            idx: [B, T] token IDs. During training, this is the original sequence.
                 During inference, this may contain mask tokens.
            targets: [B, T] target tokens for MLM loss computation.
                     If None, no loss is computed.
        
        Returns:
            logits: [B, T, vocab_size] predictions
            loss: scalar loss if targets provided, else None
        """
        B, T = idx.shape
        assert T <= self.block_size, f"Sequence length {T} exceeds block_size {self.block_size}"
        
        # During training with targets, apply random masking
        if targets is not None and self.training:
            masked_idx, mask = self._random_masking(idx, self.mlm_probability)
        else:
            masked_idx = idx
            mask = None
        
        # Get embeddings
        tok_emb = self.tok_emb(masked_idx)  # [B, T, D]
        pos = torch.arange(T, device=idx.device).unsqueeze(0)  # [1, T]
        pos_emb = self.pos_emb(pos)  # [1, T, D]
        
        x = self.dropout(tok_emb + pos_emb)
        
        # Forward through bidirectional transformer (no causal mask)
        x_out = self.transformer(x)  # [B, T, D]
        
        # Project to logits (vocab_size, excluding mask token)
        logits = self.to_logits(x_out)  # [B, T, vocab_size]
        
        # Compute MLM loss if targets provided
        loss = None
        if targets is not None:
            # Only compute loss on masked positions during training
            if mask is not None:
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                    reduction='none'
                )
                loss = (loss * mask.reshape(-1)).sum() / mask.sum().clamp(min=1)
            else:
                # During inference/eval, compute loss on all positions
                loss = F.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1)
                )
        
        return logits, loss
    
    @torch.no_grad()
    def generate(
        self,
        idx: torch.LongTensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        eos_id: Optional[int] = None,
        n_steps: int = 8,
        schedule: str = 'cosine',
        sampling_top_k: Optional[int] = None,
    ) -> torch.LongTensor:
        """
        Iterative parallel generation (MaskGIT decoding).
        
        Args:
            idx: [B, T0] prefix (unused, kept for interface compatibility)
            max_new_tokens: ignored (generates full sequence)
            temperature: sampling temperature
            top_k: ignored (use sampling_top_k instead)
            eos_id: ignored
            n_steps: number of iterative refinement steps
            schedule: 'cosine' or 'linear' masking schedule
            sampling_top_k: top-k sampling at each step
        
        Returns:
            tokens: [B, block_size] generated tokens (all in range [0, vocab_size-1])
        """
        self.eval()
        B = idx.size(0)
        T = self.block_size
        device = idx.device
        
        # Start with all tokens masked
        tokens = torch.full((B, T), self.mask_token_id, dtype=torch.long, device=device)
        mask = torch.ones(B, T, dtype=torch.bool, device=device)
        
        for step in range(n_steps):
            # Get predictions
            logits, _ = self(tokens, targets=None)  # [B, T, vocab_size]
            
            # Apply temperature
            logits = logits / max(1e-8, temperature)
            
            # Top-k filtering if specified
            if sampling_top_k is not None:
                v, _ = torch.topk(logits, min(sampling_top_k, logits.size(-1)), dim=-1)
                logits[logits < v[..., -1:]] = float("-inf")
            
            # Sample tokens (only from valid vocab range [0, vocab_size-1])
            probs = F.softmax(logits, dim=-1)
            sampled_ids = torch.multinomial(
                probs.reshape(-1, self.vocab_size),
                num_samples=1
            ).view(B, T)
            
            # Get confidence scores
            sampled_probs = torch.gather(probs, 2, sampled_ids.unsqueeze(-1)).squeeze(-1)
            
            # Don't re-sample already known tokens
            sampled_probs[~mask] = -1e9
            
            # Determine how many tokens to unmask
            n_masked = mask.sum(dim=1).float()
            
            if step == n_steps - 1:
                # Last step: unmask ALL remaining tokens
                n_to_unmask = n_masked.int()
            else:
                # Calculate target mask ratio for next step
                mask_ratio_next = self._get_mask_ratio(step + 1, n_steps, schedule)
                n_masked_target = torch.full((B,), T * mask_ratio_next, device=device, dtype=torch.float)
                n_to_unmask = (n_masked - n_masked_target).int().clamp(min=1)
            
            n_to_unmask = torch.min(n_to_unmask, n_masked.int())
            
            if (n_to_unmask == 0).all():
                # All tokens unmasked, but ensure we complete the last step
                if step < n_steps - 1:
                    break
            
            # Unmask most confident tokens per batch
            for i in range(B):
                k = n_to_unmask[i].item()
                if k == 0:
                    continue
                
                # Get top-k most confident positions
                _, top_indices = torch.topk(sampled_probs[i], k=k)
                
                # Update tokens and mask
                tokens[i, top_indices] = sampled_ids[i, top_indices]
                mask[i, top_indices] = False
        
        # CRITICAL: Ensure all mask tokens are replaced in the final step
        # If any mask tokens remain, replace them with the sampled predictions
        remaining_mask = tokens == self.mask_token_id
        if remaining_mask.any():
            # Get final predictions for remaining positions
            logits, _ = self(tokens, targets=None)
            logits = logits / max(1e-8, temperature)
            
            if sampling_top_k is not None:
                v, _ = torch.topk(logits, min(sampling_top_k, logits.size(-1)), dim=-1)
                logits[logits < v[..., -1:]] = float("-inf")
            
            probs = F.softmax(logits, dim=-1)
            final_sampled = torch.multinomial(
                probs.reshape(-1, self.vocab_size),
                num_samples=1
            ).view(B, T)
            
            # Replace any remaining mask tokens
            tokens[remaining_mask] = final_sampled[remaining_mask]
        
        # Final validation: ensure all tokens are in valid range [0, vocab_size-1]
        tokens = torch.clamp(tokens, min=0, max=self.vocab_size - 1)
        
        return tokens
    
    def _get_mask_ratio(self, step: int, total_steps: int, schedule: str = 'cosine') -> float:
        """Calculate mask ratio for iterative decoding schedule."""
        if schedule == 'cosine':
            ratio = math.cos(0.5 * math.pi * (step / total_steps))
        elif schedule == 'linear':
            ratio = 1.0 - (step / total_steps)
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        return ratio