# models/priors/pixelcnn.py
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.priors.base import ARPrior  # keep same base as GPT/BDH for compatibility


class MaskedConv2d(nn.Module):
    """
    Masked convolution used by PixelCNN. mask_type 'A' for first layer, 'B' for others.

    This implements the standard autoregressive mask over a 2D grid:
    - All pixels "in the future" (below, or to the right on the same row) are masked out.
    - For mask_type='A', the center pixel is also masked (used in the first layer).
    - For mask_type='B', the center pixel is included (used in subsequent layers).
    """
    def __init__(self, in_channels, out_channels, kernel_size, mask_type="B", padding=0, bias=True):
        super().__init__()
        assert mask_type in ("A", "B")
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

        k_h, k_w = self.kernel_size
        mask = torch.ones(out_channels, in_channels, k_h, k_w)
        center_h = k_h // 2
        center_w = k_w // 2

        # Mask out future positions
        for i in range(k_h):
            for j in range(k_w):
                if i > center_h or (i == center_h and j > center_w):
                    mask[:, :, i, j] = 0.0

        if mask_type == "A":
            # Additionally mask the center pixel for the first layer
            mask[:, :, center_h, center_w] = 0.0

        self.register_buffer("mask", mask)

    def forward(self, x):
        # Apply mask to conv weights at every forward
        w = self.conv.weight * self.mask
        return F.conv2d(x, w, bias=self.conv.bias, stride=self.conv.stride, padding=self.conv.padding)


class PixelCNNBlock(nn.Module):
    """
    Simple residual PixelCNN block as used in the VQ-VAE prior:

    x --> ReLU --> MaskedConv2d --> (+ x)

    No batchnorm, no dropout, no gating.
    """
    def __init__(self, channels, kernel_size=3, mask_type="B"):
        super().__init__()
        pad = kernel_size // 2
        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            MaskedConv2d(
                in_channels=channels,
                out_channels=channels,
                kernel_size=kernel_size,
                mask_type=mask_type,
                padding=pad,
                bias=True,
            ),
        )

    def forward(self, x):
        return x + self.net(x)


class PixelCNNPrior(ARPrior):
    """
    PixelCNN prior over VQ-VAE codes, matching the architecture described in the
    original VQ-VAE paper ("Neural Discrete Representation Learning", 2017):

    - 15 masked convolutional layers
    - 3x3 kernels
    - 128 hidden units (channels)
    - First layer uses mask type 'A', subsequent layers use 'B'
    - Residual connections
    - No batchnorm, no dropout, no gated units

    Accepts idx: [B, T] and returns logits: [B, T, Vocab]
    """

    def __init__(
        self,
        vocab_size: int = None,  # Make optional
        d_model: int = 128,
        n_layers: int = 15,
        kernel_size: int = 3,
        block_size: int = 64,
        dropout: float = 0.0,
        tie_embeddings: bool = False,
        config=None  # Add config parameter
    ):
        # Support both config object and individual parameters
        if config is not None:
            vocab_size = getattr(config, 'vocab_size', vocab_size)
            d_model = getattr(config, 'd_model', d_model)
            n_layers = getattr(config, 'n_layers', n_layers)
            kernel_size = getattr(config, 'kernel_size', kernel_size)
            block_size = getattr(config, 'block_size', block_size)
            dropout = getattr(config, 'dropout', dropout)
        
        # keep same init style as other priors
        super().__init__(vocab_size=vocab_size, d_model=d_model, block_size=block_size)

        # infer HxW from block_size
        side = int(math.isqrt(block_size))
        assert side * side == block_size, "block_size must be a perfect square (e.g., 64 -> 8x8)"
        self.side = side
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # token embedding -> channels
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # PixelCNN stack: first layer mask_type 'A', rest 'B'
        layers = []
        layers.append(PixelCNNBlock(d_model, kernel_size=kernel_size, mask_type="A"))
        for _ in range(n_layers - 1):
            layers.append(PixelCNNBlock(d_model, kernel_size=kernel_size, mask_type="B"))
        self.blocks = nn.Sequential(*layers)

        # final head: map channels -> vocab logits per spatial position
        self.head = nn.Conv2d(d_model, vocab_size, kernel_size=1, bias=True)

        # initialize
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.LongTensor, targets: torch.LongTensor = None) -> tuple:
        """
        idx: [B, T] where T <= block_size (e.g. 64 or 256)
        targets: [B, T] optional targets for loss computation
        returns: (logits, loss) or (logits, None) if targets is None
        """
        B, T = idx.shape
        
        # Handle partial sequences during generation by padding to block_size
        if T < self.block_size:
            # Pad with BOS token (vocab_size - 1) or 0, but BOS is safer
            # Actually, we should pad with a padding token, but for simplicity,
            # we'll pad with the first token (which should be BOS during generation)
            pad_token = idx[0, 0].item() if T > 0 else 0  # Use first token as padding
            padding = torch.full((B, self.block_size - T), pad_token, 
                                dtype=idx.dtype, device=idx.device)
            idx_padded = torch.cat([idx, padding], dim=1)
            T_padded = self.block_size
        elif T == self.block_size:
            idx_padded = idx
            T_padded = T
        else:
            # If T > block_size, truncate to last block_size tokens
            idx_padded = idx[:, -self.block_size:]
            T_padded = self.block_size

        # embed tokens -> [B, T_padded, D]
        x = self.token_emb(idx_padded)  # [B, T_padded, D]

        # reshape -> [B, D, H, W]
        x = x.view(B, self.side, self.side, self.d_model).permute(0, 3, 1, 2).contiguous()

        # pass through PixelCNN masked conv stack
        x = self.blocks(x)

        # head -> [B, Vocab, H, W]
        logits_spatial = self.head(x)

        # reshape to [B, T_padded, Vocab]
        logits_padded = logits_spatial.permute(0, 2, 3, 1).contiguous().view(B, T_padded, self.vocab_size)
        
        # Return only logits for the actual input length (not padding)
        logits = logits_padded[:, :T, :]
        
        # Compute loss if targets provided
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        
        return logits, loss