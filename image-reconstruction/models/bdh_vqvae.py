import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer, ResidualBlock
import math


class RoPE(nn.Module):
    """Rotary Position Embedding for BDH."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        # Generate frequencies for half the dimension (since we split into pairs)
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
    
    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        # Don't duplicate - freqs is already the right size for half the dimension
        return torch.cos(freqs), torch.sin(freqs)


def apply_rope(x, cos, sin):
    """Apply rotary position embeddings."""
    # x: [B, H, N, D]
    # cos, sin: [N, D//2]
    d = x.shape[-1]
    x1, x2 = x[..., :d//2], x[..., d//2:]
    # Broadcast cos and sin to match x1/x2 shape
    # cos/sin: [N, D//2] -> [1, 1, N, D//2]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class BDHAttention(nn.Module):
    """
    Baby Dragon Hatchling attention layer.
    Key features:
    - Q=K constraint (self-similarity matrix)
    - No softmax (raw attention scores with ReLU)
    - Sparse activations
    - RoPE for position encoding
    """
    
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_dim = n_embd // n_head
        
        # BDH uses Q=K constraint, so only one projection for queries/keys
        self.encoder = nn.Linear(n_embd, n_embd, bias=False)  # Q=K
        self.encoder_v = nn.Linear(n_embd, n_embd, bias=False)  # V
        self.decoder = nn.Linear(n_embd, n_embd, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RoPE(self.head_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        
        # Q=K: use same projection for both
        q = self.encoder(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)  # [B, H, N, D]
        k = q  # Q=K constraint!
        v = self.encoder_v(x).view(B, N, self.n_head, self.head_dim).transpose(1, 2)
        
        # Apply RoPE
        cos, sin = self.rope(N, x.device)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        
        # Attention: Q @ K^T (self-similarity matrix)
        # Scale by sqrt(d) for stability
        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # BDH: NO SOFTMAX! Use ReLU for sparse, non-negative attention
        attn = F.relu(attn)
        attn = self.dropout(attn)
        
        # Apply attention to values
        out = attn @ v  # [B, H, N, D]
        
        # Concatenate heads and project
        out = out.transpose(1, 2).contiguous().view(B, N, C)
        out = self.decoder(out)
        
        return out


class BDHBlock(nn.Module):
    """
    Complete BDH block with attention and gating.
    Features Pre-LayerNorm and multiplicative gating (no additive residual).
    """
    
    def __init__(self, n_embd, n_head, mlp_multiplier=4, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.attn = BDHAttention(n_embd, n_head, dropout)
        self.ln2 = nn.LayerNorm(n_embd)
        
        # MLP with ReLU (sparse activations)
        self.mlp = nn.Sequential(
            nn.Linear(n_embd, n_embd * mlp_multiplier),
            nn.ReLU(),  # BDH: sparse, non-negative activations
            nn.Dropout(dropout),
            nn.Linear(n_embd * mlp_multiplier, n_embd),
            nn.Dropout(dropout)
        )
        
        # Multiplicative gating (instead of additive residual)
        self.gate = nn.Parameter(torch.ones(1))
        
    def forward(self, x):
        # Pre-LayerNorm + Attention with multiplicative gating
        attn_out = self.attn(self.ln1(x))
        x = x + self.gate * attn_out
        
        # Pre-LayerNorm + MLP with multiplicative gating
        mlp_out = self.mlp(self.ln2(x))
        x = x + self.gate * mlp_out
        
        return x


class BDHEncoder(nn.Module):
    """
    Encoder with BDH (Baby Dragon Hatchling) attention refinement.
    Processes spatial features with bio-inspired sparse attention.
    """
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256], n_bdh_layers=3, n_head=4):
        super().__init__()
        self.n_bdh_layers = n_bdh_layers
        
        # Standard convolutional encoding to get features
        # Bottom level (high resolution)
        self.conv_bottom_in = nn.Conv2d(in_channels, hidden_dims[0] // 2, 4, stride=2, padding=1)
        self.conv_bottom_mid = nn.Conv2d(hidden_dims[0] // 2, hidden_dims[0], 4, stride=2, padding=1)
        self.res_bottom = nn.ModuleList([ResidualBlock(hidden_dims[0]) for _ in range(2)])
        
        # Top level (low resolution)
        self.conv_top = nn.Conv2d(hidden_dims[0], hidden_dims[1], 4, stride=2, padding=1)
        self.res_top = nn.ModuleList([ResidualBlock(hidden_dims[1]) for _ in range(2)])
        
        # BDH refinement blocks (applied to flattened spatial features)
        # These use the BDH attention mechanism for feature refinement
        self.bdh_blocks_bottom = nn.ModuleList([
            BDHBlock(hidden_dims[0], n_head, mlp_multiplier=4) 
            for _ in range(n_bdh_layers)
        ])
        self.bdh_blocks_top = nn.ModuleList([
            BDHBlock(hidden_dims[1], n_head, mlp_multiplier=4) 
            for _ in range(n_bdh_layers)
        ])
        # Trying BDH in Decoder only (No BDH in Encoder)
        # self.bdh_blocks_bottom = nn.ModuleList([
        # ])
        # self.bdh_blocks_top = nn.ModuleList([
        # ])
        
        # Output projections
        self.conv_bottom_out = nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1)
        self.conv_top_out = nn.Conv2d(hidden_dims[1], hidden_dims[1], 3, padding=1)
        
    def forward(self, x):
        # Initial CNN encoding
        x = F.relu(self.conv_bottom_in(x))
        x = F.relu(self.conv_bottom_mid(x))
        for block in self.res_bottom:
            x = block(x)
        z_bottom = x
        
        x = F.relu(self.conv_top(z_bottom))
        for block in self.res_top:
            x = block(x)
        z_top = x
        
        # BDH refinement on bottom features
        # Reshape for attention: [B, C, H, W] -> [B, H*W, C]
        B, C_bottom, H_bottom, W_bottom = z_bottom.shape
        z_bottom_flat = z_bottom.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        # Apply BDH blocks (recurrent processing with shared weights)
        for bdh_block in self.bdh_blocks_bottom:
            z_bottom_flat = bdh_block(z_bottom_flat)
        
        # Reshape back: [B, H*W, C] -> [B, C, H, W]
        z_bottom = z_bottom_flat.transpose(1, 2).reshape(B, C_bottom, H_bottom, W_bottom)
        
        # BDH refinement on top features
        B, C_top, H_top, W_top = z_top.shape
        z_top_flat = z_top.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        for bdh_block in self.bdh_blocks_top:
            z_top_flat = bdh_block(z_top_flat)
        
        z_top = z_top_flat.transpose(1, 2).reshape(B, C_top, H_top, W_top)
        
        # Final output projections
        z_bottom = self.conv_bottom_out(z_bottom)
        z_top = self.conv_top_out(z_top)
        
        return z_bottom, z_top


class BDHDecoder(nn.Module):
    """
    Decoder with BDH (Baby Dragon Hatchling) attention refinement.
    Uses bio-inspired sparse attention for feature processing.
    """
    
    def __init__(self, out_channels=3, hidden_dims=[128, 256], n_bdh_layers=3, n_head=4):
        super().__init__()
        # Input processing
        self.conv_top_in = nn.Conv2d(hidden_dims[1], hidden_dims[1], 3, padding=1)
        self.conv_bottom_in = nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1)
        
        # BDH refinement blocks
        self.bdh_blocks_top = nn.ModuleList([
            BDHBlock(hidden_dims[1], n_head, mlp_multiplier=4) 
            for _ in range(n_bdh_layers)
        ])
        self.bdh_blocks_bottom = nn.ModuleList([
            BDHBlock(hidden_dims[0], n_head, mlp_multiplier=4) 
            for _ in range(n_bdh_layers)
        ])
        # self.bdh_blocks_top = nn.ModuleList([])
        # self.bdh_blocks_bottom = nn.ModuleList([])
        
        # Residual blocks
        self.res_top = nn.ModuleList([ResidualBlock(hidden_dims[1]) for _ in range(2)])
        self.res_bottom = nn.ModuleList([ResidualBlock(hidden_dims[0]) for _ in range(2)])
        
        # Upsampling layers
        self.upsample_top = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 4, stride=2, padding=1)
        self.merge = nn.Conv2d(hidden_dims[0] * 2, hidden_dims[0], 3, padding=1)
        self.upsample_mid = nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0] // 2, 4, stride=2, padding=1)
        self.upsample_out = nn.ConvTranspose2d(hidden_dims[0] // 2, out_channels, 4, stride=2, padding=1)
        
    def forward(self, z_bottom_q, z_top_q):
        # Process inputs
        z_top = self.conv_top_in(z_top_q)
        z_bottom = self.conv_bottom_in(z_bottom_q)
        
        # BDH refinement on top features
        B, C_top, H_top, W_top = z_top.shape
        z_top_flat = z_top.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        for bdh_block in self.bdh_blocks_top:
            z_top_flat = bdh_block(z_top_flat)
        
        z_top = z_top_flat.transpose(1, 2).reshape(B, C_top, H_top, W_top)
        
        # BDH refinement on bottom features
        B, C_bottom, H_bottom, W_bottom = z_bottom.shape
        z_bottom_flat = z_bottom.flatten(2).transpose(1, 2)  # [B, H*W, C]
        
        for bdh_block in self.bdh_blocks_bottom:
            z_bottom_flat = bdh_block(z_bottom_flat)
        
        z_bottom = z_bottom_flat.transpose(1, 2).reshape(B, C_bottom, H_bottom, W_bottom)
        
        # Final decoding with residual blocks
        for block in self.res_top:
            z_top = block(z_top)
        for block in self.res_bottom:
            z_bottom = block(z_bottom)
        
        # Merge levels
        z_top_upsampled = F.relu(self.upsample_top(z_top))
        x = torch.cat([z_top_upsampled, z_bottom], dim=1)
        x = F.relu(self.merge(x))
        
        # Upsample to image
        x = F.relu(self.upsample_mid(x))
        return self.upsample_out(x)


class BDHVQVAE(nn.Module):
    """
    VQ-VAE with Baby Dragon Hatchling (BDH) attention layers.
    
    BDH features:
    - Bio-inspired sparse attention with ReLU (no softmax)
    - Q=K constraint (self-similarity matrix)
    - Multiplicative gating instead of additive residuals
    - RoPE position embeddings
    - Recurrent processing with shared weights
    
    These attention blocks refine features before and after quantization.
    """
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256], 
                 num_embeddings=[512, 512], embedding_dims=[64, 64],
                 n_bdh_layers=3, n_head=4, commitment_cost=0.25):
        super().__init__()
        self.encoder = BDHEncoder(in_channels, hidden_dims, n_bdh_layers, n_head)
        self.decoder = BDHDecoder(in_channels, hidden_dims, n_bdh_layers, n_head)
        
        # Quantizers
        self.quantizer_bottom = VectorQuantizer(
            num_embeddings[0], embedding_dims[0], commitment_cost
        )
        self.quantizer_top = VectorQuantizer(
            num_embeddings[1], embedding_dims[1], commitment_cost
        )
        
        # Projection layers
        self.pre_quant_bottom = nn.Conv2d(hidden_dims[0], embedding_dims[0], 1)
        self.post_quant_bottom = nn.Conv2d(embedding_dims[0], hidden_dims[0], 1)
        self.pre_quant_top = nn.Conv2d(hidden_dims[1], embedding_dims[1], 1)
        self.post_quant_top = nn.Conv2d(embedding_dims[1], hidden_dims[1], 1)
        
    def forward(self, x):
        # Encode with BDH attention refinement
        z_e_bottom, z_e_top = self.encoder(x)
        
        # Quantize
        z_e_bottom = self.pre_quant_bottom(z_e_bottom)
        z_q_bottom, quant_loss_bottom, perp_bottom, indices_bottom = self.quantizer_bottom(z_e_bottom)
        z_q_bottom = self.post_quant_bottom(z_q_bottom)
        
        z_e_top = self.pre_quant_top(z_e_top)
        z_q_top, quant_loss_top, perp_top, indices_top = self.quantizer_top(z_e_top)
        z_q_top = self.post_quant_top(z_q_top)
        
        # Decode with BDH attention refinement
        x_recon = self.decoder(z_q_bottom, z_q_top)
        
        return {
            'reconstruction': x_recon,
            'quantizer_loss': quant_loss_bottom + quant_loss_top,
            'perplexity': (perp_bottom + perp_top) / 2,
            'indices_bottom': indices_bottom,
            'indices_top': indices_top
        }
    
    def encode(self, x):
        """Encode to discrete codes."""
        z_e_bottom, z_e_top = self.encoder(x)
        z_e_bottom = self.pre_quant_bottom(z_e_bottom)
        z_e_top = self.pre_quant_top(z_e_top)
        _, _, _, indices_bottom = self.quantizer_bottom(z_e_bottom)
        _, _, _, indices_top = self.quantizer_top(z_e_top)
        return indices_bottom, indices_top
    
    def decode_from_indices(self, indices_bottom, indices_top):
        """Decode from discrete codes."""
        z_q_bottom = self.quantizer_bottom.embeddings(indices_bottom).permute(0, 3, 1, 2)
        z_q_bottom = self.post_quant_bottom(z_q_bottom)
        
        z_q_top = self.quantizer_top.embeddings(indices_top).permute(0, 3, 1, 2)
        z_q_top = self.post_quant_top(z_q_top)
        
        return self.decoder(z_q_bottom, z_q_top)