import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer, ResidualBlock


class Encoder(nn.Module):
    """Encoder for vanilla VQ-VAE."""
    
    def __init__(self, in_channels=3, hidden_dim=128, num_residual_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, hidden_dim // 2, 4, stride=2, padding=1)
        self.conv_mid = nn.Conv2d(hidden_dim // 2, hidden_dim, 4, stride=2, padding=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        self.conv_out = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
    def forward(self, x):
        x = F.relu(self.conv_in(x))
        x = F.relu(self.conv_mid(x))
        
        for block in self.residual_blocks:
            x = block(x)
            
        return self.conv_out(x)


class Decoder(nn.Module):
    """Decoder for vanilla VQ-VAE."""
    
    def __init__(self, out_channels=3, hidden_dim=128, num_residual_blocks=2):
        super().__init__()
        self.conv_in = nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1)
        
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(num_residual_blocks)
        ])
        
        self.conv_mid = nn.ConvTranspose2d(hidden_dim, hidden_dim // 2, 4, stride=2, padding=1)
        self.conv_out = nn.ConvTranspose2d(hidden_dim // 2, out_channels, 4, stride=2, padding=1)
        
    def forward(self, x):
        x = self.conv_in(x)
        
        for block in self.residual_blocks:
            x = block(x)
            
        x = F.relu(self.conv_mid(x))
        return self.conv_out(x)


class VQVAE(nn.Module):
    """Vanilla VQ-VAE model."""
    
    def __init__(self, in_channels=3, hidden_dim=128, num_embeddings=512, 
                 embedding_dim=64, num_residual_blocks=2, commitment_cost=0.25):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, num_residual_blocks)
        self.quantizer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(in_channels, hidden_dim, num_residual_blocks)
        
        # Projection layers to match embedding dimension
        self.pre_quant = nn.Conv2d(hidden_dim, embedding_dim, 1)
        self.post_quant = nn.Conv2d(embedding_dim, hidden_dim, 1)
        
    def forward(self, x):
        # Encode
        z_e = self.encoder(x)
        z_e = self.pre_quant(z_e)
        
        # Quantize
        z_q, quant_loss, perplexity, indices = self.quantizer(z_e)
        
        # Decode
        z_q = self.post_quant(z_q)
        x_recon = self.decoder(z_q)
        
        return {
            'reconstruction': x_recon,
            'quantizer_loss': quant_loss,
            'perplexity': perplexity,
            'indices': indices
        }
    
    def encode(self, x):
        """Encode to discrete codes."""
        z_e = self.encoder(x)
        z_e = self.pre_quant(z_e)
        _, _, _, indices = self.quantizer(z_e)
        return indices
    
    def decode_from_indices(self, indices):
        """Decode from discrete codes."""
        B, H, W = indices.shape
        quantized = self.quantizer.embeddings(indices).permute(0, 3, 1, 2)
        quantized = self.post_quant(quantized)
        return self.decoder(quantized)