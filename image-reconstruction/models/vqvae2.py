import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantizer import VectorQuantizer, ResidualBlock


class HierarchicalEncoder(nn.Module):
    """Hierarchical encoder for VQ-VAE-2."""
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256]):
        super().__init__()
        # Bottom level (high resolution)
        self.conv_bottom_in = nn.Conv2d(in_channels, hidden_dims[0] // 2, 4, stride=2, padding=1)
        self.conv_bottom_mid = nn.Conv2d(hidden_dims[0] // 2, hidden_dims[0], 4, stride=2, padding=1)
        self.res_bottom = nn.ModuleList([ResidualBlock(hidden_dims[0]) for _ in range(2)])
        self.conv_bottom_out = nn.Conv2d(hidden_dims[0], hidden_dims[0], 3, padding=1)
        
        # Top level (low resolution)
        self.conv_top = nn.Conv2d(hidden_dims[0], hidden_dims[1], 4, stride=2, padding=1)
        self.res_top = nn.ModuleList([ResidualBlock(hidden_dims[1]) for _ in range(2)])
        self.conv_top_out = nn.Conv2d(hidden_dims[1], hidden_dims[1], 3, padding=1)
        
    def forward(self, x):
        # Bottom level encoding
        x = F.relu(self.conv_bottom_in(x))
        x = F.relu(self.conv_bottom_mid(x))
        for block in self.res_bottom:
            x = block(x)
        z_bottom = self.conv_bottom_out(x)
        
        # Top level encoding
        x = F.relu(self.conv_top(z_bottom))
        for block in self.res_top:
            x = block(x)
        z_top = self.conv_top_out(x)
        
        return z_bottom, z_top


class HierarchicalDecoder(nn.Module):
    """Hierarchical decoder for VQ-VAE-2."""
    
    def __init__(self, out_channels=3, hidden_dims=[128, 256]):
        super().__init__()
        # Top level decoding
        self.conv_top_in = nn.Conv2d(hidden_dims[1], hidden_dims[1], 3, padding=1)
        self.res_top = nn.ModuleList([ResidualBlock(hidden_dims[1]) for _ in range(2)])
        self.upsample_top = nn.ConvTranspose2d(hidden_dims[1], hidden_dims[0], 4, stride=2, padding=1)
        
        # Bottom level decoding (conditioned on top)
        self.conv_bottom_in = nn.Conv2d(hidden_dims[0] * 2, hidden_dims[0], 3, padding=1)
        self.res_bottom = nn.ModuleList([ResidualBlock(hidden_dims[0]) for _ in range(2)])
        
        # Final upsampling to image
        self.upsample_mid = nn.ConvTranspose2d(hidden_dims[0], hidden_dims[0] // 2, 4, stride=2, padding=1)
        self.upsample_out = nn.ConvTranspose2d(hidden_dims[0] // 2, out_channels, 4, stride=2, padding=1)
        
    def forward(self, z_bottom_q, z_top_q):
        # Decode top level
        x = self.conv_top_in(z_top_q)
        for block in self.res_top:
            x = block(x)
        x = F.relu(self.upsample_top(x))
        
        # Combine with bottom level
        x = torch.cat([x, z_bottom_q], dim=1)
        x = F.relu(self.conv_bottom_in(x))
        for block in self.res_bottom:
            x = block(x)
        
        # Upsample to image
        x = F.relu(self.upsample_mid(x))
        return self.upsample_out(x)


class VQVAE2(nn.Module):
    """VQ-VAE-2 with hierarchical quantization."""
    
    def __init__(self, in_channels=3, hidden_dims=[128, 256], 
                 num_embeddings=[512, 512], embedding_dims=[64, 64],
                 commitment_cost=0.25):
        super().__init__()
        self.encoder = HierarchicalEncoder(in_channels, hidden_dims)
        self.decoder = HierarchicalDecoder(in_channels, hidden_dims)
        
        # Quantizers for each level
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
        # Encode at both levels
        z_e_bottom, z_e_top = self.encoder(x)
        
        # Project and quantize
        z_e_bottom = self.pre_quant_bottom(z_e_bottom)
        z_q_bottom, quant_loss_bottom, perp_bottom, indices_bottom = self.quantizer_bottom(z_e_bottom)
        z_q_bottom = self.post_quant_bottom(z_q_bottom)
        
        z_e_top = self.pre_quant_top(z_e_top)
        z_q_top, quant_loss_top, perp_top, indices_top = self.quantizer_top(z_e_top)
        z_q_top = self.post_quant_top(z_q_top)
        
        # Decode
        x_recon = self.decoder(z_q_bottom, z_q_top)
        
        return {
            'reconstruction': x_recon,
            'quantizer_loss': quant_loss_bottom + quant_loss_top,
            'perplexity': (perp_bottom + perp_top) / 2,
            'indices_bottom': indices_bottom,
            'indices_top': indices_top
        }
    
    def encode(self, x):
        """Encode to discrete codes at both levels."""
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