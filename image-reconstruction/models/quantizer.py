import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """Vector Quantization layer for VQ-VAE models."""
    
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # Codebook
        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)
        
    def forward(self, x):
        # x: [B, C, H, W]
        B, C, H, W = x.shape
        
        # Flatten input: [B, C, H, W] -> [B*H*W, C]
        flat_x = x.permute(0, 2, 3, 1).contiguous().view(-1, self.embedding_dim)
        
        # Calculate distances to codebook
        distances = (
            torch.sum(flat_x ** 2, dim=1, keepdim=True) +
            torch.sum(self.embeddings.weight ** 2, dim=1) -
            2 * torch.matmul(flat_x, self.embeddings.weight.t())
        )
        
        # Get nearest codebook entries
        encoding_indices = torch.argmin(distances, dim=1)
        quantized = self.embeddings(encoding_indices).view(B, H, W, C).permute(0, 3, 1, 2)
        
        # Compute losses
        e_latent_loss = F.mse_loss(quantized.detach(), x)
        q_latent_loss = F.mse_loss(quantized, x.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # Straight-through estimator
        quantized = x + (quantized - x).detach()
        
        # Calculate perplexity for monitoring codebook usage
        avg_probs = torch.mean(
            F.one_hot(encoding_indices, self.num_embeddings).float(), dim=0
        )
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity, encoding_indices.view(B, H, W)


class ResidualBlock(nn.Module):
    """Residual block with two convolutions."""
    
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return F.relu(x + residual)