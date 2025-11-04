# vq_vae.py
# A simple, "vanilla" VQ-VAE for CIFAR-10

import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """
    The core VQ-VAE layer.
    This layer takes the output of the encoder, finds the closest
    codebook vector (embedding), and returns it.
    """
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        
        # Initialize the codebook (embeddings)
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)
        # Initialize weights to be somewhat spread out
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        # inputs shape: (B, C, H, W)
        # We want to map each (C,) vector in the (H, W) grid
        
        # 1. Reshape to (B * H * W, C)
        # (B, C, H, W) -> (B, H, W, C)
        inputs_permuted = inputs.permute(0, 2, 3, 1).contiguous()
        flat_input = inputs_permuted.view(-1, self.embedding_dim)
        
        # 2. Calculate L2 distances between each input vector and all codebook vectors
        # distances = (x - y)^2 = x^2 + y^2 - 2xy
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
                    + torch.sum(self.embedding.weight**2, dim=1)
                    - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
                    
        # 3. Find the closest codebook vector for each input vector
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        
        # 4. Convert indices back to one-hot vectors
        # We'll use these indices to get the quantized vectors
        encodings_one_hot = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings_one_hot.scatter_(1, encoding_indices, 1)
        
        # 5. Get quantized vectors (the "z_q" from the paper)
        # (B*H*W, num_embeddings) @ (num_embeddings, embedding_dim) -> (B*H*W, embedding_dim)
        quantized = torch.matmul(encodings_one_hot, self.embedding.weight)
        # Reshape back to original grid: (B, H, W, embedding_dim)
        quantized = quantized.view(inputs_permuted.shape)
        
        # 6. Calculate the VQ loss
        # We have two terms:
        # (a) Codebook loss (e_latent_loss): Moves codebook vectors closer to encoder outputs
        e_latent_loss = F.mse_loss(quantized.detach(), inputs_permuted)
        # (b) Commitment loss (q_latent_loss): Moves encoder outputs closer to codebook vectors
        q_latent_loss = F.mse_loss(quantized, inputs_permuted.detach())
        
        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        # 7. Straight-Through Estimator (STE)
        # This is the "trick" to pass gradients back to the encoder.
        # We treat the quantized vector as if it were the original input vector
        # during the backward pass.
        quantized = inputs_permuted + (quantized - inputs_permuted).detach()
        
        # 8. Reshape back to (B, C, H, W) for the decoder
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        
        return loss, quantized

class Encoder(nn.Module):
    """
    Takes a (B, 3, 32, 32) CIFAR-10 image and encodes it
    down to a (B, embedding_dim, 8, 8) feature map.
    """
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.net = nn.Sequential(
            # 32x32 -> 16x16
            nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 8x8
            nn.Conv2d(hidden_channels, embedding_dim, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
            # Output shape: (B, embedding_dim, 8, 8)
        )

    def forward(self, x):
        return self.net(x)

class Decoder(nn.Module):
    """
    Takes a (B, embedding_dim, 8, 8) quantized feature map
    and decodes it back to a (B, 3, 32, 32) image.
    """
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.net = nn.Sequential(
            # 8x8 -> 16x16
            nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            # 16x16 -> 32x32
            nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=4, stride=2, padding=1),
            # We don't use a final sigmoid, as MSELoss is more stable
            # when operating on raw logits.
        )

    def forward(self, x):
        return self.net(x)

class VQVAE(nn.Module):
    """
    The full VQ-VAE model, putting all the pieces together.
    """
    def __init__(self, in_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.vq_layer = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def forward(self, x):
        # 1. Encode
        z_e = self.encoder(x)
        
        # 2. Vector Quantize
        vq_loss, z_q = self.vq_layer(z_e)
        
        # 3. Decode
        x_recon = self.decoder(z_q)
        
        return vq_loss, x_recon