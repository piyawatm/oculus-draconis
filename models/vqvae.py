# vqvae.py
# VQ-VAE for BDH+VQ project (PyTorch)
# - EMA vector quantization (VQ-VAE v2 style)
# - CIFAR-friendly defaults: 32x32 -> 8x8 latent grid (downsample_factor=4)
# - Slightly deeper encoder/decoder with more channels
# - Recon loss = 0.5 * MSE + 0.5 * L1
#
# API:
#   encode(x) -> [B,Hc,Wc] int codes
#   decode(indices) -> [B,C,H,W] image
#   forward(x) -> (recon, loss_dict, indices)

from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Building blocks
# -------------------------
class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.SiLU(),                              # non-inplace
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class Encoder(nn.Module):
    """
    Default: 32x32 -> 8x8 with downsample_factor=4 (two stride-2 convs).
    """
    def __init__(
        self,
        in_channels: int = 3,
        channels: int = 192,
        embed_dim: int = 256,
        downsample_factor: int = 4,
    ):
        super().__init__()
        assert downsample_factor in (4, 8, 16), "downsample_factor must be 4, 8, or 16"

        layers = []
        # First downsample (32 -> 16)
        layers += [nn.Conv2d(in_channels, channels, 4, stride=2, padding=1), nn.SiLU()]
        # Additional downsamples if needed
        factor = 2
        c = channels
        while factor < downsample_factor:
            layers += [nn.Conv2d(c, c, 4, stride=2, padding=1), nn.SiLU()]
            factor *= 2

        # Bottleneck residual stack (deeper: 3 blocks)
        layers += [
            ResidualBlock(c),
            ResidualBlock(c),
            ResidualBlock(c),
            nn.SiLU(),
        ]
        # Project to embedding dim D
        layers += [nn.Conv2d(c, embed_dim, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D, Hc, Wc]
        return self.net(x)


class Decoder(nn.Module):
    """
    Mirror of Encoder. Reconstructs images from quantized latents [B, D, Hc, Wc].
    """
    def __init__(
        self,
        out_channels: int = 3,
        channels: int = 192,
        embed_dim: int = 256,
        upsample_factor: int = 4,
    ):
        super().__init__()
        assert upsample_factor in (4, 8, 16), "upsample_factor must be 4, 8, or 16"

        layers = []
        # Project from D to hidden channels
        layers += [
            nn.Conv2d(embed_dim, channels, 3, padding=1),
            ResidualBlock(channels),
            ResidualBlock(channels),
            ResidualBlock(channels),  # deeper bottleneck
            nn.SiLU(),
        ]
        # Progressive upsampling by factor 2
        factor = 1
        while factor < upsample_factor:
            layers += [nn.ConvTranspose2d(channels, channels, 4, stride=2, padding=1), nn.SiLU()]
            factor *= 2
        # Final conv to image channels
        layers += [nn.Conv2d(channels, out_channels, 1)]

        self.net = nn.Sequential(*layers)

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        # [B, C, H, W] (range not clamped)
        return self.net(z_q)


class VectorQuantizerEMA(nn.Module):
    """
    EMA Vector Quantizer (VQ-VAE v2 style):
      - codebook: K x D
      - input: z_e [B, D, H, W]
      - output: z_q_st [B, D, H, W], indices [B, H, W]
    Maintains EMA cluster size and embedding means.

    Loss terms:
      - codebook: || sg[z_e] - e ||^2
      - commitment: beta * || z_e - sg[e] ||^2
    """

    def __init__(
        self,
        codebook_size: int,
        embed_dim: int,
        beta: float = 0.25,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.beta = beta
        self.decay = decay
        self.eps = eps

        # Codebook embeddings
        self.embedding = nn.Embedding(codebook_size, embed_dim)
        nn.init.uniform_(self.embedding.weight, -1.0 / codebook_size, 1.0 / codebook_size)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(codebook_size))
        self.register_buffer("ema_weight", self.embedding.weight.data.clone())

    @torch.no_grad()
    def _nearest_code_indices(self, z_e: torch.Tensor) -> torch.Tensor:
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)               # [B*H*W, D]
        x2 = (flat ** 2).sum(dim=1, keepdim=True)                   # [N,1]
        e2 = (self.embedding.weight ** 2).sum(dim=1)                # [K]
        x_e = flat @ self.embedding.weight.t()                      # [N,K]
        distances = x2 + e2.unsqueeze(0) - 2 * x_e                  # [N,K]
        indices = torch.argmin(distances, dim=1)                    # [N]
        return indices.view(B, H, W)                                # [B,H,W]

    @torch.no_grad()
    def _ema_update(self, z_e: torch.Tensor, indices: torch.Tensor) -> None:
        """
        EMA updates for cluster sizes and embedding weights.
        """
        B, D, H, W = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, D)               # [N,D]
        indices_flat = indices.view(-1)                             # [N]

        # One-hot assignments
        one_hot = F.one_hot(indices_flat, num_classes=self.codebook_size).float()  # [N,K]

        # Update cluster size
        cluster_size = one_hot.sum(dim=0)                           # [K]
        self.ema_cluster_size.mul_(self.decay).add_(cluster_size, alpha=1 - self.decay)

        # Update ema_weight
        embed_sum = one_hot.t() @ flat                              # [K,D]
        self.ema_weight.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        # Normalize cluster size with Laplace smoothing
        n = self.ema_cluster_size.sum()
        cluster_size = (
            (self.ema_cluster_size + self.eps)
            / (n + self.codebook_size * self.eps)
            * n
        )

        # Normalize ema_weight to get updated embeddings
        self.embedding.weight.data.copy_(self.ema_weight / cluster_size.unsqueeze(1))

    def forward(self, z_e: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        B, D, H, W = z_e.shape

        # 1) Find nearest code indices
        with torch.no_grad():
            indices = self._nearest_code_indices(z_e)               # [B,H,W]
            self._ema_update(z_e, indices)

        # 2) quantized embeddings
        z_q = self.embedding(indices).permute(0, 3, 1, 2).contiguous()  # [B,D,H,W]

        # 3) straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()

        # 4) losses
        codebook_loss = F.mse_loss(z_q.detach(), z_e)       # || sg[z_e] - e ||^2
        commitment_loss = self.beta * F.mse_loss(z_e, z_q.detach())

        # 5) perplexity (effective code usage)
        with torch.no_grad():
            one_hot = F.one_hot(indices, num_classes=self.codebook_size).float()  # [B,H,W,K]
            avg_probs = one_hot.mean(dim=(0, 1, 2)) + 1e-10
            perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs)))

        aux = {
            "codebook": codebook_loss,
            "commitment": commitment_loss,
            "perplexity": perplexity,
            "indices": indices,
        }
        return z_q_st, indices, aux


# -------------------------
# Main VQ-VAE module
# -------------------------
class VQVAE(nn.Module):
    """
    Vector-Quantized VAE suitable for BDH prior training.
      - For 32x32 images with downsample_factor=4, code grid is 8x8 (T=64).
      - encode(x)  -> [B,Hc,Wc] int codes
      - decode(ids)-> [B,C,H,W] images
      - forward(x) -> (recon, loss_dict, indices)
    """
    def __init__(
        self,
        codebook_size: int = 512,
        embed_dim: int = 256,
        encoder_channels: int = 192,
        decoder_channels: int = 192,
        beta: float = 0.25,
        downsample_factor: int = 4,
        in_channels: int = 3,
        out_channels: int = 3,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.embed_dim = embed_dim
        self.downsample_factor = downsample_factor

        self.encoder = Encoder(
            in_channels=in_channels,
            channels=encoder_channels,
            embed_dim=embed_dim,
            downsample_factor=downsample_factor,
        )
        self.vq = VectorQuantizerEMA(
            codebook_size=codebook_size,
            embed_dim=embed_dim,
            beta=beta,
        )
        self.decoder = Decoder(
            out_channels=out_channels,
            channels=decoder_channels,
            embed_dim=embed_dim,
            upsample_factor=downsample_factor,
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return discrete code indices grid [B,Hc,Wc] (dtype long).
        """
        z_e = self.encoder(x)
        _, indices, _ = self.vq(z_e)  # only need indices
        return indices

    def decode(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode from code indices [B,Hc,Wc] to image [B,C,H,W].
        """
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2).contiguous()  # [B,D,Hc,Wc]
        x_hat = self.decoder(z_q)
        return x_hat

    def forward(self, x: torch.Tensor):
        """
        Returns:
            recon: [B,C,H,W]
            loss_dict: {"recon","codebook","commitment","perplexity"}
            indices: [B,Hc,Wc] int64
        """
        z_e = self.encoder(x)                           # [B,D,Hc,Wc]
        z_q, indices, aux = self.vq(z_e)                # [B,D,Hc,Wc], [B,Hc,Wc]
        x_hat = self.decoder(z_q)                       # [B,C,H,W]

        # Mixed recon loss (assumes inputs in [0,1])
        recon_l2 = F.mse_loss(x_hat, x)
        recon_l1 = F.l1_loss(x_hat, x)
        recon_loss = 0.5 * recon_l2 + 0.5 * recon_l1

        loss_dict = {
            "recon": recon_loss,
            "codebook": aux["codebook"],
            "commitment": aux["commitment"],
            "perplexity": aux["perplexity"],
        }
        return x_hat, loss_dict, indices


# -------------------------
# Tiny sanity check
# -------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    model = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4).to(device)
    x = torch.rand(4, 3, 32, 32, device=device)  # fake CIFAR batch in [0,1]
    x_hat, losses, idx = model(x)
    total = losses["recon"] + losses["codebook"] + losses["commitment"]
    print("shapes:", x.shape, x_hat.shape, idx.shape)
    print("perplexity:", float(losses["perplexity"]))
    print("loss:", float(total.detach().cpu()))
