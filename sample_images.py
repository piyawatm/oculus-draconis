import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.pixelcnn import PixelCNNPrior

# ---------------- Setup ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"

Hc, Wc = 8, 8
T = Hc * Wc

K_code = 512
K_vocab = K_code + 1   # 513 (includes BOS)
B = 16                 # number of images to sample

# ---------------- Load Models ----------------
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = PixelCNNPrior(
    vocab_size=K_vocab,
    d_model=128,
    n_layers=12,
    block_size=T
).to(device).eval()

prior.load_state_dict(torch.load("checkpoints/pixelcnn_prior.pt", map_location=device))

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True


# ---------------- PixelCNN Sampling ----------------
@torch.no_grad()
def sample_pixelcnn(prior, side=8, temperature=1.0, batch_size=16):
    """
    Proper autoregressive 2D sampling for PixelCNN.
    """
    codes = torch.zeros(batch_size, side, side, dtype=torch.long, device=device)

    for i in range(side):
        for j in range(side):
            # Flatten [B, H, W] → [B, T]
            flat = codes.view(batch_size, -1)

            # Clamp indices to valid vocab range
            flat = flat.clamp(0, K_vocab - 1)

            logits = prior(flat)                          # [B, 64, 513]
            logits = logits.view(batch_size, side, side, K_vocab)

            # Select logits for pixel (i, j)
            probs = F.softmax(logits[:, i, j, :] / temperature, dim=-1)

            # Sample next token
            next_id = torch.multinomial(probs, 1).squeeze(-1)

            # Clamp again to valid range
            next_id = next_id.clamp(0, K_code - 1)

            codes[:, i, j] = next_id

    return codes


# ---------------- Generate images ----------------
codes = sample_pixelcnn(prior, side=Hc, temperature=1.0, batch_size=B)  # [B,8,8]

imgs = vq.decode(codes).clamp(0, 1)
save_image(imgs, "samples_pixelcnn.png", nrow=4)

print("✓ Wrote samples_pixelcnn.png")
print("Sampled codes range:", codes.min().item(), "to", codes.max().item())
