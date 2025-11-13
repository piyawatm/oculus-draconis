# eval_metrics.py (PixelCNN version)

import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F

from models.vqvae import VQVAE
from models.priors.pixelcnn import PixelCNNPrior

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- constants ----------------
B = 64
Hc, Wc = 8, 8
T = Hc * Wc      # 64
K_code = 512
K_vocab = K_code + 1   # 513 with BOS

# ---------------- helpers ----------------
def to_u8(x: torch.Tensor) -> torch.Tensor:
    """float in [0,1] → uint8 in [0,255]"""
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)


# ---------------- CIFAR10 real dataset ----------------
tfm = transforms.ToTensor()
real = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)

real_loader = DataLoader(
    real, batch_size=B, shuffle=False, num_workers=2, pin_memory=True
)


# ---------------- load models ----------------
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = PixelCNNPrior(
    vocab_size=K_vocab,
    d_model=128,
    n_layers=12,
    kernel_size=3,
    block_size=T
).to(device).eval()

prior.load_state_dict(torch.load("checkpoints/pixelcnn_prior.pt", map_location=device))


# ---------------- PixelCNN sampling ----------------
@torch.no_grad()
def sample_pixelcnn(prior, side=8, temperature=1.0, batch_size=64):
    """
    Proper 2D autoregressive PixelCNN sampling for VQ-VAE discrete codes.
    """
    codes = torch.zeros(batch_size, side, side, dtype=torch.long, device=device)

    for i in range(side):
        for j in range(side):
            flat = codes.view(batch_size, -1).clamp(0, K_vocab - 1)

            logits = prior(flat)                      # [B, 64, 513]
            logits = logits.view(batch_size, side, side, K_vocab)

            probs = F.softmax(logits[:, i, j, :] / temperature, dim=-1)

            next_id = torch.multinomial(probs, 1).squeeze(-1)
            next_id = next_id.clamp(0, K_code - 1)

            codes[:, i, j] = next_id

    return codes   # [B, side, side]


# ---------------- metrics ----------------
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)

# real images → uint8
for x, _ in real_loader:
    fid.update(to_u8(x).to(device, non_blocking=True), real=True)

# generate ~10k fake images
target = 10000
seen = 0

while seen < target:
    b = min(B, target - seen)

    # 1) sample codes using PixelCNN
    codes = sample_pixelcnn(prior, side=Hc, temperature=1.0, batch_size=b)

    # 2) decode using VQ-VAE
    imgs = vq.decode(codes).clamp(0, 1)

    # 3) convert to uint8 for FID/IS
    imgs_u8 = to_u8(imgs).to(device)

    fid.update(imgs_u8, real=False)
    iscore.update(imgs_u8)

    seen += b
    print(f"Generated {seen}/{target}", end="\r")


# ---------------- results ----------------
print("\n\nFID:", float(fid.compute()))
m, s = iscore.compute()
print("IS:", float(m), "+/-", float(s))
