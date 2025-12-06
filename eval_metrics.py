# eval_metrics.py — PixelCNN + VQ-VAE (FFHQ-64) with FID, IS, KID + model stats

import math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from datasets import load_from_disk, load_dataset

from models.vqvae import VQVAE
from models.priors.pixelcnn import PixelCNNPrior
from utils import load_config
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config + models ----------------
# VQ-VAE
vq_conf = load_config("configs/vqvae_config.yaml")
vq = VQVAE(vq_conf).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))

K_code = int(vq_conf.num_embeddings)  # e.g., 1024
K_vocab = K_code + 1                  # +1 for BOS

# PixelCNN prior
prior_cfg = yaml.safe_load(open("configs/prior_pixelcnn.yaml"))
model_cfg = dict(prior_cfg["model"])
model_name = model_cfg.pop("name", "PixelCNNPrior")
assert model_name == "PixelCNNPrior", f"eval_metrics.py is set up for PixelCNNPrior, got {model_name}"

block_size = int(prior_cfg["model"]["block_size"])  # e.g., 256
side = int(math.isqrt(block_size))                  # e.g., 16
assert side * side == block_size, f"block_size {block_size} is not a perfect square"

cfg_vocab = int(prior_cfg["model"]["vocab_size"])
assert cfg_vocab == K_vocab, (
    f"vocab_size mismatch: prior cfg has {cfg_vocab}, "
    f"but VQ-VAE has num_embeddings {K_code} → vocab_size {K_vocab}"
)

prior = PixelCNNPrior(**model_cfg).to(device).eval()
prior.load_state_dict(torch.load(prior_cfg["train"]["save_path"], map_location=device))

# ---------------- constants ----------------
B = 64
TARGET_SAMPLES = 10000
ENABLE_KID = True

# ---------------- helpers ----------------
def to_u8(x: torch.Tensor) -> torch.Tensor:
    """float in [0,1] → uint8 in [0,255]"""
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)


def count_parameters(model) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory(fn, *args, **kwargs) -> float:
    """Peak CUDA memory (MB) for a single forward pass of fn(*args, **kwargs)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = fn(*args, **kwargs)
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


@torch.no_grad()
def decode_codes(vq_model: VQVAE, codes: torch.Tensor) -> torch.Tensor:
    """
    codes: [B, Hc, Wc] integer indices in [0, K_code-1]
    returns: images [B, 3, 64, 64] in [0,1]
    """
    B, Hc, Wc = codes.shape
    flat = codes.view(B, -1)                                    # [B, T]
    z = vq_model.quantizer.embedding(flat)                      # [B, T, D]
    z = z.view(B, Hc, Wc, -1).permute(0, 3, 1, 2).contiguous()  # [B, D, Hc, Wc]
    x = vq_model.decoder(z)                                     # typically [-1,1]
    x = (x + 1.0) / 2.0                                         # → [0,1]
    return x.clamp(0.0, 1.0)


@torch.no_grad()
def sample_pixelcnn(prior, side: int, temperature: float = 1.0, batch_size: int = 64) -> torch.Tensor:
    """
    Raster-scan PixelCNN sampling over side x side grid.
    Produces integer codes in [0, K_code-1].
    """
    codes = torch.zeros(batch_size, side, side, dtype=torch.long, device=device)
    bos_id = K_code  # last vocab index

    for i in range(side):
        for j in range(side):
            flat = codes.view(batch_size, -1)                  # [B, T]
            flat = flat.clamp(0, K_vocab - 1)

            logits = prior(flat)                               # [B, T, K_vocab]
            logits = logits.view(batch_size, side, side, K_vocab)

            logits_ij = logits[:, i, j, :] / max(temperature, 1e-8)

            # forbid BOS
            logits_ij[:, bos_id] = float("-inf")

            probs = F.softmax(logits_ij, dim=-1)               # [B, K_vocab]
            next_id = torch.multinomial(probs, 1).squeeze(-1)  # [B]
            next_id = next_id.clamp(0, K_code - 1)

            codes[:, i, j] = next_id

    return codes  # [B, side, side]


# ---------------- real dataset: FFHQ-64 from disk ----------------
print("Loading FFHQ-64 real images for metrics...")

transform = transforms.ToTensor()  # [0,1]

try:
    ds = load_from_disk("data/ffhq64_local")
except Exception:
    try:
        ds = load_from_disk("data/ffhq64_test_subset")
    except Exception:
        print("Local FFHQ dataset not found, downloading now...")
        ds = load_dataset("Dmini/FFHQ-64x64", split="train")
        ds.save_to_disk("data/ffhq64_local")

def collate_fn(examples):
    imgs = [transform(ex["image"].convert("RGB")) for ex in examples]
    return torch.stack(imgs)

real_loader = DataLoader(
    ds,
    batch_size=B,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)

# ---------------- metrics ----------------
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)
kid = KernelInceptionDistance(subset_size=50).to(device) if ENABLE_KID else None

# real images → uint8
print("Accumulating real features...")
for x in real_loader:
    x_u8 = to_u8(x).to(device, non_blocking=True)
    fid.update(x_u8, real=True)
    if ENABLE_KID:
        kid.update(x_u8, real=True)

# ---------------- generate & evaluate ----------------
print("Generating samples from PixelCNN + VQ-VAE...")
seen = 0
while seen < TARGET_SAMPLES:
    b = min(B, TARGET_SAMPLES - seen)

    # 1) sample latent codes
    codes = sample_pixelcnn(prior, side=side, temperature=1.0, batch_size=b)  # [b, side, side]

    # 2) decode to images [b,3,64,64] in [0,1]
    imgs = decode_codes(vq, codes.to(device))

    # 3) uint8 for metrics
    imgs_u8 = to_u8(imgs).to(device)

    fid.update(imgs_u8, real=False)
    iscore.update(imgs_u8)
    if ENABLE_KID:
        kid.update(imgs_u8, real=False)

    seen += b
    print(f"Generated {seen}/{TARGET_SAMPLES}", end="\r")

print("\n\n--- Evaluation Metrics ---")
print(f"FID: {float(fid.compute()):.3f}")

if ENABLE_KID:
    kid_mean, kid_std = kid.compute()
    print(f"KID: {float(kid_mean):.3f} +/- {float(kid_std):.3f}")

m, s = iscore.compute()
print(f"Inception Score: {float(m):.3f} +/- {float(s):.3f}")

# ---------------- model stats ----------------
print("\n--- Model Stats ---")
prior_params = count_parameters(prior)
print(f"PixelCNN Prior Parameters: {prior_params / 1e6:.2f}M")

if torch.cuda.is_available():
    dummy_idx = torch.randint(0, K_vocab, (1, block_size), device=device)
    prior_mem = measure_memory(prior, dummy_idx)
    print(f"PixelCNN Prior Memory Footprint (1 forward): {prior_mem:.1f} MB")
else:
    print("CUDA not available — skipping memory footprint measurement.")
