# eval_metrics.py  (PixelCNN + VQ-VAE on FFHQ-64 with torchmetrics)

import math
import yaml
import torch
import torch.nn.functional as F

from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from models.vqvae import VQVAE
from models.priors.pixelcnn import PixelCNNPrior
from train_vqvae import get_data_loader
from utils import load_config

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- config + constants ----------------
BATCH_FAKE = 64          # batch size for generated images
NUM_REAL = 2000          # how many real images to use
NUM_FAKE = 2000          # how many fake images to generate
ENABLE_KID = True

# Load VQ-VAE config (same as training)
vq_conf = load_config("configs/vqvae_config.yaml")

# Load PixelCNN config
prior_cfg = yaml.safe_load(open("configs/prior_pixelcnn.yaml", "r"))
model_cfg = prior_cfg["model"]
train_cfg = prior_cfg["train"]

K_vocab = int(model_cfg["vocab_size"])
block_size = int(model_cfg["block_size"])      # e.g. 256 for 16x16 codes
side = int(math.isqrt(block_size))
assert side * side == block_size, "block_size must be a perfect square (e.g. 256 → 16x16)"

# ---------------- helpers ----------------
def to_01(x: torch.Tensor) -> torch.Tensor:
    """
    Convert images from [-1,1] to [0,1] and clamp.
    """
    return ((x + 1.0) / 2.0).clamp(0.0, 1.0)


def to_u8(x: torch.Tensor) -> torch.Tensor:
    """
    Convert float [0,1] → uint8 [0,255] as required by torch_fidelity backend.
    """
    return (x * 255.0).clamp(0, 255).to(torch.uint8)


def count_parameters(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def measure_memory(fn, *args, **kwargs) -> float:
    """Peak CUDA memory (MB) for a single forward pass of fn(*args, **kwargs)."""
    if not torch.cuda.is_available():
        return 0.0
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = fn(*args, **kwargs)
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


# ---------------- load models ----------------
print("Loading VQ-VAE...")
vq = VQVAE(vq_conf).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))

# quantizer handle (matches your training/sampling code)
if hasattr(vq, "_vq_vae"):
    quantizer = vq._vq_vae
else:
    quantizer = vq.quantizer

print("Loading PixelCNN prior...")
prior = PixelCNNPrior(
    vocab_size=K_vocab,
    d_model=int(model_cfg["d_model"]),
    n_layers=int(model_cfg["n_layers"]),
    kernel_size=int(model_cfg["kernel_size"]),
    block_size=block_size,
    dropout=float(model_cfg.get("dropout", 0.0)),
).to(device).eval()

prior.load_state_dict(torch.load(train_cfg["save_path"], map_location=device))


# ---------------- PixelCNN sampling ----------------
@torch.no_grad()
def sample_pixelcnn(prior, batch_size=64, temperature=1.0):
    """
    Sample discrete code grids [B, side, side] from PixelCNNPrior.
    """
    codes = torch.zeros(batch_size, side, side, dtype=torch.long, device=device)

    for i in range(side):
        for j in range(side):
            flat = codes.view(batch_size, -1).clamp(0, K_vocab - 1)   # [B, block_size]
            logits = prior(flat)                                      # [B, block_size, K_vocab]
            logits = logits.view(batch_size, side, side, K_vocab)     # [B, H, W, K_vocab]

            logits_ij = logits[:, i, j, :] / max(temperature, 1e-8)   # [B, K_vocab]
            probs = F.softmax(logits_ij, dim=-1)
            next_id = torch.multinomial(probs, 1).squeeze(-1)         # [B]
            next_id = next_id.clamp(0, K_vocab - 1)

            codes[:, i, j] = next_id

    return codes  # [B, side, side]


@torch.no_grad()
def decode_codes(codes: torch.Tensor) -> torch.Tensor:
    """
    codes: [B, side, side] LongTensor
    returns images in [0,1] float, shape [B, 3, 64, 64]
    """
    # quantizer.embedding: [num_embeddings, embedding_dim]
    z_q = quantizer.embedding(codes).permute(0, 3, 1, 2).contiguous()
    imgs = vq.decoder(z_q)              # typically in [-1, 1]
    imgs_01 = to_01(imgs)               # [0,1]
    return imgs_01


# ---------------- metrics ----------------
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)
kid = KernelInceptionDistance(subset_size=50).to(device) if ENABLE_KID else None

# ---------------- real images (FFHQ-64 from your loader) ----------------
print("Loading FFHQ-64 real images for metrics...")
real_loader = get_data_loader(batch_size=BATCH_FAKE)

print("Accumulating real features...")
seen_real = 0
for batch in real_loader:
    # get_data_loader returns just images (no labels), normalized to [-1,1]
    x = batch.to(device)
    x_01 = to_01(x)              # [0,1] float
    x_u8 = to_u8(x_01)           # uint8 [0,255] NCHW

    fid.update(x_u8, real=True)
    if ENABLE_KID:
        kid.update(x_u8, real=True)

    # Optional: IS on real images too (harmless)
    iscore.update(x_u8)

    seen_real += x.size(0)
    if seen_real >= NUM_REAL:
        break

# ---------------- fake images (PixelCNN + VQ-VAE) ----------------
print("Generating samples from PixelCNN + VQ-VAE...")
seen_fake = 0

while seen_fake < NUM_FAKE:
    b = min(BATCH_FAKE, NUM_FAKE - seen_fake)

    # 1) sample codes
    codes = sample_pixelcnn(prior, batch_size=b, temperature=1.0)

    # 2) decode to [0,1] float
    imgs_01 = decode_codes(codes).to(device)

    # 3) convert to uint8
    imgs_u8 = to_u8(imgs_01)

    # 4) update metrics
    fid.update(imgs_u8, real=False)
    if ENABLE_KID:
        kid.update(imgs_u8, real=False)
    iscore.update(imgs_u8)

    seen_fake += b
    print(f"Generated {seen_fake}/{NUM_FAKE} fake images", end="\r")

print("\nDone generating fakes.")

# ---------------- results ----------------
print("\n--- Evaluation Metrics (PixelCNN + VQ-VAE) ---")
print(f"FID: {float(fid.compute()):.3f}")

if ENABLE_KID:
    kid_mean, kid_std = kid.compute()
    print(f"KID: {float(kid_mean):.6f} +/- {float(kid_std):.6f}")

is_mean, is_std = iscore.compute()
print(f"Inception Score: {float(is_mean):.3f} +/- {float(is_std):.3f}")

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
