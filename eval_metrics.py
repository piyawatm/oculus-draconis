# eval_metrics.py (fixed dtype handling + fewer workers)
import torch
import os
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior  # or GPTPrior

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T, Hc, Wc, K = 64, 64, 8, 8, 512

# ------------- helpers -------------
def to_u8(x: torch.Tensor) -> torch.Tensor:
    # expects float in [0,1]; returns uint8 in [0,255]
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)

def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_memory(model, input_sample):
    """Measure GPU memory footprint (MB) during a single forward pass."""
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    with torch.no_grad():
        _ = model(input_sample)
    torch.cuda.synchronize()
    mem_used = torch.cuda.max_memory_allocated() / (1024 ** 2)
    return mem_used

# ------------- real data -------------
tfm = transforms.ToTensor()  # float in [0,1]
real = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
real_loader = DataLoader(real, batch_size=B, shuffle=False, num_workers=2, pin_memory=True)

# ------------- models -------------
vq = VQVAE(codebook_size=K, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

K_code = 512
K_vocab = K_code + 1
Hc, Wc = 8, 8
T = Hc * Wc
prior = BDHPrior(vocab_size=K_vocab, d_model=256, block_size=T+1).to(device).eval()
prior.load_state_dict(torch.load("checkpoints/bdh_prior.pt", map_location=device))

# >>> Model statistics section
print("\n--- Model Stats ---")

# Parameter counts
vq_params = count_parameters(vq)
prior_params = count_parameters(prior)
print(f"VQVAE Parameters: {vq_params/1e6:.2f}M")
print(f"BDH Prior Parameters: {prior_params/1e6:.2f}M")

# Memory footprint (forward pass)
if torch.cuda.is_available():
    sample_codes = torch.randint(0, K, (1, Hc, Wc), device=device)
    vq_mem = measure_memory(vq.decode, sample_codes)

    bos = torch.full((1, 1), K_code, dtype=torch.long, device=device)
    prior_mem = measure_memory(prior, bos)

    print(f"VQVAE Memory Footprint: {vq_mem:.1f} MB")
    print(f"BDH Prior Memory Footprint: {prior_mem:.1f} MB")
else:
    print("CUDA not available — skipping memory footprint measurement.")

# Checkpoint sizes
vq_size = os.path.getsize("checkpoints/vqvae.pt") / (1024 ** 2)
prior_size = os.path.getsize("checkpoints/bdh_prior.pt") / (1024 ** 2)
print(f"Checkpoint sizes — VQVAE: {vq_size:.1f} MB, BDH: {prior_size:.1f} MB\n")


# ------------- metrics -------------
fid = FrechetInceptionDistance(feature=2048).to(device)
kid = KernelInceptionDistance(subset_size=50).to(device)
iscore = InceptionScore().to(device)

# real features (uint8)
for x, _ in real_loader:
    x_u8 = to_u8(x).to(device, non_blocking=True)  # uint8 NCHW
    fid.update(x_u8, real=True)
    kid.update(x_u8, real=True)

# generate at least ~10k images
target = 10000
seen = 0

Hc, Wc = 8, 8
T = Hc * Wc
K_code = 512
K_vocab = K_code + 1
bos_id = K_code

while seen < target:
    b = min(B, target - seen)
    bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    ids = prior.generate(bos, max_new_tokens=T)            # [b,65]
    codes = ids[:, 1:].view(b, Hc, Wc)                     # [b,8,8]
    imgs = vq.decode(codes).clamp(0, 1)
    imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)

    fid.update(imgs_u8.to(device), real=False)
    kid.update(imgs_u8.to(device), real=False)
    iscore.update(imgs_u8.to(device))

    seen += b


print("FID:", float(fid.compute()))
kid_mean, kid_std = kid.compute()
print("KID:", float(kid_mean), "+/-", float(kid_std))
m, s = iscore.compute()
print("IS:", float(m), "+/-", float(s))
