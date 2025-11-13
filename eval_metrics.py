# eval_metrics.py (FID, IS, KID + model stats)
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from models.priors.pixelsnail import PixelSNAILPrior
import torch.nn.functional as F

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- constants ----
B, Hc, Wc, K_code = 64, 8, 8, 512
T = Hc * Wc
K_vocab = K_code + 1
bos_id = K_code
ENABLE_KID = True

# ---- helpers ----
def to_u8(x: torch.Tensor) -> torch.Tensor:
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)

def generate_without_bos(prior, bos, steps, bos_id, temperature=1.0, top_k=None):
    ids = bos
    for _ in range(steps):
        idx_cond = ids[:, -getattr(prior, "block_size", ids.size(1)):]
        logits = prior(idx_cond)[:, -1, :] / max(1e-8, temperature)
        logits[:, bos_id] = float("-inf")
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, 1:]

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_memory(fn, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = fn(*args, **kwargs)
    return torch.cuda.max_memory_allocated() / (1024**2)

# ---- dataset ----
tfm = transforms.ToTensor()
real = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
real_loader = DataLoader(real, batch_size=B, shuffle=False, num_workers=2, pin_memory=True)

# ---- models ----
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = PixelSNAILPrior(
    vocab_size=K_vocab,
    d_model=256,
    n_layer=6,
    block_size=T+1,
    n_residual=2,
    dropout=0.1,
    tie_embeddings=True,
).to(device).eval()
prior.load_state_dict(torch.load("checkpoints/pixelsnail_prior.pt", map_location=device))

# ---- metrics ----
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)
kid = KernelInceptionDistance(subset_size=50).to(device) if ENABLE_KID else None

# ---- real data updates ----
for x, _ in real_loader:
    x_u8 = to_u8(x).to(device, non_blocking=True)
    fid.update(x_u8, real=True)
    if ENABLE_KID:
        kid.update(x_u8, real=True)

# ---- generate samples ----
target = 10000
seen = 0
while seen < target:
    b = min(B, target - seen)
    bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
    codes_seq = generate_without_bos(prior, bos, steps=T, bos_id=bos_id, temperature=1.0, top_k=50)
    codes = codes_seq.view(b, Hc, Wc).clamp(0, K_code - 1)

    with torch.no_grad():
        imgs = vq.decode(codes).clamp(0, 1)
    imgs_u8 = to_u8(imgs).to(device)

    fid.update(imgs_u8, real=False)
    iscore.update(imgs_u8)
    if ENABLE_KID:
        kid.update(imgs_u8, real=False)

    seen += b

# ---- results ----
print("\n--- Evaluation Metrics ---")
print(f"FID: {float(fid.compute()):.3f}")
if ENABLE_KID:
    kid_mean, kid_std = kid.compute()
    print(f"KID: {float(kid_mean):.3f} +/- {float(kid_std):.3f}")
m, s = iscore.compute()
print(f"Inception Score: {float(m):.3f} +/- {float(s):.3f}")

# ---- model stats ----
print("\n--- Model Stats ---")
vq_params = count_parameters(vq)
prior_params = count_parameters(prior)
print(f"VQVAE Parameters: {vq_params/1e6:.2f}M")
print(f"PixelSNAIL Prior Parameters: {prior_params/1e6:.2f}M")

if torch.cuda.is_available():
    sample_codes = torch.randint(0, K_code, (1, Hc, Wc), device=device)
    vq_mem = measure_memory(vq.decode, sample_codes)

    bos = torch.full((1, 1), K_code, dtype=torch.long, device=device)
    prior_mem = measure_memory(prior, bos)

    print(f"VQVAE Memory Footprint: {vq_mem:.1f} MB")
    print(f"PixelSNAIL Prior Memory Footprint: {prior_mem:.1f} MB")
else:
    print("CUDA not available — skipping memory footprint measurement.")
