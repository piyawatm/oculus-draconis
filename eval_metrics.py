# eval_metrics.py (fixed dtype handling + fewer workers)
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior  # or GPTPrior
import torch.nn.functional as F
import os
import sys
import glob
from torch.amp import autocast
from utils import Logger

log_path = "logs/eval_metrics_log.txt"

logger = Logger(log_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T, Hc, Wc, K = 64, 64, 8, 8, 512

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def measure_memory(fn, *args, **kwargs):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = fn(*args, **kwargs)
    return torch.cuda.max_memory_allocated() / (1024**2)

@torch.no_grad()
def generate_without_bos(prior, bos, steps, bos_id, temperature=1.0, top_k=None):
    """
    Autoregressive generation:
    - BOS is provided and never generated again.
    - Uses AMP (float16) inside forward pass for memory savings.
    """
    device = bos.device
    ids = bos  # [B, 1]

    for _ in range(steps):
        # respect block_size if model has one
        block = getattr(prior, "block_size", ids.size(1))
        idx_cond = ids[:, -block:]  # [B, <=block]

        # AMP context for memory reduction
        with autocast("cuda", dtype=torch.float16):
            logits = prior(idx_cond)[:, -1, :]  # [B, vocab]
            logits = logits / max(1e-8, temperature)

        # forbid BOS from ever being re-generated
        logits[:, bos_id] = float("-inf")

        # top-k filtering
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)  # [B,1]

        ids = torch.cat([ids, next_id], dim=1)

    return ids[:, 1:]  # drop BOS column

# ------------- helpers -------------
def to_u8(x: torch.Tensor) -> torch.Tensor:
    # expects float in [0,1]; returns uint8 in [0,255]
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)

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

# Find all checkpoint files
checkpoint_pattern = "checkpoints/bdh_prior*.pt"
checkpoint_files = sorted(glob.glob(checkpoint_pattern))

# Loop through each checkpoint
for ckpt_path in checkpoint_files:
    logger.log(f"\n{'='*60}")
    logger.log(f"Evaluating: {ckpt_path}")
    logger.log(f"{'='*60}")
    
    prior = BDHPrior(
        vocab_size=K_vocab,         # 513
        d_model=256,
        n_layer=6,                  # whatever you used in train_prior
        n_head=4,
        block_size=T+1,             # 65
        mlp_internal_dim_multiplier=128,  # <<< match training
        dropout=0.1,                # or your train-time value
    ).to(device).eval()
    prior.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ------------- metrics -------------
    fid = FrechetInceptionDistance(feature=2048).to(device)
    iscore = InceptionScore().to(device)
    kid = KernelInceptionDistance(subset_size=50).to(device)

    # real features (uint8)
    for x, _ in real_loader:
        x_u8 = to_u8(x).to(device, non_blocking=True)  # uint8 NCHW
        fid.update(x_u8, real=True)
        kid.update(x_u8, real=True)

    # generate at least ~10k images
    target = 10000
    seen = 0

    bos_id = K_code

    while seen < target:
        b = min(B, target - seen)
        bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)

        with torch.no_grad(), autocast("cuda", dtype=torch.float16):
            codes_seq = generate_without_bos(
                prior, bos, steps=T, bos_id=bos_id, temperature=1.0, top_k=50
            )  # [b,64]

            codes = codes_seq.view(b, Hc, Wc)
            imgs = vq.decode(codes).clamp(0, 1)

        imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
        fid.update(imgs_u8.to(device), real=False)
        iscore.update(imgs_u8.to(device))
        kid.update(imgs_u8, real=False)

        # free intermediates
        del codes_seq, codes, imgs, imgs_u8
        torch.cuda.empty_cache()

        seen += b

    # ---- results ----
    logger.log("\n--- Evaluation Metrics ---")

    logger.log(f"FID: {float(fid.compute()):.3f}")

    kid_mean, kid_std = kid.compute()
    logger.log(f"KID: {float(kid_mean):.3f} +/- {float(kid_std):.3f}")

    m, s = iscore.compute()
    logger.log(f"Inception Score: {float(m):.3f} +/- {float(s):.3f}")

    # ---- model stats ----
    logger.log("\n--- Model Stats ---")
    prior_params = count_parameters(prior)
    logger.log(f"BDH Prior Parameters: {prior_params/1e6:.2f}M")

    if torch.cuda.is_available():
        sample_codes = torch.randint(0, K_code, (1, Hc, Wc), device=device)
        bos = torch.full((1, 1), K_code, dtype=torch.long, device=device)
        prior_mem = measure_memory(prior, bos)
        logger.log(f"BDH Prior Memory Footprint: {prior_mem:.1f} MB")
    else:
        logger.log("CUDA not available — skipping memory footprint measurement.")

logger.close()
