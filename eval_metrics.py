# eval_metrics.py (fixed dtype handling + fewer workers)
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior  # or GPTPrior
import torch.nn.functional as F
import os
import sys
from torch.amp import autocast

# ------------- logging setup (no terminal prints) -------------
log_path = "logs/eval_metrics_log.txt"
os.makedirs("logs", exist_ok=True)
log_file = open(log_path, "w")

def log(msg: str):
    log_file.write(msg + "\n")
    log_file.flush()

# Silence stdout
sys.stdout = open(os.devnull, "w")

device = "cuda" if torch.cuda.is_available() else "cpu"
B, T, Hc, Wc, K = 64, 64, 8, 8, 512

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
prior = BDHPrior(
    vocab_size=K_vocab,         # 513
    d_model=256,
    n_layer=6,                  # whatever you used in train_prior
    n_head=4,
    block_size=T+1,             # 65
    mlp_internal_dim_multiplier=128,  # <<< match training
    dropout=0.1,                # or your train-time value
).to(device).eval()
prior.load_state_dict(torch.load("checkpoints/bdh_prior.pt", map_location=device))

# ------------- metrics -------------
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)

# real features (uint8)
for x, _ in real_loader:
    x_u8 = to_u8(x).to(device, non_blocking=True)  # uint8 NCHW
    fid.update(x_u8, real=True)

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

    with torch.no_grad(), autocast("cuda", dtype=torch.float16):
        codes_seq = generate_without_bos(
            prior, bos, steps=T, bos_id=bos_id, temperature=1.0, top_k=50
        )  # [b,64]

        codes = codes_seq.view(b, Hc, Wc)
        imgs = vq.decode(codes).clamp(0, 1)

    imgs_u8 = (imgs.clamp(0, 1) * 255.0).to(torch.uint8)
    fid.update(imgs_u8.to(device), real=False)
    iscore.update(imgs_u8.to(device))

    # free intermediates
    del codes_seq, codes, imgs, imgs_u8
    torch.cuda.empty_cache()

    seen += b


log(f"FID: {float(fid.compute())}")
m, s = iscore.compute()
log(f"IS: {float(m)} +/- {float(s)}")
log_file.close()
