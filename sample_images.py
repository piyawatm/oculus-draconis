import torch
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior
import torch.nn.functional as F
import os
import sys

device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------- logging setup (no terminal prints) -------------
log_path = "logs/sample_images_log.txt"
os.makedirs("logs", exist_ok=True)
log_file = open(log_path, "w")

def log(msg: str):
    log_file.write(msg + "\n")
    log_file.flush()

# Silence stdout
sys.stdout = open(os.devnull, "w")

def generate_without_bos(prior, bos, steps, bos_id, temperature=1.0, top_k=None):
    """Autoregressive sampling that forbids BOS from being generated."""
    ids = bos  # [B,1]
    for _ in range(steps):
        idx_cond = ids[:, -getattr(prior, "block_size", ids.size(1)):]  # respect block_size if present
        logits = prior(idx_cond)[:, -1, :] / max(1e-8, temperature)      # [B, V]
        # forbid BOS token
        logits[:, bos_id] = float("-inf")
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)                             # [B,1]
        ids = torch.cat([ids, next_id], dim=1)
    return ids[:, 1:]  # drop BOS

# VQ-VAE specifics
Hc, Wc = 8, 8
T = Hc * Wc           # 64
K_code = 512
K_vocab = K_code + 1  # 513 with BOS
bos_id = K_code       # 512

# load models
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = BDHPrior(
    vocab_size=K_vocab,
    d_model=512,
    n_layer=16,
    n_head=8,
    block_size=T+1,               # 65
    mlp_internal_dim_multiplier=16,  # <<< must match train_prior config
    dropout=0.1,                     # or whatever you used
).to(device).eval()
prior.load_state_dict(torch.load("checkpoints/bdh_prior.pt", map_location=device))

# sampling
B = 16
bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
codes_seq = generate_without_bos(prior, bos, steps=T, bos_id=bos_id, temperature=1.0, top_k=50)  # [B,64]
codes = codes_seq.view(B, Hc, Wc)  # [B,8,8]

imgs = vq.decode(codes).clamp(0, 1)
save_image(imgs, "samples_bdh.png", nrow=4)
log("Wrote samples_bdh.png")
log_file.close()
