import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.pixelsnail import PixelSNAILPrior
from utils import load_config
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------------------------
# Sampling helper
# -------------------------------------------------
def generate_without_bos(prior, bos, steps, bos_id, temperature=1.0, top_k=None):
    ids = bos
    for _ in range(steps):
        idx_cond = ids[:, -prior.block_size:]
        logits = prior(idx_cond)[:, -1, :] / max(1e-8, temperature)

        # Prevent BOS from being generated
        logits[:, bos_id] = float("-inf")

        # Top-k filtering optional
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1)
        ids = torch.cat([ids, next_id], dim=1)

    return ids[:, 1:]   # drop BOS


# -------------------------------------------------
# FFHQ / New VQ-VAE settings
# -------------------------------------------------
Hc, Wc = 16, 16        # latent grid for 64x64 with downsample_factor=4
T = Hc * Wc            # 256 tokens
K_code = 1024          # num_embeddings from vqvae_config
K_vocab = K_code + 1   # extra BOS token
bos_id = K_code

# -------------------------------------------------
# Load VQ-VAE (NEW API)
# -------------------------------------------------
vq_conf = load_config("configs/vqvae_config.yaml")
vq = VQVAE(vq_conf).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))

# -------------------------------------------------
# Load PixelSNAIL Prior
# -------------------------------------------------
prior_conf = load_config("configs/pixelsnail.yaml")
prior = PixelSNAILPrior(
    vocab_size=prior_conf.vocab_size,
    d_model=prior_conf.d_model,
    n_layer=prior_conf.n_layer,
    height=prior_conf.height,
    width=prior_conf.width,
    n_residual=prior_conf.n_residual,
    dropout=prior_conf.dropout,
    tie_embeddings=prior_conf.tie_embeddings,
    attn_every=prior_conf.attn_every,
    attn_key_dim=prior_conf.attn_key_dim,
    attn_val_dim=prior_conf.attn_val_dim,
    attn_heads=prior_conf.attn_heads,
).to(device).eval()

prior.load_state_dict(torch.load("checkpoints/pixelsnail_prior.pt", map_location=device))

# -------------------------------------------------
# Generate samples
# -------------------------------------------------
B = 8
bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)

codes_seq = generate_without_bos(
    prior, bos, steps=T, bos_id=bos_id, temperature=1.0, top_k=50
)

codes = codes_seq.view(B, Hc, Wc)  # reshape to 16x16

with torch.no_grad():
    z_q = vq.quantizer.embedding(codes).permute(0,3,1,2)
    imgs = vq.decoder(z_q).clamp(-1, 1)  # output is normalized

# save output
os.makedirs("results", exist_ok=True)
save_image(imgs, "results/samples_pixelsnail.png", nrow=4, normalize=True, value_range=(-1,1))

print("✔ Saved: results/samples_pixelsnail.png")
