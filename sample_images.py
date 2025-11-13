import torch
from torchvision.utils import save_image
from models.vqvae import VQVAE
import yaml
import argparse
import os
import math

# ---- config setup ----
parser = argparse.ArgumentParser(description="Sample images from a VQ-VAE Prior")
parser.add_argument(
    '--config', 
    type=str, 
    required=True,
    help="Path to the prior's config YAML file (e.g., configs/prior_maskgit.yaml)"
)
parser.add_argument(
    '--vq_ckpt', 
    type=str, 
    default="checkpoints/vqvae.pt",
    help="Path to the VQ-VAE checkpoint"
)
parser.add_argument(
    '--n', 
    type=int, 
    default=16,
    help="Number of images to generate"
)
parser.add_argument(
    '--out', 
    type=str, 
    default=None,
    help="Path to save the output image grid. (default: samples_[config_name].png)"
)
args = parser.parse_args()

print(f"Loading config from: {args.config}")
cfg = yaml.safe_load(open(args.config))

# ---- constants ----
B = args.n # Batch size is number of samples
device = "cuda" if torch.cuda.is_available() else "cpu"

# Latent grid dimensions (hardcoded for CIFAR 8x8 grid)
Hc, Wc = 8, 8
T = Hc * Wc # 64

# ---- model config ----
model_cfg = dict(cfg["model"])
model_name = model_cfg.pop("name", "BDHPrior")

# Determine K_code (VQ codebook size) and if model is AR or MaskGIT
is_maskgit = (model_name == "MaskGITPrior")
if is_maskgit:
    # MaskGIT config has vocab_size = K_code
    K_code = int(model_cfg["vocab_size"]) # e.g., 512
    print(f"Loading MaskGITPrior (K_code={K_code}, T={T})")
else:
    # AR models have vocab_size = K_code + 1 (for BOS)
    K_code = int(model_cfg["vocab_size"]) - 1 # e.g., 513 - 1 = 512
    bos_id = K_code # BOS token is the last ID
    print(f"Loading AR Prior '{model_name}' (K_code={K_code}, T={T}, BOS_id={bos_id})")

# ---- load VQ-VAE ----
print(f"Loading VQ-VAE from: {args.vq_ckpt}")
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load(args.vq_ckpt, map_location=device))
vq.eval()

# ---- load Prior ----
prior_ckpt_path = cfg["train"]["save_path"]
print(f"Loading Prior ({model_name}) from: {prior_ckpt_path}")

if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
elif model_name == "MaskGITPrior":
    from models.priors.maskgit import MaskGITPrior as Prior
else:
    raise ValueError(f"Unknown prior name: {model_name}")

prior = Prior(**model_cfg).to(device).eval()
prior.load_state_dict(torch.load(prior_ckpt_path, map_location=device))
prior.eval()

# ---- sampling ----
print(f"Generating {B} samples...")
with torch.no_grad():
    # --- Branching generation logic ---
    if is_maskgit:
        # MaskGIT generation
        sampling_cfg = cfg.get("sampling", {}) # Get n_steps, schedule, etc.
        
        # Create a dummy prefix just to pass batch size and device
        dummy_idx = torch.zeros(B, 1, dtype=torch.long, device=device)
        
        ids = prior.generate(
            dummy_idx, 
            max_new_tokens=T, # Total tokens to generate (64)
            **sampling_cfg
        ) # [B, 64]
        codes = ids.view(B, Hc, Wc)
        
    else:
        # Autoregressive generation
        bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        ids = prior.generate(bos, max_new_tokens=T) # [B, 65] (BOS + 64 tokens)
        codes = ids[:, 1:].contiguous().view(B, Hc, Wc) # drop BOS -> [B, 8, 8]
    
    # --- Common decoding ---
    imgs = vq.decode(codes).clamp(0, 1)

# ---- save ----
if args.out is None:
    # Create default output name
    config_basename = os.path.basename(args.config).split('.')[0]
    save_path = f"samples_{config_basename}.png"
else:
    save_path = args.out

save_image(imgs, save_path, nrow=int(math.sqrt(B)))
print(f"Wrote samples to {save_path}")