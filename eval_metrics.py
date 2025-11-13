import torch
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.kid import KernelInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.vqvae import VQVAE
from utils import count_parameters
import yaml
import argparse
from tqdm import tqdm

# ---- helpers ----
def to_u8(x: torch.Tensor) -> torch.Tensor:
    # expects float in [0,1]; returns uint8 in [0,255]
    return (x.clamp(0, 1) * 255.0).to(torch.uint8)

# ---- config setup ----
parser = argparse.ArgumentParser(description="Evaluate a VQ-VAE Prior")
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
    '--batch_size', 
    type=int, 
    default=64,
    help="Batch size for generation"
)
parser.add_argument(
    '--num_samples', 
    type=int, 
    default=10000,
    help="Total number of samples to generate for metrics"
)
args = parser.parse_args()

print(f"Loading config from: {args.config}")
cfg = yaml.safe_load(open(args.config))

# ---- constants ----
B = args.batch_size
target = args.num_samples
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
else:
    # AR models have vocab_size = K_code + 1 (for BOS)
    K_code = int(model_cfg["vocab_size"]) - 1 # e.g., 513 - 1 = 512
    bos_id = K_code # BOS token is the last ID

# ---- real data loader ----
tfm = transforms.ToTensor() # float in [0,1]
real_dset = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
real_loader = DataLoader(real_dset, batch_size=B, shuffle=False, num_workers=2, pin_memory=True)
print(f"Loaded {len(real_dset)} real images.")

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
print(f"Loaded {model_name} with {count_parameters(prior):.2f}M parameters.")

# ---- init metrics ----
fid = FrechetInceptionDistance(feature=2048).to(device)
kid = KernelInceptionDistance(subset_size=50).to(device)
iscore = InceptionScore().to(device)

print("Calculating features for real images...")
for x, _ in tqdm(real_loader):
    x_u8 = to_u8(x).to(device, non_blocking=True) # uint8 NCHW
    fid.update(x_u8, real=True)
    kid.update(x_u8, real=True)

# ---- generation loop ----
print(f"Generating {target} fake images...")
seen = 0
with torch.no_grad():
    while seen < target:
        b = min(B, target - seen)
        
        # --- Branching generation logic ---
        if is_maskgit:
            # MaskGIT generation
            sampling_cfg = cfg.get("sampling", {}) # Get n_steps, schedule, etc.
            
            # Create a dummy prefix just to pass batch size and device
            dummy_idx = torch.zeros(b, 1, dtype=torch.long, device=device)
            
            ids = prior.generate(
                dummy_idx, 
                max_new_tokens=T, # Total tokens to generate (64)
                **sampling_cfg
            ) # [b, 64]
            codes = ids.view(b, Hc, Wc)
            
        else:
            # Autoregressive generation
            bos = torch.full((b, 1), bos_id, dtype=torch.long, device=device)
            ids = prior.generate(bos, max_new_tokens=T) # [b, 65] (BOS + 64 tokens)
            codes = ids[:, 1:].view(b, Hc, Wc) # [b, 8, 8]
        
        # --- Common decoding and metric update ---
        imgs_float = vq.decode(codes) # [b, 3, 32, 32]
        imgs_u8 = to_u8(imgs_float)   # [b, 3, 32, 32] (uint8)

        fid.update(imgs_u8, real=False)
        kid.update(imgs_u8, real=False)
        iscore.update(imgs_u8)

        seen += b
        print(f"\rGenerated {seen}/{target} images...", end="")

print("\nDone generating.")

# ---- compute and print results ----
print("Calculating metrics...")
fid_score = float(fid.compute())
is_mean, is_std = iscore.compute()
kid_score = float(kid.compute())

print("-" * 30)
print(f"Model: {model_name}")
print(f"Config: {args.config}")
print(f"Total Samples: {target}")
print("-" * 30)
print(f"FID: {fid_score:.4f}")
print(f"KID: {kid_score:.4f}")
print(f"IS:  {float(is_mean):.4f} +/- {float(is_std):.4f}")
print("-" * 30)