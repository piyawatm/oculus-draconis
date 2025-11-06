# eval_metrics.py (fixed dtype handling + fewer workers)
import torch
from torchmetrics.image.fid import FrechetInceptionDistance
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

# ------------- real data -------------
tfm = transforms.ToTensor()  # float in [0,1]
real = datasets.CIFAR10("./data", train=False, download=True, transform=tfm)
real_loader = DataLoader(real, batch_size=B, shuffle=False, num_workers=2, pin_memory=True)

# ------------- models -------------
vq = VQVAE(codebook_size=K, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = BDHPrior(vocab_size=K, d_model=256, block_size=T).to(device).eval()
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
prior.eval(); vq.eval()

while seen < target:
    b = min(B, target - seen)
    # sample exactly T tokens total (no BOS path)
    start = torch.randint(0, K, (b, 1), device=device)
    ids = prior.generate(start, max_new_tokens=T-1)      # [b, T]
    codes = ids.view(b, Hc, Wc)
    imgs = vq.decode(codes).clamp(0, 1)                  # float [0,1]

    # update metrics (FID needs uint8; IS can take float or uint8 — we’ll use uint8 for consistency)
    imgs_u8 = to_u8(imgs)
    fid.update(imgs_u8.to(device), real=False)
    iscore.update(imgs_u8.to(device))

    seen += b

print("FID:", float(fid.compute()))
m, s = iscore.compute()
print("IS:", float(m), "+/-", float(s))
