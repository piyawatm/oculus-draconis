# eval_metrics.py  (DO NOT MODIFY ANYTHING)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from datasets import load_from_disk
from torchvision import transforms
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.image.kid import KernelInceptionDistance

from models.vqvae import VQVAE
from models.priors.pixelsnail import PixelSNAILPrior
from utils import load_config, count_parameters
from tqdm import tqdm
import inspect

device = "cuda" if torch.cuda.is_available() else "cpu"

# Latent grid settings
Hc, Wc = 16, 16
T = Hc * Wc
num_embeddings = 1024
K_vocab = num_embeddings + 1
bos_id = num_embeddings

print(f"Eval using latent grid: {Hc}×{Wc} (T={T})")
print(f"Vocab size={K_vocab}, BOS ID={bos_id}")

# Load real images
REAL_PATH = "data/ffhq64_local"
print(f"Loading real FFHQ images from: {REAL_PATH}")

ds = load_from_disk(REAL_PATH)
tfm = transforms.Compose([transforms.ToTensor()])

def collate(examples):
    imgs = [tfm(x["image"].convert("RGB")) for x in examples]
    return torch.stack(imgs)

real_loader = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=collate)

# Metrics
fid = FrechetInceptionDistance(feature=2048).to(device)
iscore = InceptionScore().to(device)
kid = KernelInceptionDistance(subset_size=50).to(device)

for batch in tqdm(real_loader, desc="Real Images"):
    batch = (batch * 255).byte().to(device)
    fid.update(batch, real=True)
    kid.update(batch, real=True)
    iscore.update(batch)

vq_conf = load_config("configs/vqvae_config.yaml")
vq = VQVAE(vq_conf).to(device)
vq.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))
vq.eval()

p_conf = load_config("configs/pixelsnail.yaml")

# KEEP ONLY VALID KEYS
valid_keys = inspect.signature(PixelSNAILPrior).parameters.keys()
prior_kwargs = {k: v for k, v in p_conf.__dict__.items() if k in valid_keys}

# Force correct geometry + vocab
prior_kwargs["vocab_size"] = K_vocab
prior_kwargs["height"] = Hc
prior_kwargs["width"] = Wc


prior = PixelSNAILPrior(**prior_kwargs).to(device)
prior.load_state_dict(torch.load("checkpoints/pixelsnail_prior.pt", map_location=device))
prior.eval()

# Sampling function
@torch.no_grad()
def sample_codes(batch=32, temperature=1.0, top_k=50):
    bos = torch.full((batch, 1), bos_id, dtype=torch.long, device=device)
    ids = bos
    for _ in range(T):
        cond = ids[:, -prior.block_size:]
        logits = prior(cond)[:, -1, :] / temperature
        logits[:, bos_id] = float("-inf")
        if top_k:
            top_vals, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
            logits[logits < top_vals[:, [-1]]] = float("-inf")
        probs = F.softmax(logits, dim=-1)
        nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], dim=1)
    return ids[:, 1:].view(batch, Hc, Wc)


TARGET_FAKE = 2000

for _ in tqdm(range(TARGET_FAKE // 32), desc="Fake Batches"):
    codes = sample_codes(batch=32)
    emb = vq.quantizer.embedding(codes).permute(0, 3, 1, 2)
    imgs = vq.decoder(emb).clamp(0, 1)
    imgs_u8 = (imgs * 255).byte().to(device)
    fid.update(imgs_u8, real=False)
    kid.update(imgs_u8, real=False)
    iscore.update(imgs_u8)

# Results
print("\n--- Evaluation Metrics ---")
print(f"FID: {float(fid.compute()):.3f}")
kid_mean, kid_std = kid.compute()
print(f"KID: {float(kid_mean):.5f} ± {float(kid_std):.5f}")
is_mean, is_std = iscore.compute()
print(f"Inception Score: {float(is_mean):.3f} ± {float(is_std):.3f}")

# MODEL STATISTICS (PARAMS + MEMORY)
def measure_memory(model, sample_input):
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(sample_input)
    mem = torch.cuda.max_memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    return mem

print("\n--- Model Stats ---")

# Parameter counts
vq_params = count_parameters(vq)
prior_params = count_parameters(prior)

print(f"VQ-VAE Parameters:       {vq_params:.2f}M")
print(f"PixelSNAIL Parameters:   {prior_params:.2f}M")

# GPU memory footprint
if torch.cuda.is_available():
    example = torch.randint(0, K_vocab, (1, T), device=device)
    prior_mem = measure_memory(prior, example)
    print(f"PixelSNAIL Memory Footprint: {prior_mem:.1f} MB")
else:
    print("CUDA not available — cannot compute memory footprint.")

