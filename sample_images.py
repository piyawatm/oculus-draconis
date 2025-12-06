# sample_images.py — PixelCNN prior + new VQ-VAE on FFHQ-64

import math
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from models.vqvae import VQVAE
from models.priors.pixelcnn import PixelCNNPrior
from utils import load_config
import yaml

device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------- Load VQ-VAE ----------------
vq_conf = load_config("configs/vqvae_config.yaml")
vq = VQVAE(vq_conf).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))

# num_embeddings from config (must match VectorQuantizerEMA)
K_code = int(vq_conf.num_embeddings)     # e.g., 1024
K_vocab = K_code + 1                     # +1 for BOS token

# ---------------- Load PixelCNN prior ----------------
prior_cfg = yaml.safe_load(open("configs/prior_pixelcnn.yaml"))

model_cfg = dict(prior_cfg["model"])
model_name = model_cfg.pop("name", "PixelCNNPrior")
assert model_name == "PixelCNNPrior", f"sample_images.py is set up for PixelCNNPrior, got {model_name}"

block_size = int(prior_cfg["model"]["block_size"])   # 256
side = int(math.isqrt(block_size))                   # 16
assert side * side == block_size, f"block_size {block_size} is not a perfect square"

# sanity: vocab sizes must match
cfg_vocab = int(prior_cfg["model"]["vocab_size"])
assert cfg_vocab == K_vocab, (
    f"vocab_size mismatch: prior cfg has {cfg_vocab}, "
    f"but VQ-VAE has num_embeddings {K_code} → vocab_size {K_vocab}"
)

prior = PixelCNNPrior(**model_cfg).to(device).eval()
prior.load_state_dict(torch.load(prior_cfg["train"]["save_path"], map_location=device))

# ---------------- Helper: decode codes via VQ-VAE ----------------
@torch.no_grad()
def decode_codes(vq_model: VQVAE, codes: torch.Tensor) -> torch.Tensor:
    """
    codes: [B, Hc, Wc] integer code indices in [0, K_code-1]
    returns: images [B, 3, 64, 64] in [0,1]
    """
    B, Hc, Wc = codes.shape
    # 1. lookup embeddings: [B, Hc*Wc] -> [B, Hc*Wc, D]
    flat = codes.view(B, -1)  # [B, T]
    z = vq_model.quantizer.embedding(flat)  # [B, T, D]
    # 2. reshape to [B, D, Hc, Wc]
    z = z.view(B, Hc, Wc, -1).permute(0, 3, 1, 2).contiguous()  # [B, D, Hc, Wc]
    # 3. decode to image space
    x = vq_model.decoder(z)  # typically in [-1,1] because you trained with Normalize(0.5,0.5,0.5)
    # 4. map [-1,1] -> [0,1]
    x = (x + 1.0) / 2.0
    return x.clamp(0.0, 1.0)

# ---------------- PixelCNN 2D Sampling ----------------
@torch.no_grad()
def sample_pixelcnn(prior, side: int, temperature: float = 1.0, batch_size: int = 16) -> torch.Tensor:
    """
    Raster-scan autoregressive sampling for PixelCNN over a side x side grid.
    Produces integer codes in [0, K_code-1].
    """
    codes = torch.zeros(batch_size, side, side, dtype=torch.long, device=device)

    bos_id = K_code  # last vocab index

    for i in range(side):
        for j in range(side):
            # Flatten [B, H, W] -> [B, T]
            flat = codes.view(batch_size, -1)               # [B, side*side]
            flat = flat.clamp(0, K_vocab - 1)               # ensure valid for embedding

            logits = prior(flat)                            # [B, T, K_vocab]
            logits = logits.view(batch_size, side, side, K_vocab)

            logits_ij = logits[:, i, j, :] / max(temperature, 1e-8)  # [B, K_vocab]

            # forbid BOS in generated codes
            logits_ij[:, bos_id] = float("-inf")

            probs = F.softmax(logits_ij, dim=-1)            # [B, K_vocab]
            next_id = torch.multinomial(probs, 1).squeeze(-1)  # [B]

            # clamp to [0, K_code-1] (valid VQ code indices)
            next_id = next_id.clamp(0, K_code - 1)

            codes[:, i, j] = next_id

    return codes  # [B, side, side]


# ---------------- Main: generate & save ----------------
if __name__ == "__main__":
    B = 16  # number of images to sample

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    # 1. sample latent code grid from PixelCNN
    codes = sample_pixelcnn(prior, side=side, temperature=1.0, batch_size=B)  # [B,16,16]

    # 2. decode through VQ-VAE
    imgs = decode_codes(vq, codes.to(device))  # [B,3,64,64] in [0,1]

    # 3. save
    save_image(imgs, "samples_pixelcnn.png", nrow=4)
    print("✓ Wrote samples_pixelcnn.png")
    print("Sampled codes range:", int(codes.min().item()), "to", int(codes.max().item()))
    print("Codes shape:", codes.shape)
