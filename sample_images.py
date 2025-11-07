import torch
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.gpt import GPTPrior

device = "cuda" if torch.cuda.is_available() else "cpu"

# VQ-VAE specifics
Hc, Wc = 8, 8
T = Hc * Wc           # 64
K_code = 512
K_vocab = K_code + 1  # 513 with BOS
bos_id = K_code       # 512

# load models
vq = VQVAE(codebook_size=K_code, embed_dim=256, downsample_factor=4).to(device).eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt", map_location=device))

prior = GPTPrior(vocab_size=K_vocab, d_model=256, n_layer=6, n_head=4, block_size=T+1).to(device).eval()
prior.load_state_dict(torch.load("checkpoints/gpt_prior.pt", map_location=device))

# sampling
B = 16
bos = torch.full((B, 1), bos_id, dtype=torch.long, device=device)    # [B,1]
ids = prior.generate(bos, max_new_tokens=T)                          # [B, 65]
codes = ids[:, 1:].contiguous().view(B, Hc, Wc)                      # drop BOS -> [B,8,8]

imgs = vq.decode(codes).clamp(0, 1)
save_image(imgs, "samples_gpt.png", nrow=4)
print("Wrote samples_gpt.png")
