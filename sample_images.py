import torch
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior

B, T, Hc, Wc, K = 16, 64, 8, 8, 512
vq = VQVAE(codebook_size=K, embed_dim=256, downsample_factor=4).cuda().eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt"))

prior = BDHPrior(vocab_size=K, d_model=256, n_layer=6, n_head=4, block_size=T).cuda().eval()
prior.load_state_dict(torch.load("checkpoints/bdh_prior.pt"))

bos = torch.zeros(B, 1, dtype=torch.long, device="cuda")  # or your BOS id
ids = prior.generate(bos, max_new_tokens=T)[:, 1:].view(B, Hc, Wc)
imgs = vq.decode(ids).clamp(0, 1)
save_image(imgs, "samples_bdh.png", nrow=4)
