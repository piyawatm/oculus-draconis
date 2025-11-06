# extract_codes.py

import torch, torchvision
from torch.utils.data import DataLoader
from models.vqvae import VQVAE

def dump(split):
    ds = torchvision.datasets.CIFAR10("./data", train=(split=="train"),
                                      download=True, transform=torchvision.transforms.ToTensor())
    dl = DataLoader(ds, batch_size=256, shuffle=False, num_workers=4)
    codes_all = []
    with torch.no_grad():
        for x, _ in dl:
            x = x.cuda()
            idx = vq.encode(x)        # [B, Hc, Wc]
            codes_all.append(idx.cpu())
    out = torch.cat(codes_all)
    out_flat = out.flatten(1)         # [N, T]
    torch.save({"grid": out, "seq": out_flat}, f"data/codes/cifar10_{split}.pt")

vq = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4).cuda().eval()
vq.load_state_dict(torch.load("checkpoints/vqvae.pt"))
dump("train"); dump("test")
print("Extracted and saved VQ-VAE codes.")