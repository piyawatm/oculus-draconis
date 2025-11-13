# train_vqvae.py
from models.vqvae import VQVAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, torch.optim as optim
from utils import set_seed, save_checkpoint, count_parameters

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
tfm = transforms.ToTensor()
train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

model = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4).to(device)
print("Params:", count_parameters(model), "M")

opt = optim.Adam(model.parameters(), lr=2e-4)

for epoch in range(5):
    for x, _ in loader:
        x = x.to(device)
        x_hat, loss_dict, _ = model(x)
        loss = loss_dict["recon"] + loss_dict["codebook"] + loss_dict["commitment"]
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, f"checkpoints/vqvae_epoch{epoch+1}.pt")

save_checkpoint(model, "checkpoints/vqvae.pt")
print("Saved VQVAE checkpoint.")