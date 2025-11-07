# train_vqvae.py
from models.vqvae import VQVAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, torch.optim as optim
from utils import set_seed, save_checkpoint, count_parameters
import time

set_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"
tfm = transforms.ToTensor()
train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4)

model = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4).to(device)
print("Params:", count_parameters(model), "M")

opt = optim.Adam(model.parameters(), lr=2e-4)
total_training_time = 0
for epoch in range(50):
    epoch_start_time = time.time()
    epoch_loss = 0
    epoch_recon = 0
    epoch_codebook = 0
    epoch_commitment = 0 
    num_batches = 0 
    for x, _ in loader:
        x = x.to(device)
        x_hat, loss_dict, _ = model(x)
        loss = loss_dict["recon"] + loss_dict["codebook"] + loss_dict["commitment"]
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()
        epoch_recon += loss_dict["recon"].item()
        epoch_codebook += loss_dict["codebook"].item()
        epoch_commitment += loss_dict["commitment"].item()
        num_batches += 1
    epoch_time = time.time() - epoch_start_time
    total_training_time += epoch_time
    avg_loss = epoch_loss / num_batches
    avg_recon = epoch_recon / num_batches
    avg_codebook = epoch_codebook / num_batches
    avg_commitment = epoch_commitment / num_batches
    print(f"Epoch {epoch} | Loss: {avg_loss:.4f} (recon: {avg_recon:.4f}, "
          f"codebook: {avg_codebook:.4f}, commit: {avg_commitment:.4f}) | "
          f"time={epoch_time:.2f}s | total={total_training_time/60:.1f}min")
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, f"checkpoints/vqvae_epoch{epoch+1}.pt")

save_checkpoint(model, "checkpoints/vqvae.pt")
print("Saved VQVAE checkpoint.")
print(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")