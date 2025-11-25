# train_vqvae.py
from models.vqvae import VQVAE
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch, torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from utils import set_seed, save_checkpoint, count_parameters, Logger

set_seed(42)
logger = Logger("logs/train_vqvae_log.txt")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Slight augmentation: horizontal flip + ToTensor
tfm = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train = datasets.CIFAR10("./data", train=True, download=True, transform=tfm)
loader = DataLoader(train, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)

# Match updated VQVAE defaults (192 channels, EMA VQ)
model = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4).to(device)
logger.log(f"Params:{count_parameters(model)}M")

opt = optim.Adam(model.parameters(), lr=3e-4)
scheduler = CosineAnnealingLR(opt, T_max=100)  # 100 epochs

epochs = 100
best_loss = float("inf")
best_state = None

for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    epoch_perplexity = 0.0

    for x, _ in loader:
        x = x.to(device, non_blocking=True)
        x_hat, loss_dict, _ = model(x)
        loss = loss_dict["recon"] + loss_dict["codebook"] + loss_dict["commitment"]

        opt.zero_grad()
        loss.backward()
        opt.step()

        epoch_loss += loss.item()
        epoch_perplexity += loss_dict["perplexity"].item()
        num_batches += 1

    scheduler.step()

    avg_loss = epoch_loss / max(1, num_batches)
    avg_perplexity = epoch_perplexity / num_batches
    logger.log(f"Epoch {epoch:03d}, avg_loss: {avg_loss:.6f}, perplexity={avg_perplexity:.2f}")

    # track best model
    if avg_loss < best_loss:
        best_loss = avg_loss
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # checkpoint every 10 epochs
    if (epoch + 1) % 10 == 0:
        save_checkpoint(model, f"checkpoints/vqvae_epoch{epoch+1}.pt")

# save best model separately
if best_state is not None:
    best_model = VQVAE(codebook_size=512, embed_dim=256, downsample_factor=4)
    best_model.load_state_dict(best_state)
    save_checkpoint(best_model, "checkpoints/vqvae_best.pt")
    logger.log(f"Saved best VQ-VAE checkpoint with avg_loss={best_loss:.6f} -> checkpoints/vqvae_best.pt")

# save final model (last epoch weights)
save_checkpoint(model, "checkpoints/vqvae.pt")
logger.log("Saved final VQVAE checkpoint -> checkpoints/vqvae.pt")
logger.close()