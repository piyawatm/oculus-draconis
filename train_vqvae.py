import torch
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, utils
from models.vqvae import VQVAE
from utils import load_config, Logger  # Import Logger
import os
import time
import argparse
from tqdm import tqdm
# In train_vqvae.py

from datasets import load_from_disk  # Change import


def get_data_loader(batch_size):
    print("Loading FFHQ-64 from local disk...")

    # FIX: Load from local path
    try:
        dataset = load_from_disk("data/ffhq64_local")
    except:
        print("Local dataset not found, downloading now...")
        from datasets import load_dataset
        dataset = load_dataset("Dmini/FFHQ-64x64", split="train")
        dataset.save_to_disk("data/ffhq64_local")

    # Remove .with_format("torch") if it causes issues with transforms,
    # but usually good for speed.

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def collate_fn(examples):
        valid_imgs = [ex["image"].convert("RGB") for ex in examples]
        images = [transform(img) for img in valid_imgs]
        return torch.stack(images)

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=4,  # Now this works perfectly!
        shuffle=True,  # Important for VQ-VAE
        pin_memory=True
    )
    return dataloader


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/vqvae_config.yaml')
    args = parser.parse_args()

    conf = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Create dirs
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Initialize Logger
    logger = Logger("logs/vqvae_train.log")
    logger.log(f"Starting VQ-VAE training on {device} with config: {conf}")

    model = VQVAE(conf).to(device)
    optimizer = optim.Adam(model.parameters(), lr=conf.learning_rate)

    loader = get_data_loader(conf.batch_size)
    iterator = iter(loader)

    start_time = time.time()

    for i in range(conf.max_iters):
        try:
            images = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            images = next(iterator)

        images = images.to(device)

        optimizer.zero_grad()
        reconstruction, quantization_loss, perplexity = model(images)

        # MSE Loss
        recon_loss = F.mse_loss(reconstruction, images)
        loss = recon_loss + conf.commitment_cost * quantization_loss

        loss.backward()
        optimizer.step()

        if i % conf.log_interval == 0:
            elapsed = time.time() - start_time
            steps_per_sec = conf.log_interval / elapsed
            logger.log(f"Iter {i}: Loss={loss.item():.4f} | Recon={recon_loss.item():.4f} | "
                       f"Q={quantization_loss.item():.4f} | PPL={perplexity.item():.2f} | "
                       f"Speed={steps_per_sec:.2f} it/s")
            start_time = time.time()

        if i % conf.save_interval == 0:
            torch.save(model.state_dict(), "checkpoints/vqvae_best.pt")
            logger.log(f"Saved checkpoint to checkpoints/vqvae_best.pt")

            # Save visual sample
            utils.save_image(
                torch.cat([images[:8], reconstruction[:8]], dim=0),
                f"results/recon_{i}.png",
                nrow=8, normalize=True, value_range=(-1, 1)
            )

    logger.close()


if __name__ == "__main__":
    train()