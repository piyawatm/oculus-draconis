# train_cifar10.py
# Trains the vanilla VQ-VAE on CIFAR-10

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import save_image
import os

# Import our model
from vq_vae import VQVAE

# --- Configuration ---
# Check for Mac's MPS (Metal Performance Shaders)
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS (Mac GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using NVIDIA CUDA GPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")

EPOCHS = 20
BATCH_SIZE = 64
LEARNING_RATE = 1e-3

# Model Hyperparameters
IN_CHANNELS = 3  # CIFAR-10 is RGB
HIDDEN_CHANNELS = 128
EMBEDDING_DIM = 64   # Dimension of each codebook vector
NUM_EMBEDDINGS = 512 # Size of the codebook (K)
COMMITMENT_COST = 0.25 # Beta parameter from the paper

# Create a directory to save results
if not os.path.exists('results'):
    os.makedirs('results')

# --- 1. Load Data ---
transform = transforms.Compose([
    transforms.ToTensor(),
    # We don't normalize, as we'll use MSELoss on pixel values [0, 1]
    # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Get a fixed batch from the test set for visual comparison
fixed_test_images, _ = next(iter(test_loader))
fixed_test_images = fixed_test_images.to(DEVICE)
save_image(fixed_test_images, 'results/original_images.png', nrow=8)

# --- 2. Initialize Model, Optimizer, and Loss ---
model = VQVAE(
    in_channels=IN_CHANNELS,
    hidden_channels=HIDDEN_CHANNELS,
    embedding_dim=EMBEDDING_DIM,
    num_embeddings=NUM_EMBEDDINGS,
    commitment_cost=COMMITMENT_COST
).to(DEVICE)

optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
recon_criterion = nn.MSELoss() # Mean Squared Error for reconstruction

print("Model, Optimizer, and Data are ready.")
print(f"Training on {DEVICE} for {EPOCHS} epochs...")

# --- 3. Training Loop ---
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    total_recon_loss = 0
    
    for i, (images, _) in enumerate(train_loader):
        images = images.to(DEVICE)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Forward pass
        vq_loss, x_recon = model(images)
        
        # Calculate loss
        recon_loss = recon_criterion(x_recon, images)
        loss = recon_loss + vq_loss
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        
        if (i + 1) % 100 == 0:
            print(f"  [Epoch {epoch+1}/{EPOCHS}, Batch {i+1}/{len(train_loader)}] "
                  f"Total Loss: {loss.item():.4f}, "
                  f"Recon Loss: {recon_loss.item():.4f}, "
                  f"VQ Loss: {vq_loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    avg_recon_loss = total_recon_loss / len(train_loader)
    print(f"--- EPOCH {epoch+1} COMPLETE ---")
    print(f"Avg. Total Loss: {avg_loss:.4f}, Avg. Recon Loss: {avg_recon_loss:.4f}")
    print("---------------------------------")
    
    # --- 4. Save Reconstruction Sample ---
    model.eval()
    with torch.no_grad():
        _, recon_images = model(fixed_test_images)
        # Clamp values to [0, 1] for saving
        recon_images = torch.clamp(recon_images, 0.0, 1.0)
        save_image(recon_images, f'results/recon_epoch_{epoch+1}.png', nrow=8)

print("Training Complete!")
print("Original images saved to results/original_images.png")
print("Reconstructions saved to results/ folder.")