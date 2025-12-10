import torch
from models.vqvae import VQVAE
from train_vqvae import get_data_loader
from utils import load_config
import os
from tqdm import tqdm

def extract():
    conf = load_config("configs/vqvae_config.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = VQVAE(conf).to(device)
    model.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=device))
    model.eval()
    
    loader = get_data_loader(batch_size=128)
    all_codes = []
    
    print("Extracting codes from FFHQ...")
    # Extract ~100k images (FFHQ is 70k, so just run until empty or cap it)
    MAX_IMAGES = 70000
    count = 0
    
    with torch.no_grad():
        for images in tqdm(loader):
            images = images.to(device)
            # Encode: [B, C, H, W] -> [B, H', W'] (indices)
            _, _, _, _, indices = model.encode(images)
            
            # Flatten to [B, Sequence_Len]
            indices = indices.view(images.shape[0], -1)
            all_codes.append(indices.cpu())
            
            count += images.shape[0]
            if count >= MAX_IMAGES:
                break
                
    all_codes = torch.cat(all_codes, dim=0)
    os.makedirs("data/codes", exist_ok=True)
    torch.save(all_codes, "data/codes/ffhq_train.pt")
    print(f"Saved {all_codes.shape} codes to data/codes/ffhq_train.pt")

if __name__ == "__main__":
    extract()
