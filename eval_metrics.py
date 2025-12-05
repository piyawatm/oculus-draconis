import torch
import os
import shutil
import yaml
import argparse
from argparse import Namespace
from tqdm import tqdm
import torch.nn.functional as F
from torchvision.utils import save_image
from cleanfid import fid
from torch_fidelity import calculate_metrics

from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior
from train_vqvae import get_data_loader
from utils import Logger

# Constants
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REAL_DIR = "eval_real"
FAKE_DIR = "eval_fake"
LOG_FILE = "logs/evaluation.log"

def load_conf(path):
    with open(path, 'r') as f:
        raw = yaml.safe_load(f)
        if 'model' in raw: raw.update(raw['model'])
        if 'data' in raw: raw.update(raw['data'])
        return Namespace(**raw)

def generate_fake_images(logger, num_imgs=2000, batch_size=50):
    if os.path.exists(FAKE_DIR): shutil.rmtree(FAKE_DIR)
    os.makedirs(FAKE_DIR)
    
    logger.log(f"Generating {num_imgs} fake images to {FAKE_DIR}...")
    
    # Load Models
    v_conf = load_conf("configs/vqvae_config.yaml")
    p_conf = load_conf("configs/bdh_config.yaml")
    
    vqvae = VQVAE(v_conf).to(DEVICE)
    vqvae.load_state_dict(torch.load("checkpoints/vqvae_best.pt", map_location=DEVICE))
    vqvae.eval()
    
    prior = BDHPrior(p_conf).to(DEVICE)
    prior.load_state_dict(torch.load("checkpoints/prior_best.pt", map_location=DEVICE))
    prior.eval()
    
    bos_token = p_conf.vocab_size - 1
    H = W = int(p_conf.block_size ** 0.5)
    
    count = 0
    pbar = tqdm(total=num_imgs, desc="Generating Fakes")
    
    while count < num_imgs:
        curr_batch = min(batch_size, num_imgs - count)
        idx = torch.full((curr_batch, 1), bos_token, dtype=torch.long).to(DEVICE)
        
        with torch.no_grad():
            for _ in range(p_conf.block_size):
                logits, _ = prior(idx)
                last_logits = logits[:, -1, :]
                last_logits[:, bos_token] = float('-inf')
                
                probs = F.softmax(last_logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_idx), dim=1)
            
            codes = idx[:, 1:].view(curr_batch, H, W)
            z_q = vqvae.quantizer.embedding(codes).permute(0, 3, 1, 2)
            images = vqvae.decoder(z_q)
            
            for j in range(curr_batch):
                save_image(images[j], f"{FAKE_DIR}/{count}.png", normalize=True, value_range=(-1, 1))
                count += 1
                pbar.update(1)
    pbar.close()

def extract_real_images(logger, num_imgs=2000):
    if os.path.exists(REAL_DIR):
        if len(os.listdir(REAL_DIR)) >= num_imgs:
            logger.log(f"Found {len(os.listdir(REAL_DIR))} existing real images. Skipping extraction.")
            return

    if os.path.exists(REAL_DIR): shutil.rmtree(REAL_DIR)
    os.makedirs(REAL_DIR)
    
    logger.log(f"Extracting {num_imgs} real images from FFHQ to {REAL_DIR}...")
    loader = get_data_loader(batch_size=50)
    
    count = 0
    pbar = tqdm(total=num_imgs, desc="Extracting Reals")
    
    for batch in loader:
        batch = batch.to(DEVICE)
        for j in range(batch.size(0)):
            if count >= num_imgs: break
            save_image(batch[j], f"{REAL_DIR}/{count}.png", normalize=True, value_range=(-1, 1))
            count += 1
            pbar.update(1)
        if count >= num_imgs: break
    pbar.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_imgs', type=int, default=2000)
    args = parser.parse_args()
    
    logger = Logger(LOG_FILE)
    logger.log("--- Starting Evaluation ---")
    
    try:
        extract_real_images(logger, num_imgs=args.num_imgs)
        generate_fake_images(logger, num_imgs=args.num_imgs)
        
        logger.log("Calculating FID and KID...")
        fid_score = fid.compute_fid(FAKE_DIR, REAL_DIR)
        kid_score = fid.compute_kid(FAKE_DIR, REAL_DIR)
        
        logger.log(f"FID Score: {fid_score:.2f}")
        logger.log(f"KID Score: {kid_score:.5f}")
        
        logger.log("Calculating Inception Score...")
        metrics = calculate_metrics(input1=FAKE_DIR, isc=True, verbose=False)
        is_mean = metrics['inception_score_mean']
        is_std = metrics['inception_score_std']
        
        logger.log(f"IS Score:  {is_mean:.2f} +/- {is_std:.2f}")
        logger.log("Evaluation Complete.")
        
    except Exception as e:
        logger.log(f"ERROR during evaluation: {e}")
        raise e
    finally:
        logger.close()

if __name__ == "__main__":
    main()
