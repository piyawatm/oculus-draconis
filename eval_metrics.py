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
from models.priors.pixelcnn import PixelCNNPrior
from models.priors.pixelsnail import PixelSNAILPrior
from models.priors.maskgit import MaskGITPrior
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
        if 'training' in raw: raw.update(raw['training'])
        return Namespace(**raw)

def get_prior_class(config):
    """Factory function to get the appropriate prior class based on config."""
    model_name = getattr(config, 'name', None) or getattr(config, 'type', None)
    
    if model_name == 'PixelCNNPrior' or 'pixelcnn' in str(model_name).lower():
        return PixelCNNPrior
    elif model_name == 'PixelSNAILPrior' or 'pixelsnail' in str(model_name).lower():
        return PixelSNAILPrior
    elif model_name == 'MaskGITPrior' or 'maskgit' in str(model_name).lower():
        return MaskGITPrior
    elif model_name == 'BDHPrior' or 'bdh' in str(model_name).lower():
        return BDHPrior
    else:
        # Default to BDH for backward compatibility
        return BDHPrior

def load_state_dict_safe(model, ckpt_path):
    """Loads state dict handling torch.compile prefix."""
    print(f"Loading checkpoint from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=DEVICE)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_key = k[10:] # remove "_orig_mod."
            new_state_dict[new_key] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    return model

def generate_fake_images(logger, num_imgs=2000, batch_size=50, prior_config_path=None, prior_checkpoint_path=None):
    if os.path.exists(FAKE_DIR): shutil.rmtree(FAKE_DIR)
    os.makedirs(FAKE_DIR)
    
    logger.log(f"Generating {num_imgs} fake images to {FAKE_DIR}...")
    
    # Load Models
    v_conf = load_conf("configs/vqvae_config.yaml")
    p_conf = load_conf(prior_config_path or "configs/bdh_config.yaml")
    
    # Load sampling config if available
    with open(prior_config_path or "configs/bdh_config.yaml", 'r') as f:
        raw_conf = yaml.safe_load(f)
        if 'sampling' in raw_conf:
            for k, v in raw_conf['sampling'].items():
                setattr(p_conf, k, v)
    
    vqvae = VQVAE(v_conf).to(DEVICE)
    # Apply Safe Load
    vqvae = load_state_dict_safe(vqvae, "checkpoints/vqvae_best.pt")
    vqvae.eval()
    
    # Get appropriate prior class and instantiate
    prior_cls = get_prior_class(p_conf)
    if prior_cls in (PixelCNNPrior, PixelSNAILPrior, MaskGITPrior):
        prior = prior_cls(config=p_conf).to(DEVICE)
    else:
        prior = prior_cls(p_conf).to(DEVICE)
    
    # Apply Safe Load
    prior_ckpt = prior_checkpoint_path or "checkpoints/prior_best.pt"
    prior = load_state_dict_safe(prior, prior_ckpt)
    prior.eval()
    
    # Check if using MaskGIT
    is_maskgit = prior_cls == MaskGITPrior
    
    if is_maskgit:
        # MaskGIT parameters
        n_steps = getattr(p_conf, 'n_steps', 8)
        schedule = getattr(p_conf, 'schedule', 'cosine')
        sampling_top_k = getattr(p_conf, 'sampling_top_k', 100)
        temperature = getattr(p_conf, 'temperature', 1.5)
    else:
        # Autoregressive parameters
        bos_token = p_conf.vocab_size - 1
    
    H = W = int(p_conf.block_size ** 0.5)
    
    count = 0
    pbar = tqdm(total=num_imgs, desc="Generating Fakes")
    
    while count < num_imgs:
        curr_batch = min(batch_size, num_imgs - count)
        
        with torch.no_grad():
            if is_maskgit:
                # MaskGIT: Iterative parallel generation
                dummy_prefix = torch.zeros((curr_batch, 1), dtype=torch.long).to(DEVICE)
                codes = prior.generate(
                    idx=dummy_prefix,
                    max_new_tokens=p_conf.block_size,
                    temperature=temperature,
                    n_steps=n_steps,
                    schedule=schedule,
                    sampling_top_k=sampling_top_k,
                )  # [B, block_size]
            else:
                # Autoregressive generation
                idx = torch.full((curr_batch, 1), bos_token, dtype=torch.long).to(DEVICE)
                
                for _ in range(p_conf.block_size):
                    logits, _ = prior(idx)
                    last_logits = logits[:, -1, :]
                    last_logits[:, bos_token] = float('-inf')
                    
                    probs = F.softmax(last_logits, dim=-1)
                    next_idx = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, next_idx), dim=1)
                
                codes = idx[:, 1:].view(curr_batch, H, W)
            
            # Reshape codes to grid
            codes = codes.view(curr_batch, H, W)
            
            # Handle VQVAE naming safely
            if hasattr(vqvae, '_vq_vae'):
                quantizer = vqvae._vq_vae
            else:
                quantizer = vqvae.quantizer
                
            z_q = quantizer.embedding(codes).permute(0, 3, 1, 2)
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
    parser.add_argument('--prior_config', type=str, default=None, help='Path to prior config file')
    parser.add_argument('--prior_checkpoint', type=str, default=None, help='Path to prior checkpoint')
    args = parser.parse_args()
    
    logger = Logger(LOG_FILE)
    logger.log("--- Starting Evaluation ---")
    
    try:
        extract_real_images(logger, num_imgs=args.num_imgs)
        generate_fake_images(logger, num_imgs=args.num_imgs, 
                           prior_config_path=args.prior_config,
                           prior_checkpoint_path=args.prior_checkpoint)
        
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
