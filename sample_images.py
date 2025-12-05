import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior
from utils import load_config, Logger
import os
import argparse
from argparse import Namespace
import yaml
import tqdm

# Constants
VQVAE_CHECKPOINT = "checkpoints/vqvae_best.pt"
PRIOR_CHECKPOINT = "checkpoints/prior_best.pt"
VQVAE_CONFIG = "configs/vqvae_config.yaml"
PRIOR_CONFIG = "configs/bdh_config.yaml"
LOG_FILE = "logs/sampling.log"

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering """
    top_k = min(top_k, logits.size(-1))  # Safety check
    
    if top_k > 0:
        # Remove all tokens with a probability less than the last token of the top-k
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above the threshold
        sorted_indices_to_remove = cumulative_probs > top_p
        
        # Shift the indices to the right to keep also the first token above the threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = filter_value
        
    return logits

def load_model(config_path, model_cls, ckpt_path, logger):
    with open(config_path, 'r') as f:
        raw_conf = yaml.safe_load(f)
        if 'model' in raw_conf: raw_conf.update(raw_conf['model'])
        if 'data' in raw_conf: raw_conf.update(raw_conf['data'])
        if 'training' in raw_conf: raw_conf.update(raw_conf['training'])
        config = Namespace(**raw_conf)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model_cls(config).to(device)
    
    if os.path.exists(ckpt_path):
        logger.log(f"Loading checkpoint from {ckpt_path}")
        
        # --- FIX 1: Actually load the file ---
        state_dict = torch.load(ckpt_path, map_location=device)
        
        new_state_dict = {}
        for k, v in state_dict.items():
            # Handle torch.compile prefix
            if k.startswith("_orig_mod."):
                new_key = k[10:] # remove "_orig_mod."
                new_state_dict[new_key] = v
            else:
                new_state_dict[k] = v
                
        model.load_state_dict(new_state_dict)
    else:
        logger.log(f"WARNING: {ckpt_path} not found! Using random initialization.")
        
    model.eval()
    return model, config, device

def sample_images(n_samples=16, temperature=1.0, top_p=0.9):
    logger = Logger(LOG_FILE)
    logger.log(f"--- Starting Image Sampling (N={n_samples}, T={temperature}, Top-p={top_p}) ---")
    
    try:
        # Load Models
        vqvae, v_conf, device = load_model(VQVAE_CONFIG, VQVAE, VQVAE_CHECKPOINT, logger)
        prior, p_conf, _ = load_model(PRIOR_CONFIG, BDHPrior, PRIOR_CHECKPOINT, logger)
        
        logger.log(f"Vocab Size: {p_conf.vocab_size} | Block Size: {p_conf.block_size}")
        
        # 1. Initialize with BOS
        bos_token = p_conf.vocab_size - 1
        idx = torch.full((n_samples, 1), bos_token, dtype=torch.long).to(device)
        
        logger.log("Starting Autoregressive Generation...")
        
        # 2. Generation Loop
        for _ in tqdm.tqdm(range(p_conf.block_size), desc="Sampling"):
            with torch.no_grad():
                logits, _ = prior(idx)
                last_logits = logits[:, -1, :] / temperature
                
                # Mask BOS
                last_logits[:, bos_token] = float('-inf')

                # --- FILTERING ---
                filtered_logits = top_k_top_p_filtering(last_logits, top_k=0, top_p=top_p)
                # -----------------
                
                # --- FIX 2: Use filtered_logits here ---
                probs = F.softmax(filtered_logits, dim=-1)
                
                next_idx = torch.multinomial(probs, num_samples=1)
                idx = torch.cat((idx, next_idx), dim=1)
                
        # 3. Decode
        generated_codes = idx[:, 1:] # Remove BOS
        H = W = int(p_conf.block_size ** 0.5)
        codes_grid = generated_codes.view(n_samples, H, W)
        
        with torch.no_grad():
            # Handle VQ-VAE variable naming differences safely
            if hasattr(vqvae, '_vq_vae'):
                quantizer = vqvae._vq_vae
            else:
                quantizer = vqvae.quantizer
                
            z_q = quantizer.embedding(codes_grid).permute(0, 3, 1, 2)
            images = vqvae.decoder(z_q)
            
        # 4. Save
        os.makedirs("results", exist_ok=True)
        save_path = "results/generated_faces.png"
        save_image(images, save_path, nrow=4, normalize=True, value_range=(-1, 1))
        
        logger.log(f"Success! Saved grid to {save_path}")
        
    except Exception as e:
        logger.log(f"ERROR during sampling: {e}")
        raise e
    finally:
        logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=16)
    parser.add_argument('--temp', type=float, default=1.0)
    parser.add_argument('--top_p', type=float, default=0.9)
    args = parser.parse_args()
    
    sample_images(n_samples=args.n_samples, temperature=args.temp, top_p=args.top_p)
