import torch
import torch.nn.functional as F
from torchvision.utils import save_image
from models.vqvae import VQVAE
from models.priors.bdh import BDHPrior
from models.priors.pixelcnn import PixelCNNPrior
from models.priors.pixelsnail import PixelSNAILPrior
from models.priors.maskgit import MaskGITPrior
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
PRIOR_CONFIG = "configs/bdh_config.yaml"  # Can be overridden via --config
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

def load_model(config_path, model_cls, ckpt_path, logger):
    with open(config_path, 'r') as f:
        raw_conf = yaml.safe_load(f)
        if 'model' in raw_conf: raw_conf.update(raw_conf['model'])
        if 'data' in raw_conf: raw_conf.update(raw_conf['data'])
        if 'training' in raw_conf: raw_conf.update(raw_conf['training'])
        config = Namespace(**raw_conf)
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Handle both config-based and direct instantiation
    if model_cls in (PixelCNNPrior, PixelSNAILPrior, MaskGITPrior):
        model = model_cls(config=config).to(device)
    else:
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

def sample_images(n_samples=16, temperature=1.0, top_p=0.9, prior_config=None, prior_checkpoint=None):
    logger = Logger(LOG_FILE)
    logger.log(f"--- Starting Image Sampling (N={n_samples}, T={temperature}, Top-p={top_p}) ---")
    
    # Allow overriding config/checkpoint paths
    prior_cfg = prior_config or PRIOR_CONFIG
    prior_ckpt = prior_checkpoint or PRIOR_CHECKPOINT
    
    try:
        # Load Models
        vqvae, v_conf, device = load_model(VQVAE_CONFIG, VQVAE, VQVAE_CHECKPOINT, logger)
        
        # Determine prior class from config
        with open(prior_cfg, 'r') as f:
            raw_conf = yaml.safe_load(f)
            if 'model' in raw_conf: raw_conf.update(raw_conf['model'])
            if 'sampling' in raw_conf: raw_conf.update(raw_conf['sampling'])
            temp_config = Namespace(**raw_conf)
        prior_cls = get_prior_class(temp_config)
        
        prior, p_conf, _ = load_model(prior_cfg, prior_cls, prior_ckpt, logger)
        
        logger.log(f"Vocab Size: {p_conf.vocab_size} | Block Size: {p_conf.block_size}")
        
        # Check if using MaskGIT (non-autoregressive)
        is_maskgit = prior_cls == MaskGITPrior
        
        if is_maskgit:
            logger.log("Starting MaskGIT Iterative Parallel Generation...")
            
            # MaskGIT: Use generate() method with iterative parallel decoding
            # Create dummy prefix (MaskGIT ignores it, just needs batch size)
            dummy_prefix = torch.zeros((n_samples, 1), dtype=torch.long).to(device)
            
            # Get sampling parameters from config
            n_steps = getattr(p_conf, 'n_steps', 8)
            schedule = getattr(p_conf, 'schedule', 'cosine')
            sampling_top_k = getattr(p_conf, 'sampling_top_k', 100)
            
            with torch.no_grad():
                generated_codes = prior.generate(
                    idx=dummy_prefix,
                    max_new_tokens=p_conf.block_size,
                    temperature=temperature,
                    n_steps=n_steps,
                    schedule=schedule,
                    sampling_top_k=sampling_top_k,
                )  # [B, block_size]
        else:
            # Autoregressive generation for other priors
            logger.log("Starting Autoregressive Generation...")
            
            # 1. Initialize with BOS
            bos_token = p_conf.vocab_size - 1
            idx = torch.full((n_samples, 1), bos_token, dtype=torch.long).to(device)
            
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
                    
                    probs = F.softmax(filtered_logits, dim=-1)
                    
                    next_idx = torch.multinomial(probs, num_samples=1)
                    idx = torch.cat((idx, next_idx), dim=1)
            
            # Remove BOS token
            generated_codes = idx[:, 1:]  # [B, block_size]
        
        # 3. Decode
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
    parser.add_argument('--config', type=str, default=None, help='Path to prior config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to prior checkpoint')
    args = parser.parse_args()
    
    sample_images(n_samples=args.n_samples, temperature=args.temp, top_p=args.top_p,
                  prior_config=args.config, prior_checkpoint=args.checkpoint)
