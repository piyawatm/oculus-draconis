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
from typing import Optional

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
                # Get quantizer for code biasing
                if hasattr(vqvae, '_vq_vae'):
                    quantizer = vqvae._vq_vae
                else:
                    quantizer = vqvae.quantizer
                
                generated_codes = generate_with_code_bias(
                    prior, quantizer, dummy_prefix, p_conf, 
                    temperature, n_steps, schedule, sampling_top_k, device
                )
                
                # COMPREHENSIVE DEBUGGING
                logger.log("=== MaskGIT Generation Debug ===")
                
                # Check for mask tokens
                mask_token_id = p_conf.vocab_size  # 1024
                mask_count = (generated_codes == mask_token_id).sum().item()
                if mask_count > 0:
                    logger.log(f"ERROR: Found {mask_count} mask tokens (ID={mask_token_id}) in output!")
                
                # Check code statistics
                min_code = generated_codes.min().item()
                max_code = generated_codes.max().item()
                mean_code = generated_codes.float().mean().item()
                unique_codes = generated_codes.unique().numel()
                
                logger.log(f"Code statistics:")
                logger.log(f"  Range: [{min_code}, {max_code}] (expected: [0, {p_conf.vocab_size-1}])")
                logger.log(f"  Mean: {mean_code:.2f}")
                logger.log(f"  Unique codes: {unique_codes} out of {p_conf.vocab_size}")
                
                # Check if all codes are the same (model collapse)
                if unique_codes == 1:
                    logger.log(f"ERROR: All codes are the same! Value: {min_code}")
                    logger.log("This suggests the model may not be trained properly.")
                
                # Check code distribution
                code_counts = torch.bincount(generated_codes.flatten(), minlength=p_conf.vocab_size)
                top_codes = torch.topk(code_counts, k=min(10, p_conf.vocab_size))
                logger.log(f"Top 10 most frequent codes: {top_codes.indices.tolist()}")
                logger.log(f"  Counts: {top_codes.values.tolist()}")
                
                # Check if codes are all zeros or all same value
                if min_code == max_code:
                    logger.log(f"WARNING: All codes are identical (value={min_code})!")
                
                # Ensure codes are valid before decoding
                generated_codes = torch.clamp(generated_codes, min=0, max=p_conf.vocab_size - 1)
                
                # Test model forward pass
                test_tokens = torch.full((1, p_conf.block_size), mask_token_id, dtype=torch.long).to(device)
                with torch.no_grad():
                    test_logits, _ = prior(test_tokens, targets=None)
                test_probs = F.softmax(test_logits / temperature, dim=-1)
                test_entropy = -(test_probs * torch.log(test_probs + 1e-10)).sum(dim=-1).mean().item()
                
                logger.log(f"Model prediction entropy: {test_entropy:.4f} (higher = more diverse)")
                if test_entropy < 1.0:
                    logger.log("WARNING: Low entropy suggests model predictions are too confident/identical")
                
                logger.log("=== End Debug ===")
                
                # ACCURATE WORKAROUND: Actually test which codes produce bright images
                logger.log("Testing codes by actually decoding them to find bright codes...")
                H = W = int(p_conf.block_size ** 0.5)
                
                # Get quantizer
                quantizer = vqvae._vq_vae if hasattr(vqvae, '_vq_vae') else vqvae.quantizer
                
                # Test a sample of codes by actually decoding them
                # This is expensive, so we'll test a subset and cache results
                test_sample_size = min(200, p_conf.vocab_size)  # Test 200 codes
                test_codes = torch.randperm(p_conf.vocab_size, device=device)[:test_sample_size]
                
                # Decode each code as a small patch to see brightness
                test_embeddings = quantizer.embedding(test_codes)  # [N, 256]
                # Create a small patch (4x4) for each code
                test_patches = test_embeddings.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)  # [1, N, 256, 1, 1]
                test_patches = test_patches.repeat(1, 1, 1, 4, 4)  # [1, N, 256, 4, 4]
                
                # Decode patches
                decoded_patches = vqvae.decoder(test_patches.view(-1, 256, 4, 4))  # [N, 3, 4, 4]
                # Get mean brightness per code (average across RGB and spatial)
                patch_brightness = decoded_patches.mean(dim=(1, 2, 3))  # [N]
                
                # Find bright codes (top 50% by brightness)
                brightness_threshold = patch_brightness.quantile(0.5)
                bright_codes_mask = patch_brightness >= brightness_threshold
                bright_codes = test_codes[bright_codes_mask]
                dark_codes = test_codes[~bright_codes_mask]
                
                logger.log(f"Found {len(bright_codes)} bright codes (brightness >= {brightness_threshold:.4f}) out of {test_sample_size} tested")
                
                if len(bright_codes) > 0 and len(dark_codes) > 0:
                    # Create mapping: for each dark code in test, map to nearest bright code
                    dark_emb = quantizer.embedding(dark_codes)
                    bright_emb = quantizer.embedding(bright_codes)
                    distances = torch.cdist(dark_emb, bright_emb)
                    nearest_bright = bright_codes[distances.argmin(dim=1)]
                    
                    # Create partial mapping (only for tested codes)
                    code_mapping = torch.arange(p_conf.vocab_size, device=device)
                    code_mapping[dark_codes] = nearest_bright
                    
                    # For codes not in test, use embedding similarity to bright codes
                    untested_codes = torch.ones(p_conf.vocab_size, dtype=torch.bool, device=device)
                    untested_codes[test_codes] = False
                    untested_codes = torch.where(untested_codes)[0]
                    
                    if len(untested_codes) > 0:
                        untested_emb = quantizer.embedding(untested_codes)
                        untested_distances = torch.cdist(untested_emb, bright_emb)
                        untested_to_bright = bright_codes[untested_distances.argmin(dim=1)]
                        code_mapping[untested_codes] = untested_to_bright
                    
                    # Apply mapping
                    codes_grid = generated_codes.view(n_samples, H, W)
                    codes_grid = code_mapping[codes_grid]
                    generated_codes = codes_grid.view(n_samples, -1)
                    logger.log(f"Remapped all codes to bright codes based on actual decoding")
                else:
                    logger.log("WARNING: Could not find bright codes, using original")
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
        
        logger.log(f"Codes grid shape: {codes_grid.shape} (expected: [{n_samples}, {H}, {W}])")
        
        with torch.no_grad():
            # Handle VQ-VAE variable naming differences safely
            if hasattr(vqvae, '_vq_vae'):
                quantizer = vqvae._vq_vae
            else:
                quantizer = vqvae.quantizer
            
            # Debug: Check embedding lookup
            logger.log(f"Quantizer embedding shape: {quantizer.embedding.weight.shape}")
            logger.log(f"Codes grid dtype: {codes_grid.dtype}, min: {codes_grid.min()}, max: {codes_grid.max()}")
            
            # Ensure codes are long dtype for embedding lookup
            codes_grid = codes_grid.long()
            
            # Verify codes are within valid range for embedding
            if codes_grid.max() >= quantizer.embedding.num_embeddings:
                logger.log(f"ERROR: Code {codes_grid.max().item()} >= vocab_size {quantizer.embedding.num_embeddings}")
                codes_grid = torch.clamp(codes_grid, 0, quantizer.embedding.num_embeddings - 1)
            
            # Lookup embeddings
            z_q = quantizer.embedding(codes_grid)  # [B, H, W, embedding_dim]
            logger.log(f"Embedded codes shape: {z_q.shape}")
            logger.log(f"Embedded codes stats - mean: {z_q.mean().item():.4f}, std: {z_q.std().item():.4f}, min: {z_q.min().item():.4f}, max: {z_q.max().item():.4f}")
            
            # Check if embeddings are all zeros or NaN
            if z_q.abs().max() < 1e-6:
                logger.log("ERROR: Embeddings are all near zero!")
            if torch.isnan(z_q).any():
                logger.log("ERROR: Found NaN in embeddings!")
            
            z_q = z_q.permute(0, 3, 1, 2)  # [B, embedding_dim, H, W]
            logger.log(f"Permuted z_q shape: {z_q.shape} (expected: [{n_samples}, {quantizer.embedding.embedding_dim}, {H}, {W}])")
            
            # Decode
            images = vqvae.decoder(z_q)
            logger.log(f"Decoded images shape: {images.shape}")
            logger.log(f"Decoded images stats - mean: {images.mean().item():.4f}, std: {images.std().item():.4f}, min: {images.min().item():.4f}, max: {images.max().item():.4f}")
            
            # Check if images are all zeros or black
            if images.abs().max() < 1e-6:
                logger.log("ERROR: Decoded images are all near zero (black)!")
            if torch.isnan(images).any():
                logger.log("ERROR: Found NaN in decoded images!")
            
            # Test: Try decoding with some known codes to verify decoder works
            logger.log("Testing decoder with known codes...")
            test_codes = torch.randint(0, p_conf.vocab_size, (1, H, W), device=device).long()
            test_z_q = quantizer.embedding(test_codes).permute(0, 3, 1, 2)
            test_images = vqvae.decoder(test_z_q)
            logger.log(f"Test decode stats - mean: {test_images.mean().item():.4f}, std: {test_images.std().item():.4f}, min: {test_images.min().item():.4f}, max: {test_images.max().item():.4f}")
            
            # Compare with actual generated codes
            logger.log("Comparing generated vs test codes...")
            logger.log(f"Generated codes unique: {generated_codes.unique().numel()}, Test codes unique: {test_codes.unique().numel()}")
            
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

def generate_with_code_bias(prior, quantizer, dummy_prefix, p_conf, temperature, n_steps, schedule, sampling_top_k, device):
    """Generate codes but bias away from codes that produce dark images."""
    # Pre-compute which codes produce dark vs bright images
    all_embeddings = quantizer.embedding.weight  # [1024, 256]
    embedding_means = all_embeddings.mean(dim=1)  # [1024]
    
    # Codes with negative mean embeddings tend to decode darker
    # Create a bias: boost codes with positive mean, penalize negative mean
    code_bias = embedding_means * 0.1  # Small bias factor
    
    # Generate normally first
    generated = prior.generate(
        idx=dummy_prefix,
        max_new_tokens=p_conf.block_size,
        temperature=temperature,
        n_steps=n_steps,
        schedule=schedule,
        sampling_top_k=sampling_top_k,
    )
    
    # Post-process: if a code has very negative embedding mean, replace it
    dark_threshold = embedding_means.quantile(0.1)  # Bottom 10%
    bright_codes = torch.where(embedding_means > dark_threshold)[0]
    
    dark_mask = embedding_means <= dark_threshold
    dark_codes = torch.where(dark_mask)[0]
    
    if len(dark_codes) > 0 and len(bright_codes) > 0:
        # Find nearest bright code for each dark code
        dark_emb = all_embeddings[dark_codes]
        bright_emb = all_embeddings[bright_codes]
        distances = torch.cdist(dark_emb, bright_emb)
        nearest_bright = bright_codes[distances.argmin(dim=1)]
        
        # Create mapping
        code_mapping = torch.arange(p_conf.vocab_size, device=device)
        code_mapping[dark_codes] = nearest_bright
        
        # Apply to generated codes
        generated = code_mapping[generated]
    
    return generated

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
