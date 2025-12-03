"""Evaluation script for VQ-VAE models."""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import json

from models import VQVAE, VQVAE2, BDHVQVAE
from config import get_config
from utils import (get_dataloader, load_checkpoint, save_image_grid,
                  MetricsCalculator, denormalize)


def get_model(config):
    """Initialize model based on configuration."""
    if config.model_name == 'vqvae':
        model = VQVAE(
            in_channels=config.in_channels,
            hidden_dim=config.hidden_dim,
            num_embeddings=config.num_embeddings,
            embedding_dim=config.embedding_dim,
            num_residual_blocks=config.num_residual_blocks,
            commitment_cost=config.commitment_cost
        )
    elif config.model_name == 'vqvae2':
        model = VQVAE2(
            in_channels=config.in_channels,
            hidden_dims=config.hidden_dims,
            num_embeddings=config.num_embeddings,
            embedding_dims=config.embedding_dims,
            commitment_cost=config.commitment_cost
        )
    elif config.model_name == 'bdh_vqvae':
        model = BDHVQVAE(
            in_channels=config.in_channels,
            hidden_dims=config.hidden_dims,
            num_embeddings=config.num_embeddings,
            embedding_dims=config.embedding_dims,
            n_bdh_layers=config.n_bdh_layers,
            n_head=config.n_head,
            commitment_cost=config.commitment_cost
        )
    else:
        raise ValueError(f"Unknown model: {config.model_name}")
    
    return model


@torch.no_grad()
def evaluate_reconstruction(model, dataloader, device, metrics_calc, config):
    """Evaluate reconstruction quality."""
    model.eval()
    
    psnr_scores = []
    lpips_scores = []
    mse_scores = []
    all_indices = []
    
    print("Evaluating reconstruction quality...")
    for images, _ in tqdm(dataloader):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        recons = outputs['reconstruction']
        
        # Calculate metrics
        mse = F.mse_loss(recons, images, reduction='none').mean(dim=[1, 2, 3])
        mse_scores.extend(mse.cpu().numpy())
        
        # PSNR for each image
        for i in range(images.size(0)):
            psnr = metrics_calc.calculate_psnr(images[i:i+1], recons[i:i+1])
            psnr_scores.append(psnr.cpu().item())
        
        # LPIPS (process in batches for efficiency)
        lpips_score = metrics_calc.calculate_lpips(images, recons)
        lpips_scores.extend([lpips_score.cpu().item()] * images.size(0))
        
        # Collect indices for codebook usage
        if 'indices' in outputs:
            all_indices.append(outputs['indices'].cpu())
        elif 'indices_bottom' in outputs:
            all_indices.append(outputs['indices_bottom'].cpu())
    
    # Calculate codebook usage
    codebook_usage = None
    if all_indices:
        all_indices = torch.cat(all_indices, dim=0)
        if config.model_name == 'vqvae':
            codebook_usage = metrics_calc.calculate_codebook_usage(
                all_indices, config.num_embeddings
            )
        else:
            codebook_usage = metrics_calc.calculate_codebook_usage(
                all_indices, config.num_embeddings[0]
            )
    
    return {
        'psnr_mean': np.mean(psnr_scores),
        'psnr_std': np.std(psnr_scores),
        'lpips_mean': np.mean(lpips_scores),
        'lpips_std': np.std(lpips_scores),
        'mse_mean': np.mean(mse_scores),
        'mse_std': np.std(mse_scores),
        'codebook_usage': codebook_usage
    }


@torch.no_grad()
def generate_reconstructions(model, dataloader, device, config, num_samples=64):
    """Generate and save reconstruction examples."""
    model.eval()
    
    # Get samples
    images, _ = next(iter(dataloader))
    images = images[:num_samples].to(device)
    
    # Reconstruct
    outputs = model(images)
    recons = outputs['reconstruction']
    
    # Save original images
    filepath = os.path.join(config.output_dir, config.model_name, 'eval_originals.png')
    save_image_grid(images, filepath, nrow=8)
    
    # Save reconstructions
    filepath = os.path.join(config.output_dir, config.model_name, 'eval_reconstructions.png')
    save_image_grid(recons, filepath, nrow=8)
    
    # Save side-by-side comparison
    comparison = torch.cat([images, recons], dim=0)
    filepath = os.path.join(config.output_dir, config.model_name, 'eval_comparison.png')
    save_image_grid(comparison, filepath, nrow=8)
    
    print(f"Reconstruction samples saved to {config.output_dir}/{config.model_name}/")


@torch.no_grad()
def test_encoding_decoding(model, dataloader, device, config):
    """Test encode/decode cycle."""
    model.eval()
    
    # Get a sample
    images, _ = next(iter(dataloader))
    images = images[:8].to(device)
    
    # Encode to indices
    if config.model_name == 'vqvae':
        indices = model.encode(images)
        # Decode from indices
        decoded = model.decode_from_indices(indices)
    else:  # vqvae2 or bdh_vqvae
        indices_bottom, indices_top = model.encode(images)
        # Decode from indices
        decoded = model.decode_from_indices(indices_bottom, indices_top)
    
    # Save results
    comparison = torch.cat([images, decoded], dim=0)
    filepath = os.path.join(config.output_dir, config.model_name, 'encode_decode_test.png')
    save_image_grid(comparison, filepath, nrow=8)
    
    print(f"Encode/decode test saved to {filepath}")
    
    # Print index statistics
    if config.model_name == 'vqvae':
        print(f"Indices shape: {indices.shape}")
        print(f"Unique codes used: {torch.unique(indices).size(0)}/{config.num_embeddings}")
    else:
        print(f"Bottom indices shape: {indices_bottom.shape}")
        print(f"Top indices shape: {indices_top.shape}")
        print(f"Bottom unique codes: {torch.unique(indices_bottom).size(0)}/{config.num_embeddings[0]}")
        print(f"Top unique codes: {torch.unique(indices_top).size(0)}/{config.num_embeddings[1]}")


def main(args):
    # Get configuration
    config = get_config(args.model)
    
    if args.dataset:
        config.dataset = args.dataset
    if args.batch_size:
        config.batch_size = args.batch_size
    
    print(f"\n{'='*60}")
    print(f"Evaluating {config.model_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    # Load data
    test_loader = get_dataloader(
        config.dataset, config.batch_size, config.img_size,
        config.num_workers, train=False
    )
    
    # Initialize model
    model = get_model(config).to(config.device)
    
    # Load checkpoint
    checkpoint_name = args.checkpoint if args.checkpoint else 'best.pth'
    checkpoint_path = os.path.join(config.checkpoint_dir, config.model_name, checkpoint_name)
    
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
    
    load_checkpoint(model, None, checkpoint_path)
    
    # Initialize metrics calculator
    metrics_calc = MetricsCalculator(config.device)
    
    # Run evaluations
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    # 1. Reconstruction quality
    recon_metrics = evaluate_reconstruction(model, test_loader, config.device, 
                                           metrics_calc, config)
    
    print(f"\nReconstruction Metrics:")
    print(f"  PSNR: {recon_metrics['psnr_mean']:.2f} ± {recon_metrics['psnr_std']:.2f} dB")
    print(f"  LPIPS: {recon_metrics['lpips_mean']:.4f} ± {recon_metrics['lpips_std']:.4f}")
    print(f"  MSE: {recon_metrics['mse_mean']:.6f} ± {recon_metrics['mse_std']:.6f}")
    if recon_metrics['codebook_usage'] is not None:
        print(f"  Codebook Usage: {recon_metrics['codebook_usage']:.2%}")
    
    # 2. Generate reconstruction examples
    print("\nGenerating reconstruction samples...")
    generate_reconstructions(model, test_loader, config.device, config, 
                           num_samples=args.num_samples)
    
    # 3. Test encoding/decoding
    print("\nTesting encode/decode cycle...")
    test_encoding_decoding(model, test_loader, config.device, config)
    
    # Save metrics to file
    results = {
        'model': config.model_name,
        'dataset': config.dataset,
        'metrics': {k: float(v) if v is not None else None 
                   for k, v in recon_metrics.items()}
    }
    
    results_path = os.path.join(config.output_dir, config.model_name, 'eval_results.json')
    os.makedirs(os.path.dirname(results_path), exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")
    print("="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate VQ-VAE models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vqvae', 'vqvae2', 'bdh_vqvae'],
                       help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Checkpoint file name (default: best.pth)')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset to use (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--num-samples', type=int, default=64,
                       help='Number of samples to generate')
    
    args = parser.parse_args()
    main(args)