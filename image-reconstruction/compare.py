"""Compare multiple VQ-VAE models."""

import os
import argparse
import torch
import json
import pandas as pd
from tqdm import tqdm

from models import VQVAE, VQVAE2, BDHVQVAE
from config import get_config
from utils import (get_dataloader, load_checkpoint, save_image_grid,
                  MetricsCalculator)


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
    return model


@torch.no_grad()
def evaluate_model(model_name, test_loader, device, metrics_calc):
    """Evaluate a single model."""
    config = get_config(model_name)
    model = get_model(config).to(device)
    
    # Load best checkpoint
    checkpoint_path = os.path.join(config.checkpoint_dir, model_name, 'best.pth')
    if not os.path.exists(checkpoint_path):
        print(f"Warning: No checkpoint found for {model_name} at {checkpoint_path}")
        return None
    
    load_checkpoint(model, None, checkpoint_path)
    model.eval()
    
    psnr_scores = []
    lpips_scores = []
    mse_scores = []
    all_indices = []
    
    for images, _ in tqdm(test_loader, desc=f"Evaluating {model_name}"):
        images = images.to(device)
        outputs = model(images)
        recons = outputs['reconstruction']
        
        # MSE
        mse = torch.nn.functional.mse_loss(recons, images, reduction='none').mean(dim=[1, 2, 3])
        mse_scores.extend(mse.cpu().numpy())
        
        # PSNR
        for i in range(images.size(0)):
            psnr = metrics_calc.calculate_psnr(images[i:i+1], recons[i:i+1])
            psnr_scores.append(psnr.cpu().item())
        
        # LPIPS
        lpips_score = metrics_calc.calculate_lpips(images, recons)
        lpips_scores.extend([lpips_score.cpu().item()] * images.size(0))
        
        # Indices
        if 'indices' in outputs:
            all_indices.append(outputs['indices'].cpu())
        elif 'indices_bottom' in outputs:
            all_indices.append(outputs['indices_bottom'].cpu())
    
    # Codebook usage
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
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    
    return {
        'model': model_name,
        'psnr_mean': float(torch.tensor(psnr_scores).mean()),
        'psnr_std': float(torch.tensor(psnr_scores).std()),
        'lpips_mean': float(torch.tensor(lpips_scores).mean()),
        'lpips_std': float(torch.tensor(lpips_scores).std()),
        'mse_mean': float(torch.tensor(mse_scores).mean()),
        'codebook_usage': float(codebook_usage) if codebook_usage else None,
        'num_params': num_params
    }


@torch.no_grad()
def generate_comparison(models, test_loader, device, output_dir, num_samples=8):
    """Generate side-by-side comparison of reconstructions."""
    # Get sample images
    images, _ = next(iter(test_loader))
    images = images[:num_samples].to(device)
    
    all_recons = [images]
    
    for model_name in models:
        config = get_config(model_name)
        model = get_model(config).to(device)
        
        checkpoint_path = os.path.join(config.checkpoint_dir, model_name, 'best.pth')
        if not os.path.exists(checkpoint_path):
            print(f"Skipping {model_name} - no checkpoint found")
            continue
        
        load_checkpoint(model, None, checkpoint_path)
        model.eval()
        
        outputs = model(images)
        all_recons.append(outputs['reconstruction'])
    
    # Create comparison grid
    # Format: [Original, VQVAE, VQVAE2, BDHVQVAE] for each sample
    comparison = torch.cat(all_recons, dim=0)
    
    filepath = os.path.join(output_dir, 'model_comparison.png')
    save_image_grid(comparison, filepath, nrow=num_samples)
    print(f"Comparison saved to {filepath}")


def main(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Comparing VQ-VAE Models")
    print(f"{'='*60}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load test data (use config from first model)
    config = get_config(args.models[0])
    test_loader = get_dataloader(
        args.dataset, 
        batch_size=32,
        img_size=config.img_size,
        num_workers=4,
        train=False
    )
    
    # Initialize metrics
    metrics_calc = MetricsCalculator(device)
    
    # Evaluate each model
    results = []
    for model_name in args.models:
        print(f"\nEvaluating {model_name}...")
        result = evaluate_model(model_name, test_loader, device, metrics_calc)
        if result:
            results.append(result)
    
    # Create results table
    if results:
        df = pd.DataFrame(results)
        df = df.round(4)
        
        print("\n" + "="*60)
        print("COMPARISON RESULTS")
        print("="*60)
        print(df.to_string(index=False))
        print("="*60)
        
        # Save to CSV
        output_dir = './outputs/comparison'
        os.makedirs(output_dir, exist_ok=True)
        csv_path = os.path.join(output_dir, 'comparison_results.csv')
        df.to_csv(csv_path, index=False)
        print(f"\nResults saved to {csv_path}")
        
        # Save to JSON
        json_path = os.path.join(output_dir, 'comparison_results.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {json_path}")
        
        # Generate visual comparison
        if args.generate_images:
            print("\nGenerating visual comparison...")
            generate_comparison(args.models, test_loader, device, output_dir, 
                              num_samples=args.num_samples)
    
    print("\n" + "="*60 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare VQ-VAE models')
    parser.add_argument('--models', nargs='+', 
                       default=['vqvae', 'vqvae2', 'bdh_vqvae'],
                       choices=['vqvae', 'vqvae2', 'bdh_vqvae'],
                       help='Models to compare')
    parser.add_argument('--dataset', type=str, default='cifar10',
                       help='Dataset to use')
    parser.add_argument('--generate-images', action='store_true',
                       help='Generate visual comparison images')
    parser.add_argument('--num-samples', type=int, default=8,
                       help='Number of samples for visual comparison')
    
    args = parser.parse_args()
    main(args)