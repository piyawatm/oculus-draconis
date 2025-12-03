"""Training script for VQ-VAE models."""

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from models import VQVAE, VQVAE2, BDHVQVAE
from config import get_config
from utils import get_dataloader, save_checkpoint, load_checkpoint, save_image_grid


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


def train_epoch(model, dataloader, optimizer, device, config, epoch, writer, global_step):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_recon_loss = 0
    total_quant_loss = 0
    total_perplexity = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, (images, _) in enumerate(pbar):
        images = images.to(device)
        
        # Forward pass
        outputs = model(images)
        recon = outputs['reconstruction']
        quant_loss = outputs['quantizer_loss']
        perplexity = outputs['perplexity']
        
        # Calculate reconstruction loss
        recon_loss = F.mse_loss(recon, images)
        
        # Total loss
        loss = config.recon_weight * recon_loss + quant_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Accumulate metrics
        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_quant_loss += quant_loss.item()
        total_perplexity += perplexity.item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'recon': f"{recon_loss.item():.4f}",
            'perp': f"{perplexity.item():.1f}"
        })
        
        # Log to tensorboard
        if batch_idx % config.log_interval == 0:
            writer.add_scalar('Train/Loss', loss.item(), global_step)
            writer.add_scalar('Train/Recon_Loss', recon_loss.item(), global_step)
            writer.add_scalar('Train/Quant_Loss', quant_loss.item(), global_step)
            writer.add_scalar('Train/Perplexity', perplexity.item(), global_step)
        
        global_step += 1
    
    # Return average metrics
    n_batches = len(dataloader)
    return {
        'loss': total_loss / n_batches,
        'recon_loss': total_recon_loss / n_batches,
        'quant_loss': total_quant_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }, global_step


@torch.no_grad()
def validate(model, dataloader, device, config):
    """Validate the model."""
    model.eval()
    total_recon_loss = 0
    total_perplexity = 0
    
    for images, _ in tqdm(dataloader, desc="Validation"):
        images = images.to(device)
        outputs = model(images)
        
        recon_loss = F.mse_loss(outputs['reconstruction'], images)
        total_recon_loss += recon_loss.item()
        total_perplexity += outputs['perplexity'].item()
    
    n_batches = len(dataloader)
    return {
        'recon_loss': total_recon_loss / n_batches,
        'perplexity': total_perplexity / n_batches
    }


@torch.no_grad()
def generate_samples(model, dataloader, device, config, epoch):
    """Generate reconstruction samples."""
    model.eval()
    
    # Get a batch of images
    images, _ = next(iter(dataloader))
    images = images[:config.num_eval_samples].to(device)
    
    # Reconstruct
    outputs = model(images)
    recons = outputs['reconstruction']
    
    # Save comparison
    comparison = torch.cat([images, recons], dim=0)
    filepath = os.path.join(config.output_dir, config.model_name, f'recon_epoch_{epoch}.png')
    save_image_grid(comparison, filepath, nrow=8)


def main(args):
    # Get configuration
    config = get_config(args.model)
    
    # Override config with command line args
    if args.dataset:
        config.dataset = args.dataset
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.epochs:
        config.num_epochs = args.epochs
    if args.lr:
        config.learning_rate = args.lr
    
    print(f"\n{'='*60}")
    print(f"Training {config.model_name.upper()}")
    print(f"{'='*60}")
    print(f"Dataset: {config.dataset}")
    print(f"Batch size: {config.batch_size}")
    print(f"Epochs: {config.num_epochs}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print(f"{'='*60}\n")
    
    # Set up data loaders
    train_loader = get_dataloader(
        config.dataset, config.batch_size, config.img_size,
        config.num_workers, train=True
    )
    val_loader = get_dataloader(
        config.dataset, config.batch_size, config.img_size,
        config.num_workers, train=False
    )
    
    # Initialize model
    model = get_model(config).to(config.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Tensorboard
    log_path = os.path.join(config.log_dir, config.model_name)
    writer = SummaryWriter(log_path)
    
    # Load checkpoint if resuming
    start_epoch = 0
    global_step = 0
    if args.resume:
        checkpoint_path = os.path.join(
            config.checkpoint_dir, config.model_name, 'latest.pth'
        )
        if os.path.exists(checkpoint_path):
            start_epoch, _ = load_checkpoint(model, optimizer, checkpoint_path)
            start_epoch += 1
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, config.num_epochs):
        # Train
        train_metrics, global_step = train_epoch(
            model, train_loader, optimizer, config.device,
            config, epoch, writer, global_step
        )
        
        print(f"\nEpoch {epoch} - Train Loss: {train_metrics['loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"Perplexity: {train_metrics['perplexity']:.1f}")
        
        # Validate
        if epoch % config.eval_interval == 0:
            val_metrics = validate(model, val_loader, config.device, config)
            print(f"Epoch {epoch} - Val Recon: {val_metrics['recon_loss']:.4f}, "
                  f"Perplexity: {val_metrics['perplexity']:.1f}")
            
            writer.add_scalar('Val/Recon_Loss', val_metrics['recon_loss'], epoch)
            writer.add_scalar('Val/Perplexity', val_metrics['perplexity'], epoch)
            
            # Generate samples
            generate_samples(model, val_loader, config.device, config, epoch)
            
            # Save best model
            if val_metrics['recon_loss'] < best_val_loss:
                best_val_loss = val_metrics['recon_loss']
                checkpoint_path = os.path.join(
                    config.checkpoint_dir, config.model_name, 'best.pth'
                )
                save_checkpoint(model, optimizer, epoch, best_val_loss, checkpoint_path)
        
        # Save checkpoint
        if epoch % config.save_interval == 0:
            checkpoint_path = os.path.join(
                config.checkpoint_dir, config.model_name, f'epoch_{epoch}.pth'
            )
            save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_path)
        
        # Always save latest
        checkpoint_path = os.path.join(
            config.checkpoint_dir, config.model_name, 'latest.pth'
        )
        save_checkpoint(model, optimizer, epoch, train_metrics['loss'], checkpoint_path)
    
    writer.close()
    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VQ-VAE models')
    parser.add_argument('--model', type=str, required=True,
                       choices=['vqvae', 'vqvae2', 'bdh_vqvae'],
                       help='Model to train')
    parser.add_argument('--dataset', type=str, default=None,
                       help='Dataset to use (default: from config)')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='Batch size (default: from config)')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of epochs (default: from config)')
    parser.add_argument('--lr', type=float, default=None,
                       help='Learning rate (default: from config)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume training from latest checkpoint')
    
    args = parser.parse_args()
    main(args)