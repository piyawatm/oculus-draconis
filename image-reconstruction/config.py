"""Configuration file for VQ-VAE experiments."""


class BaseConfig:
    """Base configuration."""
    # Data
    dataset = 'cifar10'
    img_size = 32
    batch_size = 128
    num_workers = 4
    
    # Training
    num_epochs = 100
    learning_rate = 2e-4
    device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    # Checkpointing
    checkpoint_dir = './checkpoints'
    save_interval = 10  # Save every N epochs
    
    # Logging
    log_dir = './logs'
    log_interval = 100  # Log every N iterations
    
    # Evaluation
    eval_interval = 5  # Evaluate every N epochs
    num_eval_samples = 64
    
    # Output
    output_dir = './outputs'


class VQVAEConfig(BaseConfig):
    """Configuration for vanilla VQ-VAE."""
    model_name = 'vqvae'
    
    # Model architecture
    in_channels = 3
    hidden_dim = 128
    num_embeddings = 512
    embedding_dim = 64
    num_residual_blocks = 2
    commitment_cost = 0.25
    
    # Loss weights
    recon_weight = 1.0


class VQVAE2Config(BaseConfig):
    """Configuration for VQ-VAE-2."""
    model_name = 'vqvae2'
    
    # Model architecture
    in_channels = 3
    hidden_dims = [128, 256]
    num_embeddings = [512, 512]
    embedding_dims = [64, 64]
    commitment_cost = 0.25
    
    # Loss weights
    recon_weight = 1.0


class BDHVQVAEConfig(BaseConfig):
    """Configuration for BDH-VQVAE (Baby Dragon Hatchling)."""
    model_name = 'bdh_vqvae'
    
    # Model architecture
    in_channels = 3
    hidden_dims = [128, 256]
    num_embeddings = [512, 512] # Increased codebook size for BDH-VQVAE
    embedding_dims = [64, 64]
    n_bdh_layers = 1  # Number of BDH attention layers (recurrent depth)
    n_head = 4  # Number of attention heads in BDH
    commitment_cost = 1.0 # Increased commitment cost for BDH-VQVAE
    
    # Loss weights
    recon_weight = 1.0


def get_config(model_name):
    """Get configuration for specified model."""
    configs = {
        'vqvae': VQVAEConfig,
        'vqvae2': VQVAE2Config,
        'bdh_vqvae': BDHVQVAEConfig
    }
    
    if model_name not in configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(configs.keys())}")
    
    return configs[model_name]()