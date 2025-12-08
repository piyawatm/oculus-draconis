import os
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import lpips
import numpy as np
from scipy import linalg


def get_dataloader(dataset_name, batch_size, img_size=32, num_workers=4, train=True):
    """Get dataloader for specified dataset."""
    
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    if dataset_name.lower() == 'cifar10':
        dataset = torchvision.datasets.CIFAR10(
            root='./data', train=train, download=True, transform=transform
        )
    elif dataset_name.lower() == 'celeba':
        dataset = torchvision.datasets.CelebA(
            root='./data', split='train' if train else 'test',
            download=True, transform=transform
        )
    elif dataset_name.lower() == 'imagenet':
        # For ImageNet, you need to download manually
        split = 'train' if train else 'val'
        dataset = torchvision.datasets.ImageFolder(
            root=f'./data/imagenet/{split}', transform=transform
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=train,
        num_workers=num_workers, pin_memory=True
    )


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, filepath)
    print(f"Checkpoint saved: {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded: {filepath} (epoch {epoch})")
    return epoch, loss


class MetricsCalculator:
    """Calculate various metrics for model evaluation."""
    
    def __init__(self, device):
        self.device = device
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        
    def calculate_psnr(self, img1, img2):
        """Calculate PSNR between two images."""
        mse = torch.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * torch.log10(2.0 / torch.sqrt(mse))
    
    def calculate_lpips(self, img1, img2):
        """Calculate LPIPS perceptual distance."""
        with torch.no_grad():
            return self.lpips_fn(img1, img2).mean()
    
    def calculate_fid(self, real_features, fake_features):
        """
        Calculate FID score.
        Features should be [N, D] arrays.
        """
        mu1, sigma1 = real_features.mean(axis=0), np.cov(real_features, rowvar=False)
        mu2, sigma2 = fake_features.mean(axis=0), np.cov(fake_features, rowvar=False)
        
        diff = mu1 - mu2
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        
        fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
        return fid
    
    def calculate_codebook_usage(self, indices, num_embeddings):
        """Calculate codebook usage statistics."""
        unique_codes = torch.unique(indices)
        usage_rate = len(unique_codes) / num_embeddings
        return usage_rate


def denormalize(tensor):
    """Denormalize tensor from [-1, 1] to [0, 1]."""
    return (tensor + 1.0) / 2.0


def save_image_grid(images, filepath, nrow=8):
    """Save a grid of images."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    images = denormalize(images)
    grid = torchvision.utils.make_grid(images, nrow=nrow, padding=2)
    torchvision.utils.save_image(grid, filepath)
    print(f"Image grid saved: {filepath}")