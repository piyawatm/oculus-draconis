import os
import random
import numpy as np
import torch
import yaml
from argparse import Namespace
from torchvision.utils import save_image as tv_save_image


def set_seed(seed: int = 42):
    """Set RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model):
    """Return number of trainable parameters (in millions)."""
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total / 1e6


def save_checkpoint(model, path: str):
    """Create dir and save model state."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[checkpoint] Saved → {path}")


def load_checkpoint(model, path: str, device="cuda"):
    """Load weights into model."""
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"[checkpoint] Loaded ← {path}")
    return model


def load_config(path):
    """
    Load YAML config and return it as a flat Namespace.
    Handles nested 'model', 'data', 'training' keys by flattening them.
    """
    with open(path, 'r') as f:
        raw_config = yaml.safe_load(f)

    # Flatten the config so we can access attributes directly (e.g. conf.batch_size)
    flat_config = {}
    for key, value in raw_config.items():
        if isinstance(value, dict):
            flat_config.update(value)
        else:
            flat_config[key] = value

    return Namespace(**flat_config)


def save_image(tensor, path, **kwargs):
    """Wrapper around torchvision save_image to ensure dir exists."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tv_save_image(tensor, path, **kwargs)


class Logger:
    """Tiny text logger; writes both to stdout and file."""

    def __init__(self, path):
        # Ensure directory exists
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.f = open(path, "w")

    def log(self, msg):
        print(msg)
        self.f.write(str(msg) + "\n")
        self.f.flush()

    def close(self):
        self.f.close()