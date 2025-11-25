# utils.py
import os, random, numpy as np, torch

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


class Logger:
    """Tiny text logger; writes both to stdout and file."""
    def __init__(self, path):
        os.makedirs("logs", exist_ok=True)
        self.f = open(path, "w")
    def log(self, msg):
        print(msg)
        self.f.write(msg + "\n")
        self.f.flush()
    def close(self):
        self.f.close()
