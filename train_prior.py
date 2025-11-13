import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed, save_checkpoint, count_parameters
import yaml
import argparse # Added for config file argument

# ---- config setup
parser = argparse.ArgumentParser(description="Train a VQ-VAE Prior")
parser.add_argument(
    '--config', 
    type=str, 
    default="configs/prior_bdh.yaml",
    help="Path to the config YAML file (e.g., configs/prior_maskgit.yaml)"
)
args = parser.parse_args()

print(f"Loading config from: {args.config}")
cfg = yaml.safe_load(open(args.config))
set_seed(int(cfg["train"]["seed"]))

# ---- training config
BATCH = int(cfg["train"]["batch_size"])
LR = float(cfg["train"]["lr"])
SAVE = cfg["train"]["save_path"]
EPOCHS = int(cfg["train"]["epochs"])
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---- model config
model_cfg = dict(cfg["model"])
model_name = model_cfg.pop("name", "BDHPrior")

# ---- build prior by name
if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
    is_maskgit = False
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
    is_maskgit = False
elif model_name == "MaskGITPrior":
    from models.priors.maskgit import MaskGITPrior as Prior
    is_maskgit = True
else:
    raise ValueError(f"Unknown prior name: {model_name}")

prior = Prior(**model_cfg).to(device)
print(f"Loaded model: {model_name} with {count_parameters(prior):.2f}M parameters")

# ---- dataset: load flattened code seq [N, T]
# This is the raw sequence of VQ codes, e.g., [N, 64]
data = torch.load("data/codes/cifar10_train.pt")
seq = data["seq"] # [N, 64]

# ---- setup dataset based on model type
if is_maskgit:
    # MaskGIT trains on the raw sequences [N, T]
    # Masking is applied on-the-fly in the training loop
    print("Setting up MaskGIT data (T = 64)")
    K_vocab = int(cfg["model"]["vocab_size"]) # e.g., 512
    ds = TensorDataset(seq)

    # Get mask token ID from the model instance
    mask_token_id = prior.mask_token_id
    # Masking schedule (could be in config, hardcoded for simplicity)
    mask_prob = 0.15 

else:
    # Autoregressive models (BDH, GPT) need BOS-aware, shifted targets
    print("Setting up Autoregressive (BOS-aware) data (T+1 = 65)")
    K_vocab = int(cfg["model"]["vocab_size"]) # e.g., 513
    
    # BOS id is the extra vocab id at the end
    K_code = K_vocab - 1      # 512
    bos_id = K_code           # 512
    bos_col = torch.full((seq.size(0), 1), bos_id, dtype=torch.long)
    
    xb = torch.cat([bos_col, seq[:, :-1]], dim=1) # [N, 64] ; first token = BOS
    yb = seq.clone()                             # [N, 64] ; predict original codes
    
    ds = TensorDataset(xb, yb)

# ---- common dataloader and optimizer
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)
opt = torch.optim.AdamW(prior.parameters(), lr=LR)

# ---- train
print(f"Starting training on {device}...")
for epoch in range(EPOCHS):
    prior.train()
    
    for batch in dl:
        # ---- Branching training logic ----
        if is_maskgit:
            x_true = batch[0].to(device, non_blocking=True) # [B, T]
            B, T = x_true.shape
            
            # 1. Create mask: True = to be masked
            mask = torch.rand(x_true.shape, device=device) < mask_prob
            
            # 2. Create masked input
            x_masked = x_true.clone()
            x_masked[mask] = mask_token_id
            
            # 3. Create labels: -100 at unmasked positions
            labels = x_true.clone()
            labels[~mask] = -100 # ignore_index for cross_entropy
            
            # 4. Forward pass
            logits = prior(x_masked) # [B, T, K_vocab]
            
            # 5. Compute loss only on masked positions
            loss = F.cross_entropy(
                logits.view(-1, K_vocab), # [B*T, K_vocab]
                labels.view(-1),          # [B*T]
                ignore_index=-100
            )
            
        else:
            # Autoregressive training
            x, y = batch # x=[B, 64] (BOS + seq), y=[B, 64] (seq)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            
            logits = prior(x) # [B, 64, K_vocab]
            
            # Compute loss on all positions
            loss = F.cross_entropy(
                logits.reshape(-1, K_vocab), # [B*64, K_vocab]
                y.reshape(-1)                # [B*64]
            )
        
        # ---- Common optimizer step ----
        opt.zero_grad()
        loss.backward()
        if cfg["train"]["clip_grad"] is not None:
             torch.nn.utils.clip_grad_norm_(prior.parameters(), float(cfg["train"]["clip_grad"]))
        opt.step()
        
    print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

# ---- save
save_checkpoint(prior, SAVE)
print(f"Saved model checkpoint → {SAVE}")