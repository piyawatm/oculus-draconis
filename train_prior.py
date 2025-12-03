# train_prior.py (BOS-aware, AMP, silent stdout, logs to file)
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed, Logger
import yaml
import time

# ------------- logging setup (no terminal prints) -------------
log_path = "logs/train_prior_log.txt"

logger  = Logger(log_path)

# Silence stdout
sys.stdout = open(os.devnull, "w")

# ---- load config / seed ----
cfg = yaml.safe_load(open("configs/prior_bdh.yaml"))
set_seed(int(cfg["train"]["seed"]))

K_vocab = int(cfg["model"]["vocab_size"])  # e.g., 513
T_plus_1 = int(cfg["model"]["block_size"]) # e.g., 65
T = T_plus_1 - 1                           # 64 tokens from VQ codes
BATCH = int(cfg["train"]["batch_size"])
LR = float(cfg["train"]["lr"])
SAVE = cfg["train"]["save_path"]

# ---- dataset: load flattened code seq [N, T]
data = torch.load("data/codes/cifar10_train.pt")
seq = data["seq"]                  # [N, 64] from VQ-VAE extraction

# ---- build BOS-aware inputs/targets
# BOS id is the extra vocab id at the end
K_code = K_vocab - 1               # 512
bos_id = K_code                    # 512
bos_col = torch.full((seq.size(0), 1), bos_id, dtype=torch.long)
xb = torch.cat([bos_col, seq[:, :-1]], dim=1)  # [N, 64] ; first token = BOS
yb = seq.clone()                               # [N, 64] ; predict original codes

ds = TensorDataset(xb, yb)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                num_workers=2, pin_memory=True)

# ---- build prior by name ----
model_cfg  = dict(cfg["model"])
model_name = model_cfg.pop("name", "BDHPrior")
if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
else:
    raise ValueError(f"Unknown prior name: {model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
prior  = Prior(**model_cfg).to(device)

opt = torch.optim.AdamW(prior.parameters(), lr=LR)

# ---- AMP setup ----
use_amp = (device == "cuda")
if use_amp:
    from torch.amp import GradScaler, autocast
    scaler = GradScaler("cuda")
else:
    scaler = None

total_training_time = 0
epoch_times = []

# ---- train ----
for epoch in range(int(cfg["train"]["epochs"])):
    epoch_start_time = time.time()
    epoch_loss = 0
    num_batches = 0
    prior.train()
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        opt.zero_grad()

        if use_amp:
            with autocast("cuda", dtype=torch.float16):
                logits = prior(x)
                loss = F.cross_entropy(
                    logits.reshape(-1, K_vocab),
                    y.reshape(-1)
                )
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            logits = prior(x)
            loss = F.cross_entropy(
                logits.reshape(-1, K_vocab),
                y.reshape(-1)
            )
            loss.backward()
            opt.step()
        epoch_loss += loss.item()
        num_batches += 1
    epoch_time = time.time() - epoch_start_time 
    epoch_times.append(epoch_time)  
    total_training_time += epoch_time           

    avg_loss = epoch_loss / num_batches

    # log last loss of epoch to file only
    logger.log(f"Epoch {epoch:03d} | loss={loss.item():.6f} | time={epoch_time:.2f}s | total={total_training_time/60:.1f}min")
    
    # Save checkpoint every 5 epochs
    # if (epoch + 1) % 5 == 0:
    ckpt_path = f"{SAVE}_epoch{epoch+1:03d}.pt"
    torch.save(prior.state_dict(), ckpt_path)
    logger.log(f"Checkpoint saved → {ckpt_path}")

# ---- save final checkpoint ----
torch.save(prior.state_dict(), SAVE)
logger.log(f"Final checkpoint saved → {SAVE}")

avg_epoch_time = sum(epoch_times) / len(epoch_times)
logger.log(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
logger.log(f"Average epoch time: {avg_epoch_time:.2f} seconds")

logger.close()
