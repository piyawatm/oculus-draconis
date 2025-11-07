# train_prior.py (BOS-aware)
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed
import yaml
import time

# ---- load config / seed
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
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

# ---- build prior by name
model_cfg = dict(cfg["model"])
model_name = model_cfg.pop("name", "BDHPrior")
if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
else:
    raise ValueError(f"Unknown prior name: {model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
prior = Prior(**model_cfg).to(device)

opt = torch.optim.AdamW(prior.parameters(), lr=LR)

total_training_time = 0 
# ---- train
for epoch in range(int(cfg["train"]["epochs"])):
    epoch_start_time = time.time()
    prior.train()
    epoch_loss = 0
    num_batches = 0 
    for x, y in dl:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)  # [B,64]
        logits = prior(x)                         # [B,64,K_vocab]
        loss = F.cross_entropy(logits.reshape(-1, K_vocab), y.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        epoch_loss += loss.item()
        num_batches += 1 
    epoch_time = time.time() - epoch_start_time  # ADD THIS
    total_training_time += epoch_time  # ADD THIS
    avg_loss = epoch_loss / num_batches 
    print(f"Epoch {epoch:03d} | loss={avg_loss:.4f} | time={epoch_time:.2f}s | total={total_training_time/60:.1f}min")

# ---- save
torch.save(prior.state_dict(), SAVE)
print(f"Saved → {SAVE}")
print(f"Total training time: {total_training_time/60:.2f} minutes ({total_training_time/3600:.2f} hours)")
