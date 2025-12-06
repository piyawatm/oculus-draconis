# train_prior.py (BOS-aware)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed
import yaml
import os

# ---- load config / seed
cfg = yaml.safe_load(open("configs/prior_pixelcnn.yaml"))
set_seed(int(cfg["train"]["seed"]))

K_vocab = int(cfg["model"]["vocab_size"])    # 1025
T = int(cfg["model"]["block_size"])          # 64
BATCH = int(cfg["train"]["batch_size"])
LR = float(cfg["train"]["lr"])
SAVE = cfg["train"]["save_path"]

# ---- dataset: load flattened code seq [N, T]
codes_path = "data/codes/ffhq_train.pt"
assert os.path.exists(codes_path), f"Codes file not found at {codes_path}"

data = torch.load(codes_path)

# NEW: handle both tensor and dict formats
if isinstance(data, torch.Tensor):
    seq = data                      # saved directly as tensor [N, T]
elif isinstance(data, dict):
    if "seq" in data:
        seq = data["seq"]
    elif "codes" in data:
        seq = data["codes"]
    else:
        raise KeyError(f"Unknown keys in codes file: {list(data.keys())}")
else:
    raise TypeError(f"Unexpected type loaded from {codes_path}: {type(data)}")

# sanity checks on seq shape
assert seq.ndim == 2, f"Expected seq [N, T], got shape {seq.shape}"
assert seq.size(1) == T, f"Seq length {seq.size(1)} != block_size {T}"

# ---- build BOS-aware inputs/targets
K_code = K_vocab - 1                         # 1024
bos_id = K_code                              # 1024

min_code = int(seq.min().item())
max_code = int(seq.max().item())
assert 0 <= min_code, f"Found negative code index {min_code}"
assert max_code < K_code, (
    f"Found code index {max_code} >= K_code({K_code}). "
    "Update vocab_size in prior_pixelcnn.yaml to num_embeddings+1 from VQ-VAE."
)

bos_col = torch.full((seq.size(0), 1), bos_id, dtype=torch.long)
xb = torch.cat([bos_col, seq[:, :-1]], dim=1)  # [N, 64]
yb = seq.clone()                               # [N, 64]

ds = TensorDataset(xb, yb)
dl = DataLoader(
    ds,
    batch_size=BATCH,
    shuffle=True,
    num_workers=2,
    pin_memory=True,
)

# ---- build prior by name (unchanged) ----
model_cfg = dict(cfg["model"])
model_name = model_cfg.pop("name", "PixelCNNPrior")
if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
elif model_name == "PixelCNNPrior":
    from models.priors.pixelcnn import PixelCNNPrior as Prior
else:
    raise ValueError(f"Unknown prior name: {model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"
prior = Prior(**model_cfg).to(device)

opt = torch.optim.AdamW(prior.parameters(), lr=LR)

# ---- train (unchanged) ----
for epoch in range(int(cfg["train"]["epochs"])):
    prior.train()
    for x, y in dl:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = prior(x)                    # [B, 64, K_vocab]
        loss = F.cross_entropy(
            logits.reshape(-1, K_vocab),
            y.reshape(-1)
        )
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

torch.save(prior.state_dict(), SAVE)
print(f"Saved → {SAVE}")
