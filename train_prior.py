# train_prior.py
import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed
import yaml

# ---- load config / seed
cfg = yaml.safe_load(open("configs/prior_bdh.yaml"))
set_seed(cfg["train"]["seed"])

K = cfg["model"]["vocab_size"]
T = cfg["model"]["block_size"]
BATCH = cfg["train"]["batch_size"]

# ---- dataset
data = torch.load("data/codes/cifar10_train.pt")
seq = data["seq"]                       # [N, T]
ds  = TensorDataset(seq[:, :-1], seq[:, 1:])
dl  = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2)  # <= 2 workers

# ---- build prior from YAML 'name'
model_cfg = dict(cfg["model"])          # shallow copy
model_name = model_cfg.pop("name", "BDHPrior")

if model_name == "BDHPrior":
    from models.priors.bdh import BDHPrior as Prior
elif model_name == "GPTPrior":
    from models.priors.gpt import GPTPrior as Prior
else:
    raise ValueError(f"Unknown prior name in config: {model_name}")

prior = Prior(**model_cfg).cuda()

opt = torch.optim.AdamW(prior.parameters(), lr=cfg["train"]["lr"])

# ---- train
for epoch in range(cfg["train"]["epochs"]):
    prior.train()
    for xb, yb in dl:
        xb, yb = xb.cuda(non_blocking=True), yb.cuda(non_blocking=True)
        logits = prior(xb)                              # [B, T-1, K]
        loss = F.cross_entropy(logits.reshape(-1, K), yb.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"Epoch {epoch:03d} | loss={loss.item():.4f}")

# ---- save
torch.save(prior.state_dict(), cfg["train"]["save_path"])
print(f"Saved → {cfg['train']['save_path']}")
