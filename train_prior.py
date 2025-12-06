import torch, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from utils import set_seed
import yaml
import time

# -------- load config --------
cfg = yaml.safe_load(open("configs/pixelsnail.yaml"))
set_seed(int(cfg["train"]["seed"]))

K_vocab = int(cfg["model"]["vocab_size"])      # 1025
T_plus_1 = int(cfg["model"]["block_size"])     # 257
T = T_plus_1 - 1                               # 256 tokens
HEIGHT = int(cfg["model"]["height"])           # 16
WIDTH = int(cfg["model"]["width"])             # 16

BATCH = int(cfg["train"]["batch_size"])
LR = float(cfg["train"]["lr"])
SAVE = cfg["train"]["save_path"]

# -------- load FFHQ codes --------
codes = torch.load("data/codes/ffhq_train.pt")     # [N, 256]
assert codes.shape[1] == T, f"Expected {T} tokens, got {codes.shape[1]}"

# -------- BOS handling --------
BOS_ID = K_vocab - 1       # 1024

bos_col = torch.full((codes.size(0), 1), BOS_ID, dtype=torch.long)
xb = torch.cat([bos_col, codes[:, :-1]], dim=1)   # shift
yb = codes.clone()

ds = TensorDataset(xb, yb)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, num_workers=2, pin_memory=True)

# -------- build model --------
model_cfg = dict(cfg["model"])
model_name = model_cfg.pop("name")
from models.priors.pixelsnail import PixelSNAILPrior as Prior

device = "cuda" if torch.cuda.is_available() else "cpu"
prior = Prior(**model_cfg).to(device)

opt = torch.optim.AdamW(prior.parameters(), lr=LR)

# -------- training loop --------
total_training_time = 0

for epoch in range(int(cfg["train"]["epochs"])):
    start = time.time()
    prior.train()

    epoch_loss = 0
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = prior(x)
        loss = F.cross_entropy(logits.reshape(-1, K_vocab), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        epoch_loss += loss.item()

    dt = time.time() - start
    total_training_time += dt
    print(f"Epoch {epoch:03d} | loss={epoch_loss/len(dl):.4f} | time={dt:.2f}")

torch.save(prior.state_dict(), SAVE)
print(f"Saved → {SAVE}")
