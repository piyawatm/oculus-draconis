import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.priors.bdh import BDHPrior
from utils import load_config, Logger 
import argparse
import os
import time
from torch.cuda.amp import GradScaler

def calculate_accuracy(logits, targets, k=1):
    """Compute top-k accuracy."""
    with torch.no_grad():
        _, pred_indices = torch.topk(logits, k=k, dim=-1)
        # pred_indices: [B*T, k], targets: [B*T]
        correct = pred_indices.eq(targets.view(-1, 1).expand_as(pred_indices))
        return correct.sum().float() / targets.numel()

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/bdh_config.yaml')
    args = parser.parse_args()
    
    conf = load_config(args.config)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Initialize Logger
    logger = Logger("logs/prior_train.log")
    logger.log(f"Starting BDH Prior training on {device}")
    
    # Load Codes
    if not os.path.exists("data/codes/ffhq_train.pt"):
        logger.log("Error: Codes not found! Run extract_codes.py first.")
        return

    data = torch.load("data/codes/ffhq_train.pt") # Shape [N, 256]
    dataset = TensorDataset(data)
    
    loader = DataLoader(dataset, batch_size=conf.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    model = BDHPrior(conf).to(device)

    try:
        model = torch.compile(model)
        logger.log("Enabled torch.compile() for speedup.")
    except:
        pass
    
    # Explicitly cast LR to float to prevent config type errors
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(conf.learning_rate), weight_decay=0.05)
    scaler = GradScaler()
    
    step = 0
    model.train()
    total_steps = conf.max_iters
    
    # BOS Token
    bos_token = conf.vocab_size - 1
    logger.log(f"Training with BOS token index: {bos_token}")

    start_time = time.time()
    
    while step < total_steps:
        for batch in loader:
            codes = batch[0].to(device, non_blocking=True).long()
            
            # Prepend BOS
            bs = codes.size(0)
            bos = torch.full((bs, 1), bos_token, device=device, dtype=torch.long)
            full_seq = torch.cat((bos, codes), dim=1)
            
            inp = full_seq[:, :-1]
            tgt = full_seq[:, 1:]
            
            optimizer.zero_grad(set_to_none=True)
            
            with torch.amp.autocast('cuda'):
                logits, loss = model(inp, tgt)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            
            step += 1
            
            if step % conf.log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = conf.log_interval / elapsed
                start_time = time.time()
                
                # --- METRICS CALCULATION ---
                # Flatten for metrics
                flat_logits = logits.reshape(-1, logits.size(-1))
                flat_targets = tgt.reshape(-1)
                
                acc1 = calculate_accuracy(flat_logits, flat_targets, k=1)
                acc5 = calculate_accuracy(flat_logits, flat_targets, k=5)
                bpd = loss.item() / 0.693147 # Loss is Nats, divide by ln(2) for Bits
                # ---------------------------
                
                logger.log(f"Step {step}: Loss={loss.item():.4f} | BPD={bpd:.2f} | "
                           f"Acc@1={acc1.item():.2%} | Acc@5={acc5.item():.2%} | "
                           f"Speed={steps_per_sec:.2f} it/s")
            
            # Save 'Best' (Latest) frequently
            if step % conf.save_interval == 0:
                torch.save(model.state_dict(), "checkpoints/prior_best.pt")
                logger.log(f"Saved checkpoint to checkpoints/prior_best.pt")
            
            # Save Historic Checkpoint every 10k
            if step % 10000 == 0:
                ckpt_name = f"checkpoints/prior_{step}.pt"
                torch.save(model.state_dict(), ckpt_name)
                logger.log(f"Saved historic checkpoint to {ckpt_name}")
                
            if step >= total_steps:
                break
    
    torch.save(model.state_dict(), "checkpoints/prior_final.pt")
    logger.log("Training Complete.")
    logger.close()

if __name__ == "__main__":
    train()
