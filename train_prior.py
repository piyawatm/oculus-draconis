import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from models.priors.bdh import BDHPrior
from utils import load_config, Logger # Import Logger
import argparse
import os
import time
from torch.cuda.amp import GradScaler

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
    optimizer = torch.optim.AdamW(model.parameters(), lr=conf.learning_rate, weight_decay=0.05)
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
                logger.log(f"Step {step}: Loss={loss.item():.4f}| Elapsed Time={elapsed:.2f}sec | Speed={steps_per_sec:.2f} it/s")
                
            if step % conf.save_interval == 0:
                torch.save(model.state_dict(), "checkpoints/prior_best.pt")
                logger.log(f"Saved checkpoint to checkpoints/prior_best.pt")
                
            if step >= total_steps:
                break
    
    torch.save(model.state_dict(), "checkpoints/prior_best.pt")
    logger.log("Training Complete.")
    logger.close()

if __name__ == "__main__":
    train()
