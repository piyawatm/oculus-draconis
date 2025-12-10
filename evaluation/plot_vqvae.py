import matplotlib.pyplot as plt
import re

def parse_vqvae_log(filepath):
    """
    Parses VQ-VAE logs to extract Iteration, Loss, Recon, Q, and PPL.
    Expected format: 
    Iter 100: Loss=0.3225 | Recon=0.1215 | Q=0.8039 | PPL=1.86 ...
    """
    data = {
        "iter": [],
        "loss": [],
        "recon": [],
        "q": [],
        "ppl": []
    }
    
    # Regex to capture all 5 values
    pattern = re.compile(r"Iter (\d+): Loss=([\d\.]+) \| Recon=([\d\.]+) \| Q=([\d\.]+) \| PPL=([\d\.]+)")
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    data["iter"].append(int(match.group(1)))
                    data["loss"].append(float(match.group(2)))
                    data["recon"].append(float(match.group(3)))
                    data["q"].append(float(match.group(4)))
                    data["ppl"].append(float(match.group(5)))
    except FileNotFoundError:
        print(f"Error: File {filepath} not found.")
        return None
            
    return data

# --- Configuration ---
log_file = "training_loss/vqvae_training_log.txt" 

# --- Plotting ---
data = parse_vqvae_log(log_file)

if data and data["iter"]:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VQ-VAE Training Progress', fontsize=16)

    # 1. Total Loss
    axes[0, 0].plot(data["iter"], data["loss"], color='#1f77b4', label='Total Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_ylabel('Loss Value')
    axes[0, 0].grid(True, linestyle='--', alpha=0.6)

    # 2. Reconstruction Loss
    axes[0, 1].plot(data["iter"], data["recon"], color='#2ca02c', label='Recon Loss')
    axes[0, 1].set_title('Reconstruction Loss')
    axes[0, 1].grid(True, linestyle='--', alpha=0.6)

    # 3. Quantization Loss
    axes[1, 0].plot(data["iter"], data["q"], color='#d62728', label='Quantization Loss')
    axes[1, 0].set_title('Quantization Loss')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Loss Value')
    axes[1, 0].grid(True, linestyle='--', alpha=0.6)

    # 4. Perplexity
    axes[1, 1].plot(data["iter"], data["ppl"], color='#9467bd', label='Perplexity')
    axes[1, 1].set_title('Codebook Perplexity (PPL)')
    axes[1, 1].set_xlabel('Iteration')
    axes[1, 1].grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("vqvae_metrics.png", dpi=300)
    plt.show()
    print("Plot saved as vqvae_metrics.png")
else:
    print("No valid log data found.")