import matplotlib.pyplot as plt
import re
import sys

# --- CONFIGURATION: Update these paths to your actual log files ---
log_files = {
    "MaskGIT": "training_loss/maskgit_train_cifar.log",
    "PixelCNN": "training_loss/pixelcnn_train_cifar.log",
    "PixelSNAIL": "training_loss/pixelsnail_train_cifar.log",
    "BDH (Ours)": "training_loss/bdh_train_cifar.log"
}

# Define distinct colors and styles for clarity
styles = {
    "MaskGIT":    {"color": "#e41a1c", "marker": "o", "style": "--"}, # Red, dashed
    "PixelCNN":   {"color": "#377eb8", "marker": "s", "style": "--"}, # Blue, dashed
    "PixelSNAIL": {"color": "#4daf4a", "marker": "^", "style": "--"}, # Green, dashed
    "BDH (Ours)": {"color": "#984ea3", "marker": "D", "style": "-"}   # Purple, solid (Highlight)
}

def parse_log_file(filepath):
    """
    Parses a log file. Automatically detects if it uses 'Step' or 'Epoch' format.
    Returns: (x_values, loss_values, label_type)
    """
    x_values = []
    losses = []
    
    # Regex 1: "Step 100: Loss=6.5411"
    pattern_step = re.compile(r"Step (\d+): Loss=([\d\.]+)")
    # Regex 2: "Epoch 000 | loss=0.199104"
    pattern_epoch = re.compile(r"Epoch (\d+) \| loss=([\d\.]+)")
    
    detected_type = "Steps" # Default

    try:
        with open(filepath, 'r') as f:
            for line in f:
                # Try matching Step format
                match_step = pattern_step.search(line)
                if match_step:
                    x_values.append(int(match_step.group(1)))
                    losses.append(float(match_step.group(2)))
                    detected_type = "Steps"
                    continue # Found match, move to next line

                # Try matching Epoch format
                match_epoch = pattern_epoch.search(line)
                if match_epoch:
                    x_values.append(int(match_epoch.group(1)))
                    losses.append(float(match_epoch.group(2)))
                    detected_type = "Epochs"

    except FileNotFoundError:
        print(f"Warning: File not found at {filepath}")
        return [], [], "Unknown"

    return x_values, losses, detected_type

# --- PLOTTING ---
plt.figure(figsize=(12, 7))

for model_name, filepath in log_files.items():
    x, y, x_label = parse_log_file(filepath)
    
    if not x:
        print(f"Skipping {model_name} (No data found)")
        continue
        
    style = styles.get(model_name, {"color": "black", "marker": "x", "style": "-"})
    
    # Plotting
    plt.plot(x, y, 
             label=model_name,
             color=style["color"],
             linestyle=style["style"],
             linewidth=2 if model_name == "BDH (Ours)" else 1.5, # Make ours slightly thicker
             marker=style["marker"],
             markevery=max(1, int(len(x)/10)), # Prevent marker clutter
             markersize=5,
             alpha=0.8)

    # Update label based on the last detected file type (assuming all are consistent, 
    # or you might want to normalize them if they are mixed)
    final_x_label = x_label

plt.title("Training Loss Comparison: 4 Priors", fontsize=16)
plt.xlabel(final_x_label, fontsize=14)
plt.ylabel("Loss", fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.5)

# Optional: Log scale if early losses are huge compared to final losses
# plt.yscale('log')

plt.tight_layout()
plt.savefig("priors_comparison.png", dpi=300)
plt.show()