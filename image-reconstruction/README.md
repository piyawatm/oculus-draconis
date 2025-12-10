# VQ-VAE with Baby Dragon Hatchling Attention

This codebase compares three VQ-VAE architectures for image reconstruction:
- Vanilla VQ-VAE (single-level baseline)
- VQ-VAE-2 (hierarchical, two quantization levels)
- BDH-VQVAE (uses Baby Dragon Hatchling attention)

## What is BDH?

Baby Dragon Hatchling (BDH) is an attention mechanism from ["The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain"](https://arxiv.org/abs/2509.26507).

Main differences from standard attention:
- Uses ReLU instead of softmax (sparse, non-negative activations)
- Q=K constraint (queries and keys share the same projection)
- RoPE position embeddings
- Multiplicative gating instead of additive residuals

## Setup

pip install -r requirements.txt

## Running

Follow the steps in `oscar_script.sh`:

### 1. Check files
python check_files.py

### 2. Verify BDH implementation
python verify_bdh.py

### 3. Train
python train.py --model bdh_vqvae --dataset cifar10 --epochs 100

### 4. Evaluate
python evaluate.py --model bdh_vqvae --dataset cifar10

### 5. Compare models
python compare.py --models vqvae vqvae2 bdh_vqvae --generate-imagesFor cluster/SLURM:
sbatch oscar_script.sh

## Results

Results are saved in `outputs/`:

### Per-model results
- `outputs/{model_name}/eval_results.json` - Metrics (PSNR, LPIPS, MSE, codebook usage)
- `outputs/{model_name}/eval_originals.png` - Original test images
- `outputs/{model_name}/eval_reconstructions.png` - Reconstructed images
- `outputs/{model_name}/eval_comparison.png` - Side-by-side comparison

Example `eval_results.json`:
{
  "model": "bdh_vqvae",
  "dataset": "cifar10",
  "metrics": {
    "psnr_mean": 25.46,
    "lpips_mean": 0.044,
    "mse_mean": 0.013,
    "codebook_usage": 0.525
  }
}

### Comparison results
- `outputs/comparison/comparison_results.csv` - Table format
- `outputs/comparison/comparison_results.json` - JSON format
- `outputs/comparison/model_comparison.png` - Visual comparison

Metrics:
- PSNR: Higher is better
- LPIPS: Lower is better
- MSE: Lower is better
- Codebook Usage: Percentage of codebook entries used

## Architecture

BDH attention replaces standard attention in the VQ-VAE encoder/decoder:

q = encoder(x)

k = q  # Q=K constraint

v = encoder_v(x)

q, k = apply_rope(q, k)  # Rotary position embeddings

attn = (q @ k.T) / sqrt(d)

attn = F.relu(attn)  # ReLU instead of softmax

out = attn @ vPipeline: 

Image → CNN Encoder → BDH Attention → Quantize → BDH Attention → CNN Decoder → Reconstruction

## Configuration

Edit `config.py` to change:
- `n_bdh_layers` - Number of BDH attention layers (default: 3)
- `n_head` - Attention heads (default: 4)
- `hidden_dims` - Feature dimensions
- `num_embeddings` - Codebook sizes

## Files

- `models/bdh_vqvae.py` - BDH-VQVAE implementation
- `models/vqvae.py` - Vanilla VQ-VAE
- `models/vqvae2.py` - VQ-VAE-2
- `train.py` - Training script
- `evaluate.py` - Evaluation script
- `compare.py` - Model comparison
- `verify_bdh.py` - BDH verification
- `oscar_script.sh` - SLURM batch script

## Output Structure

```
outputs/
├── bdh_vqvae/
│   ├── eval_results.json
│   ├── eval_originals.png
│   ├── eval_reconstructions.png
│   ├── eval_comparison.png
│   └── recon_epoch_*.png
├── vqvae/
│   └── ...
├── vqvae2/
│   └── ...
└── comparison/
    ├── comparison_results.csv
    ├── comparison_results.json
    └── model_comparison.png
```

## References

- BDH: Kosowski et al. (2025). "The Dragon Hatchling: The Missing Link between the Transformer and Models of the Brain". arXiv:2509.26507
- VQ-VAE: van den Oord et al. (2017). "Neural Discrete Representation Learning". NeurIPS.
- VQ-VAE-2: Razavi et al. (2019). "Generating Diverse High-Fidelity Images with VQ-VAE-2". NeurIPS.
