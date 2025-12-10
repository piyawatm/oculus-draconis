# Oculus Draconis

Biologically Inspired Sparse Graph Model for Image Generation

Uses the Dragon Hatchling (BDH) architecture as a sparse, Hebbian graph-based decoder over VQ-VAE image tokens to generate images efficiently while exploring biologically plausible and interpretable neural dynamics.

## Pipeline

The image generation pipeline has 5 steps:

1. **Train VQ-VAE**: Encode images into discrete tokens
2. **Extract codes**: Extract token sequences from training images
3. **Train prior**: Train a prior model to generate token sequences
4. **Sample images**: Generate new images from the prior
5. **Evaluate**: Compute metrics on generated images

## Running

### Step 1: Train VQ-VAE

`python train_vqvae.py`

### Step 2: Extract codesh

`python extract_codes.py`

### Step 3: Train prior

For BDH prior:

`python train_prior.py --config configs/prior_bdh.yaml`

For MaskGIT prior:

`python train_prior.py --config configs/prior_maskgit.yaml`

For PixelCNN prior:

`python train_prior.py --config configs/prior_pixelcnn.yaml`

For PixelSNAIL prior:

`python train_prior.py --config configs/prior_pixelsnail.yaml`

### Step 4: Sample images

For BDH prior:

`python sample_images.py --config configs/prior_bdh.yaml`

For MaskGIT prior:

`python sample_images.py --config configs/prior_maskgit.yaml`

For PixelCNN prior:

`python sample_images.py --config configs/prior_pixelcnn.yaml`

For PixelSNAIL prior:

`python sample_images.py --config configs/prior_pixelsnail.yaml`

### Step 5: Evaluate

For BDH prior:

`python eval_metrics.py --prior_config configs/prior_bdh.yaml`

For MaskGIT prior:

`python eval_metrics.py --prior_config configs/prior_maskgit.yaml`

For PixelCNN prior:

`python eval_metrics.py --prior_config configs/prior_pixelcnn.yaml`

For PixelSNAIL prior:

`python eval_metrics.py --prior_config configs/prior_pixelsnail.yaml`

## SLURM/Cluster

Use `oscar_script.sh` for cluster execution. Uncomment the lines for the prior you want to train/evaluate.

## Results

### Generated images
- `evaluation/sample_images/` - Generated image samples
  - `samples_bdh_prior.png` - BDH prior samples
  - `samples_bdh_prior.pt_epoch*.png` - Samples at different epochs

### Training logs
- `evaluation/training_loss/` - Training loss logs
  - `bdh_train.log` - BDH training log
  - `maskgit_train.log` - MaskGIT training log
  - `pixelcnn_train.log` - PixelCNN training log
  - `pixelsnail_train.log` - PixelSNAIL training log
  - `vqvae_training_log.txt` - VQ-VAE training log

### Checkpoints
- `checkpoints/` - Model checkpoints
  - `vqvae_best.pt` - Best VQ-VAE checkpoint
  - `prior_best.pt` - Best prior checkpoint
  - `prior_final.pt` - Final prior checkpoint
  - `prior_*.pt` - Checkpoints at different iterations

### Evaluation metrics
- `evaluation/evaluation_loss_curve_*.png` - Loss curves
- `evaluation/vqvae_metrics.png` - VQ-VAE metrics

## Configuration

Prior models are configured in `configs/`:
- `configs/prior_bdh.yaml` - BDH prior config
- `configs/prior_maskgit.yaml` - MaskGIT prior config
- `configs/prior_pixelcnn.yaml` - PixelCNN prior config
- `configs/prior_pixelsnail.yaml` - PixelSNAIL prior config

Edit these files to change model architecture, training parameters, or dataset settings.
