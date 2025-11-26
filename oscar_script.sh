#!/bin/bash

# Request a GPU partition node and access to 2 GPUs for 6 hours
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH --mem=32G
#SBATCH -t 06:00:00

# Load a CUDA module
module load cuda

# Run program
# python train_vqvae.py
# python extract_codes.py
# python train_prior.py
python sample_images.py
python eval_metrics.py