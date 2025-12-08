#!/bin/bash
# Request a GPU partition node and access to 2 GPUs for 6 hours
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 1
#SBATCH -t 06:00:00

# Load a CUDA module
module load cuda
# Complete training pipeline for BDH-VQVAE

# 1. Check necessary files
python check_files.py

# 2. Verify the BDH implementation
python verify_bdh.py

# 3. Train BDH-VQVAE
python train.py --model bdh_vqvae --dataset cifar10 --epochs 100

# 4. Evaluate
python evaluate.py --model bdh_vqvae --dataset cifar10

# 5. Compare all models
python compare.py --models vqvae vqvae2 bdh_vqvae --generate-images