#!/bin/bash

# Request a GPU partition node and access to 4 GPUs for 20 minutes
#SBATCH -p gpu --gres=gpu:4
#SBATCH -n 1
#SBATCH -t 00:20:00

# Load a CUDA module
module load cuda

# Run program
python train_prior.py --config configs/prior_maskgit.yaml
python eval_metrics.py --config configs/prior_maskgit.yaml