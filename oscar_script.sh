#!/bin/bash

# Request a GPU partition node and access to 2 GPUs for 6 hours
#SBATCH -p gpu --gres=gpu:2
#SBATCH -n 4
#SBATCH -t 06:00:00

# Load a CUDA module
module load cuda

# Run program
# python train_prior.py --config configs/prior_maskgit.yaml
python eval_metrics.py --config configs/prior_maskgit.yaml