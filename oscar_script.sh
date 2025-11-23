#!/bin/bash

# Request a GPU partition node and access to 2 GPUs
#SBATCH -p gpu --gres=gpu:2

# Request 1 CPU core
#SBATCH -n 1
#SBATCH -t 04:00:00

# Load a CUDA module
module load cuda

# Run program
 python train_prior.py
python sample_images.py
python eval_metrics.py
