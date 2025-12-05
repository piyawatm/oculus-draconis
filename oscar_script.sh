#!/bin/bash

#!/bin/bash
#SBATCH -p gpu
#SBATCH --constraint=geforce3090|quadrortx|titanrtx  # Request ONLY the best cards in the free pool
#SBATCH --gres=gpu:2
#SBATCH -n 1                 # 1 Task (Python script)
#SBATCH -c 10                # 10 CPU cores (Near your 12-core limit, leaves buffer)
#SBATCH --mem=64G            # Double your memory (Safe limit, well below 192G max)
#SBATCH -t 12:00:00           # (Max is 48 hours for 2 GPUs if you need longer)
#SBATCH -J optimized_job
#SBATCH -o output_%j.txt

module load cuda        # Always specify a version

# Debug: Print what GPU you actually got
nvidia-smi

# Run program
# python ffhq_dataset.py
# python train_vqvae.py
# python extract_codes.py
# python train_prior.py
python sample_images.py --temp 0.8

# python eval_metrics.py