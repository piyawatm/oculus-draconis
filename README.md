# Dragon
Biologically Inspired Sparse Graph Model for Image Generation : We use the Dragon Hatchling (BDH) architecture as a sparse, Hebbian graph-based decoder over VQ-VAE image tokens to generate images efficiently while exploring biologically plausible and interpretable neural dynamics.

VQ-VAE and Generative Priors (BDH & MaskGIT)

This project implements a Vector-Quantized VAE (VQ-VAE) and two types of generative priors to model the latent code distribution:

BDHPrior: An autoregressive (sequential) prior.

MaskGITPrior: A non-autoregressive (parallel) prior.

The scripts are configured using .yaml files in the configs/ directory.

Setup

Clone the repository and navigate to the project directory.

Install the required Python packages.

pip install torch torchvision torchmetrics pyyaml tqdm


Create a directory for checkpoints:

mkdir checkpoints


Training Workflow

The process is split into three main stages:

Train the VQ-VAE tokenizer.

Extract the latent codes from the dataset using the trained VQ-VAE.

Train a prior (MaskGIT or BDH) on the extracted codes.

Step 1: Train the VQ-VAE

First, train the VQ-VAE, which learns to compress images into a grid of discrete codes.

python train_vqvae.py


This will train the VQ-VAE on CIFAR-10 and save the model to checkpoints/vqvae.pt.

Step 2: Extract Dataset Codes

Next, use the trained VQ-VAE to "tokenize" the entire training dataset into latent codes. These codes will be the training data for the prior.

(Note: The script extract_codes.py is assumed to exist for this step.)

# This is a placeholder command, as extract_codes.py was not provided.
# This script would load 'checkpoints/vqvae.pt' and save the codes.
python extract_codes.py --vq_ckpt checkpoints/vqvae.pt --out_path data/codes/cifar10_train.pt


This script should save the extracted code sequences to data/codes/cifar10_train.pt, which is loaded by train_prior.py.

Step 3: Train the Prior

Now you can train a prior to learn the distribution of the latent codes. You can choose which prior to train by specifying its config file.

To train the MaskGITPrior:

python train_prior.py --config configs/prior_maskgit.yaml


This will save the model to checkpoints/maskgit_prior.pt.

To train the BDHPrior:

python train_prior.py --config configs/prior_bdh.yaml


This will save the model to checkpoints/bdh_prior.pt.

Evaluation and Sampling

Once a prior is trained, you can use it to generate new images.

Step 4: Sample Images

Generate a grid of sample images to visually inspect the model's quality.

To sample from MaskGIT:
(This will create samples_prior_maskgit.png)

python sample_images.py --config configs/prior_maskgit.yaml


To sample from BDH:
(This will create samples_prior_bdh.png)

python sample_images.py --config configs/prior_bdh.yaml


You can generate a different number of images using the --n flag:

python sample_images.py --config configs/prior_maskgit.yaml --n 64 --out samples_64.png


Step 5: Evaluate Metrics (FID/IS)

Calculate the Fréchet Inception Distance (FID) and Inception Score (IS) for a large number of generated samples (e.g., 10,000).

To evaluate MaskGIT:

python eval_metrics.py --config configs/prior_maskgit.yaml


To evaluate BDH:

python eval_metrics.py --config configs/prior_bdh.yaml


You can change the number of samples used for evaluation:

python eval_metrics.py --config configs/prior_maskgit.yaml --num_samples 50000
