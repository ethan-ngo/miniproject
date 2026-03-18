# LVM Disaster Assessment Miniproject (VAE Segment)

This repository contains the Variational Autoencoder (VAE) component of my disaster assessment pipeline, developed for the CS4990 Generative AI Miniproject.

## Project Overview

The objective of this miniproject is to leverage a Large Vision Model (LVM) to learn rich, conceptual representations of post-disaster satellite/aerial imagery. I utilize the [LADI v2 Dataset](https://huggingface.co/datasets/MITLL/LADI-v2-dataset) (Low Altitude Disaster Imagery).

To fulfill the LVM requirements of the assignment, this specific repository focuses purely on the **Variational Autoencoder (VAE)**. The VAE is designed to ingest raw aerial imagery and compress it into a continuous 256-dimensional latent space. By training the model to reconstruct the images from this compressed bottleneck, the VAE inherently learns fundamental, high-level features about the disaster scenes (e.g., the presence of water, damaged structures, and debris).

*Note: The subsequent stages of this pipeline (e.g., training a downstream multi-label classifier on the VAE's latent variables) are reserved for the final project implementation.*

## Environment Setup

This project uses `conda` to manage dependencies. An RTX 4090 GPU (or equivalent) is recommended for training.

1. Clone this repository.
2. Create and activate the conda environment:
   ```bash
   conda create -n ladi python=3.10
   conda activate ladi
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Running the VAE

The dataset is automatically downloaded and cached natively using Hugging Face datasets and pandas.

To begin training the VAE (defaults to 150 epochs, saving checkpoints every 50 epochs):

```bash
python train_vae.py
```

### MLOps Tracking

This project heavily integrates **Weights & Biases (W&B)** for professional MLOps tracking. Before running the training script, ensure you are logged into your W&B account:

```bash
wandb login
```

The script will automatically:
1. Stream live training metrics (Mean Squared Error, KL Divergence) to your W&B cloud dashboard.
2. Upload image reconstruction visualization grids per epoch to visually track the VAE's generative progress.
3. Automatically upload the `.pth` model checkpoints to the W&B Artifacts model registry every 50 epochs.
4. Trigger alerts if gradient explosion (NaN loss) is detected.
