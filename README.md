# Graph VQ-Transformer (GVT)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-310/)

This is the official PyTorch implementation for the paper: **"Graph VQ-Transformer (GVT): Fast and Accurate Molecular Generation via High-Fidelity Discrete Latents"**.

<!-- <p align="center">
  <img src="figures/framework.png" width="800" alt="GVT Framework">
  </p> -->

## Abstract

GVT is a two-stage generative framework designed for efficient and high-quality molecular generation. At its core is a novel Graph Vector Quantized Variational Autoencoder (Graph VQ-VAE) that compresses molecular graphs into high-fidelity discrete latent sequences. Subsequently, an autoregressive Transformer is trained on these discrete sequences, converting the complex task of graph generation into a well-structured sequence modeling problem. GVT achieves state-of-the-art or highly competitive performance across major molecular generation benchmarks.

## Table of Contents
- [Installation](#installation)
- [Dataset Preparation](#dataset-preparation)
- [Training](#training)
  - [Stage 1: Train the Graph VQ-VAE](#stage-1-train-the-graph-vq-vae)
  - [Stage 2: Train the Autoregressive Transformer](#stage-2-train-the-autoregressive-transformer)
- [Generation](#generation)
- [Pre-trained Models](#pre-trained-models)
- [Results](#results)
- [Citation](#citation)
- [License](#license)

## Installation

We recommend using Conda to manage the environment.

1.  **Clone the repository**

2.  **Install Envs**
    We provide an `pyproject.toml` file for easy setup (recommended).
    ```bash
    # install uv before
    uv sync
    ```

## Dataset Preparation

We use the QM9, ZINC250k, MOSES, and GuacaMol datasets. The data loading scripts are located in `src/mydatasets/`.

The data is automatically processed when used. It takes a while to prepare dataset. You can download preprocessed dataset from [here](https://data.zzdirty.com/data.tar.gz) and unzip it at root directory.

## Training

### Stage 1: Train the Graph VQ-VAE

This stage trains the VQ-VAE to learn a high-fidelity discrete representation of molecules.

```bash
uv run src/train_vqvae.py --dataset.name <DATASET_NAME>
```
**Example (for ZINC250k):**
```bash
uv run src/train_vqvae.py --train.batch_size 256 --model.config.lamb_edge 0.2 --model.config.lamb_node 0.1 --model.config.codebook_size 2048 --train.num_epochs 2001 --model.config.hidden_dim 192 --model.config.num_layers 8 --model.config.output_dim 64 --model.config.edge_dim 3 --model.config.input_dim 9 --dataset.name ZINC
```
The trained VQ-VAE checkpoint will be saved in the `runs/` directory.

### Stage 2: Train the Autoregressive Transformer

After training the VQ-VAE, use its encoder to convert the dataset into discrete code sequences. 

```bash
uv run src/vqvae_gen_dataset.py --ckpt_path <PATH_TO_VQ_VAE_CHECKPOINT>
```

Then, train the AR model on these sequences.

```bash
python src/train_ar_model.py --train.dataset_path <PATH_TO_VQ_VAE_GENERATED_DATASET>
```
**Example (for ZINC250k):**
```bash
uv run src/train_ar_model.py --model.config.n_embd 512 --model.config.n_head 16 --model.config.n_positions 16 --train.dataset_path runs/vqvae/QM9/radiant-resonance-16/QM9_vqdataset_valid_reduced.pt --model.config.n_layer 16
```

## Generation

Use a trained model to generate new molecules. The scripts `test_ar_model.py`, `test_ar_model_moses.py`, and `test_ar_model_guacamol.py` contain the logic for generation and evaluation.

## Pre-trained Models

We provide the pre-trained model weights for all datasets to facilitate reproduction and further research.

Available at: [VQVAE Pre-trained Models](https://data.zzdirty.com/vqvae_ckpt.tar.gz), [AR Pre-trained Models](https://data.zzdirty.com/ar_ckpt.tar.gz).