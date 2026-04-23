# PCam CNN Classification

A PyTorch project for PatchCamelyon (PCam) binary histopathology classification.

## Project structure

- `src/` - model definitions, dataset, utilities
- `scripts/` - data download, visualization, training entry points
- `notebooks/` - optional exploration and Colab-ready notebook
- `data/` - dataset storage (gitignored)
- `results/` - training plots and logs

## Setup

1. Create a Python virtual environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the dataset using `scripts/download_data.py`.

## Goals

- Start with a simple CNN baseline
- Add augmentation, batch normalization, dropout
- Tune learning rate and optimizer
- Train locally and then on Colab

## Notes

- `data/` is excluded from git to keep dataset files out of version control.
- Use `scripts/train.py` to run training experiments.
