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
- If you already have the unzipped PCam data in `data/`, use `scripts/split_train_data.py` to create a local train/val/test split with labels.

## Local split workflow

This project expects `data/train/` and `data/train_labels.csv` to exist locally.
Run the split script from the repository root:

```bash
python scripts/split_train_data.py --data_dir data --output_dir data/splits
```

That creates:
- `data/splits/train_new.zip`
- `data/splits/val_new.zip`
- `data/splits/test_new.zip`
- `data/splits/labels_train.csv`
- `data/splits/labels_val.csv`
- `data/splits/labels_test.csv`

The zip files contain the image files at the top level (no nested folder). The CSV files preserve the same `id,label` columns as `train_labels.csv`.
