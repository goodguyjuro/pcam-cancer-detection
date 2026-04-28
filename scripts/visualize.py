import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
import numpy as np

"""Visualize PCam image data and statistics."""

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"


def load_labels():
    labels_path = DATA_DIR / "labels_train.csv"
    if not labels_path.exists():
        labels_path = DATA_DIR / "train_labels.csv"
    if not labels_path.exists():
        print(f"Labels file not found: {DATA_DIR / 'labels_train.csv'} or {DATA_DIR / 'train_labels.csv'}")
        return None
    return pd.read_csv(labels_path, dtype={"id": str})


def plot_class_distribution(labels_df):
    if labels_df is None:
        return

    class_counts = labels_df['label'].value_counts()
    plt.figure(figsize=(6, 4))
    class_counts.plot(kind='bar')
    plt.title('Class Distribution in PCam Training Set')
    plt.xlabel('Class (0: No Tumor, 1: Tumor)')
    plt.ylabel('Number of Samples')
    plt.xticks(rotation=0)
    plt.savefig(RESULTS_DIR / "class_distribution.png")
    plt.close()
    print("Saved class distribution plot.")


def plot_sample_images(labels_df):
    if labels_df is None:
        return

    train_dir = DATA_DIR / "train_new" if (DATA_DIR / "train_new").exists() else DATA_DIR / "train"
    if not train_dir.exists():
        print(f"Train images directory not found: {train_dir}")
        return

    train_files = {p.stem for p in train_dir.glob("*.tif")}
    existing_labels = labels_df[labels_df['id'].isin(train_files)]

    def choose_sample(label_value):
        subset = existing_labels[existing_labels['label'] == label_value]
        if subset.empty:
            return None
        return subset.sample(1)['id'].values[0]

    sample_0 = choose_sample(0)
    sample_1 = choose_sample(1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, (sample_id, title) in zip(
        axes,
        [(sample_0, 'No Tumor'), (sample_1, 'Tumor')],
    ):
        if sample_id is None:
            ax.text(0.5, 0.5, f'No {title.lower()} images found in train folder.', ha='center', va='center')
            ax.set_title(title)
            ax.axis('off')
            continue

        img_name = f"{sample_id}.tif"
        img_path = train_dir / img_name
        img = Image.open(img_path)
        ax.imshow(img)
        ax.set_title(f"{title} ({sample_id[:8]}...)")
        ax.axis('off')

    plt.savefig(RESULTS_DIR / "sample_images.png")
    plt.close()
    print("Saved sample images plot.")


def compute_normalization_stats(labels_df):
    if labels_df is None:
        return

    train_dir = DATA_DIR / "train_new" if (DATA_DIR / "train_new").exists() else DATA_DIR / "train"
    if not train_dir.exists():
        print(f"Train images directory not found: {train_dir}")
        return

    # Sample a subset for stats (e.g., 1000 images)
    sample_ids = labels_df.sample(1000)['id'].values
    means = []
    stds = []

    for img_id in sample_ids:
        img_path = train_dir / f"{img_id}.tif"
        if img_path.exists():
            img = np.array(Image.open(img_path)) / 255.0  # Normalize to [0,1]
            means.append(img.mean(axis=(0,1)))
            stds.append(img.std(axis=(0,1)))

    if means:
        overall_mean = np.mean(means, axis=0)
        overall_std = np.mean(stds, axis=0)
        print(f"Computed normalization stats from {len(means)} images:")
        print(f"Mean: {overall_mean}")
        print(f"Std: {overall_std}")
        # Save to file
        with open(RESULTS_DIR / "normalization_stats.txt", "w") as f:
            f.write(f"Mean: {overall_mean}\nStd: {overall_std}\n")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    labels_df = load_labels()
    if labels_df is not None:
        print(f"Loaded labels for {len(labels_df)} images.")
        plot_class_distribution(labels_df)
        plot_sample_images(labels_df)
        compute_normalization_stats(labels_df)
    else:
        print("Could not load labels. Ensure data is downloaded and unzipped.")


if __name__ == "__main__":
    main()
