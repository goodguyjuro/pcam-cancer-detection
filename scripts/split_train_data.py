import argparse
import csv
import random
from pathlib import Path
import zipfile

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(description="Split PCam train data into train/val/test zips and label CSVs")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Path to the data folder containing train/ and train_labels.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/splits",
        help="Directory where split zip files and CSV labels will be written",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.10,
        help="Fraction of the filtered dataset to reserve for the final test split",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.20,
        help="Fraction of the remaining training split to reserve for validation",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible split",
    )
    return parser.parse_args()


def get_image_ids(image_dir):
    image_dir = Path(image_dir)
    image_paths = sorted(image_dir.glob("*.tif"))
    return [path.stem for path in image_paths]


def write_csv(labels_df, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_csv(output_path, index=False)
    print(f"Saved labels CSV: {output_path} ({len(labels_df)} rows)")


def zip_images(image_ids, image_dir, zip_path):
    image_dir = Path(image_dir)
    zip_path = Path(zip_path)
    zip_path.parent.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for image_id in image_ids:
            file_path = image_dir / f"{image_id}.tif"
            if not file_path.exists():
                raise FileNotFoundError(f"Missing image file: {file_path}")
            zf.write(file_path, arcname=file_path.name)

    print(f"Saved zip file: {zip_path} ({len(image_ids)} images)")


def main():
    args = parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    train_dir = data_dir / "train"
    labels_path = data_dir / "train_labels.csv"

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Label CSV not found: {labels_path}")

    labels_df = pd.read_csv(labels_path)
    image_ids = set(get_image_ids(train_dir))
    label_ids = set(labels_df["id"].astype(str).tolist())

    matched_ids = sorted(image_ids & label_ids)
    missing_labels = sorted(image_ids - label_ids)
    missing_images = sorted(label_ids - image_ids)

    print(f"Total images in train/: {len(image_ids)}")
    print(f"Total label rows in train_labels.csv: {len(label_ids)}")
    print(f"Matched image IDs: {len(matched_ids)}")
    print(f"Images without labels: {len(missing_labels)}")
    print(f"Labels without images: {len(missing_images)}")

    if len(missing_labels) > 0:
        print("Note: The following image IDs exist on disk but not in labels.csv:")
        print(" ", ", ".join(missing_labels[:10]) + ("..." if len(missing_labels) > 10 else ""))
    if len(missing_images) > 0:
        print("Note: The following label IDs have no matching image file:")
        print(" ", ", ".join(missing_images[:10]) + ("..." if len(missing_images) > 10 else ""))

    random.seed(args.seed)
    random.shuffle(matched_ids)

    n_total = len(matched_ids)
    n_test = int(round(n_total * args.test_ratio))
    n_remaining = n_total - n_test
    n_val = int(round(n_remaining * args.val_ratio))
    n_train = n_remaining - n_val

    test_ids = matched_ids[:n_test]
    remaining_ids = matched_ids[n_test:]
    val_ids = remaining_ids[:n_val]
    train_ids = remaining_ids[n_val:]

    print(f"Split counts -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    labels_train = labels_df[labels_df["id"].isin(train_ids)].copy()
    labels_val = labels_df[labels_df["id"].isin(val_ids)].copy()
    labels_test = labels_df[labels_df["id"].isin(test_ids)].copy()

    write_csv(labels_train, output_dir / "labels_train.csv")
    write_csv(labels_val, output_dir / "labels_val.csv")
    write_csv(labels_test, output_dir / "labels_test.csv")

    zip_images(train_ids, train_dir, output_dir / "train_new.zip")
    zip_images(val_ids, train_dir, output_dir / "val_new.zip")
    zip_images(test_ids, train_dir, output_dir / "test_new.zip")

    print("Split complete. Upload the generated zip files and CSVs to Drive for Colab.")


if __name__ == "__main__":
    main()
