import os
import warnings
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class PCamDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None):
        """
        Args:
            image_dir: Path to folder with .tif images
            labels_df: DataFrame with 'id' and 'label' columns
            transform: Optional torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.labels_df = labels_df
        self.image_paths = sorted(self.image_dir.glob("*.tif"))
        self.image_paths = [p for p in self.image_paths if self._is_valid_image(p)]

    def _is_valid_image(self, image_path):
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as e:
            warnings.warn(f"Skipping invalid image file: {image_path} ({e})")
            return False

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        if self.labels_df is not None:
            img_id = image_path.stem
            row = self.labels_df.loc[self.labels_df["id"] == img_id, "label"]
            if row.empty:
                raise KeyError(f"Label not found for image ID: {img_id}")
            label = int(row.values[0])
        else:
            label = 0

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label


def get_transforms(augment: bool = False):
    """
    augment=False: v1 baseline (no augmentation, just normalize)
    augment=True: v2+ (with augmentation)
    """
    base_transforms = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]

    if augment:
        augmentation = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(15),
        ]
        return transforms.Compose(augmentation + base_transforms)
    else:
        return transforms.Compose(base_transforms)


def _read_labels(path):
    return pd.read_csv(path, dtype={"id": str})


def create_dataloaders(data_dir, batch_size=64, num_workers=2, augment=False):
    """
    Create train and validation dataloaders.
    """
    data_dir = Path(data_dir)

    if (data_dir / "train_new").exists() and (data_dir / "labels_train.csv").exists():
        train_dir = data_dir / "train_new"
        val_dir = data_dir / "val_new"
        train_labels_path = data_dir / "labels_train.csv"
        val_labels_path = data_dir / "labels_val.csv"
    else:
        train_dir = data_dir / "train"
        val_dir = data_dir / "test"
        train_labels_path = data_dir / "train_labels.csv"
        val_labels_path = None

    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")
    if not train_labels_path.exists():
        raise FileNotFoundError(f"Train labels not found: {train_labels_path}")
    if val_dir.exists() and val_labels_path is None:
        raise FileNotFoundError(
            f"Validation labels not found. For proper validation, use split data with labels in {data_dir}/labels_val.csv"
        )
    if val_labels_path is not None and not val_labels_path.exists():
        raise FileNotFoundError(f"Validation labels not found: {val_labels_path}")

    train_labels_df = _read_labels(train_labels_path)
    val_labels_df = _read_labels(val_labels_path) if val_labels_path is not None else None

    train_transform = get_transforms(augment=augment)
    val_transform = get_transforms(augment=False)

    train_dataset = PCamDataset(train_dir, labels_df=train_labels_df, transform=train_transform)
    val_dataset = PCamDataset(val_dir, labels_df=val_labels_df, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    return train_loader, val_loader
