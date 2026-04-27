import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd


class PCamDataset(Dataset):
    def __init__(self, image_dir, labels_df=None, transform=None):
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.tif"))
        self.labels_df = labels_df

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        if self.labels_df is not None:
            img_id = image_path.stem  # id without .tif
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
        else:
            label = 0  # For test set
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transforms(train: bool = True):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


def create_dataloaders(data_dir, batch_size=64, num_workers=4):
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    test_dir = data_dir / "test"
    labels_path = data_dir / "train_labels.csv"

    labels_df = pd.read_csv(labels_path) if labels_path.exists() else None

    train_dataset = PCamDataset(train_dir, labels_df=labels_df, transform=get_transforms(train=True))
    val_dataset = PCamDataset(test_dir, labels_df=None, transform=get_transforms(train=False))  # Test has no labels

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader
