import os
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
            labels_df: DataFrame with 'id' and 'label' columns (only for train set)
            transform: Optional torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.tif"))
        self.labels_df = labels_df

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        
        # Get label from DataFrame if available
        if self.labels_df is not None:
            img_id = image_path.stem  # filename without extension
            label = self.labels_df.loc[self.labels_df['id'] == img_id, 'label'].values[0]
        else:
            label = 0  # For test set without labels
        
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


def create_dataloaders(data_dir, batch_size=64, num_workers=4, augment=False):
    """
    Create train and validation dataloaders.
    
    Args:
        data_dir: Path to data folder (contains train/, test/, train_labels.csv)
        batch_size: Batch size for training
        num_workers: Number of workers for data loading
        augment: Whether to apply augmentation (False for v1, True for v2+)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / "train"
    val_dir = data_dir / "test"
    labels_path = data_dir / "train_labels.csv"

    # Load labels for training set
    labels_df = pd.read_csv(labels_path) if labels_path.exists() else None

    # Create datasets with appropriate transforms
    train_transform = get_transforms(augment=augment)
    val_transform = get_transforms(augment=False)  # Never augment validation

    train_dataset = PCamDataset(train_dir, labels_df=labels_df, transform=train_transform)
    val_dataset = PCamDataset(val_dir, labels_df=None, transform=val_transform)

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers
    )
    
    return train_loader, val_loader
