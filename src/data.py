import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import torch


class PCamDataset(Dataset):
    def __init__(
        self,
        image_dir,
        labels_df=None,
        transform=None,
        extensions=(".png", ".jpg", ".jpeg", ".tif", ".tiff"),
    ):
        """
        Args:
            image_dir: Path to folder with image files
            labels_df: DataFrame with 'id' and 'label' columns
            transform: Optional torchvision transforms
        """
        self.image_dir = Path(image_dir)
        self.transform = transform

        self.image_paths = []
        for ext in extensions:
            self.image_paths.extend(self.image_dir.glob(f"*{ext}"))

        self.image_paths = sorted(self.image_paths)

        if labels_df is not None:
            self.label_map = dict(
                zip(labels_df["id"].astype(str), labels_df["label"].astype(int))
            )
        else:
            self.label_map = None

        print(f"Loaded {len(self.image_paths)} images from {self.image_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]

        with Image.open(image_path) as img:
            image = img.convert("RGB")

        if self.label_map is not None:
            img_id = image_path.stem

            if img_id not in self.label_map:
                raise KeyError(f"Label not found for image ID: {img_id}")

            label = self.label_map[img_id]
        else:
            label = 0

        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        return image, label

def get_transforms(augment: bool = False):
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

    return transforms.Compose(base_transforms)


def _read_labels(path):
    return pd.read_csv(path, dtype={"id": str})


def create_dataloaders(data_dir, batch_size=64, num_workers=2, augment=False):
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
    if val_labels_path is not None and not val_labels_path.exists():
        raise FileNotFoundError(f"Validation labels not found: {val_labels_path}")

    train_labels_df = _read_labels(train_labels_path)
    val_labels_df = _read_labels(val_labels_path) if val_labels_path is not None else None

    train_dataset = PCamDataset(
        train_dir,
        labels_df=train_labels_df,
        transform=get_transforms(augment=augment),
    )

    val_dataset = PCamDataset(
        val_dir,
        labels_df=val_labels_df,
        transform=get_transforms(augment=False),
    )

    use_cuda = torch.cuda.is_available()

    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": use_cuda,
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        **loader_kwargs,
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs,
    )

    return train_loader, val_loader