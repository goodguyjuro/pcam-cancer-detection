import argparse
import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure the repository root is on PYTHONPATH so imports like `from src.data` work
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.data import create_dataloaders
from src.model import get_model
from src.utils import save_checkpoint, plot_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train a PCam CNN model")
    parser.add_argument("--model_version", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_workers", type=int, default=2,
                        help="Number of DataLoader workers. Lower values are safer in Colab.")
    return parser.parse_args()


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    augment = args.model_version >= 2
    print(f"Model v{args.model_version}: augmentation={'ON' if augment else 'OFF'}")

    train_loader, val_loader = create_dataloaders(
        args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=augment,
    )

    model = get_model(args.model_version).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    output_dir = os.path.join(args.output_dir, f"version_{args.model_version}")
    os.makedirs(output_dir, exist_ok=True)

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        print(
            f"Epoch {epoch}/{args.epochs}:"
            f" train_loss={train_loss:.4f}, train_acc={train_acc:.4f},"
            f" val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_checkpoint(model, optimizer, epoch, os.path.join(output_dir, "best_checkpoint.pth"))

    plot_metrics(history, output_dir)
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Saved training curves and best checkpoint in {output_dir}")


if __name__ == "__main__":
    main()
