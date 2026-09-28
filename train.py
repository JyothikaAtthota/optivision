"""
OPTIVISION — Training Script
Forward pass → CrossEntropyLoss → Adam → Backprop → Save best model
"""

import os
import argparse
import time
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from utils.dataset import get_dataloaders, NUM_CLASSES, DR_CLASSES
from models.resnet_model import build_model


# ── Config defaults ───────────────────────────────────────────────────────────
DEFAULT_DATA_DIR   = "data"
DEFAULT_EPOCHS     = 25
DEFAULT_BATCH_SIZE = 32
DEFAULT_LR         = 1e-4
DEFAULT_SAVE_PATH  = "outputs/best_model.pth"


# ── Training / Validation loops ───────────────────────────────────────────────
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)                 # forward pass
        loss    = criterion(outputs, labels)    # cross-entropy loss
        loss.backward()                         # backprop
        optimizer.step()                        # weight update

        running_loss += loss.item() * images.size(0)
        preds   = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss    = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds   = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

    epoch_loss = running_loss / total
    epoch_acc  = correct / total
    return epoch_loss, epoch_acc


# ── Main training loop ────────────────────────────────────────────────────────
def train(data_dir: str, epochs: int, batch_size: int,
          lr: float, save_path: str):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[OPTIVISION] Training on: {device}")
    print(f"[OPTIVISION] DR Classes : {DR_CLASSES}")

    # Data
    train_loader, val_loader = get_dataloaders(data_dir, batch_size)
    print(f"[OPTIVISION] Train samples: {len(train_loader.dataset)} | "
          f"Val samples: {len(val_loader.dataset)}")

    # Model
    model     = build_model(num_classes=NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", patience=3,
                                  factor=0.5, verbose=True)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    best_val_loss = float("inf")
    history = {"train_loss": [], "val_loss": [],
                "train_acc":  [], "val_acc":  []}

    for epoch in range(1, epochs + 1):
        t0 = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device)
        val_loss,   val_acc   = validate(
            model, val_loader,   criterion, device)

        scheduler.step(val_loss)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:>3}/{epochs}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            print(f"  >> Best model saved -> {save_path}")

    print(f"\n[OPTIVISION] Training complete. Best Val Loss: {best_val_loss:.4f}")
    return history


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OPTIVISION — Train DR Classifier")
    parser.add_argument("--data_dir",   default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs",     type=int,   default=DEFAULT_EPOCHS)
    parser.add_argument("--batch_size", type=int,   default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr",         type=float, default=DEFAULT_LR)
    parser.add_argument("--save_path",  default=DEFAULT_SAVE_PATH)
    args = parser.parse_args()

    train(
        data_dir   = args.data_dir,
        epochs     = args.epochs,
        batch_size = args.batch_size,
        lr         = args.lr,
        save_path  = args.save_path,
    )
