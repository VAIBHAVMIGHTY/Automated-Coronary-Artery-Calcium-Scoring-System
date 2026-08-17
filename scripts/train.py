#!/usr/bin/env python3
"""Train the MONAI UNet for coronary calcium segmentation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cac_scoring.config import load_config
from cac_scoring.dataset import CACSegmentationDataset
from cac_scoring.model import build_loss, build_model, save_checkpoint


def dice_coefficient(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return float((2 * intersection + eps) / (union + eps))


def train_epoch(model, loader, optimizer, loss_fn, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()

        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        total_dice += dice_coefficient(probs, labels.float())

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def validate(model, loader, loss_fn, device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        total_dice += dice_coefficient(probs, labels.float())

    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CAC segmentation model")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    train_cfg = config["training"]
    epochs = args.epochs or train_cfg["num_epochs"]
    batch_size = args.batch_size or train_cfg["batch_size"]

    manifest = Path(config["data"]["synthetic_dir"]) / "manifest.csv"
    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifest not found at {manifest}. Run scripts/generate_synthetic_data.py first."
        )

    dataset = CACSegmentationDataset(
        manifest_path=manifest,
        hu_window=tuple(config["preprocessing"]["hu_window"]),
        augment=True,
    )

    val_size = max(1, int(len(dataset) * train_cfg["val_split"]))
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=train_cfg["num_workers"],
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=train_cfg["num_workers"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg["learning_rate"])

    checkpoint_dir = Path(train_cfg["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")

    print(f"Training on {device} — {train_size} train / {val_size} val samples")
    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = validate(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train loss={train_loss:.4f} dice={train_dice:.4f} | "
            f"val loss={val_loss:.4f} dice={val_dice:.4f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                model,
                str(checkpoint_dir / "best_model.pt"),
                epoch,
                val_loss,
                optimizer,
            )

    save_checkpoint(
        model,
        str(checkpoint_dir / "last_model.pt"),
        epochs,
        val_loss,
        optimizer,
    )
    print(f"Training complete. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()
