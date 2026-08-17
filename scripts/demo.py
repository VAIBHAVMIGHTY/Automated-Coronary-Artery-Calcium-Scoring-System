#!/usr/bin/env python3
"""
End-to-end demo: generate synthetic data, train a model, score, and visualize.

This script produces portfolio-ready outputs without requiring external datasets.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from cac_scoring.agatston import compute_agatston_from_hu, compute_agatston_score
from cac_scoring.config import load_config
from cac_scoring.dataset import CACSegmentationDataset
from cac_scoring.inference import CACScorer
from cac_scoring.model import build_loss, build_model, save_checkpoint
from cac_scoring.synthetic import generate_synthetic_dataset
from cac_scoring.visualize import create_score_report_figure, save_comparison


def _dice_coefficient(pred, target, eps=1e-6):
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum()
    union = pred_bin.sum() + target.sum()
    return float((2 * intersection + eps) / (union + eps))


def _train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = total_dice = 0.0
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
        total_dice += _dice_coefficient(probs, labels.float())
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


@torch.no_grad()
def _validate(model, loader, loss_fn, device):
    model.eval()
    total_loss = total_dice = 0.0
    for batch in loader:
        images = batch["image"].to(device)
        labels = batch["label"].to(device).long()
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        probs = torch.softmax(outputs, dim=1)[:, 1]
        total_loss += loss.item()
        total_dice += _dice_coefficient(probs, labels.float())
    n = max(len(loader), 1)
    return total_loss / n, total_dice / n


def quick_train(config, epochs: int = 10) -> Path:
    """Fast training loop for demo purposes."""
    manifest = Path(config["data"]["synthetic_dir"]) / "manifest.csv"
    dataset = CACSegmentationDataset(
        manifest_path=manifest,
        hu_window=tuple(config["preprocessing"]["hu_window"]),
        augment=True,
    )
    val_size = max(1, len(dataset) // 5)
    train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=1, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(config).to(device)
    loss_fn = build_loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    checkpoint_dir = Path(config["training"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, train_dice = _train_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss, val_dice = _validate(model, val_loader, loss_fn, device)
        print(f"  Epoch {epoch}/{epochs} — val dice={val_dice:.3f}")
        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, str(checkpoint_dir / "best_model.pt"), epoch, val_loss)

    return checkpoint_dir / "best_model.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description="CAC scoring end-to-end demo")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--num-cases", type=int, default=12)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/4] Generating synthetic cardiac CT volumes...")
    generate_synthetic_dataset(
        output_dir=config["data"]["synthetic_dir"],
        num_cases=args.num_cases,
        seed=42,
    )

    if not args.skip_train:
        print(f"\n[2/4] Training MONAI UNet ({args.epochs} epochs)...")
        quick_train(config, epochs=args.epochs)
    else:
        print("\n[2/4] Skipping training (--skip-train)")

    print("\n[3/4] Running inference and Agatston scoring...")
    scorer = CACScorer(config)
    manifest_path = Path(config["data"]["synthetic_dir"]) / "manifest.csv"
    import pandas as pd

    manifest = pd.read_csv(manifest_path)
    summary_rows = []

    for _, row in manifest.iterrows():
        case_id = row["case_id"]
        hu = np.load(row["image"], allow_pickle=True).item()
        gt_mask = np.load(row["mask"])

        from cac_scoring.dicom_io import CTVolume

        volume = CTVolume(
            hu=hu["hu"],
            spacing=tuple(hu["spacing"]),
            patient_id=case_id,
        )

        ml_mask, ml_result = scorer.score_volume(volume)
        baseline_result = compute_agatston_from_hu(
            hu_volume=volume.hu,
            spacing=volume.spacing,
            hu_threshold=config["preprocessing"]["hu_threshold"],
            min_lesion_area_mm2=config["agatston"]["min_lesion_area_mm2"],
        )
        gt_result = compute_agatston_score(
            hu_volume=volume.hu,
            calcium_mask=gt_mask,
            spacing=volume.spacing,
            hu_threshold=config["preprocessing"]["hu_threshold"],
            min_lesion_area_mm2=config["agatston"]["min_lesion_area_mm2"],
        )

        save_comparison(
            hu_volume=volume.hu,
            pred_mask=ml_mask,
            gt_mask=gt_mask,
            output_path=output_dir / f"{case_id}_comparison.png",
            agatston_score=ml_result.total_score,
            risk_category=ml_result.risk_category,
            case_id=case_id,
        )

        summary_rows.append(
            {
                "case_id": case_id,
                "gt_score": round(gt_result.total_score, 2),
                "ml_score": round(ml_result.total_score, 2),
                "baseline_score": round(baseline_result.total_score, 2),
                "risk": ml_result.risk_category,
            }
        )

    print("\n[4/4] Saving summary report...")
    summary_path = output_dir / "demo_summary.json"
    summary_path.write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")

    if summary_rows:
        fig = create_score_report_figure(
            {
                "lesions": [
                    {"label_id": i + 1, "agatston_score": row["ml_score"]}
                    for i, row in enumerate(summary_rows[:8])
                ],
                "total_agatston_score": sum(r["ml_score"] for r in summary_rows),
                "risk_category": "demo",
            }
        )
        fig.savefig(str(output_dir / "score_summary.png"), dpi=150, bbox_inches="tight")

    print(f"\nDemo complete! Results in: {output_dir}")
    print("\nSample scores:")
    for row in summary_rows[:5]:
        print(
            f"  {row['case_id']}: ML={row['ml_score']}, "
            f"Baseline={row['baseline_score']}, GT={row['gt_score']}, Risk={row['risk']}"
        )


if __name__ == "__main__":
    main()
