"""Visualization utilities for CAC scoring results."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def pick_best_slice(calcium_mask: np.ndarray) -> int:
    """Return the axial slice index with the most calcium pixels."""
    per_slice = calcium_mask.sum(axis=(1, 2))
    if per_slice.max() == 0:
        return calcium_mask.shape[0] // 2
    return int(np.argmax(per_slice))


def create_comparison_figure(
    hu_slice: np.ndarray,
    pred_mask: np.ndarray,
    gt_mask: np.ndarray | None,
    agatston_score: float,
    risk_category: str,
    case_id: str,
    overlay_alpha: float = 0.45,
) -> plt.Figure:
    """
    Side-by-side visualization: original CT | prediction overlay | ground truth.
    """
    num_panels = 3 if gt_mask is not None else 2
    fig, axes = plt.subplots(1, num_panels, figsize=(5 * num_panels, 5))
    if num_panels == 2:
        axes = list(axes)

    axes[0].imshow(hu_slice, cmap="gray", vmin=-200, vmax=400)
    axes[0].set_title(f"Original CT — {case_id}")
    axes[0].axis("off")

    axes[1].imshow(hu_slice, cmap="gray", vmin=-200, vmax=400)
    pred_overlay = np.ma.masked_where(pred_mask == 0, pred_mask)
    axes[1].imshow(pred_overlay, cmap="autumn", alpha=overlay_alpha, vmin=0, vmax=1)
    axes[1].set_title(f"Prediction — Agatston: {agatston_score:.1f}\nRisk: {risk_category}")
    axes[1].axis("off")

    if gt_mask is not None:
        axes[2].imshow(hu_slice, cmap="gray", vmin=-200, vmax=400)
        gt_overlay = np.ma.masked_where(gt_mask == 0, gt_mask)
        axes[2].imshow(gt_overlay, cmap="winter", alpha=overlay_alpha, vmin=0, vmax=1)
        axes[2].set_title("Ground Truth")
        axes[2].axis("off")

    fig.suptitle("Coronary Artery Calcium Scoring", fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


def save_comparison(
    hu_volume: np.ndarray,
    pred_mask: np.ndarray,
    output_path: str | Path,
    agatston_score: float,
    risk_category: str,
    case_id: str,
    gt_mask: np.ndarray | None = None,
    slice_index: int | None = None,
    overlay_alpha: float = 0.45,
) -> Path:
    """Render and save a comparison figure to disk."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if slice_index is None:
        slice_index = pick_best_slice(pred_mask)

    fig = create_comparison_figure(
        hu_slice=hu_volume[slice_index],
        pred_mask=pred_mask[slice_index],
        gt_mask=gt_mask[slice_index] if gt_mask is not None else None,
        agatston_score=agatston_score,
        risk_category=risk_category,
        case_id=case_id,
        overlay_alpha=overlay_alpha,
    )
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def create_score_report_figure(result_dict: dict) -> plt.Figure:
    """Bar chart of per-lesion Agatston contributions."""
    lesions = result_dict.get("lesions", [])
    fig, ax = plt.subplots(figsize=(8, 4))
    if not lesions:
        ax.text(0.5, 0.5, "No calcified lesions detected", ha="center", va="center")
        ax.set_axis_off()
        return fig

    labels = [f"L{lesion['label_id']}" for lesion in lesions]
    scores = [lesion["agatston_score"] for lesion in lesions]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, len(scores)))

    ax.bar(labels, scores, color=colors)
    ax.set_xlabel("Lesion")
    ax.set_ylabel("Agatston Score")
    ax.set_title(
        f"Per-Lesion Scores — Total: {result_dict['total_agatston_score']:.1f} "
        f"({result_dict['risk_category']})"
    )
    fig.tight_layout()
    return fig
