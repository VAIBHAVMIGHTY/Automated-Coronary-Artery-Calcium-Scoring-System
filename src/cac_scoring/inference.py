"""Inference pipeline: segment calcium and compute Agatston score."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from cac_scoring.agatston import AgatstonResult, compute_agatston_score
from cac_scoring.dicom_io import CTVolume, load_dicom_series, load_nifti_or_numpy
from cac_scoring.model import build_model, load_checkpoint
from cac_scoring.preprocessing import normalize_hu, threshold_calcium_mask
from cac_scoring.visualize import save_comparison


class CACScorer:
    """End-to-end CAC scoring: load CT → segment → Agatston score → visualize."""

    def __init__(self, config: dict[str, Any], device: str | None = None) -> None:
        self.config = config
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = build_model(config).to(self.device)
        self.model.eval()

        checkpoint_path = config["inference"]["checkpoint"]
        if Path(checkpoint_path).exists():
            load_checkpoint(self.model, checkpoint_path, self.device)

    @torch.no_grad()
    def predict_mask(self, hu_volume: np.ndarray) -> np.ndarray:
        """Run the UNet and return a binary calcium mask with HU-gated post-processing."""
        window = tuple(self.config["preprocessing"]["hu_window"])
        hu_threshold = self.config["preprocessing"]["hu_threshold"]
        min_area = self.config["agatston"]["min_lesion_area_mm2"]

        normalized = normalize_hu(hu_volume, window)
        tensor = torch.from_numpy(normalized[np.newaxis, np.newaxis].astype(np.float32))
        tensor = tensor.to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        pred = (probs[:, 1] > 0.5).cpu().numpy()[0].astype(np.uint8)

        # Gate predictions by HU threshold to suppress false positives
        pred = pred & (hu_volume >= hu_threshold).astype(np.uint8)

        # If the model predicts almost nothing, fall back to classical thresholding
        if pred.sum() < 3:
            spacing = (3.0, 0.8, 0.8)  # default; overridden when scoring with volume.spacing
            pred = threshold_calcium_mask(
                hu_volume,
                threshold=hu_threshold,
                min_area_mm2=min_area,
                pixel_area_mm2=spacing[1] * spacing[2],
            )

        return pred

    def score_volume(
        self,
        volume: CTVolume,
        mask: np.ndarray | None = None,
    ) -> tuple[np.ndarray, AgatstonResult]:
        """Segment (or use provided mask) and compute Agatston score."""
        if mask is None:
            mask = self.predict_mask(volume.hu)

        # Final cleanup: enforce minimum lesion size on the segmentation
        from scipy import ndimage

        min_area = self.config["agatston"]["min_lesion_area_mm2"]
        pixel_area = volume.spacing[1] * volume.spacing[2]
        min_pixels = max(1, int(np.ceil(min_area / pixel_area)))
        labeled, num = ndimage.label(mask.astype(bool))
        cleaned = np.zeros_like(mask, dtype=np.uint8)
        for label_id in range(1, num + 1):
            component = labeled == label_id
            if component.sum() >= min_pixels:
                cleaned[component] = 1
        mask = cleaned

        result = compute_agatston_score(
            hu_volume=volume.hu,
            calcium_mask=mask,
            spacing=volume.spacing,
            hu_threshold=self.config["preprocessing"]["hu_threshold"],
            min_lesion_area_mm2=self.config["agatston"]["min_lesion_area_mm2"],
            risk_categories=self.config.get("risk_categories"),
        )
        return mask, result

    def score_from_dicom(self, dicom_dir: str | Path) -> tuple[CTVolume, np.ndarray, AgatstonResult]:
        volume = load_dicom_series(dicom_dir)
        mask, result = self.score_volume(volume)
        return volume, mask, result

    def score_from_npy(self, npy_path: str | Path) -> tuple[CTVolume, np.ndarray, AgatstonResult]:
        volume = load_nifti_or_numpy(npy_path)
        mask, result = self.score_volume(volume)
        return volume, mask, result

    def score_and_visualize(
        self,
        volume: CTVolume,
        case_id: str,
        output_dir: str | Path,
        gt_mask: np.ndarray | None = None,
    ) -> AgatstonResult:
        mask, result = self.score_volume(volume, mask=None)
        save_comparison(
            hu_volume=volume.hu,
            pred_mask=mask,
            gt_mask=gt_mask,
            output_path=Path(output_dir) / f"{case_id}_comparison.png",
            agatston_score=result.total_score,
            risk_category=result.risk_category,
            case_id=case_id,
            overlay_alpha=self.config["inference"]["overlay_alpha"],
        )
        return result
