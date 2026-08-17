"""Agatston score calculation from HU volumes and calcium segmentations."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class LesionScore:
    """Score for a single calcified lesion."""

    label_id: int
    area_mm2: float
    max_hu: float
    density_weight: int
    agatston_score: float
    centroid: tuple[float, float, float]


@dataclass
class AgatstonResult:
    """Full Agatston scoring result for one CT volume."""

    total_score: float
    num_lesions: int
    lesions: list[LesionScore] = field(default_factory=list)
    risk_category: str = "none"

    def to_dict(self) -> dict:
        return {
            "total_agatston_score": round(self.total_score, 2),
            "num_lesions": self.num_lesions,
            "risk_category": self.risk_category,
            "lesions": [
                {
                    "label_id": lesion.label_id,
                    "area_mm2": round(lesion.area_mm2, 2),
                    "max_hu": round(lesion.max_hu, 1),
                    "density_weight": lesion.density_weight,
                    "agatston_score": round(lesion.agatston_score, 2),
                }
                for lesion in self.lesions
            ],
        }


def density_weight(max_hu: float) -> int:
    """Map maximum lesion HU to Agatston density weighting factor."""
    if max_hu >= 400:
        return 4
    if max_hu >= 300:
        return 3
    if max_hu >= 200:
        return 2
    if max_hu >= 130:
        return 1
    return 0


def classify_risk(score: float, categories: dict[str, list[int]] | None = None) -> str:
    """Assign cardiovascular risk category from total Agatston score."""
    if categories is None:
        categories = {
            "none": [0, 0],
            "minimal": [1, 10],
            "mild": [11, 100],
            "moderate": [101, 400],
            "severe": [401, 999999],
        }
    for name, (low, high) in categories.items():
        if low <= score <= high:
            return name
    return "severe"


def compute_agatston_score(
    hu_volume: np.ndarray,
    calcium_mask: np.ndarray,
    spacing: tuple[float, float, float],
    hu_threshold: float = 130.0,
    min_lesion_area_mm2: float = 1.0,
    risk_categories: dict[str, list[int]] | None = None,
) -> AgatstonResult:
    """
    Compute the Agatston score from a calcium segmentation mask.

    For each connected lesion:
      score = area_mm² × density_weight(max_HU)

    Only lesions with max HU ≥ threshold and area ≥ 1 mm² are counted.
    """
    spacing_z, spacing_y, spacing_x = spacing
    pixel_area_mm2 = spacing_y * spacing_x

    labeled, num_lesions = ndimage.label(calcium_mask.astype(bool))
    lesions: list[LesionScore] = []
    total = 0.0

    for label_id in range(1, num_lesions + 1):
        component = labeled == label_id
        area_pixels = int(component.sum())
        area_mm2 = area_pixels * pixel_area_mm2

        if area_mm2 < min_lesion_area_mm2:
            continue

        lesion_hu = hu_volume[component]
        max_hu = float(lesion_hu.max())
        if max_hu < hu_threshold:
            continue

        weight = density_weight(max_hu)
        if weight == 0:
            continue

        lesion_score = area_mm2 * weight
        total += lesion_score

        coords = ndimage.center_of_mass(component)
        lesions.append(
            LesionScore(
                label_id=label_id,
                area_mm2=area_mm2,
                max_hu=max_hu,
                density_weight=weight,
                agatston_score=lesion_score,
                centroid=(float(coords[0]), float(coords[1]), float(coords[2])),
            )
        )

    return AgatstonResult(
        total_score=total,
        num_lesions=len(lesions),
        lesions=lesions,
        risk_category=classify_risk(total, risk_categories),
    )


def compute_agatston_from_hu(
    hu_volume: np.ndarray,
    spacing: tuple[float, float, float],
    hu_threshold: float = 130.0,
    min_lesion_area_mm2: float = 1.0,
    risk_categories: dict[str, list[int]] | None = None,
) -> AgatstonResult:
    """Rule-based Agatston score directly from HU thresholding (no ML)."""
    spacing_z, spacing_y, spacing_x = spacing
    pixel_area_mm2 = spacing_y * spacing_x
    min_pixels = max(1, int(np.ceil(min_lesion_area_mm2 / pixel_area_mm2)))

    binary = hu_volume >= hu_threshold
    labeled, num = ndimage.label(binary)
    mask = np.zeros_like(hu_volume, dtype=np.uint8)
    for label_id in range(1, num + 1):
        component = labeled == label_id
        if component.sum() >= min_pixels:
            mask[component] = 1

    return compute_agatston_score(
        hu_volume=hu_volume,
        calcium_mask=mask,
        spacing=spacing,
        hu_threshold=hu_threshold,
        min_lesion_area_mm2=min_lesion_area_mm2,
        risk_categories=risk_categories,
    )
