"""Unit tests for Agatston scoring."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cac_scoring.agatston import compute_agatston_score, density_weight


def test_density_weight():
    assert density_weight(150) == 1
    assert density_weight(250) == 2
    assert density_weight(350) == 3
    assert density_weight(450) == 4
    assert density_weight(100) == 0


def test_single_lesion_score():
    """One 2×2 mm lesion at 250 HU → area=4 mm², weight=2, score=8."""
    hu = np.full((5, 10, 10), 40.0, dtype=np.float32)
    mask = np.zeros((5, 10, 10), dtype=np.uint8)
    hu[2, 4:6, 4:6] = 250
    mask[2, 4:6, 4:6] = 1

    result = compute_agatston_score(
        hu_volume=hu,
        calcium_mask=mask,
        spacing=(3.0, 1.0, 1.0),
        hu_threshold=130,
        min_lesion_area_mm2=1.0,
    )
    assert result.num_lesions == 1
    assert result.total_score == pytest.approx(8.0, rel=0.01)
    assert result.lesions[0].density_weight == 2


def test_small_lesion_filtered():
    """Lesions smaller than 1 mm² should be excluded."""
    hu = np.full((3, 5, 5), 40.0, dtype=np.float32)
    mask = np.zeros((3, 5, 5), dtype=np.uint8)
    hu[1, 2, 2] = 300
    mask[1, 2, 2] = 1

    result = compute_agatston_score(
        hu_volume=hu,
        calcium_mask=mask,
        spacing=(3.0, 0.5, 0.5),  # 0.25 mm² per pixel — below 1 mm² threshold
        min_lesion_area_mm2=1.0,
    )
    assert result.num_lesions == 0
    assert result.total_score == 0.0
