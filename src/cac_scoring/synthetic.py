"""Synthetic cardiac CT volumes with embedded calcium deposits for demo/training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import ndimage

from cac_scoring.dicom_io import CTVolume, save_volume


def _soft_tissue_background(shape: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Simulate soft-tissue HU background (~40 HU) with mild noise."""
    base = rng.normal(loc=40.0, scale=8.0, size=shape).astype(np.float32)
    return ndimage.gaussian_filter(base, sigma=1.5)


def _add_heart_region(hu: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add a central heart-shaped region at ~50 HU."""
    z, y, x = hu.shape
    cz, cy, cx = z // 2, y // 2, x // 2
    zz, yy, xx = np.ogrid[:z, :y, :x]
    heart_mask = ((zz - cz) / (z * 0.35)) ** 2 + ((yy - cy) / (y * 0.28)) ** 2 + (
        (xx - cx) / (x * 0.28)
    ) ** 2 <= 1.0
    hu[heart_mask] = np.clip(hu[heart_mask] + 15, -200, 800)
    return hu


def _add_calcium_lesion(
    hu: np.ndarray,
    center: tuple[int, int, int],
    radius: tuple[int, int, int],
    peak_hu: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Insert a spherical calcium deposit and return updated HU + binary mask."""
    z, y, x = hu.shape
    cz, cy, cx = center
    rz, ry, rx = radius
    zz, yy, xx = np.ogrid[:z, :y, :x]
    ellipsoid = (
        ((zz - cz) / max(rz, 1)) ** 2
        + ((yy - cy) / max(ry, 1)) ** 2
        + ((xx - cx) / max(rx, 1)) ** 2
    ) <= 1.0

    mask = np.zeros_like(hu, dtype=np.uint8)
    lesion_hu = peak_hu + rng.normal(0, 15, size=hu.shape)
    hu = hu.copy()
    hu[ellipsoid] = np.clip(lesion_hu[ellipsoid], 130, 800)
    mask[ellipsoid] = 1
    return hu, mask


def generate_synthetic_case(
    case_id: str,
    spacing: tuple[float, float, float] = (3.0, 0.8, 0.8),
    shape: tuple[int, int, int] = (32, 128, 128),
    num_lesions: int | None = None,
    seed: int | None = None,
) -> tuple[CTVolume, np.ndarray]:
    """
    Generate one synthetic non-contrast cardiac CT with coronary calcium.

    Returns
    -------
    volume : CTVolume in HU
    mask   : binary calcium segmentation (Z, Y, X)
    """
    rng = np.random.default_rng(seed)
    if num_lesions is None:
        num_lesions = int(rng.integers(1, 6))

    hu = _soft_tissue_background(shape, rng)
    hu = _add_heart_region(hu, rng)
    mask = np.zeros(shape, dtype=np.uint8)

    z, y, x = shape
    for _ in range(num_lesions):
        center = (
            int(rng.integers(z // 3, 2 * z // 3)),
            int(rng.integers(y // 4, 3 * y // 4)),
            int(rng.integers(x // 4, 3 * x // 4)),
        )
        radius = (
            int(rng.integers(1, 3)),
            int(rng.integers(2, 5)),
            int(rng.integers(2, 5)),
        )
        peak_hu = float(rng.choice([180, 250, 350, 450]))
        hu, lesion_mask = _add_calcium_lesion(hu, center, radius, peak_hu, rng)
        mask = np.maximum(mask, lesion_mask)

    volume = CTVolume(
        hu=hu.astype(np.float32),
        spacing=spacing,
        patient_id=case_id,
    )
    return volume, mask


def generate_synthetic_dataset(
    output_dir: str | Path,
    num_cases: int = 20,
    seed: int = 42,
) -> list[dict]:
    """Generate a full synthetic dataset for training and demo."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    manifest: list[dict] = []

    for index in range(num_cases):
        case_id = f"SYN_{index:03d}"
        case_seed = int(rng.integers(0, 1_000_000))
        volume, mask = generate_synthetic_case(case_id=case_id, seed=case_seed)

        image_path = output_dir / f"{case_id}_hu.npy"
        mask_path = output_dir / f"{case_id}_mask.npy"
        save_volume(volume, image_path)
        np.save(str(mask_path), mask)

        manifest.append(
            {
                "case_id": case_id,
                "image": str(image_path),
                "mask": str(mask_path),
                "spacing": list(volume.spacing),
            }
        )

    manifest_path = output_dir / "manifest.csv"
    lines = ["case_id,image,mask,spacing_z,spacing_y,spacing_x"]
    for entry in manifest:
        sp = entry["spacing"]
        lines.append(
            f"{entry['case_id']},{entry['image']},{entry['mask']},{sp[0]},{sp[1]},{sp[2]}"
        )
    manifest_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest
