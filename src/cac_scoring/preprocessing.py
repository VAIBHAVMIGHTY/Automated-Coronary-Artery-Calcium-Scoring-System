"""Volume preprocessing for CAC segmentation and scoring."""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from cac_scoring.dicom_io import CTVolume


def clip_hu(volume: CTVolume, window: tuple[float, float] = (-200, 800)) -> np.ndarray:
    """Clip HU values to a cardiac CT window."""
    low, high = window
    return np.clip(volume.hu, low, high)


def normalize_hu(
    hu: np.ndarray,
    window: tuple[float, float] = (-200, 800),
) -> np.ndarray:
    """Normalize clipped HU to [0, 1] for neural network input."""
    low, high = window
    clipped = np.clip(hu, low, high)
    return (clipped - low) / (high - low)


def resample_volume(
    volume: CTVolume,
    target_spacing: tuple[float, float, float],
) -> CTVolume:
    """Resample a CT volume to isotropic or target spacing (mm)."""
    current = np.array(volume.spacing, dtype=np.float64)
    target = np.array(target_spacing, dtype=np.float64)
    zoom = current / target

    resampled = ndimage.zoom(volume.hu, zoom, order=1)
    return CTVolume(
        hu=resampled.astype(np.float32),
        spacing=tuple(float(s) for s in target),
        origin=volume.origin,
        patient_id=volume.patient_id,
        series_uid=volume.series_uid,
    )


def extract_patch(
    array: np.ndarray,
    center: tuple[int, int, int],
    patch_size: tuple[int, int, int],
) -> np.ndarray:
    """Extract a 3-D patch centered at `center`, zero-padded at boundaries."""
    z, y, x = array.shape
    pz, py, px = patch_size
    cz, cy, cx = center

    z0, z1 = cz - pz // 2, cz + pz // 2
    y0, y1 = cy - py // 2, cy + py // 2
    x0, x1 = cx - px // 2, cx + px // 2

    patch = np.zeros(patch_size, dtype=array.dtype)
    src_z0, src_z1 = max(0, z0), min(z, z1)
    src_y0, src_y1 = max(0, y0), min(y, y1)
    src_x0, src_x1 = max(0, x0), min(x, x1)

    dst_z0 = src_z0 - z0
    dst_y0 = src_y0 - y0
    dst_x0 = src_x0 - x0

    patch[
        dst_z0 : dst_z0 + (src_z1 - src_z0),
        dst_y0 : dst_y0 + (src_y1 - src_y0),
        dst_x0 : dst_x0 + (src_x1 - src_x0),
    ] = array[src_z0:src_z1, src_y0:src_y1, src_x0:src_x1]
    return patch


def threshold_calcium_mask(
    hu: np.ndarray,
    threshold: float = 130.0,
    min_area_mm2: float = 1.0,
    pixel_area_mm2: float = 1.0,
) -> np.ndarray:
    """
    Classical Agatston-style binary calcium mask (HU > threshold, area ≥ 1 mm²).

    Used as pseudo-labels for training and as a rule-based baseline.
    """
    binary = hu >= threshold
    labeled, num = ndimage.label(binary)
    if num == 0:
        return np.zeros_like(hu, dtype=np.uint8)

    min_pixels = max(1, int(np.ceil(min_area_mm2 / pixel_area_mm2)))
    mask = np.zeros_like(hu, dtype=np.uint8)
    for label_id in range(1, num + 1):
        component = labeled == label_id
        if component.sum() >= min_pixels:
            mask[component] = 1
    return mask
