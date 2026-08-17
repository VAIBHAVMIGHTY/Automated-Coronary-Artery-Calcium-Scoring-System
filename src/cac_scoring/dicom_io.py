"""DICOM loading and Hounsfield Unit conversion for cardiac CT."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pydicom
from pydicom.dataset import FileDataset


@dataclass
class CTVolume:
    """A 3-D CT volume in Hounsfield Units with physical spacing."""

    hu: np.ndarray  # shape (Z, Y, X)
    spacing: tuple[float, float, float]  # (z, y, x) in mm
    origin: tuple[float, float, float] | None = None
    patient_id: str | None = None
    series_uid: str | None = None

    @property
    def shape(self) -> tuple[int, int, int]:
        return self.hu.shape

    @property
    def pixel_area_mm2(self) -> float:
        """In-plane pixel area (y × x spacing)."""
        return self.spacing[1] * self.spacing[2]


def _apply_rescale(ds: FileDataset, pixels: np.ndarray) -> np.ndarray:
    """Convert stored pixel values to Hounsfield Units."""
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    return pixels.astype(np.float32) * slope + intercept


def load_dicom_series(dicom_dir: str | Path) -> CTVolume:
    """
    Load a DICOM series directory and return a sorted 3-D HU volume.

    Parameters
    ----------
    dicom_dir : path to folder containing .dcm files for one series
    """
    dicom_dir = Path(dicom_dir)
    files = sorted(dicom_dir.glob("*.dcm"))
    if not files:
        files = sorted(dicom_dir.rglob("*.dcm"))
    if not files:
        raise FileNotFoundError(f"No DICOM files found in {dicom_dir}")

    datasets: list[tuple[float, FileDataset, np.ndarray]] = []
    for path in files:
        ds = pydicom.dcmread(str(path))
        if not hasattr(ds, "PixelData"):
            continue
        pixels = ds.pixel_array
        hu_slice = _apply_rescale(ds, pixels)

        position = 0.0
        if hasattr(ds, "ImagePositionPatient"):
            position = float(ds.ImagePositionPatient[2])
        elif hasattr(ds, "InstanceNumber"):
            position = float(ds.InstanceNumber)

        datasets.append((position, ds, hu_slice))

    if not datasets:
        raise ValueError(f"No readable DICOM slices in {dicom_dir}")

    datasets.sort(key=lambda item: item[0])
    slices = [item[2] for item in datasets]
    ref_ds = datasets[0][1]

    spacing_y = float(getattr(ref_ds, "PixelSpacing", [1.0, 1.0])[0])
    spacing_x = float(getattr(ref_ds, "PixelSpacing", [1.0, 1.0])[1])
    spacing_z = float(getattr(ref_ds, "SliceThickness", 3.0))
    if hasattr(ref_ds, "SpacingBetweenSlices"):
        spacing_z = float(ref_ds.SpacingBetweenSlices)

    volume = np.stack(slices, axis=0)
    patient_id = str(getattr(ref_ds, "PatientID", "unknown"))
    series_uid = str(getattr(ref_ds, "SeriesInstanceUID", "unknown"))

    origin = None
    if hasattr(ref_ds, "ImagePositionPatient"):
        origin = tuple(float(v) for v in ref_ds.ImagePositionPatient)

    return CTVolume(
        hu=volume,
        spacing=(spacing_z, spacing_y, spacing_x),
        origin=origin,
        patient_id=patient_id,
        series_uid=series_uid,
    )


def load_nifti_or_numpy(path: str | Path) -> CTVolume:
    """Load a saved .npy volume (used for synthetic / processed data)."""
    path = Path(path)
    if path.suffix == ".npy":
        payload = np.load(str(path), allow_pickle=True).item()
        return CTVolume(
            hu=payload["hu"],
            spacing=tuple(payload["spacing"]),
            patient_id=payload.get("patient_id"),
        )
    raise ValueError(f"Unsupported format: {path.suffix}")

def save_volume(volume: CTVolume, path: str | Path) -> None:
    """Persist a CTVolume as .npy for reuse in training."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(
        str(path),
        {
            "hu": volume.hu,
            "spacing": volume.spacing,
            "patient_id": volume.patient_id,
        },
    )
