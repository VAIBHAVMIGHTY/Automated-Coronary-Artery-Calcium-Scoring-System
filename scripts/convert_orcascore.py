#!/usr/bin/env python3
"""Convert orCaScore NIfTI/MHA volumes to project .npy format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cac_scoring.dicom_io import save_volume, CTVolume


def load_volume(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    """Load a 3-D volume via SimpleITK."""
    import SimpleITK as sitk

    image = sitk.ReadImage(str(path))
    array = sitk.GetArrayFromImage(image).astype(np.float32)
    spacing = image.GetSpacing()  # (x, y, z)
    # Convert to (z, y, x) spacing
    vol_spacing = (float(spacing[2]), float(spacing[1]), float(spacing[0]))
    return array, vol_spacing


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert orCaScore data to .npy format")
    parser.add_argument("--input-dir", type=str, required=True, help="Folder with CT volumes")
    parser.add_argument("--mask-dir", type=str, required=True, help="Folder with label masks")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    mask_dir = Path(args.mask_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_lines = ["case_id,image,mask,spacing_z,spacing_y,spacing_x"]
    volumes = sorted(input_dir.glob("*.mha")) + sorted(input_dir.glob("*.nii.gz"))

    for vol_path in volumes:
        case_id = vol_path.stem.replace(".nii", "")
        mask_path = mask_dir / vol_path.name
        if not mask_path.exists():
            mask_path = mask_dir / f"{case_id}_R.mha"
        if not mask_path.exists():
            print(f"Skipping {case_id}: no matching mask")
            continue

        hu, spacing = load_volume(vol_path)
        mask, _ = load_volume(mask_path)
        binary_mask = (mask > 0).astype(np.uint8)

        image_out = output_dir / f"{case_id}_hu.npy"
        mask_out = output_dir / f"{case_id}_mask.npy"
        save_volume(CTVolume(hu=hu, spacing=spacing, patient_id=case_id), image_out)
        np.save(str(mask_out), binary_mask)

        manifest_lines.append(
            f"{case_id},{image_out},{mask_out},{spacing[0]},{spacing[1]},{spacing[2]}"
        )
        print(f"Converted {case_id}")

    manifest_path = output_dir / "manifest.csv"
    manifest_path.write_text("\n".join(manifest_lines) + "\n", encoding="utf-8")
    print(f"Manifest written to {manifest_path}")


if __name__ == "__main__":
    main()
