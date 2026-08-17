#!/usr/bin/env python3
"""Run CAC scoring inference on DICOM or saved volumes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cac_scoring.agatston import compute_agatston_from_hu
from cac_scoring.config import load_config
from cac_scoring.inference import CACScorer


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict CAC score from CT")
    parser.add_argument("--input", type=str, required=True, help="DICOM dir or .npy volume")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--baseline", action="store_true", help="HU-threshold baseline only")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(args.output_dir or config["inference"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    input_path = Path(args.input)

    if args.baseline:
        from cac_scoring.dicom_io import load_dicom_series, load_nifti_or_numpy

        volume = (
            load_dicom_series(input_path)
            if input_path.is_dir()
            else load_nifti_or_numpy(input_path)
        )
        result = compute_agatston_from_hu(
            hu_volume=volume.hu,
            spacing=volume.spacing,
            hu_threshold=config["preprocessing"]["hu_threshold"],
            min_lesion_area_mm2=config["agatston"]["min_lesion_area_mm2"],
            risk_categories=config.get("risk_categories"),
        )
        case_id = volume.patient_id or input_path.stem
        report = result.to_dict()
    else:
        scorer = CACScorer(config)
        if input_path.is_dir():
            volume, mask, result = scorer.score_from_dicom(input_path)
        else:
            volume, mask, result = scorer.score_from_npy(input_path)
        case_id = volume.patient_id or input_path.stem
        report = result.to_dict()
        scorer.score_and_visualize(volume, case_id, output_dir)

    report_path = output_dir / f"{case_id}_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n{'=' * 50}")
    print(f"  Case: {case_id}")
    print(f"  Agatston Score: {report['total_agatston_score']}")
    print(f"  Lesions: {report['num_lesions']}")
    print(f"  Risk Category: {report['risk_category'].upper()}")
    print(f"{'=' * 50}")
    print(f"Report saved to {report_path}")


if __name__ == "__main__":
    main()
