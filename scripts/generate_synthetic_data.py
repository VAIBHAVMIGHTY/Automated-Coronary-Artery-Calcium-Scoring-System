#!/usr/bin/env python3
"""Generate synthetic cardiac CT data for training and demo."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from cac_scoring.config import load_config
from cac_scoring.synthetic import generate_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic CAC dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config.yaml")
    parser.add_argument("--num-cases", type=int, default=20, help="Number of cases")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = config["data"]["synthetic_dir"]
    manifest = generate_synthetic_dataset(
        output_dir=output_dir,
        num_cases=args.num_cases,
        seed=args.seed,
    )
    print(f"Generated {len(manifest)} synthetic cases in {output_dir}")


if __name__ == "__main__":
    main()
