"""Load and expose project configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config.yaml"


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load YAML configuration from disk."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key in ("raw_dir", "processed_dir", "synthetic_dir"):
        rel = config["data"][key]
        config["data"][key] = str(PROJECT_ROOT / rel)

    config["training"]["checkpoint_dir"] = str(
        PROJECT_ROOT / config["training"]["checkpoint_dir"]
    )
    config["training"]["log_dir"] = str(PROJECT_ROOT / config["training"]["log_dir"])
    config["inference"]["checkpoint"] = str(
        PROJECT_ROOT / config["inference"]["checkpoint"]
    )
    config["inference"]["output_dir"] = str(
        PROJECT_ROOT / config["inference"]["output_dir"]
    )
    return config
