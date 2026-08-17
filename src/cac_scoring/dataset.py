"""PyTorch datasets for CAC segmentation training."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from cac_scoring.preprocessing import normalize_hu


class CACSegmentationDataset(Dataset):
    """Load synthetic or processed volumes with calcium masks."""

    def __init__(
        self,
        manifest_path: str | Path,
        hu_window: tuple[float, float] = (-200, 800),
        augment: bool = False,
    ) -> None:
        self.manifest = pd.read_csv(manifest_path)
        self.hu_window = hu_window
        self.augment = augment

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = self.manifest.iloc[index]
        hu = np.load(row["image"], allow_pickle=True).item()["hu"]
        mask = np.load(row["mask"]).astype(np.float32)

        if self.augment:
            hu, mask = self._augment(hu, mask)

        image = normalize_hu(hu, self.hu_window)
        image = image[np.newaxis]  # (C, Z, Y, X)
        mask = mask[np.newaxis]

        return {
            "image": torch.from_numpy(image.astype(np.float32)),
            "label": torch.from_numpy(mask.astype(np.float32)),
            "case_id": row["case_id"],
        }

    @staticmethod
    def _augment(hu: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Simple random flips for 3-D augmentation."""
        if np.random.rand() > 0.5:
            hu = np.flip(hu, axis=1).copy()
            mask = np.flip(mask, axis=1).copy()
        if np.random.rand() > 0.5:
            hu = np.flip(hu, axis=2).copy()
            mask = np.flip(mask, axis=2).copy()
        return hu, mask
