"""MONAI UNet for 3-D coronary calcium segmentation."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from monai.networks.nets import UNet
from monai.losses import DiceCELoss


def build_model(config: dict[str, Any]) -> nn.Module:
    """Construct a 3-D UNet for binary calcium segmentation."""
    model_cfg = config["model"]
    return UNet(
        spatial_dims=model_cfg["spatial_dims"],
        in_channels=model_cfg["in_channels"],
        out_channels=model_cfg["out_channels"],
        channels=tuple(model_cfg["channels"]),
        strides=tuple(model_cfg["strides"]),
        num_res_units=2,
    )


def build_loss() -> DiceCELoss:
    """Combined Dice + cross-entropy loss with class weighting for sparse calcium."""
    import torch

    # Background heavily outweighs calcium voxels — upweight the positive class
    class_weight = torch.tensor([0.2, 0.8])
    return DiceCELoss(to_onehot_y=True, softmax=True, weight=class_weight)


def save_checkpoint(
    model: nn.Module,
    path: str,
    epoch: int,
    val_loss: float,
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    payload = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "val_loss": val_loss,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(model: nn.Module, path: str, device: torch.device) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint
