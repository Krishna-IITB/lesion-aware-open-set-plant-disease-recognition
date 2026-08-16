"""Segmentation losses used by the lesion decoder."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.nn import functional as F


def dice_loss(logits: Tensor, targets: Tensor, epsilon: float = 1e-6) -> Tensor:
    if logits.shape != targets.shape:
        raise ValueError("logits and targets must have identical shapes")
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).flatten(1).sum(dim=1)
    denominator = probabilities.flatten(1).sum(dim=1) + targets.flatten(1).sum(dim=1)
    return (1.0 - (2.0 * intersection + epsilon) / (denominator + epsilon)).mean()


def bce_dice_loss(logits: Tensor, targets: Tensor, dice_weight: float = 0.5) -> Tensor:
    if not 0.0 <= dice_weight <= 1.0:
        raise ValueError("dice_weight must lie in [0, 1]")
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    return (1.0 - dice_weight) * bce + dice_weight * dice_loss(logits, targets)
