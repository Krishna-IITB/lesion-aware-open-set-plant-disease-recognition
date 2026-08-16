"""Lightweight lesion localization and robust lesion-weighted pooling."""

from __future__ import annotations

from typing import cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class LesionDecoder(nn.Module):
    """Parameter-efficient pointwise decoder over frozen DINO patch features."""

    def __init__(self, feature_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.normalization = nn.LayerNorm(feature_dim)
        self.classifier = nn.Sequential(
            nn.Conv2d(feature_dim, hidden_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden_dim, 1, kernel_size=1),
        )

    def forward(
        self,
        patch_features: Tensor,
        grid_size: tuple[int, int],
        output_size: tuple[int, int] | None = None,
    ) -> Tensor:
        if patch_features.ndim != 3:
            raise ValueError("patch_features must have shape [batch, patches, dimensions]")
        height, width = grid_size
        if height * width != patch_features.shape[1]:
            raise ValueError("grid_size does not match the number of patch features")
        features = self.normalization(patch_features)
        features = features.transpose(1, 2).reshape(
            features.shape[0], features.shape[2], height, width
        )
        logits = self.classifier(features)
        if output_size is not None:
            logits = F.interpolate(logits, size=output_size, mode="bilinear", align_corners=False)
        return cast(Tensor, logits)


def lesion_aware_pool(
    patch_features: Tensor,
    mask_logits: Tensor,
    grid_size: tuple[int, int],
    minimum_mass: float = 1e-4,
) -> tuple[Tensor, Tensor]:
    """Soft-pool lesion tokens, falling back to the global patch mean for empty masks."""
    if patch_features.ndim != 3 or mask_logits.ndim != 4 or mask_logits.shape[1] != 1:
        raise ValueError("expected patch features [B,P,D] and mask logits [B,1,H,W]")
    height, width = grid_size
    masks = torch.sigmoid(
        F.interpolate(mask_logits, size=(height, width), mode="bilinear", align_corners=False)
    ).flatten(1)
    mass = masks.sum(dim=1, keepdim=True)
    normalized = masks / mass.clamp_min(minimum_mass)
    pooled = torch.einsum("bp,bpd->bd", normalized, patch_features)
    fallback = patch_features.mean(dim=1)
    use_fallback = mass.squeeze(1) < minimum_mass
    pooled = torch.where(use_fallback[:, None], fallback, pooled)
    coverage = masks.mean(dim=1)
    return F.normalize(pooled, dim=-1), coverage
