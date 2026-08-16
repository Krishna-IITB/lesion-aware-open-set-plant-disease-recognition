"""Deterministic multi-prototype construction and cosine scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class PrototypeBank:
    values: Tensor  # [classes, prototypes, feature_dim]
    class_names: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.values.ndim != 3:
            raise ValueError("prototype values must have shape [classes, prototypes, dimensions]")
        if self.values.shape[0] != len(self.class_names):
            raise ValueError("class_names length must equal the first prototype dimension")

    def score(self, queries: Tensor, reduction: str = "max") -> Tensor:
        queries = F.normalize(queries, dim=-1)
        prototypes = F.normalize(self.values.to(queries), dim=-1)
        similarities = torch.einsum("bd,cpd->bcp", queries, prototypes)
        if reduction == "max":
            return similarities.max(dim=-1).values
        if reduction == "mean":
            return similarities.mean(dim=-1)
        raise ValueError("reduction must be 'max' or 'mean'")


def build_prototypes(
    features: Tensor,
    labels: Tensor,
    class_names: tuple[str, ...],
    prototypes_per_class: int,
    seed: int,
    iterations: int = 25,
) -> PrototypeBank:
    """Build per-class spherical k-means prototypes from training features only."""
    if features.ndim != 2 or labels.ndim != 1 or features.shape[0] != labels.shape[0]:
        raise ValueError("expected features [N,D] and labels [N]")
    generator = torch.Generator(device=features.device).manual_seed(seed)
    banks: list[Tensor] = []
    for class_index, class_name in enumerate(class_names):
        points = F.normalize(features[labels == class_index], dim=-1)
        if not len(points):
            raise ValueError(f"no prototype features for class {class_name!r}")
        k = min(prototypes_per_class, len(points))
        indices = torch.randperm(len(points), generator=generator, device=points.device)[:k]
        centers = points[indices].clone()
        for _ in range(iterations):
            assignments = (points @ centers.T).argmax(dim=1)
            updated = []
            for cluster in range(k):
                members = points[assignments == cluster]
                updated.append(centers[cluster] if not len(members) else members.mean(dim=0))
            next_centers = F.normalize(torch.stack(updated), dim=-1)
            if torch.allclose(next_centers, centers, atol=1e-6):
                centers = next_centers
                break
            centers = next_centers
        if k < prototypes_per_class:
            centers = torch.cat([centers, centers[:1].expand(prototypes_per_class - k, -1)])
        banks.append(centers)
    return PrototypeBank(torch.stack(banks), class_names)
