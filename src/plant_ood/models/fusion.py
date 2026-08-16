"""Interpretable per-sample evidence gating."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class ThreeViewGate(nn.Module):
    """Fuse class scores with normalized, inspectable per-view weights."""

    def __init__(self, views: tuple[str, ...], hidden_dim: int = 32, learned: bool = True) -> None:
        super().__init__()
        if not views or len(set(views)) != len(views):
            raise ValueError("views must be a non-empty unique tuple")
        self.views = views
        self.learned = learned
        self.network: nn.Module | None = None
        if learned:
            # Two quality signals per view: peak score and score margin.
            self.network = nn.Sequential(
                nn.Linear(2 * len(views), hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, len(views)),
            )

    def forward(self, view_scores: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        if tuple(view_scores) != self.views:
            raise ValueError(f"expected ordered views {self.views}, got {tuple(view_scores)}")
        scores = torch.stack([view_scores[name] for name in self.views], dim=1)
        if any(score.shape != scores[:, 0].shape for score in view_scores.values()):
            raise ValueError("all view score tensors must have the same shape")
        if self.network is None:
            weights = scores.new_full((scores.shape[0], len(self.views)), 1.0 / len(self.views))
        else:
            top = scores.topk(k=min(2, scores.shape[-1]), dim=-1).values
            peak = top[..., 0]
            margin = top[..., 0] - (top[..., 1] if top.shape[-1] == 2 else 0.0)
            quality = torch.stack((peak, margin), dim=-1).flatten(1)
            weights = torch.softmax(self.network(quality), dim=-1)
        fused = (scores * weights.unsqueeze(-1)).sum(dim=1)
        return fused, weights
