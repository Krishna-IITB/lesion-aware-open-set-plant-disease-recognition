"""Validation-only calibration and open-set novelty scoring."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def energy_score(logits: Tensor, temperature: float | Tensor = 1.0) -> Tensor:
    """Return novelty-oriented energy: larger values indicate less-known samples."""
    temp = torch.as_tensor(temperature, device=logits.device, dtype=logits.dtype)
    if torch.any(temp <= 0):
        raise ValueError("temperature must be positive")
    return -temp * torch.logsumexp(logits / temp, dim=-1)


def prototype_distance(logits: Tensor, logit_scale: float = 10.0) -> Tensor:
    if logit_scale <= 0:
        raise ValueError("logit_scale must be positive")
    return 1.0 - logits.max(dim=-1).values / logit_scale


def normalized_hybrid_score(energy: Tensor, distance: Tensor, weight: float) -> Tensor:
    if not 0.0 <= weight <= 1.0:
        raise ValueError("weight must lie in [0, 1]")

    def standardize(values: Tensor) -> Tensor:
        return (values - values.mean()) / values.std(unbiased=False).clamp_min(1e-8)

    return weight * standardize(energy) + (1.0 - weight) * standardize(distance)


@dataclass(frozen=True)
class RejectionRule:
    threshold: float
    score_method: str
    fitted_split: str = "validation"

    def __post_init__(self) -> None:
        if self.fitted_split != "validation":
            raise ValueError("open-set thresholds may only be fitted on the validation split")

    def reject(self, novelty_scores: Tensor) -> Tensor:
        return novelty_scores >= self.threshold


def fit_rejection_threshold(
    known_scores: Tensor,
    unknown_scores: Tensor,
    *,
    split: str = "validation",
    method: str = "energy",
) -> RejectionRule:
    """Select the validation threshold maximizing balanced known/unknown accuracy."""
    if split != "validation":
        raise ValueError("refusing to fit an OOD threshold on a non-validation split")
    if known_scores.numel() == 0 or unknown_scores.numel() == 0:
        raise ValueError("known and unknown validation scores must both be non-empty")
    values = torch.cat((known_scores.flatten(), unknown_scores.flatten())).sort().values
    candidates = torch.cat(
        (
            values[:1] - 1e-6,
            (values[:-1] + values[1:]) / 2,
            values[-1:] + 1e-6,
        )
    )
    known_acceptance = (known_scores[:, None] < candidates[None, :]).float().mean(dim=0)
    unknown_rejection = (unknown_scores[:, None] >= candidates[None, :]).float().mean(dim=0)
    balanced = (known_acceptance + unknown_rejection) / 2
    best = int(balanced.argmax())
    return RejectionRule(float(candidates[best].detach()), method, fitted_split=split)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_temperature = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> Tensor:
        return self.log_temperature.exp().clamp(0.05, 20.0)

    def forward(self, logits: Tensor) -> Tensor:
        return logits / self.temperature

    def fit(
        self, validation_logits: Tensor, validation_labels: Tensor, iterations: int = 50
    ) -> float:
        if validation_logits.shape[0] != validation_labels.shape[0]:
            raise ValueError("validation logits and labels must contain the same sample count")
        optimizer = torch.optim.LBFGS([self.log_temperature], lr=0.1, max_iter=iterations)

        def closure() -> Tensor:
            optimizer.zero_grad()
            loss = F.cross_entropy(self(validation_logits), validation_labels)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        return float(self.temperature.detach())
