"""Stable inference output schema for known-class predictions and abstentions."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import Tensor

from plant_ood.openset import RejectionRule


@dataclass(frozen=True)
class OpenSetPrediction:
    predicted_label: str
    known_label: str
    rejected: bool
    known_confidence: float
    novelty_score: float
    gate_weights: dict[str, float]

    def to_dict(self) -> dict[str, str | bool | float | dict[str, float]]:
        return asdict(self)


def format_predictions(
    logits: Tensor,
    novelty_scores: Tensor,
    gate_weights: Tensor,
    *,
    class_names: tuple[str, ...],
    view_names: tuple[str, ...],
    rejection_rule: RejectionRule,
) -> list[OpenSetPrediction]:
    if logits.ndim != 2 or logits.shape[1] != len(class_names):
        raise ValueError("logits and class_names do not match")
    if novelty_scores.shape != (logits.shape[0],):
        raise ValueError("novelty_scores must have shape [batch]")
    if gate_weights.shape != (logits.shape[0], len(view_names)):
        raise ValueError("gate_weights and view_names do not match")
    probabilities = torch.softmax(logits, dim=-1)
    confidence, indices = probabilities.max(dim=-1)
    rejected = rejection_rule.reject(novelty_scores)
    predictions = []
    for row in range(logits.shape[0]):
        known_label = class_names[int(indices[row])]
        is_rejected = bool(rejected[row])
        predictions.append(
            OpenSetPrediction(
                predicted_label="unknown" if is_rejected else known_label,
                known_label=known_label,
                rejected=is_rejected,
                known_confidence=float(confidence[row]),
                novelty_score=float(novelty_scores[row]),
                gate_weights={
                    view: float(gate_weights[row, index]) for index, view in enumerate(view_names)
                },
            )
        )
    return predictions
