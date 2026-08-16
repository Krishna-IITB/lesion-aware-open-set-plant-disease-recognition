"""Composable recognition system joining prototype evidence and the gate."""

from __future__ import annotations

from dataclasses import dataclass

from torch import Tensor, nn

from plant_ood.models.fusion import ThreeViewGate
from plant_ood.models.prototypes import PrototypeBank


@dataclass(frozen=True)
class RecognitionOutput:
    logits: Tensor
    gate_weights: Tensor
    view_scores: dict[str, Tensor]


class LesionAwareRecognizer(nn.Module):
    def __init__(
        self, banks: dict[str, PrototypeBank], gate: ThreeViewGate, logit_scale: float = 10.0
    ) -> None:
        super().__init__()
        if tuple(banks) != gate.views:
            raise ValueError("prototype-bank order must match gate views")
        names = [bank.class_names for bank in banks.values()]
        if any(item != names[0] for item in names[1:]):
            raise ValueError("prototype banks must use identical class ordering")
        self.banks = banks
        self.gate = gate
        self.logit_scale = logit_scale

    def forward(self, features: dict[str, Tensor]) -> RecognitionOutput:
        if tuple(features) != tuple(self.banks):
            raise ValueError("feature-view order must match prototype banks")
        view_scores = {
            name: self.banks[name].score(features[name]) * self.logit_scale for name in self.banks
        }
        logits, weights = self.gate(view_scores)
        return RecognitionOutput(logits=logits, gate_weights=weights, view_scores=view_scores)
