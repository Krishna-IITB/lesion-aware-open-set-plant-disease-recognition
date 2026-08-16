"""Synthetic integration check; its metrics are never project results."""

from __future__ import annotations

from typing import Any

import torch

from plant_ood.evaluation.metrics import accuracy, macro_f1, segmentation_metrics
from plant_ood.losses import bce_dice_loss
from plant_ood.models.fusion import ThreeViewGate
from plant_ood.models.lesion import LesionDecoder, lesion_aware_pool
from plant_ood.models.prototypes import build_prototypes
from plant_ood.models.system import LesionAwareRecognizer
from plant_ood.openset import energy_score, fit_rejection_threshold
from plant_ood.utils import parameter_counts, seed_everything


def run_synthetic_smoke_test() -> dict[str, Any]:
    """Exercise every lightweight stage with deterministic random tensors."""
    seed_everything(7)
    batch, classes, dimension, grid = 12, 3, 16, (4, 4)
    labels = torch.arange(batch) % classes
    patch_features = torch.randn(batch, grid[0] * grid[1], dimension)
    masks = torch.zeros(batch, 1, 8, 8)
    masks[:, :, 2:6, 2:6] = 1.0
    decoder = LesionDecoder(dimension, hidden_dim=8)
    mask_logits = decoder(patch_features, grid, output_size=(8, 8))
    segmentation_loss = bce_dice_loss(mask_logits, masks)
    segmentation_loss.backward()  # type: ignore[no-untyped-call]
    lesion_features, coverage = lesion_aware_pool(patch_features, mask_logits, grid)
    global_features = torch.nn.functional.normalize(torch.randn(batch, dimension), dim=-1)
    text_queries = global_features

    views = {
        "text": text_queries,
        "global": global_features,
        "lesion": lesion_features.detach(),
    }
    banks = {
        name: build_prototypes(features[:9], labels[:9], ("a", "b", "c"), 2, seed=7)
        for name, features in views.items()
    }
    recognizer = LesionAwareRecognizer(banks, ThreeViewGate(tuple(views), hidden_dim=8))
    output = recognizer(views)
    novelty = energy_score(output.logits)
    rule = fit_rejection_threshold(novelty[:6], novelty[6:9] + 1.0)
    rejected = rule.reject(novelty)
    prediction = output.logits.argmax(dim=-1)
    segmentation = segmentation_metrics(mask_logits.detach(), masks)
    return {
        "synthetic_only": True,
        "pipeline_shape": list(output.logits.shape),
        "gate_shape": list(output.gate_weights.shape),
        "weights_normalized": bool(
            torch.allclose(output.gate_weights.sum(dim=-1), torch.ones(batch), atol=1e-6)
        ),
        "coverage_shape": list(coverage.shape),
        "abstentions": int(rejected.sum()),
        "classification_sanity": {
            "accuracy": accuracy(prediction, labels),
            "macro_f1": macro_f1(prediction, labels, classes),
        },
        "segmentation_sanity": segmentation,
        "parameters": parameter_counts(recognizer),
    }
