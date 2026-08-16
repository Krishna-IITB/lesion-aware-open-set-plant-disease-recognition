import pytest
import torch

from plant_ood.losses import bce_dice_loss, dice_loss
from plant_ood.models.backbones import freeze_module
from plant_ood.models.fusion import ThreeViewGate
from plant_ood.models.lesion import LesionDecoder, lesion_aware_pool
from plant_ood.models.prototypes import build_prototypes
from plant_ood.models.system import LesionAwareRecognizer
from plant_ood.utils import parameter_counts


def test_lesion_shapes_loss_and_empty_fallback() -> None:
    patches = torch.randn(2, 16, 8)
    decoder = LesionDecoder(8, 4)
    logits = decoder(patches, (4, 4), (12, 10))
    assert logits.shape == (2, 1, 12, 10)
    targets = torch.zeros_like(logits)
    assert torch.isfinite(bce_dice_loss(logits, targets))
    assert torch.isfinite(dice_loss(logits, targets))
    pooled, coverage = lesion_aware_pool(patches, torch.full((2, 1, 4, 4), -100.0), (4, 4))
    expected = torch.nn.functional.normalize(patches.mean(dim=1), dim=-1)
    assert torch.allclose(pooled, expected)
    assert torch.all(coverage < 1e-6)


def test_gate_weights_and_recognizer_schema() -> None:
    labels = torch.tensor([0, 0, 1, 1])
    features = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]], dtype=torch.float32)
    views = ("text", "global", "lesion")
    banks = {name: build_prototypes(features, labels, ("a", "b"), 1, 3) for name in views}
    gate = ThreeViewGate(views, hidden_dim=4)
    output = LesionAwareRecognizer(banks, gate)({name: features for name in views})
    assert output.logits.shape == (4, 2)
    assert output.gate_weights.shape == (4, 3)
    assert torch.allclose(output.gate_weights.sum(dim=1), torch.ones(4))
    assert tuple(output.view_scores) == views
    counts = parameter_counts(gate)
    assert counts["trainable"] == counts["total"]


def test_prototype_construction_has_no_missing_class() -> None:
    with pytest.raises(ValueError, match="no prototype"):
        build_prototypes(torch.randn(3, 4), torch.zeros(3, dtype=torch.long), ("a", "b"), 2, 1)


def test_backbone_freezing_is_explicit() -> None:
    backbone = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Dropout())
    assert freeze_module(backbone) is backbone
    assert backbone.training is False
    assert not any(parameter.requires_grad for parameter in backbone.parameters())
