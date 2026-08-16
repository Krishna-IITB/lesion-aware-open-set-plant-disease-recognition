import pytest
import torch

from plant_ood.evaluation.metrics import accuracy, auroc, fpr95, macro_f1, segmentation_metrics
from plant_ood.openset import (
    TemperatureScaler,
    energy_score,
    fit_rejection_threshold,
    prototype_distance,
)
from plant_ood.prediction import format_predictions


def test_classification_and_segmentation_metrics() -> None:
    targets = torch.tensor([0, 0, 1, 1])
    predictions = torch.tensor([0, 1, 1, 1])
    assert accuracy(predictions, targets) == 0.75
    assert macro_f1(predictions, targets, 2) == pytest.approx((2 / 3 + 0.8) / 2)
    mask = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]])
    logits = torch.where(mask.bool(), torch.tensor(20.0), torch.tensor(-20.0))
    assert segmentation_metrics(logits, mask) == {"dice": 1.0, "miou": 1.0}


def test_ood_metrics_and_threshold_direction() -> None:
    scores = torch.tensor([0.0, 0.1, 0.2, 0.8, 0.9, 1.0])
    labels = torch.tensor([False, False, False, True, True, True])
    assert auroc(scores, labels) == 1.0
    assert fpr95(scores, labels) == 0.0
    rule = fit_rejection_threshold(scores[:3], scores[3:], split="validation")
    assert not rule.reject(scores[:3]).any()
    assert rule.reject(scores[3:]).all()
    with pytest.raises(ValueError, match="non-validation"):
        fit_rejection_threshold(scores[:3], scores[3:], split="test")


def test_energy_distance_and_temperature() -> None:
    confident = torch.tensor([[10.0, 0.0]])
    uncertain = torch.tensor([[1.0, 1.0]])
    assert energy_score(confident) < energy_score(uncertain)
    assert prototype_distance(confident) < prototype_distance(uncertain)
    scaler = TemperatureScaler()
    value = scaler.fit(torch.tensor([[3.0, 0.0], [0.0, 3.0]]), torch.tensor([0, 1]), iterations=5)
    assert value > 0


def test_inference_output_schema() -> None:
    scores = torch.tensor([[3.0, 1.0], [1.0, 3.0]])
    novelty = torch.tensor([0.1, 0.9])
    weights = torch.tensor([[0.6, 0.4], [0.2, 0.8]])
    rule = fit_rejection_threshold(torch.tensor([0.0, 0.1]), torch.tensor([0.8, 0.9]))
    output = format_predictions(
        scores,
        novelty,
        weights,
        class_names=("a", "b"),
        view_names=("text", "global"),
        rejection_rule=rule,
    )
    assert output[0].predicted_label == "a"
    assert output[1].predicted_label == "unknown"
    assert output[1].known_label == "b"
