"""Dependency-light multiclass, segmentation, and OOD metrics."""

from __future__ import annotations

import torch
from torch import Tensor


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    _same_vector_shape(predictions, targets)
    return float((predictions == targets).float().mean())


def macro_f1(predictions: Tensor, targets: Tensor, number_of_classes: int) -> float:
    _same_vector_shape(predictions, targets)
    if number_of_classes < 1:
        raise ValueError("number_of_classes must be positive")
    scores = []
    for class_index in range(number_of_classes):
        predicted = predictions == class_index
        actual = targets == class_index
        true_positive = (predicted & actual).sum().float()
        false_positive = (predicted & ~actual).sum().float()
        false_negative = (~predicted & actual).sum().float()
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(torch.where(denominator > 0, 2 * true_positive / denominator, 0.0))
    return float(torch.stack(scores).mean())


def segmentation_metrics(
    logits: Tensor, targets: Tensor, threshold: float = 0.5
) -> dict[str, float]:
    if logits.shape != targets.shape:
        raise ValueError("segmentation logits and targets must have identical shapes")
    predicted = torch.sigmoid(logits) >= threshold
    actual = targets >= 0.5
    intersection = (predicted & actual).flatten(1).sum(dim=1).float()
    predicted_mass = predicted.flatten(1).sum(dim=1).float()
    actual_mass = actual.flatten(1).sum(dim=1).float()
    union = predicted_mass + actual_mass - intersection
    # Empty-empty masks are a correct localization with score 1.
    dice = torch.where(
        predicted_mass + actual_mass > 0,
        2 * intersection / (predicted_mass + actual_mass).clamp_min(1),
        1.0,
    )
    iou = torch.where(union > 0, intersection / union.clamp_min(1), 1.0)
    return {"dice": float(dice.mean()), "miou": float(iou.mean())}


def auroc(novelty_scores: Tensor, is_unknown: Tensor) -> float:
    """Area under ROC with ties handled by trapezoidal integration."""
    _same_vector_shape(novelty_scores, is_unknown)
    labels = is_unknown.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("AUROC requires at least one known and one unknown sample")
    order = torch.argsort(novelty_scores, descending=True, stable=True)
    scores = novelty_scores[order]
    labels = labels[order]
    distinct = torch.ones_like(labels, dtype=torch.bool)
    distinct[:-1] = scores[:-1] != scores[1:]
    true_positive = labels.cumsum(0)[distinct].float()
    false_positive = (~labels).cumsum(0)[distinct].float()
    tpr = torch.cat((torch.zeros(1), true_positive / positives))
    fpr = torch.cat((torch.zeros(1), false_positive / negatives))
    return float(torch.trapz(tpr, fpr))


def fpr95(novelty_scores: Tensor, is_unknown: Tensor) -> float:
    """Known-sample false-positive rate at >=95% unknown true-positive rate."""
    _same_vector_shape(novelty_scores, is_unknown)
    labels = is_unknown.bool()
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if not positives or not negatives:
        raise ValueError("FPR95 requires at least one known and one unknown sample")
    thresholds = torch.unique(novelty_scores).sort(descending=True).values
    best = 1.0
    for threshold in thresholds:
        rejected = novelty_scores >= threshold
        tpr = float((rejected & labels).sum()) / positives
        fpr = float((rejected & ~labels).sum()) / negatives
        if tpr >= 0.95:
            best = min(best, fpr)
    return best


def _same_vector_shape(first: Tensor, second: Tensor) -> None:
    if first.ndim != 1 or second.ndim != 1 or first.shape != second.shape:
        raise ValueError("inputs must be one-dimensional tensors with identical shapes")
