"""Small-head training loops with validation and early stopping."""

from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from plant_ood.losses import bce_dice_loss


def train_gate(
    model: nn.Module,
    batches: Iterable[tuple[dict[str, Tensor], Tensor]],
    validation: tuple[dict[str, Tensor], Tensor],
    *,
    learning_rate: float,
    epochs: int,
    patience: int,
) -> list[dict[str, float]]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("the recognition model has no trainable parameters")
    optimizer = torch.optim.AdamW(parameters, lr=learning_rate)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state = deepcopy(model.state_dict())
    stale = 0
    materialized = list(batches)
    for epoch in range(epochs):
        model.train()
        train_total = 0.0
        for features, labels in materialized:
            optimizer.zero_grad()
            loss = F.cross_entropy(model(features).logits, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            train_total += float(loss.detach())
        model.eval()
        with torch.no_grad():
            validation_loss = float(F.cross_entropy(model(validation[0]).logits, validation[1]))
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": train_total / len(materialized),
                "val_loss": validation_loss,
            }
        )
        if validation_loss < best_loss - 1e-6:
            best_loss, best_state, stale = validation_loss, deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(best_state)
    return history


def lesion_training_step(
    decoder: nn.Module,
    optimizer: torch.optim.Optimizer,
    patch_features: Tensor,
    grid_size: tuple[int, int],
    masks: Tensor,
) -> float:
    decoder.train()
    optimizer.zero_grad()
    logits = decoder(patch_features.detach(), grid_size, output_size=masks.shape[-2:])
    loss = bce_dice_loss(logits, masks)
    loss.backward()  # type: ignore[no-untyped-call]
    optimizer.step()
    return float(loss.detach())
