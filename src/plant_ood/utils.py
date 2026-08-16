"""Reproducibility and experiment-metadata helpers."""

from __future__ import annotations

import json
import os
import platform
import random
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn


def seed_everything(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)


def parameter_counts(model: nn.Module) -> dict[str, float | int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    percentage = 0.0 if total == 0 else 100.0 * trainable / total
    return {"total": total, "trainable": trainable, "trainable_percent": percentage}


def git_commit() -> str | None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    return result.stdout.strip() if result.returncode == 0 else None


def write_run_metadata(path: str | Path, fields: dict[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        **fields,
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "pid": os.getpid(),
    }
    destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
