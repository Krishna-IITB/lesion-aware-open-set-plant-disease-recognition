"""Portable JSONL manifest schema for images and optional lesion masks."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

VALID_SPLITS = frozenset({"train", "validation", "test"})


@dataclass(frozen=True)
class Sample:
    sample_id: str
    image_path: str
    label: str
    split: str
    domain: str
    mask_path: str | None = None

    def validate(self) -> None:
        if not self.sample_id or not self.image_path or not self.label or not self.domain:
            raise ValueError("sample_id, image_path, label, and domain must be non-empty")
        if self.split not in VALID_SPLITS:
            raise ValueError(
                f"invalid split {self.split!r}; expected one of {sorted(VALID_SPLITS)}"
            )


def load_manifest(path: str | Path) -> list[Sample]:
    """Read and validate a JSON Lines manifest."""
    source = Path(path)
    samples: list[Sample] = []
    seen: set[str] = set()
    with source.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                sample = Sample(**json.loads(line))
                sample.validate()
            except (TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(f"invalid manifest row {line_number}: {error}") from error
            if sample.sample_id in seen:
                raise ValueError(f"duplicate sample_id in manifest: {sample.sample_id}")
            seen.add(sample.sample_id)
            samples.append(sample)
    if not samples:
        raise ValueError(f"manifest is empty: {source}")
    return samples


def write_manifest(samples: Iterable[Sample], path: str | Path) -> None:
    """Write samples atomically in stable order."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(samples, key=lambda sample: sample.sample_id)
    for sample in ordered:
        sample.validate()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for sample in ordered:
            handle.write(json.dumps(asdict(sample), sort_keys=True) + "\n")
    temporary.replace(destination)
