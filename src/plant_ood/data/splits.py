"""Leakage-safe few-shot and class-held-out split construction."""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path

from plant_ood.data.manifest import Sample


@dataclass(frozen=True)
class SplitManifest:
    seed: int
    shots: int
    known_classes: tuple[str, ...]
    unknown_classes: tuple[str, ...]
    prototype_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    test_ids: tuple[str, ...]
    source_digest: str

    def write(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n")

    @classmethod
    def read(cls, path: str | Path) -> SplitManifest:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            return cls(
                seed=int(raw["seed"]),
                shots=int(raw["shots"]),
                known_classes=tuple(raw["known_classes"]),
                unknown_classes=tuple(raw["unknown_classes"]),
                prototype_ids=tuple(raw["prototype_ids"]),
                validation_ids=tuple(raw["validation_ids"]),
                test_ids=tuple(raw["test_ids"]),
                source_digest=str(raw["source_digest"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"invalid split manifest: {error}") from error


def manifest_digest(samples: Iterable[Sample]) -> str:
    rows = [json.dumps(asdict(sample), sort_keys=True) for sample in samples]
    return hashlib.sha256("\n".join(sorted(rows)).encode()).hexdigest()


def create_few_shot_split(
    samples: list[Sample], shots: int, seed: int, unknown_classes: Iterable[str]
) -> SplitManifest:
    """Select exactly ``shots`` training samples per known class.

    Validation and test membership are inherited, never sampled into prototypes. Unknown
    classes are excluded from prototypes and retained for class-held-out evaluation.
    """
    if shots not in {1, 5, 10, 20}:
        raise ValueError("shots must be one of 1, 5, 10, or 20")
    unknown = tuple(sorted(set(unknown_classes)))
    labels = {sample.label for sample in samples}
    missing = set(unknown) - labels
    if missing:
        raise ValueError(f"unknown classes absent from manifest: {sorted(missing)}")
    known = tuple(sorted(labels - set(unknown)))
    if not known:
        raise ValueError("at least one known class is required")

    rng = random.Random(seed)
    prototype_ids: list[str] = []
    for label in known:
        candidates = sorted(
            sample.sample_id
            for sample in samples
            if sample.split == "train" and sample.label == label
        )
        if len(candidates) < shots:
            raise ValueError(f"class {label!r} has {len(candidates)} train samples, needs {shots}")
        prototype_ids.extend(rng.sample(candidates, shots))

    validation_ids = tuple(sorted(s.sample_id for s in samples if s.split == "validation"))
    test_ids = tuple(sorted(s.sample_id for s in samples if s.split == "test"))
    split = SplitManifest(
        seed=seed,
        shots=shots,
        known_classes=known,
        unknown_classes=unknown,
        prototype_ids=tuple(sorted(prototype_ids)),
        validation_ids=validation_ids,
        test_ids=test_ids,
        source_digest=manifest_digest(samples),
    )
    validate_split(split, {sample.sample_id: sample for sample in samples})
    return split


def validate_split(split: SplitManifest, samples: Mapping[str, Sample]) -> None:
    groups = [set(split.prototype_ids), set(split.validation_ids), set(split.test_ids)]
    if any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
        raise ValueError("prototype, validation, and test sample IDs must be disjoint")
    unknown = set(split.unknown_classes)
    for sample_id in split.prototype_ids:
        sample = samples.get(sample_id)
        if sample is None:
            raise ValueError(f"split references absent sample: {sample_id}")
        if sample.split != "train":
            raise ValueError(f"prototype sample {sample_id} is not from train")
        if sample.label in unknown:
            raise ValueError(f"unknown class leaked into prototypes: {sample.label}")
    if manifest_digest(samples.values()) != split.source_digest:
        raise ValueError("split source digest does not match the current data manifest")
