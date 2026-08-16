from dataclasses import replace
from pathlib import Path

import pytest

from plant_ood.data.manifest import Sample, load_manifest, write_manifest
from plant_ood.data.mapping import apply_class_mapping, load_class_mapping
from plant_ood.data.splits import create_few_shot_split, validate_split


def samples() -> list[Sample]:
    rows = []
    for label in ("known_a", "known_b", "unknown"):
        for split, count in (("train", 5), ("validation", 2), ("test", 2)):
            for index in range(count):
                rows.append(
                    Sample(
                        sample_id=f"{label}-{split}-{index}",
                        image_path=f"{label}/{index}.jpg",
                        label=label,
                        split=split,
                        domain="lab" if split == "train" else "field",
                    )
                )
    return rows


def test_manifest_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    write_manifest(samples(), path)
    assert load_manifest(path) == sorted(samples(), key=lambda item: item.sample_id)


def test_few_shot_split_is_reproducible_and_leakage_safe() -> None:
    first = create_few_shot_split(samples(), 5, 13, ["unknown"])
    second = create_few_shot_split(samples(), 5, 13, ["unknown"])
    assert first == second
    assert len(first.prototype_ids) == 10
    assert not set(first.prototype_ids) & set(first.test_ids)
    assert all("unknown" not in item for item in first.prototype_ids)


def test_split_detects_manifest_drift() -> None:
    rows = samples()
    split = create_few_shot_split(rows, 5, 13, ["unknown"])
    changed = {sample.sample_id: sample for sample in rows}
    key = next(iter(changed))
    changed[key] = replace(changed[key], image_path="moved.jpg")
    with pytest.raises(ValueError, match="digest"):
        validate_split(split, changed)


def test_class_mapping_is_explicit(tmp_path: Path) -> None:
    path = tmp_path / "map.json"
    path.write_text('{"source": "target", "drop": null}')
    mapping = load_class_mapping(path)
    assert apply_class_mapping("source", mapping) == "target"
    assert apply_class_mapping("drop", mapping) is None
    with pytest.raises(KeyError):
        apply_class_mapping("implicit", mapping)
