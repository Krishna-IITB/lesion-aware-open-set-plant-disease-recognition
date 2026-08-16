"""Explicit, auditable class taxonomy mappings."""

from __future__ import annotations

import json
from pathlib import Path


def load_class_mapping(path: str | Path) -> dict[str, str | None]:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or not raw:
        raise ValueError("class mapping must be a non-empty JSON object")
    mapping: dict[str, str | None] = {}
    for source, target in raw.items():
        if not isinstance(source, str) or not source:
            raise ValueError("class mapping keys must be non-empty strings")
        if target is not None and (not isinstance(target, str) or not target):
            raise ValueError(f"invalid mapping target for {source!r}")
        mapping[source] = target
    return mapping


def apply_class_mapping(label: str, mapping: dict[str, str | None]) -> str | None:
    if label not in mapping:
        raise KeyError(f"class {label!r} is not explicitly mapped")
    return mapping[label]
