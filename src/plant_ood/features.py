"""Feature-cache storage with provenance checks and atomic writes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor


@dataclass(frozen=True)
class CacheMetadata:
    schema_version: int
    sample_id: str
    image_sha256: str
    dataset: str
    split: str
    backbone: str
    checkpoint: str
    preprocessing: str
    representation: str

    @classmethod
    def for_file(
        cls,
        image_path: str | Path,
        *,
        sample_id: str,
        dataset: str,
        split: str,
        backbone: str,
        checkpoint: str,
        preprocessing: str,
        representation: str,
    ) -> CacheMetadata:
        digest = hashlib.sha256(Path(image_path).read_bytes()).hexdigest()
        return cls(
            schema_version=1,
            sample_id=sample_id,
            image_sha256=digest,
            dataset=dataset,
            split=split,
            backbone=backbone,
            checkpoint=checkpoint,
            preprocessing=preprocessing,
            representation=representation,
        )


class FeatureCache:
    """One-tensor-per-entry cache whose key includes all relevant provenance."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    @staticmethod
    def key(metadata: CacheMetadata) -> str:
        payload = json.dumps(asdict(metadata), sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()

    def store(self, metadata: CacheMetadata, tensor: Tensor) -> Path:
        destination = self.root / metadata.representation / f"{self.key(metadata)}.pt"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        torch.save({"metadata": asdict(metadata), "tensor": tensor.detach().cpu()}, temporary)
        temporary.replace(destination)
        return destination

    def load(self, metadata: CacheMetadata) -> Tensor | None:
        source = self.root / metadata.representation / f"{self.key(metadata)}.pt"
        if not source.exists():
            return None
        payload = torch.load(source, map_location="cpu", weights_only=True)
        if payload.get("metadata") != asdict(metadata):
            raise ValueError(f"cache metadata mismatch: {source}")
        tensor = payload.get("tensor")
        if not isinstance(tensor, Tensor):
            raise ValueError(f"cache entry contains no tensor: {source}")
        return tensor
