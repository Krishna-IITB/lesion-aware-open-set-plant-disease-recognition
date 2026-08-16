from pathlib import Path

import torch

from plant_ood.features import CacheMetadata, FeatureCache


def test_feature_cache_rejects_stale_image(tmp_path: Path) -> None:
    image = tmp_path / "image.bin"
    image.write_bytes(b"first")
    cache = FeatureCache(tmp_path / "cache")
    fields = {
        "sample_id": "a",
        "dataset": "tiny",
        "split": "train",
        "backbone": "fake",
        "checkpoint": "v1",
        "preprocessing": "resize-8",
        "representation": "patch",
    }
    metadata = CacheMetadata.for_file(image, **fields)
    expected = torch.randn(4, 8)
    cache.store(metadata, expected)
    assert torch.equal(cache.load(metadata), expected)
    image.write_bytes(b"second")
    changed = CacheMetadata.for_file(image, **fields)
    assert cache.load(changed) is None
