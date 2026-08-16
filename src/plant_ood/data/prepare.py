"""Dataset indexing and integrity validation."""

from __future__ import annotations

import hashlib
from pathlib import Path

from PIL import Image

from plant_ood.data.manifest import Sample

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp"})


def scan_class_folders(
    root: str | Path,
    *,
    domain: str,
    masks_root: str | Path | None = None,
) -> list[Sample]:
    """Index ``root/{train,validation,test}/{class}/image`` without moving data."""
    source = Path(root).resolve()
    mask_source = Path(masks_root).resolve() if masks_root else None
    samples: list[Sample] = []
    for split in ("train", "validation", "test"):
        split_directory = source / split
        if not split_directory.is_dir():
            continue
        for class_directory in sorted(path for path in split_directory.iterdir() if path.is_dir()):
            for image_path in sorted(class_directory.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative = image_path.relative_to(source)
                mask_path: str | None = None
                if mask_source:
                    candidate = (mask_source / relative).with_suffix(".png")
                    if candidate.is_file():
                        mask_path = str(candidate)
                identifier = hashlib.sha256(str(relative).encode()).hexdigest()[:16]
                samples.append(
                    Sample(
                        sample_id=f"{domain}-{identifier}",
                        image_path=str(image_path),
                        label=class_directory.name,
                        split=split,
                        domain=domain,
                        mask_path=mask_path,
                    )
                )
    if not samples:
        raise ValueError(f"no supported images found below {source}")
    return samples


def validate_dataset(samples: list[Sample], *, check_files: bool = True) -> dict[str, int]:
    """Reject cross-split duplicate image content and unreadable image/mask pairs."""
    content_splits: dict[str, set[str]] = {}
    masked = 0
    for sample in samples:
        sample.validate()
        if not check_files:
            continue
        image_path = Path(sample.image_path)
        if not image_path.is_file():
            raise ValueError(f"missing image: {image_path}")
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        content_splits.setdefault(digest, set()).add(sample.split)
        with Image.open(image_path) as image:
            image.verify()
        if sample.mask_path:
            mask_path = Path(sample.mask_path)
            if not mask_path.is_file():
                raise ValueError(f"missing mask: {mask_path}")
            with Image.open(mask_path) as mask:
                mask.verify()
            masked += 1
    leaked = [digest for digest, splits in content_splits.items() if len(splits) > 1]
    if leaked:
        raise ValueError(f"detected {len(leaked)} image-content duplicates across splits")
    return {
        "samples": len(samples),
        "masked_samples": masked,
        "classes": len({s.label for s in samples}),
    }
