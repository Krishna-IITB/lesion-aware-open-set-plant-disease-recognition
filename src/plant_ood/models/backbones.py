"""Lazy, frozen Hugging Face wrappers for CLIP and DINOv3."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def _transformers() -> Any:
    try:
        import transformers  # type: ignore[import-not-found]
    except ImportError as error:
        raise RuntimeError(
            "Foundation backbones require `pip install -e '.[backbones]'`. "
            "DINOv3 access may additionally require accepting its model terms and setting HF_TOKEN."
        ) from error
    return transformers


def freeze_module(module: nn.Module) -> nn.Module:
    """Freeze parameters and switch to evaluation mode, returning the same module."""
    return module.requires_grad_(False).eval()


class FrozenCLIP(nn.Module):
    """Frozen CLIP image/text encoder using an exact Hugging Face model identifier."""

    def __init__(self, model_id: str = "openai/clip-vit-base-patch32") -> None:
        super().__init__()
        transformers = _transformers()
        self.model_id = model_id
        self.processor = transformers.AutoProcessor.from_pretrained(model_id)
        self.model = transformers.CLIPModel.from_pretrained(model_id)
        freeze_module(self.model)

    @torch.inference_mode()
    def encode_images(self, images: list[Any], device: torch.device) -> Tensor:
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        return F.normalize(self.model.get_image_features(**inputs), dim=-1)

    @torch.inference_mode()
    def encode_texts(self, texts: list[str], device: torch.device) -> Tensor:
        inputs = self.processor(text=texts, padding=True, return_tensors="pt").to(device)
        return F.normalize(self.model.get_text_features(**inputs), dim=-1)


class FrozenDINOv3(nn.Module):
    """Frozen DINOv3 ViT-S/16 dense patch encoder.

    The small distilled variant is the default because its 21M parameters are credible on a
    single research GPU while retaining DINOv3's dense-token interface.
    """

    def __init__(self, model_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m") -> None:
        super().__init__()
        transformers = _transformers()
        self.model_id = model_id
        self.processor = transformers.AutoImageProcessor.from_pretrained(model_id)
        self.model = transformers.AutoModel.from_pretrained(model_id)
        freeze_module(self.model)

    @torch.inference_mode()
    def encode_patches(
        self, images: list[Any], device: torch.device
    ) -> tuple[Tensor, tuple[int, int]]:
        inputs = self.processor(images=images, return_tensors="pt").to(device)
        output = self.model(**inputs).last_hidden_state
        # DINOv3 prepends CLS and register tokens; the rectangular patch grid is the tail.
        patches = output[:, 1:]
        pixel_values = inputs["pixel_values"]
        patch_size = int(getattr(self.model.config, "patch_size", 16))
        grid = (pixel_values.shape[-2] // patch_size, pixel_values.shape[-1] // patch_size)
        expected = grid[0] * grid[1]
        if patches.shape[1] < expected:
            raise RuntimeError("DINOv3 returned fewer tokens than the inferred patch grid")
        return F.normalize(patches[:, -expected:], dim=-1), grid
