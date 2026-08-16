"""Feature extraction and leakage-safe experiment execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import Tensor
from torch.nn import functional as F

from plant_ood.config import ProjectConfig
from plant_ood.data.manifest import Sample, load_manifest
from plant_ood.data.splits import SplitManifest, validate_split
from plant_ood.evaluation.metrics import accuracy, auroc, fpr95, macro_f1
from plant_ood.losses import bce_dice_loss
from plant_ood.models.backbones import FrozenCLIP, FrozenDINOv3
from plant_ood.models.fusion import ThreeViewGate
from plant_ood.models.lesion import LesionDecoder, lesion_aware_pool
from plant_ood.models.prototypes import PrototypeBank, build_prototypes
from plant_ood.models.system import LesionAwareRecognizer
from plant_ood.openset import (
    TemperatureScaler,
    energy_score,
    fit_rejection_threshold,
    prototype_distance,
)
from plant_ood.training.loops import lesion_training_step, train_gate
from plant_ood.utils import parameter_counts, seed_everything, write_run_metadata


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def extract_feature_bundle(
    config: ProjectConfig,
    output: str | Path,
    *,
    prompts_path: str | Path | None = None,
) -> None:
    """Extract frozen CLIP and DINOv3 features into a provenance-bearing bundle."""
    samples = load_manifest(config.dataset.manifest)
    device = choose_device(config.training.device)
    clip = FrozenCLIP(config.model.clip_id).to(device)
    dino = FrozenDINOv3(config.model.dino_id).to(device) if config.model.use_lesion else None
    class_names = tuple(sorted({sample.label for sample in samples}))
    prompts = _load_prompts(class_names, prompts_path)
    text_values: list[Tensor] = []
    max_prompts = max(len(prompts[name]) for name in class_names)
    for name in class_names:
        descriptions = prompts[name]
        encoded = clip.encode_texts(descriptions, device).cpu()
        if len(encoded) < max_prompts:
            encoded = torch.cat((encoded, encoded[:1].expand(max_prompts - len(encoded), -1)))
        text_values.append(encoded)

    global_features: list[Tensor] = []
    patch_features: list[Tensor] = []
    masks: list[Tensor] = []
    grid_size: tuple[int, int] | None = None
    for sample in samples:
        with Image.open(sample.image_path) as raw:
            image = raw.convert("RGB")
            global_features.append(clip.encode_images([image], device).cpu()[0])
            if dino is not None:
                patches, current_grid = dino.encode_patches([image], device)
                if grid_size is not None and grid_size != current_grid:
                    raise RuntimeError("inconsistent DINO patch grids after preprocessing")
                grid_size = current_grid
                patch_features.append(patches.cpu()[0])
                masks.append(_read_mask(sample, current_grid))

    payload: dict[str, Any] = {
        "schema_version": 1,
        "class_names": class_names,
        "sample_ids": tuple(sample.sample_id for sample in samples),
        "labels": torch.tensor([class_names.index(sample.label) for sample in samples]),
        "splits": tuple(sample.split for sample in samples),
        "domains": tuple(sample.domain for sample in samples),
        "clip_global": torch.stack(global_features),
        "text_prototypes": torch.stack(text_values),
        "clip_id": config.model.clip_id,
        "dino_id": config.model.dino_id if dino else None,
        "manifest": str(config.dataset.manifest),
        "backbone_parameters": {
            "clip": parameter_counts(clip.model),
            "dino": parameter_counts(dino.model) if dino is not None else None,
        },
    }
    if dino is not None and grid_size is not None:
        payload.update(
            {
                "dino_patches": torch.stack(patch_features),
                "lesion_masks": torch.stack(masks),
                "grid_size": grid_size,
            }
        )
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def train_lesion_decoder(
    feature_bundle: str | Path,
    output: str | Path,
    *,
    epochs: int,
    learning_rate: float,
    hidden_dim: int,
    split_path: str | Path | None = None,
) -> dict[str, float]:
    bundle = _load_bundle(feature_bundle)
    patches: Tensor = bundle["dino_patches"]
    masks: Tensor = bundle["lesion_masks"]
    grid_size = tuple(bundle["grid_size"])
    split_names = tuple(bundle["splits"])
    excluded_classes: tuple[str, ...] = ()
    split_source_digest: str | None = None
    if split_path is not None:
        split_manifest = SplitManifest.read(split_path)
        excluded_classes = split_manifest.unknown_classes
        split_source_digest = split_manifest.source_digest
    class_names = tuple(bundle["class_names"])
    labels: Tensor = bundle["labels"]
    excluded_indices = {class_names.index(name) for name in excluded_classes}
    train_indices = [
        index
        for index, split_name in enumerate(split_names)
        if split_name == "train" and int(labels[index]) not in excluded_indices
    ]
    validation_indices = [index for index, split in enumerate(split_names) if split == "validation"]
    usable_train = [index for index in train_indices if masks[index].sum() > 0]
    usable_validation = [index for index in validation_indices if masks[index].sum() > 0]
    if not usable_train or not usable_validation:
        raise ValueError("lesion training requires non-empty annotated train and validation masks")
    decoder = LesionDecoder(patches.shape[-1], hidden_dim)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=learning_rate)
    for _ in range(epochs):
        lesion_training_step(
            decoder,
            optimizer,
            patches[usable_train],
            grid_size,
            masks[usable_train],
        )
    decoder.eval()
    with torch.no_grad():
        logits = decoder(patches[usable_validation], grid_size, output_size=masks.shape[-2:])
        validation_loss = float(bce_dice_loss(logits, masks[usable_validation]))
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": 1,
            "feature_dim": patches.shape[-1],
            "hidden_dim": hidden_dim,
            "state_dict": decoder.state_dict(),
            "validation_loss": validation_loss,
            "parameters": parameter_counts(decoder),
            "excluded_classes": excluded_classes,
            "split_source_digest": split_source_digest,
        },
        destination,
    )
    return {"validation_loss": validation_loss}


def run_experiment(
    config: ProjectConfig,
    feature_bundle: str | Path,
    split_path: str | Path,
    *,
    lesion_checkpoint: str | Path | None = None,
) -> dict[str, Any]:
    """Train lightweight heads and evaluate once using a persisted split manifest."""
    seed_everything(config.training.seeds[0])
    bundle = _load_bundle(feature_bundle)
    samples = load_manifest(config.dataset.manifest)
    split = SplitManifest.read(split_path)
    validate_split(split, {sample.sample_id: sample for sample in samples})
    if split.seed != config.training.seeds[0] or split.shots != config.training.shots:
        raise ValueError("split seed/shots do not match the experiment configuration")
    ids = tuple(bundle["sample_ids"])
    index_by_id = {sample_id: index for index, sample_id in enumerate(ids)}
    labels: Tensor = bundle["labels"]
    all_class_names = tuple(bundle["class_names"])
    known_original = [all_class_names.index(name) for name in split.known_classes]
    original_to_known = {original: known for known, original in enumerate(known_original)}
    prototype_indices = torch.tensor([index_by_id[item] for item in split.prototype_ids])
    validation_indices = torch.tensor([index_by_id[item] for item in split.validation_ids])
    test_indices = torch.tensor([index_by_id[item] for item in split.test_ids])

    representations: dict[str, Tensor] = {
        "text": bundle["clip_global"],
        "global": bundle["clip_global"],
    }
    if config.model.use_lesion:
        if config.experiment.variant in {"C", "D", "E"} and lesion_checkpoint is None:
            raise ValueError(
                f"experiment {config.experiment.variant} requires --lesion-checkpoint; "
                "only variant B uses unmasked global DINO pooling"
            )
        representations["lesion"] = _lesion_representations(bundle, lesion_checkpoint)
        if config.open_set.enabled and lesion_checkpoint is not None:
            lesion_payload = torch.load(lesion_checkpoint, map_location="cpu", weights_only=True)
            excluded = set(lesion_payload.get("excluded_classes", ()))
            missing_exclusions = set(split.unknown_classes) - excluded
            if missing_exclusions:
                raise ValueError(
                    "lesion checkpoint was not trained with held-out classes excluded: "
                    f"{sorted(missing_exclusions)}"
                )
            if lesion_payload.get("split_source_digest") != split.source_digest:
                raise ValueError("lesion checkpoint split provenance does not match this manifest")
    enabled = tuple(
        view
        for view, active in (
            ("text", config.model.use_text),
            ("global", config.model.use_global),
            ("lesion", config.model.use_lesion),
        )
        if active
    )
    banks: dict[str, PrototypeBank] = {}
    for view in enabled:
        if view == "text":
            text = bundle["text_prototypes"][known_original]
            banks[view] = PrototypeBank(text, split.known_classes)
        else:
            prototype_labels = torch.tensor(
                [original_to_known[int(label)] for label in labels[prototype_indices]]
            )
            banks[view] = build_prototypes(
                representations[view][prototype_indices],
                prototype_labels,
                split.known_classes,
                config.model.prototypes_per_class,
                split.seed,
            )
    gate = ThreeViewGate(enabled, config.model.hidden_dim, learned=config.model.learned_gate)
    model = LesionAwareRecognizer(banks, gate)

    known_validation = torch.tensor(
        [int(labels[index]) in original_to_known for index in validation_indices], dtype=torch.bool
    )
    known_val_labels = torch.tensor(
        [original_to_known[int(labels[index])] for index in validation_indices[known_validation]]
    )
    gate_train_indices = prototype_indices
    gate_labels = torch.tensor(
        [original_to_known[int(labels[index])] for index in gate_train_indices], dtype=torch.long
    )
    gate_features = {
        view: values[gate_train_indices]
        for view, values in representations.items()
        if view in enabled
    }
    history: list[dict[str, float]] = []
    if config.model.learned_gate:
        if not len(gate_train_indices) or not known_validation.any():
            raise ValueError("learned gating needs known training and validation samples")
        batches = [(gate_features, gate_labels)]
        validation = (
            {view: representations[view][validation_indices[known_validation]] for view in enabled},
            known_val_labels,
        )
        history = train_gate(
            model,
            batches,
            validation,
            learning_rate=config.training.learning_rate,
            epochs=config.training.epochs,
            patience=config.training.patience,
        )

    model.eval()
    with torch.no_grad():
        validation_output = model(
            {view: representations[view][validation_indices] for view in enabled}
        )
        test_output = model({view: representations[view][test_indices] for view in enabled})
    scaler = TemperatureScaler()
    known_val_logits = validation_output.logits[known_validation]
    temperature = (
        scaler.fit(known_val_logits, known_val_labels)
        if config.open_set.temperature_scaling
        else 1.0
    )
    validation_logits = validation_output.logits / temperature
    test_logits = test_output.logits / temperature
    test_unknown = torch.tensor(
        [int(labels[index]) not in original_to_known for index in test_indices], dtype=torch.bool
    )
    known_test = ~test_unknown
    if not known_test.any():
        raise ValueError("test split contains no known-class samples")
    known_targets = torch.tensor(
        [original_to_known[int(labels[index])] for index in test_indices[known_test]]
    )
    known_predictions = test_logits[known_test].argmax(dim=-1)
    result: dict[str, Any] = {
        "status": "measured",
        "seed": split.seed,
        "shots": split.shots,
        "known_accuracy": accuracy(known_predictions, known_targets),
        "known_macro_f1": macro_f1(known_predictions, known_targets, len(split.known_classes)),
        "temperature": temperature,
        "gate_views": enabled,
        "mean_gate_weights": test_output.gate_weights.mean(dim=0).tolist(),
        "epochs_recorded": len(history),
    }
    result["parameters"] = _complete_parameter_counts(
        bundle, model, lesion_checkpoint if config.model.use_lesion else None
    )
    if config.open_set.enabled:
        validation_novelty, test_novelty = _novelty_pair(validation_logits, test_logits, config)
        unknown_validation = ~known_validation
        if not unknown_validation.any() or not test_unknown.any():
            raise ValueError(
                "open-set evaluation needs held-out-class samples in validation and test"
            )
        rule = fit_rejection_threshold(
            validation_novelty[known_validation],
            validation_novelty[unknown_validation],
            split="validation",
            method=config.open_set.method,
        )
        result.update(
            {
                "ood_auroc": auroc(test_novelty, test_unknown),
                "ood_fpr95": fpr95(test_novelty, test_unknown),
                "rejection_threshold": rule.threshold,
                "threshold_fitted_split": rule.fitted_split,
            }
        )
    else:
        result["open_set_status"] = "not evaluated (disabled by configuration)"
    run_directory = config.experiment.output_dir / config.experiment.name / f"seed-{split.seed}"
    run_directory.mkdir(parents=True, exist_ok=True)
    (run_directory / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    write_run_metadata(
        run_directory / "metadata.json",
        {
            "config": str(config.source_path),
            "split_manifest": str(Path(split_path).resolve()),
            "feature_bundle": str(Path(feature_bundle).resolve()),
            "clip_id": config.model.clip_id,
            "dino_id": config.model.dino_id,
        },
    )
    return result


def _novelty_pair(
    validation_logits: Tensor, test_logits: Tensor, config: ProjectConfig
) -> tuple[Tensor, Tensor]:
    validation_energy = energy_score(validation_logits)
    test_energy = energy_score(test_logits)
    validation_distance = prototype_distance(validation_logits)
    test_distance = prototype_distance(test_logits)
    if config.open_set.method == "energy":
        return validation_energy, test_energy
    if config.open_set.method == "prototype_distance":
        return validation_distance, test_distance

    def normalize_from_validation(validation: Tensor, test: Tensor) -> tuple[Tensor, Tensor]:
        center = validation.mean()
        scale = validation.std(unbiased=False).clamp_min(1e-8)
        return (validation - center) / scale, (test - center) / scale

    validation_energy, test_energy = normalize_from_validation(validation_energy, test_energy)
    validation_distance, test_distance = normalize_from_validation(
        validation_distance, test_distance
    )
    weight = config.open_set.energy_weight
    return (
        weight * validation_energy + (1.0 - weight) * validation_distance,
        weight * test_energy + (1.0 - weight) * test_distance,
    )


def _complete_parameter_counts(
    bundle: dict[str, Any], model: LesionAwareRecognizer, lesion_checkpoint: str | Path | None
) -> dict[str, float | int | str]:
    backbone_counts = bundle.get("backbone_parameters", {})
    use_clip = any(view in model.gate.views for view in ("text", "global"))
    use_dino = "lesion" in model.gate.views
    clip_total = int((backbone_counts.get("clip") or {}).get("total", 0)) if use_clip else 0
    dino_total = int((backbone_counts.get("dino") or {}).get("total", 0)) if use_dino else 0
    head_counts = parameter_counts(model)
    decoder_total = 0
    if lesion_checkpoint is not None:
        decoder_payload = torch.load(lesion_checkpoint, map_location="cpu", weights_only=True)
        decoder_total = int((decoder_payload.get("parameters") or {}).get("total", 0))
    total = clip_total + dino_total + decoder_total + int(head_counts["total"])
    task_trainable = decoder_total + int(head_counts["trainable"])
    return {
        "scope": "complete configured system; frozen backbone counts recorded at extraction",
        "total": total,
        "task_trainable": task_trainable,
        "task_trainable_percent": 0.0 if total == 0 else 100.0 * task_trainable / total,
    }


def _lesion_representations(bundle: dict[str, Any], checkpoint: str | Path | None) -> Tensor:
    patches: Tensor = bundle["dino_patches"]
    grid_size = tuple(bundle["grid_size"])
    if checkpoint is None:
        return F.normalize(patches.mean(dim=1), dim=-1)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    decoder = LesionDecoder(int(payload["feature_dim"]), int(payload["hidden_dim"]))
    decoder.load_state_dict(payload["state_dict"])
    decoder.eval()
    with torch.no_grad():
        mask_logits = decoder(patches, grid_size)
        pooled, _ = lesion_aware_pool(patches, mask_logits, grid_size)
    return pooled


def _load_prompts(class_names: tuple[str, ...], path: str | Path | None) -> dict[str, list[str]]:
    if path is None:
        return {
            name: [f"a photograph of a plant affected by {name.replace('_', ' ')}"]
            for name in class_names
        }
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    if set(raw) != set(class_names):
        raise ValueError("prompt file classes must exactly match manifest classes")
    if any(not isinstance(value, list) or not value for value in raw.values()):
        raise ValueError("each prompt-file class must contain a non-empty list")
    return {name: [str(prompt) for prompt in raw[name]] for name in class_names}


def _read_mask(sample: Sample, size: tuple[int, int]) -> Tensor:
    if sample.mask_path is None:
        return torch.zeros((1, *size), dtype=torch.float32)
    with Image.open(sample.mask_path) as raw:
        mask = raw.convert("L").resize((size[1], size[0]), resample=Image.Resampling.NEAREST)
        array = np.asarray(mask, dtype=np.float32)
    return torch.from_numpy((array > 0).astype(np.float32)).unsqueeze(0)


def _load_bundle(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("unsupported feature-bundle schema")
    return payload
