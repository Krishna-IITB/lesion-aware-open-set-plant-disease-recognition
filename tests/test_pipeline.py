from pathlib import Path

import pytest
import torch

from plant_ood.config import (
    DatasetConfig,
    ExperimentConfig,
    ModelConfig,
    OpenSetConfig,
    ProjectConfig,
    TrainingConfig,
)
from plant_ood.data.manifest import Sample, write_manifest
from plant_ood.data.splits import create_few_shot_split
from plant_ood.pipeline import run_experiment, train_lesion_decoder


def test_real_pipeline_contract_on_synthetic_features(tmp_path: Path) -> None:
    samples = []
    for label in ("a", "b", "u"):
        for split in ("train", "validation", "test"):
            for index in range(2):
                samples.append(
                    Sample(
                        sample_id=f"{label}-{split}-{index}",
                        image_path=f"unused/{label}-{split}-{index}.jpg",
                        label=label,
                        split=split,
                        domain="lab" if split == "train" else "field",
                        mask_path=f"unused/{label}-{split}-{index}.png",
                    )
                )
    manifest = tmp_path / "manifest.jsonl"
    write_manifest(samples, manifest)
    split = create_few_shot_split(samples, 1, 13, ["u"])
    split_path = tmp_path / "split.json"
    split.write(split_path)

    class_names = ("a", "b", "u")
    base = {
        "a": torch.tensor([1.0, 0.0, 0.0, 0.0]),
        "b": torch.tensor([0.0, 1.0, 0.0, 0.0]),
        "u": torch.tensor([-1.0, -1.0, 0.0, 0.0]),
    }
    clip = torch.stack([base[sample.label] + 0.01 * torch.randn(4) for sample in samples])
    patches = torch.stack([base[sample.label].repeat(4, 1) for sample in samples])
    masks = torch.ones(len(samples), 1, 2, 2)
    bundle = tmp_path / "features.pt"
    torch.save(
        {
            "schema_version": 1,
            "class_names": class_names,
            "sample_ids": tuple(sample.sample_id for sample in samples),
            "labels": torch.tensor([class_names.index(sample.label) for sample in samples]),
            "splits": tuple(sample.split for sample in samples),
            "domains": tuple(sample.domain for sample in samples),
            "clip_global": clip,
            "text_prototypes": torch.stack([base[name][None, :] for name in class_names]),
            "dino_patches": patches,
            "lesion_masks": masks,
            "grid_size": (2, 2),
            "backbone_parameters": {
                "clip": {"total": 100, "trainable": 0, "trainable_percent": 0.0},
                "dino": {"total": 50, "trainable": 0, "trainable_percent": 0.0},
            },
        },
        bundle,
    )
    lesion = tmp_path / "lesion.pt"
    train_lesion_decoder(
        bundle,
        lesion,
        epochs=1,
        learning_rate=1e-3,
        hidden_dim=4,
        split_path=split_path,
    )
    config = ProjectConfig(
        dataset=DatasetConfig("tiny", manifest),
        model=ModelConfig(
            use_text=True,
            use_global=True,
            use_lesion=True,
            learned_gate=True,
            prototypes_per_class=1,
            hidden_dim=4,
        ),
        training=TrainingConfig(
            shots=1,
            seeds=(13,),
            epochs=2,
            patience=1,
            learning_rate=1e-3,
        ),
        open_set=OpenSetConfig(enabled=True, unknown_classes=("u",)),
        experiment=ExperimentConfig("synthetic-contract", "E", tmp_path / "runs"),
    )
    unsafe = tmp_path / "unsafe-lesion.pt"
    unsafe_payload = torch.load(lesion, map_location="cpu", weights_only=True)
    unsafe_payload["excluded_classes"] = ()
    torch.save(unsafe_payload, unsafe)
    with pytest.raises(ValueError, match="held-out classes"):
        run_experiment(config, bundle, split_path, lesion_checkpoint=unsafe)
    result = run_experiment(config, bundle, split_path, lesion_checkpoint=lesion)
    assert result["status"] == "measured"
    assert result["threshold_fitted_split"] == "validation"
    assert set(("known_accuracy", "known_macro_f1", "ood_auroc", "ood_fpr95")) <= result.keys()
    assert result["parameters"]["total"] >= 150
