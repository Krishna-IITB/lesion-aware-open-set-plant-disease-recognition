"""Typed configuration loading and cross-field validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration is internally inconsistent."""


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    manifest: Path
    source_domain: str = "lab"
    target_domain: str = "field"
    class_map: Path | None = None


@dataclass(frozen=True)
class ModelConfig:
    clip_id: str = "openai/clip-vit-base-patch32"
    dino_id: str = "facebook/dinov3-vits16-pretrain-lvd1689m"
    use_text: bool = True
    use_global: bool = True
    use_lesion: bool = True
    learned_gate: bool = True
    prototypes_per_class: int = 1
    hidden_dim: int = 32


@dataclass(frozen=True)
class TrainingConfig:
    shots: int = 5
    seeds: tuple[int, ...] = (13, 37, 71, 101, 137)
    batch_size: int = 32
    epochs: int = 30
    learning_rate: float = 1e-3
    patience: int = 5
    device: str = "auto"


@dataclass(frozen=True)
class OpenSetConfig:
    enabled: bool = True
    method: str = "hybrid"
    energy_weight: float = 0.5
    temperature_scaling: bool = True
    unknown_classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class ExperimentConfig:
    name: str
    variant: str
    output_dir: Path = Path("runs")


@dataclass(frozen=True)
class ProjectConfig:
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    open_set: OpenSetConfig
    experiment: ExperimentConfig
    source_path: Path = field(default=Path("<memory>"), repr=False)

    def validate(self) -> None:
        if self.training.shots not in {1, 5, 10, 20}:
            raise ConfigError("training.shots must be one of 1, 5, 10, or 20")
        if not self.training.seeds or len(set(self.training.seeds)) != len(self.training.seeds):
            raise ConfigError("training.seeds must be a non-empty sequence of unique integers")
        if self.model.prototypes_per_class < 1:
            raise ConfigError("model.prototypes_per_class must be positive")
        if not any((self.model.use_text, self.model.use_global, self.model.use_lesion)):
            raise ConfigError("at least one evidence view must be enabled")
        if (
            self.model.learned_gate
            and sum((self.model.use_text, self.model.use_global, self.model.use_lesion)) < 2
        ):
            raise ConfigError("learned gating requires at least two enabled views")
        if self.open_set.method not in {"energy", "prototype_distance", "hybrid"}:
            raise ConfigError("open_set.method must be energy, prototype_distance, or hybrid")
        if not 0.0 <= self.open_set.energy_weight <= 1.0:
            raise ConfigError("open_set.energy_weight must lie in [0, 1]")
        if self.training.epochs < 1 or self.training.batch_size < 1:
            raise ConfigError("epochs and batch_size must be positive")


def _require_mapping(raw: object, key: str) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise ConfigError(f"{key} must be a mapping")
    return raw


def load_config(path: str | Path) -> ProjectConfig:
    """Load a YAML configuration, resolve its paths, and validate it."""
    source = Path(path).resolve()
    with source.open(encoding="utf-8") as handle:
        root = _require_mapping(yaml.safe_load(handle), "configuration")

    for section in ("dataset", "model", "training", "open_set", "experiment"):
        if section not in root:
            raise ConfigError(f"missing required section: {section}")

    dataset_raw = _require_mapping(root["dataset"], "dataset").copy()
    dataset_raw["manifest"] = _resolve_path(source, dataset_raw["manifest"])
    if dataset_raw.get("class_map"):
        dataset_raw["class_map"] = _resolve_path(source, dataset_raw["class_map"])
    training_raw = _require_mapping(root["training"], "training").copy()
    training_raw["seeds"] = tuple(int(seed) for seed in training_raw.get("seeds", ()))
    open_raw = _require_mapping(root["open_set"], "open_set").copy()
    open_raw["unknown_classes"] = tuple(open_raw.get("unknown_classes", ()))
    experiment_raw = _require_mapping(root["experiment"], "experiment").copy()
    experiment_raw["output_dir"] = _resolve_path(source, experiment_raw.get("output_dir", "runs"))

    try:
        config = ProjectConfig(
            dataset=DatasetConfig(**dataset_raw),
            model=ModelConfig(**_require_mapping(root["model"], "model")),
            training=TrainingConfig(**training_raw),
            open_set=OpenSetConfig(**open_raw),
            experiment=ExperimentConfig(**experiment_raw),
            source_path=source,
        )
    except TypeError as error:
        raise ConfigError(f"invalid configuration field: {error}") from error
    config.validate()
    return config


def _resolve_path(config_path: Path, value: object) -> Path:
    path = Path(str(value))
    if path.is_absolute():
        return path
    # Repository configs live at configs/**; paths are intentionally repository-relative.
    repository = config_path.parent
    while repository.parent != repository and not (repository / "pyproject.toml").exists():
        repository = repository.parent
    return (repository / path).resolve()
