from pathlib import Path

import pytest

from plant_ood.config import ConfigError, load_config


def test_all_experiment_configs_validate() -> None:
    for path in Path("configs/experiments").glob("*.yaml"):
        config = load_config(path)
        assert config.training.seeds == (13, 37, 71, 101, 137)
        assert config.dataset.manifest.is_absolute()


def test_invalid_shots_fail(tmp_path: Path) -> None:
    source = Path("configs/experiments/e_open_set.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source.replace("shots: 5", "shots: 3"))
    with pytest.raises(ConfigError, match="shots"):
        load_config(path)
