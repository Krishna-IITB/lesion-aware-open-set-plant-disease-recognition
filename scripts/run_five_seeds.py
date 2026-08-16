"""Generate split manifests for the configured five-seed protocol."""

from __future__ import annotations

import argparse
from pathlib import Path

from plant_ood.config import load_config
from plant_ood.data.manifest import load_manifest
from plant_ood.data.splits import create_few_shot_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    samples = load_manifest(config.dataset.manifest)
    output = Path(args.output_dir)
    for seed in config.training.seeds:
        split = create_few_shot_split(
            samples, config.training.shots, seed, config.open_set.unknown_classes
        )
        split.write(output / f"shot-{config.training.shots}-seed-{seed}.json")


if __name__ == "__main__":
    main()
