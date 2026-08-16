"""Command-line interface for preparation, extraction, training, and evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from plant_ood.config import load_config
from plant_ood.data.manifest import load_manifest, write_manifest
from plant_ood.data.mapping import apply_class_mapping, load_class_mapping
from plant_ood.data.prepare import scan_class_folders, validate_dataset
from plant_ood.data.splits import create_few_shot_split
from plant_ood.pipeline import extract_feature_bundle, run_experiment, train_lesion_decoder
from plant_ood.smoke import run_synthetic_smoke_test


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="plant-ood", description="Lesion-aware open-set plant disease research CLI"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_config = subparsers.add_parser("validate-config", help="validate an experiment YAML")
    validate_config.add_argument("--config", required=True)

    prepare = subparsers.add_parser("prepare-data", help="index split/class image folders")
    prepare.add_argument("--root", required=True)
    prepare.add_argument("--domain", required=True)
    prepare.add_argument("--masks-root")
    prepare.add_argument("--output", required=True)

    validate_data = subparsers.add_parser("validate-data", help="check manifest files and leakage")
    validate_data.add_argument("--manifest", required=True)
    validate_data.add_argument("--skip-files", action="store_true")

    mapping = subparsers.add_parser("map-manifest", help="apply an explicit class taxonomy map")
    mapping.add_argument("--manifest", required=True)
    mapping.add_argument("--mapping", required=True)
    mapping.add_argument("--output", required=True)

    split = subparsers.add_parser("make-splits", help="persist a reproducible few/open-set split")
    split.add_argument("--manifest", required=True)
    split.add_argument("--shots", type=int, required=True)
    split.add_argument("--seed", type=int, required=True)
    split.add_argument("--unknown", nargs="*", default=[])
    split.add_argument("--output", required=True)

    extract = subparsers.add_parser("extract-features", help="cache frozen CLIP/DINOv3 features")
    extract.add_argument("--config", required=True)
    extract.add_argument("--output", required=True)
    extract.add_argument("--prompts")

    lesion = subparsers.add_parser("train-lesion", help="train the lightweight lesion decoder")
    lesion.add_argument("--features", required=True)
    lesion.add_argument("--output", required=True)
    lesion.add_argument("--epochs", type=int, default=30)
    lesion.add_argument("--learning-rate", type=float, default=1e-3)
    lesion.add_argument("--hidden-dim", type=int, default=128)
    lesion.add_argument(
        "--split", help="split manifest whose unknown classes must be excluded from supervision"
    )

    experiment = subparsers.add_parser("run-experiment", help="train heads and evaluate a split")
    experiment.add_argument("--config", required=True)
    experiment.add_argument("--features", required=True)
    experiment.add_argument("--split", required=True)
    experiment.add_argument("--lesion-checkpoint")

    subparsers.add_parser("smoke-test", help="run the synthetic end-to-end integration check")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = _dispatch(args)
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _dispatch(args: argparse.Namespace) -> dict[str, Any] | None:
    if args.command == "validate-config":
        config = load_config(args.config)
        return {"status": "valid", "experiment": config.experiment.name}
    if args.command == "prepare-data":
        samples = scan_class_folders(args.root, domain=args.domain, masks_root=args.masks_root)
        write_manifest(samples, args.output)
        return validate_dataset(samples)
    if args.command == "validate-data":
        return validate_dataset(load_manifest(args.manifest), check_files=not args.skip_files)
    if args.command == "map-manifest":
        samples = load_manifest(args.manifest)
        mapping = load_class_mapping(args.mapping)
        mapped = []
        for sample in samples:
            target = apply_class_mapping(sample.label, mapping)
            if target is not None:
                mapped.append(replace(sample, label=target))
        if not mapped:
            raise ValueError("class mapping excluded every sample")
        write_manifest(mapped, args.output)
        return {"input_samples": len(samples), "output_samples": len(mapped)}
    if args.command == "make-splits":
        samples = load_manifest(args.manifest)
        split = create_few_shot_split(samples, args.shots, args.seed, args.unknown)
        split.write(args.output)
        return {
            "output": str(Path(args.output).resolve()),
            "prototype_samples": len(split.prototype_ids),
            "known_classes": len(split.known_classes),
            "unknown_classes": len(split.unknown_classes),
        }
    if args.command == "extract-features":
        extract_feature_bundle(load_config(args.config), args.output, prompts_path=args.prompts)
        return {"output": str(Path(args.output).resolve())}
    if args.command == "train-lesion":
        return train_lesion_decoder(
            args.features,
            args.output,
            epochs=args.epochs,
            learning_rate=args.learning_rate,
            hidden_dim=args.hidden_dim,
            split_path=args.split,
        )
    if args.command == "run-experiment":
        return run_experiment(
            load_config(args.config),
            args.features,
            args.split,
            lesion_checkpoint=args.lesion_checkpoint,
        )
    if args.command == "smoke-test":
        return run_synthetic_smoke_test()
    raise AssertionError(f"unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
