# Repository instructions

This repository implements a reproducible, student-scale extension of the ACM MM'24 MVPDR plant
disease baseline. Keep the core scope to CLIP text/global prototypes, frozen DINOv3 dense features,
a lightweight lesion decoder, three-view gating, and simple open-set rejection.

## Non-negotiable research integrity

- Never fabricate, estimate, or silently copy metrics. Mark unexecuted work `not yet evaluated`.
- Label external values `paper-reported`; label repository results with their artifact provenance.
- Never tune epochs, temperature, prototypes, mappings, or rejection thresholds on test data.
- Ground-truth test masks are evaluation data and may not enter classification inference.
- Keep split manifests deterministic and preserve their source-manifest digest.
- Never commit datasets, weights, feature caches, checkpoints, credentials, or large run artifacts.
- The academic project period is September–November 2025. Do not backdate Git/files or imply this
  repository existed then; it is a later reproducibility reconstruction.

## Structure and conventions

Put reusable code under `src/plant_ood`, experiments under `configs/experiments`, data taxonomy
metadata under `configs/datasets`, tests under `tests`, and generated products under ignored
`runs`, `features`, or `checkpoints`. Use typed, small modules and explicit errors. Do not add
FieldGuard extras (retrieval, conformal prediction, deployment, multiple backbones) by default.

Any new metric, sampler, cache key, pooling rule, or threshold logic needs a focused synthetic test.
README commands must be real CLI commands. Optional external dependencies must fail with a useful
setup message and must not be required by CPU CI.

Before completing changes, run:

```bash
ruff check .
ruff format --check .
mypy src
pytest
plant-ood smoke-test
plant-ood --help
```

Inspect `git diff`, tracked file sizes, and likely credential patterns before committing.
