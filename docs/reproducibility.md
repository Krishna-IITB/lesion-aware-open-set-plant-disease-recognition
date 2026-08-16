# Reproducibility

## Stored artifacts

- Data manifest: stable sample IDs, paths, class, split, domain, optional mask.
- Split manifest: seed, shots, exact prototype/validation/test IDs, known/unknown classes, SHA-256
  digest of the source manifest.
- Feature bundle/cache: image digest, model/checkpoint, preprocessing, representation, split, and
  dataset provenance.
- Checkpoint: schema version, epoch, model and optimizer state, metadata.
- Run directory: metrics plus config/split/feature paths, model IDs, Git SHA, Python/PyTorch/platform.

The large artifacts are ignored by Git. Archive them in approved research storage and record their
checksums if results are published.

## Determinism

Python, NumPy, CPU/CUDA PyTorch, few-shot sampling, and k-means initialization are seeded. PyTorch
deterministic algorithms are requested with warnings. Exact bitwise reproduction is not guaranteed
across CUDA, driver, hardware, and library versions; record the environment. Five-seed reporting is
the primary uncertainty summary.

## Leakage audit

1. Manifest validation rejects duplicate IDs and content duplicated across splits.
2. Prototypes can reference only `train` samples from known classes.
3. Open-set lesion checkpoints record and exclude every held-out disease class from supervision.
4. Gate optimization uses prototype/training IDs and early-stops on known validation IDs.
5. Temperature uses known validation data.
6. Rejection thresholds require the literal split name `validation`.
7. Test images are scored only after all fitted values are frozen.
8. Ground-truth lesion masks are absent from recognition inference.
