# Experiment protocol

## Progression

| ID | Configuration | Required checkpoint |
|---|---|---|
| A | CLIP text + global prototypes | none |
| B | A + mean-pooled frozen DINO tokens | none |
| C | B + predicted-lesion DINO pooling | lesion decoder |
| D | C + learned per-image three-view gate | lesion decoder |
| E | D + temperature and open-set rejection | lesion decoder |

Run A–E with the matching files under `configs/experiments/`. For 1/5/10/20 shots, change only the
shot field and generate a new persisted split for every configured seed. Never regenerate a split
under an existing run identifier.

## Evaluation contract

- recognition: known-class accuracy and macro-F1;
- segmentation: Dice and mIoU, including a defined score for empty-empty masks;
- open set: AUROC and known-sample FPR at 95% unknown recall;
- efficiency: total, trainable, and trainable-percent parameters;
- interpretability: mean and per-sample gate weights.

Aggregate five seeds with mean, sample standard deviation, and all individual values. A real result
row must include dataset versions, mapping, shot count, unknown classes, split digests, checkpoint,
backbones, and Git SHA.

## Current status

Synthetic unit/integration checks have been designed to run on CPU. They validate interfaces and
numerics but are not experiments. Author-supplied results from the September–November 2025 project
are recorded in [`results/README.md`](../results/README.md). Real-data A–E runs are **not yet
evaluated by this reconstruction** because data, model access, and GPU execution are external.
