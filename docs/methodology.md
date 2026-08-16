# Methodology and project history

## Chronology

The academic project period was **December 2025–January 2026**. DINOv3's technical report and public
model announcement appeared in August 2025, so its use is chronologically plausible for that period.
The present repository is a later reproducibility reconstruction and does not claim the repository
or its Git history existed during that project period. The current PlantSeg archival dataset/paper
citation is later; that bibliographic date is preserved rather than backdated.

## MVPDR-style baseline

Wei et al. encode disease descriptions and disease images with CLIP and represent each disease with
multiple textual and visual prototypes. Their paper and public implementation use per-class visual
clustering, trainable prototype adapters, text mean/max evidence, and visual affinity evidence.

This repository preserves the research idea but makes several explicit engineering deviations:

- Hugging Face `openai/clip-vit-base-patch32` replaces the authors' OpenAI CLIP package/RN101 default.
- Spherical k-means is implemented deterministically in PyTorch.
- Per-view cosine class scores are fused uniformly or by a small learned gate.
- Checkpoint selection is validation-only; no test metric selects an epoch.
- Split manifests and feature provenance are mandatory.

Consequently, results must be labeled “repository reproduction” only after actual execution and
must not be compared as if this were the authors' identical code path.

## Local representation and lesion supervision

The frozen `facebook/dinov3-vits16-pretrain-lvd1689m` checkpoint is selected because ViT-S/16 is the
smallest published DINOv3 ViT (about 21M backbone parameters per its model card) and exposes dense
tokens. A small lesion decoder learns spatial disease evidence while the backbone stays frozen.
Predicted masks pool patch features; target masks never enter classification inference.

## Historical project evaluation

The December 2025–January 2026 evaluation covered parameter-efficient three-view fusion, binary
lesion localization, 20-shot lab-to-field recognition against a same-split MVPDR reproduction, and
held-out-class open-set detection. The author-supplied outcomes are consolidated in
[`results/README.md`](../results/README.md); this reconstruction has not independently rerun them.

## Few-shot and lab-to-field protocol

For every seed and shot count, only source-domain training samples construct visual/lesion
prototypes and train the gate. Validation controls early stopping, temperature, and OOD threshold.
The target-domain test split is evaluated once. Class mappings are explicit, and unknown diseases
are held out at class level. Report individual seeds plus mean and standard deviation across the
five fixed seeds; do not select the best seed.

## Ablations

The same code supports disabling text/global/lesion branches, uniform versus learned fusion,
globally pooled versus predicted-lesion DINO tokens, and energy versus prototype-distance rejection.
SigLIP 2, SAM 2, prompt tuning, conformal prediction, retrieval, deployment, and broad corruption
tests are intentionally outside the required scope.
