# Project timeline: December 2025-January 2026

This timeline summarizes the academic project period at phase level. It is a retrospective account
of the work and outcomes; it is not a reconstructed commit log or a claim that this Git repository
existed during the project period. The current repository is a later reproducibility reconstruction.

## December 2025 - problem definition and core recognition system

### Early December: scope and evaluation design

- Defined a student-scale extension of the ACM MM'24 MVPDR plant-disease baseline.
- Fixed the core scope: CLIP textual/global evidence, frozen DINOv3 dense features, lesion
  supervision, three-view fusion, and open-set rejection.
- Established the few-shot lab-to-field setting, including 1/5/10/20-shot evaluation, class-held-out
  unknown diseases, validation-only calibration, and leakage-safe split requirements.

### Mid December: MVPDR-style baseline and local features

- Structured the CLIP text-prototype and global visual-prototype branches.
- Added frozen DINOv3 patch features to represent local lesion texture, colour, and shape while
  limiting the number of trainable parameters.
- Defined feature caching and provenance requirements so backbone, preprocessing, split, and image
  identity could be tracked consistently.

### Late December: lesion supervision

- Added a lightweight lesion decoder over frozen DINOv3 features.
- Used binary lesion masks for BCE + Dice supervision and predicted-mask pooling for classification.
- Kept ground-truth test masks strictly outside classification inference.

## January 2026 - fusion, open-set evaluation, and consolidation

### Early January: three-view gated fusion

- Combined textual prototypes, global visual prototypes, and lesion-localized prototypes.
- Added a small per-image gate to weight the three evidence views.
- Maintained the parameter-efficient design, with **0.8% trainable parameters** in the historical
  project result.

### Mid January: open-set rejection and evaluation

- Added temperature calibration and validation-selected energy/prototype-distance rejection.
- Evaluated recognition under 20-shot lab-to-field transfer and held-out disease classes.
- Recorded the historical outcomes supplied by the project author:
  - **0.76 mean binary-lesion Dice**;
  - **67% 20-shot lab-to-field macro-F1**;
  - **+6 pp vs same-split MVPDR reproduction**;
  - **0.90 AUROC and 40% FPR95** for open-set disease detection.

### Late January: analysis and project handoff

- Consolidated the A-E experimental progression: baseline, DINOv3 local features, lesion
  supervision, three-view fusion, and open-set rejection.
- Documented method boundaries, provenance categories, and research-integrity constraints.
- Closed the academic project period in January 2026.

## Repository status

The Git repository and its commit history reflect the later cleaned implementation and
reproducibility work. The historical results above are `author-supplied historical`; the current
repository has not independently rerun the corresponding real-data experiments.

Regenerate the PDF edition from the repository root with:

```bash
uv run --extra docs python scripts/generate_project_timeline_pdf.py
```
