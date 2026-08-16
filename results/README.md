# Historical and repository results

## December 2025–January 2026 historical results

| Outcome | Historical result |
|---|---:|
| Trainable parameters | **0.8%** |
| Lesion segmentation | **0.76 mean binary-lesion Dice** |
| 20-shot lab-to-field macro-F1 | **67%** |
| Improvement | **+6 pp vs same-split MVPDR reproduction** |
| Open-set AUROC | **0.90** |
| FPR95 | **40%** |

These genuine historical project results were supplied by the project author. They are recorded as
`author-supplied historical`; the current reconstruction has not independently reproduced them.

## Current repository status

No real-data experiment has been executed in this repository snapshot. Repository-reproduced and
repository-new recognition, segmentation, and open-set metrics are **not yet evaluated**. Synthetic
smoke-test values are integration diagnostics and must not be reported as research results.

After real runs, promote only reviewed `runs/**/metrics.json` files into a clearly labeled table
that records dataset version, split-manifest digest, seed set, checkpoint, Git commit, and result
provenance.
