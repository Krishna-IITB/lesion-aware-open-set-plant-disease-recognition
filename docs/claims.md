# Claims and provenance

| Claim type | Status | Allowed wording |
|---|---|---|
| Published MVPDR findings | paper-reported only | Attribute explicitly to Wei et al. (2024) |
| MVPDR reproduction in this repository | not yet evaluated | Do not report a metric |
| DINOv3 local-feature extension | implemented; not yet evaluated | Describe design, not improvement |
| PlantSeg lesion decoder | implemented; not yet evaluated | Describe BCE+Dice and predicted-mask pooling |
| Three-view gate | implemented; not yet evaluated | Describe inspectable weights, not superiority |
| Open-set rejection | implemented; not yet evaluated | Describe held-out protocol, not AUROC/FPR95 |

## Résumé template—do not fill without artifacts

- Extended an MVPDR-style baseline with frozen DINOv3 lesion features and a PlantSeg-supervised
  decoder.
- Designed three-view gated fusion using `[verified trainable %]` trainable parameters.
- Changed 20-shot target-domain macro-F1 by `[verified pp]` against the repository baseline across
  five seeds.
- Added open-set abstention with `[verified AUROC] / [verified FPR95]` on held-out diseases.

Every bracket must be replaced from reviewed run artifacts or the bullet must be omitted. Synthetic
smoke metrics, paper-reported numbers, and planned experiments cannot support résumé claims.
