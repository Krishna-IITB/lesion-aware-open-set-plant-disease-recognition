# Model card

## Intended use

Research on few-shot and open-set plant disease image recognition, particularly controlled-to-field
shift and interpretable fusion of textual, global, and lesion evidence. It is not a diagnostic tool
and must not drive treatment decisions.

## Components

Frozen CLIP ViT-B/32 and DINOv3 ViT-S/16 backbones feed prototype classifiers, a lightweight lesion
decoder, and a small gate. The backbones retain their upstream terms. Repository code is MIT; this
does not relicense pretrained weights or datasets.

## Training and evaluation status

The software path is implemented and covered by synthetic tests. No real-data model card metrics or
trained checkpoint are distributed. Accuracy, macro-F1, lesion Dice/mIoU, AUROC, FPR95, subgroup
behavior, and field reliability are **not yet evaluated**.

## Risks

Symptoms can be visually ambiguous; class taxonomies differ across datasets; lighting/background,
crop, geography, and device shift can cause errors; open-set scores provide no safety guarantee.
Inspect rejection behavior, class mappings, gate weights, and failure cases before research use.
