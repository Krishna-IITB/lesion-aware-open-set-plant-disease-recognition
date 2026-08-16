# Architecture

## Tensor contract

For batch size `B`, known classes `C`, CLIP dimension `Dc`, DINO dimension `Dd`, and DINO patch
grid `H × W`:

| Stage | Shape | Notes |
|---|---:|---|
| CLIP image | `B × Dc` | L2-normalized global query |
| CLIP text bank | `C × Pt × Dc` | `Pt` descriptions per disease |
| CLIP visual bank | `C × Pv × Dc` | deterministic spherical k-means |
| DINO patches | `B × (H·W) × Dd` | frozen dense tokens, CLS/register tokens removed |
| lesion logits | `B × 1 × H × W` | lightweight pointwise decoder |
| lesion vector | `B × Dd` | soft mask-weighted and L2-normalized |
| view scores | `B × 3 × C` | text, global, lesion cosine logits |
| gate weights | `B × 3` | softmax; sum is one per sample |
| fused logits | `B × C` | weighted view sum |

The gate consumes each view's maximum class score and top-1/top-2 margin. This makes the decision
trace inspectable without a large cross-modal network. Uniform fusion is available by disabling the
learned gate.

## Lesion behavior

The decoder applies LayerNorm to each DINO token, reshapes tokens to the patch grid, and uses two
1×1 convolutions with GELU. Training combines binary cross entropy and soft Dice. Masks use nearest
neighbor resizing; decoder logits use bilinear resizing. Images without masks are skipped for
supervised decoder training. At inference, an effectively empty predicted mask falls back to mean
patch pooling, avoiding NaNs and unstable division.

## Open-set decision

The recognition head emits known-class logits. Temperature is fitted on known validation samples.
Novelty is either negative log-sum-exp energy, one minus maximum prototype similarity, or a
standardized weighted hybrid. A threshold maximizing validation balanced known/unknown accuracy is
persisted and then frozen for test evaluation. Larger novelty means `unknown / abstain`.
