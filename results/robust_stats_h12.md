# Robustness statistics, H = 12, folds [1, 2, 3, 4, 5, 6]

Inflation = % QLIKE reduction of the arm versus its matched control (positive = looks better). Losses are averaged over the training seeds listed. CI = moving-block bootstrap (block 48 origins). RE = DerSimonian-Laird random-effects pooling of the per-fold estimates. Holm correction is applied within each family.

## Original leak channels

| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |
|---|---|---|---|---|---|---|---|
| Ch1 window norm uses horizon | direct | 3/3 | +63.39% | +61.77% to +65.04% | <0.001 | +62.03% [+58.80, +65.26] | 0% |
| Ch1 window norm uses horizon | generative | 3/3 | +62.17% | +60.59% to +63.71% | <0.001 | +61.45% [+57.38, +65.52] | 0% |
| Ch2 merge table on train+test | direct | 3/3 | +0.64% | -0.14% to +1.44% | 0.725 | +0.39% [-0.65, +1.43] | 26% |
| Ch2 merge table on train+test | generative | 3/3 | -2.58% | -4.49% to -0.60% | 0.134 | -1.18% [-6.63, +4.28] | 91% |
| Ch3 decode tables on train+test | generative | 3/3 | +0.24% | +0.11% to +0.39% | 0.038 | +0.20% [+0.06, +0.33] | 44% |
| Ch4 tokenize stream, then split | direct | 3/3 | +0.47% | -0.15% to +1.05% | 0.725 | +0.46% [-0.16, +1.08] | 0% |
| Ch4 tokenize stream, then split | generative | 3/3 | -0.95% | -2.18% to +0.29% | 0.725 | -0.28% [-2.50, +1.93] | 62% |
| Ch5 centered gate FFT | direct | 3/3 | +1.78% | +0.81% to +2.84% | 0.034 | +1.61% [+0.00, +3.21] | 40% |
| Ch5 centered gate FFT | generative | 3/3 | +0.26% | -1.28% to +1.73% | 0.758 | +0.32% [-5.53, +6.18] | 95% |
| Positive control | direct | 3/3 | +99.98% | +99.96% to +99.98% | <0.001 | +99.98% [+95.79, +104.16] | 0% |
| Positive control | generative | 3/3 | +72.22% | +70.68% to +73.64% | <0.001 | +71.60% [+64.09, +79.11] | 44% |

## Partial-token bias

| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |
|---|---|---|---|---|---|---|---|
| Unconstrained BPE (context only) vs hour-aligned | direct | 3/3 | +0.42% | -0.82% to +1.70% | 0.586 | +0.02% [-1.59, +1.63] | 24% |
| Unconstrained BPE (context only) vs hour-aligned | generative | 3/3 | +1.89% | +0.29% to +3.45% | 0.115 | +2.65% [-3.13, +8.42] | 90% |

## Extension: cross-asset leakage (ETH)

| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |
|---|---|---|---|---|---|---|---|
| ETH through test period vs ETH to split | direct | 2/2 | +0.95% | +0.16% to +1.78% | 0.188 | +0.85% [-0.58, +2.27] | 51% |
| ETH through test period vs ETH to split | generative | 2/2 | +1.02% | -0.53% to +2.50% | 0.287 | +1.91% [-2.71, +6.53] | 85% |
| ETH to split vs BTC only (more data, no leak) | direct | 2/3 | +3.06% | +2.21% to +3.91% | <0.001 | +3.37% [+2.13, +4.61] | 46% |
| ETH to split vs BTC only (more data, no leak) | generative | 2/3 | +1.89% | +0.02% to +3.65% | 0.188 | +2.69% [-1.28, +6.66] | 84% |

## Extension: token healing

| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |
|---|---|---|---|---|---|---|---|
| Plain unconstrained vs hour-aligned (healing run) | generative | 1/3 | +0.12% | -2.04% to +2.17% | 0.926 | +1.55% [-10.77, +13.87] | 96% |
| Healed vs plain (same model) | generative | 1/1 | +1.70% | +0.07% to +3.38% | 0.295 | +1.89% [+0.12, +3.66] | 14% |
| Healed unconstrained vs hour-aligned | generative | 1/3 | +1.82% | -0.39% to +4.05% | 0.367 | +3.75% [-8.69, +16.19] | 96% |

## Extension: Ch1 dose-response (future bars in the window)

| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |
|---|---|---|---|---|---|---|---|
| k = 1 future bar | direct | 1/3 | +2.70% | +0.76% to +4.92% | 0.040 | +2.45% [+0.98, +3.92] | 3% |
| k = 3 future bars | direct | 1/3 | +6.95% | +5.31% to +8.58% | <0.001 | +7.39% [+4.03, +10.76] | 67% |
| k = 6 future bars | direct | 1/3 | +18.38% | +13.92% to +22.60% | <0.001 | +17.26% [+13.69, +20.82] | 46% |
| k = 12 future bars (= Ch1) | direct | 3/3 | +63.39% | +61.77% to +65.04% | <0.001 | +62.03% [+58.80, +65.26] | 0% |

## Model Confidence Set (pooled folds, α = 0.1)

| Model | Mean QLIKE | MCS p-value | In MCS |
|---|---|---|---|
| GPT unconstrained BPE, direct | 0.5556 | 1.000 | yes |
| GPT reference, direct | 0.5580 | 0.576 | yes |
| GARCH(1,1) | 0.5747 | 0.042 | no |
| GPT window-norm control, direct | 0.5758 | 0.042 | no |
| GPT reference, generative | 0.6453 | 0.000 | no |
| HAR-RV | 0.6811 | 0.000 | no |

## Model Confidence Set (2025 holdout (already-scored forecasts), α = 0.1)

| Model | Mean QLIKE | MCS p-value | In MCS |
|---|---|---|---|
| GARCH(1,1) | 0.4267 | 1.000 | yes |
| GPT reference, direct | 0.4416 | 0.039 | no |
| GPT unconstrained BPE, direct | 0.4452 | 0.015 | no |
| GPT reference, generative | 0.4603 | 0.004 | no |
| GPT window-norm control, direct | 0.5093 | 0.000 | no |
| HAR-RV | 0.5667 | 0.000 | no |

