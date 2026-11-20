# Look-ahead inflation, H = 12 bars, folds [1, 2, 3, 4, 5, 6], seed 0

Inflation = % reduction in QLIKE versus the matched control (positive = the leak makes the forecast look better). Pooled over all test origins of the listed folds.

## Direct head

| Arm | Control | Inflation | DM p | Holm p | GW p | TOST equivalent (±5%) | Folds with gain | Folds significant |
|---|---|---|---|---|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +63.67% | <0.001 | <0.001 | <0.001 | no | 6/6 | 6/6 |
| Ch2: merge table on train+test | reference | +2.06% | 0.038 | 0.075 | 0.089 | yes | 3/6 | 0/6 |
| Ch4: tokenize stream, then split | ch4_control_context_only | +0.42% | 0.469 | 0.469 | 0.452 | yes | 3/6 | 0/6 |
| Ch5: centered (non-causal) gate FFT | reference | +3.29% | 0.002 | 0.006 | 0.006 | no | 5/6 | 2/6 |
| Positive control: true future RV norm | reference | +99.96% | <0.001 | <0.001 | <0.001 | no | 6/6 | 6/6 |
| Partial-token bias (Ch4 control vs reference) | reference | +1.82% | 0.148 | – | 0.191 | yes | – | – |

| Model | Pooled QLIKE | 95% CI |
|---|---|---|
| har_rv | 0.6811 | 0.6479 – 0.7206 |
| garch_11 | 0.5747 | 0.5439 – 0.6092 |
| GPT: Causal reference (leak-free) | 0.5651 | 0.5314 – 0.6000 |
| GPT: Ch1 control: window norm, causal | 0.5820 | 0.5477 – 0.6250 |
| GPT: Ch1: window norm uses horizon | 0.2114 | 0.2023 – 0.2204 |
| GPT: Ch2: merge table on train+test | 0.5535 | 0.5232 – 0.5848 |
| GPT: Ch3: decode tables on train+test | 0.5651 | 0.5314 – 0.6000 |
| GPT: Ch4 control: unconstrained BPE, context only | 0.5548 | 0.5260 – 0.5829 |
| GPT: Ch4: tokenize stream, then split | 0.5525 | 0.5223 – 0.5832 |
| GPT: Ch5: centered (non-causal) gate FFT | 0.5465 | 0.5153 – 0.5781 |
| GPT: Positive control: true future RV norm | 0.0002 | 0.0001 – 0.0004 |

- Causal reference (leak-free) vs har_rv: +17.04% (better), DM p <0.001

- Causal reference (leak-free) vs garch_11: +1.67% (better), DM p 0.326

- Ch1 control: window norm, causal vs har_rv: +14.55% (better), DM p <0.001

- Ch1 control: window norm, causal vs garch_11: -1.27% (worse), DM p 0.454

- Ch4 control: unconstrained BPE, context only vs har_rv: +18.54% (better), DM p <0.001

- Ch4 control: unconstrained BPE, context only vs garch_11: +3.45% (better), DM p 0.006

## Generative head

| Arm | Control | Inflation | DM p | Holm p | GW p | TOST equivalent (±5%) | Folds with gain | Folds significant |
|---|---|---|---|---|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +65.72% | <0.001 | <0.001 | <0.001 | no | 6/6 | 6/6 |
| Ch2: merge table on train+test | reference | -2.65% | 0.082 | 0.163 | 0.218 | no | 3/6 | 1/6 |
| Ch3: decode tables on train+test | reference | +0.13% | 0.040 | 0.121 | 0.070 | yes | 3/6 | 1/6 |
| Ch4: tokenize stream, then split | ch4_control_context_only | -3.21% | 0.011 | 0.045 | 0.014 | no | 1/6 | 0/6 |
| Ch5: centered (non-causal) gate FFT | reference | +2.37% | 0.239 | 0.239 | 0.262 | no | 2/6 | 2/6 |
| Positive control: true future RV norm | reference | +61.50% | <0.001 | <0.001 | <0.001 | no | 6/6 | 6/6 |
| Partial-token bias (Ch4 control vs reference) | reference | +1.01% | 0.570 | – | 0.851 | yes | – | – |

| Model | Pooled QLIKE | 95% CI |
|---|---|---|
| har_rv | 0.6811 | 0.6479 – 0.7206 |
| garch_11 | 0.5747 | 0.5439 – 0.6092 |
| GPT: Causal reference (leak-free) | 0.6366 | 0.5959 – 0.6780 |
| GPT: Ch1 control: window norm, causal | 0.7304 | 0.6896 – 0.7771 |
| GPT: Ch1: window norm uses horizon | 0.2504 | 0.2411 – 0.2592 |
| GPT: Ch2: merge table on train+test | 0.6534 | 0.6059 – 0.7054 |
| GPT: Ch3: decode tables on train+test | 0.6357 | 0.5952 – 0.6776 |
| GPT: Ch4 control: unconstrained BPE, context only | 0.6301 | 0.5949 – 0.6688 |
| GPT: Ch4: tokenize stream, then split | 0.6504 | 0.6099 – 0.6949 |
| GPT: Ch5: centered (non-causal) gate FFT | 0.6215 | 0.5735 – 0.6745 |
| GPT: Positive control: true future RV norm | 0.2451 | 0.2384 – 0.2519 |

- Causal reference (leak-free) vs har_rv: +6.54% (better), DM p 0.004

- Causal reference (leak-free) vs garch_11: -10.77% (worse), DM p <0.001

- Ch1 control: window norm, causal vs har_rv: -7.23% (worse), DM p 0.003

- Ch1 control: window norm, causal vs garch_11: -27.09% (worse), DM p <0.001

- Ch4 control: unconstrained BPE, context only vs har_rv: +7.49% (better), DM p 0.001

- Ch4 control: unconstrained BPE, context only vs garch_11: -9.65% (worse), DM p <0.001

## Seed robustness

fold_1, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +63.29%, +60.18%, +62.68% | +62.05 ± 1.65% |
| Ch2: merge table on train+test | reference | -0.25%, -1.50%, +4.57% | +0.94 ± 3.21% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -2.22%, +1.89%, -2.99% | -1.10 ± 2.62% |
| Ch5: centered (non-causal) gate FFT | reference | +1.68%, -0.32%, +5.44% | +2.26 ± 2.92% |
| Positive control: true future RV norm | reference | +99.98%, +99.99%, +99.99% | +99.99 ± 0.00% |
| Ch4 control: unconstrained BPE, context only | reference | -0.50%, -6.09%, +1.66% | -1.64 ± 4.00% |

fold_1, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +57.52%, +53.31%, +54.48% | +55.10 ± 2.18% |
| Ch2: merge table on train+test | reference | -11.97%, -5.08%, +14.80% | -0.75 ± 13.90% |
| Ch3: decode tables on train+test | reference | +0.03%, +0.39%, +0.01% | +0.14 ± 0.22% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -0.99%, -4.94%, +8.95% | +1.00 ± 7.16% |
| Ch5: centered (non-causal) gate FFT | reference | -11.07%, -17.78%, +11.85% | -5.67 ± 15.53% |
| Positive control: true future RV norm | reference | +59.15%, +81.58%, +69.00% | +69.91 ± 11.24% |
| Ch4 control: unconstrained BPE, context only | reference | -10.31%, +15.75%, +14.07% | +6.50 ± 14.58% |

fold_2, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +44.13%, +63.84%, +58.44% | +55.47 ± 10.19% |
| Ch2: merge table on train+test | reference | -0.27%, +0.41%, -1.88% | -0.58 ± 1.18% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -0.97%, +3.38%, +0.80% | +1.07 ± 2.19% |
| Ch5: centered (non-causal) gate FFT | reference | +2.28%, -2.86%, -9.35% | -3.31 ± 5.83% |
| Positive control: true future RV norm | reference | +99.97%, +99.96%, +99.99% | +99.98 ± 0.01% |
| Ch4 control: unconstrained BPE, context only | reference | +0.09%, -5.79%, -2.91% | -2.87 ± 2.94% |

fold_2, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +59.04%, +56.46%, +52.09% | +55.86 ± 3.51% |
| Ch2: merge table on train+test | reference | +7.34%, +16.19%, -1.13% | +7.47 ± 8.66% |
| Ch3: decode tables on train+test | reference | -0.17%, +0.02%, +1.94% | +0.60 ± 1.17% |
| Ch4: tokenize stream, then split | ch4_control_context_only | +2.82%, +4.96%, +7.48% | +5.09 ± 2.33% |
| Ch5: centered (non-causal) gate FFT | reference | -3.22%, +10.92%, +6.13% | +4.61 ± 7.19% |
| Positive control: true future RV norm | reference | +65.55%, +86.87%, +79.17% | +77.20 ± 10.80% |
| Ch4 control: unconstrained BPE, context only | reference | -6.83%, +23.50%, -2.51% | +4.72 ± 16.41% |

fold_3, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +57.69%, +64.24%, +59.71% | +60.55 ± 3.35% |
| Ch2: merge table on train+test | reference | +1.04%, +3.40%, -3.17% | +0.42 ± 3.33% |
| Ch4: tokenize stream, then split | ch4_control_context_only | +0.89%, -2.46%, +1.79% | +0.07 ± 2.24% |
| Ch5: centered (non-causal) gate FFT | reference | -0.28%, +2.13%, -0.97% | +0.29 ± 1.63% |
| Positive control: true future RV norm | reference | +99.88%, +99.96%, +99.97% | +99.93 ± 0.05% |
| Ch4 control: unconstrained BPE, context only | reference | +0.75%, +2.17%, -5.20% | -0.76 ± 3.91% |

fold_3, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +61.33%, +61.20%, +58.40% | +60.31 ± 1.65% |
| Ch2: merge table on train+test | reference | -13.24%, -3.94%, -29.82% | -15.67 ± 13.11% |
| Ch3: decode tables on train+test | reference | -0.12%, +0.35%, -0.16% | +0.02 ± 0.28% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -2.98%, -8.37%, +5.12% | -2.08 ± 6.79% |
| Ch5: centered (non-causal) gate FFT | reference | +5.53%, +1.98%, -5.61% | +0.63 ± 5.69% |
| Positive control: true future RV norm | reference | +47.18%, +80.71%, +77.39% | +68.43 ± 18.48% |
| Ch4 control: unconstrained BPE, context only | reference | +0.62%, +10.14%, -9.66% | +0.36 ± 9.90% |

fold_4, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +65.51%, +61.37%, +63.76% | +63.55 ± 2.08% |
| Ch2: merge table on train+test | reference | +3.16%, +3.88%, -1.37% | +1.89 ± 2.85% |
| Ch4: tokenize stream, then split | ch4_control_context_only | +1.37%, +0.27%, -1.04% | +0.20 ± 1.20% |
| Ch5: centered (non-causal) gate FFT | reference | +5.44%, +2.64%, +1.83% | +3.30 ± 1.89% |
| Positive control: true future RV norm | reference | +99.99%, +99.98%, +99.99% | +99.99 ± 0.00% |
| Ch4 control: unconstrained BPE, context only | reference | +5.09%, +4.79%, -0.19% | +3.23 ± 2.96% |

fold_4, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +66.64%, +58.07%, +60.16% | +61.62 ± 4.47% |
| Ch2: merge table on train+test | reference | +4.45%, +15.68%, -6.86% | +4.42 ± 11.27% |
| Ch3: decode tables on train+test | reference | +0.32%, +0.31%, -0.43% | +0.06 ± 0.43% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -3.14%, -2.32%, -6.22% | -3.90 ± 2.06% |
| Ch5: centered (non-causal) gate FFT | reference | +19.33%, +10.88%, -0.92% | +9.76 ± 10.17% |
| Positive control: true future RV norm | reference | +79.54%, +89.93%, +71.60% | +80.36 ± 9.19% |
| Ch4 control: unconstrained BPE, context only | reference | +18.97%, +15.49%, -7.93% | +8.84 ± 14.63% |

fold_5, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +74.19%, +71.85%, +67.12% | +71.05 ± 3.60% |
| Ch2: merge table on train+test | reference | +6.32%, -1.42%, -0.55% | +1.45 ± 4.24% |
| Ch4: tokenize stream, then split | ch4_control_context_only | +2.56%, +2.47%, +0.32% | +1.78 ± 1.27% |
| Ch5: centered (non-causal) gate FFT | reference | +5.57%, +3.08%, +2.90% | +3.85 ± 1.49% |
| Positive control: true future RV norm | reference | +99.98%, +99.99%, +99.99% | +99.99 ± 0.00% |
| Ch4 control: unconstrained BPE, context only | reference | +3.62%, +0.71%, +1.76% | +2.03 ± 1.48% |

fold_5, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +72.61%, +66.27%, +66.18% | +68.35 ± 3.69% |
| Ch2: merge table on train+test | reference | -6.57%, -8.58%, -12.69% | -9.28 ± 3.12% |
| Ch3: decode tables on train+test | reference | -0.05%, +1.03%, +0.05% | +0.35 ± 0.60% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -8.52%, +6.01%, +2.38% | -0.04 ± 7.56% |
| Ch5: centered (non-causal) gate FFT | reference | -11.98%, +0.04%, -2.98% | -4.97 ± 6.25% |
| Positive control: true future RV norm | reference | +74.17%, +78.02%, +69.86% | +74.02 ± 4.08% |
| Ch4 control: unconstrained BPE, context only | reference | -4.17%, +7.65%, -10.00% | -2.17 ± 8.99% |

fold_6, direct head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +64.48%, +61.13%, +60.82% | +62.14 ± 2.03% |
| Ch2: merge table on train+test | reference | -0.26%, -2.12%, +0.18% | -0.74 ± 1.22% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -1.17%, +1.26%, +0.49% | +0.19 ± 1.24% |
| Ch5: centered (non-causal) gate FFT | reference | +3.02%, +0.81%, +0.04% | +1.29 ± 1.55% |
| Positive control: true future RV norm | reference | +99.97%, +99.99%, +99.99% | +99.98 ± 0.01% |
| Ch4 control: unconstrained BPE, context only | reference | -0.41%, -3.28%, +1.22% | -0.82 ± 2.27% |

fold_6, generative head, seeds [0, 1, 2]:

| Arm | Control | Inflation per seed | Mean ± sd |
|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +67.24%, +60.97%, +59.75% | +62.65 ± 4.02% |
| Ch2: merge table on train+test | reference | +0.56%, -0.70%, +0.40% | +0.09 ± 0.69% |
| Ch3: decode tables on train+test | reference | +0.40%, +0.15%, +0.40% | +0.32 ± 0.14% |
| Ch4: tokenize stream, then split | ch4_control_context_only | -2.21%, -0.88%, -0.64% | -1.24 ± 0.85% |
| Ch5: centered (non-causal) gate FFT | reference | -0.64%, -17.28%, +0.65% | -5.76 ± 10.00% |
| Positive control: true future RV norm | reference | +43.46%, +77.93%, +68.34% | +63.24 ± 17.79% |
| Ch4 control: unconstrained BPE, context only | reference | -7.19%, -3.45%, -7.90% | -6.18 ± 2.39% |

