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

