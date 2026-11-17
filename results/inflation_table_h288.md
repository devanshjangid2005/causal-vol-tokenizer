# Look-ahead inflation, H = 288 bars, folds [1], seed 0

Inflation = % reduction in QLIKE versus the matched control (positive = the leak makes the forecast look better). Pooled over all test origins of the listed folds.

## Direct head

| Arm | Control | Inflation | DM p | Holm p | GW p | TOST equivalent (±5%) | Folds with gain | Folds significant |
|---|---|---|---|---|---|---|---|---|
| Ch1: window norm uses horizon | ch1_control_window | +96.36% | <0.001 | <0.001 | <0.001 | no | 1/1 | 1/1 |
| Ch2: merge table on train+test | reference | +5.67% | 0.016 | 0.016 | 0.045 | no | 1/1 | 1/1 |
| Ch4: tokenize stream, then split | ch4_control_context_only | +16.10% | <0.001 | <0.001 | <0.001 | no | 1/1 | 1/1 |
| Ch5: centered (non-causal) gate FFT | reference | +7.02% | 0.007 | 0.014 | 0.025 | no | 1/1 | 1/1 |
| Positive control: true future RV norm | reference | +99.99% | <0.001 | <0.001 | <0.001 | no | 1/1 | 1/1 |
| Partial-token bias (Ch4 control vs reference) | reference | -19.53% | <0.001 | – | <0.001 | no | – | – |

| Model | Pooled QLIKE | 95% CI |
|---|---|---|
| har_rv | 0.2558 | 0.2094 – 0.3030 |
| garch_11 | 0.4448 | 0.3611 – 0.5202 |
| GPT: Causal reference (leak-free) | 0.3128 | 0.2393 – 0.4041 |
| GPT: Ch1 control: window norm, causal | 0.2268 | 0.1715 – 0.2999 |
| GPT: Ch1: window norm uses horizon | 0.0082 | 0.0061 – 0.0112 |
| GPT: Ch2: merge table on train+test | 0.2951 | 0.2340 – 0.3720 |
| GPT: Ch3: decode tables on train+test | 0.3128 | 0.2393 – 0.4041 |
| GPT: Ch4 control: unconstrained BPE, context only | 0.3739 | 0.2876 – 0.4942 |
| GPT: Ch4: tokenize stream, then split | 0.3137 | 0.2492 – 0.4048 |
| GPT: Ch5: centered (non-causal) gate FFT | 0.2908 | 0.2314 – 0.3696 |
| GPT: Positive control: true future RV norm | 0.0000 | 0.0000 – 0.0001 |

- Causal reference (leak-free) vs har_rv: -22.30% (worse), DM p 0.095

- Causal reference (leak-free) vs garch_11: +29.67% (better), DM p 0.001

- Ch1 control: window norm, causal vs har_rv: +11.34% (better), DM p 0.236

- Ch1 control: window norm, causal vs garch_11: +49.01% (better), DM p <0.001

- Ch4 control: unconstrained BPE, context only vs har_rv: -46.18% (worse), DM p 0.008

- Ch4 control: unconstrained BPE, context only vs garch_11: +15.93% (better), DM p 0.158
