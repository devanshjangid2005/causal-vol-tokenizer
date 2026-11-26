# Video script — final presentation (DA4022, causal-vol-tokenizer)

Narration for `presentation/final_presentation.pptx`, built by `presentation/build_final_presentation.py`. Each slide's narration below is **identical** to that slide's speaker notes in the deck. The builder stores it in `NOTES` and checks every number in it against `results/` before writing the deck.

**Total:** about 6:33 (879 words at ~150 words per minute, plus about 3 s per slide transition). Target 5–7 min.

**Reading guide for the numbers:**
- Headline H = 12 figures come from `results/robust_stats_h12.md` (seeds 0–2 averaged per origin).
- The labels printed on `inflation_all_folds_h12.png` are seed-0 values from `results/inflation_table_h12.md`. Slide 8 says so out loud.
- The 2025 holdout was scored once on 17 Nov (seed 0) and was not re-scored.
- The extensions use Folds 1–6 only.

| # | Slide | On screen | Start – end |
|---|---|---|---|
| 1 | Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series | Title slide | 0:00 – 0:21 |
| 2 | Which parts of a tokenizer leak the future — and by how much? | Two-column text: leak types and contributions | 0:21 – 0:56 |
| 3 | A causal reference pipeline with five leakage switches | `pipeline_leak_switches.png` + arm/control table | 0:56 – 1:26 |
| 4 | Data, target, model and statistics | Setup bullets + folds table (26,290 pooled test origins; 8,760 holdout) | 1:26 – 1:59 |
| 5 | An audit withdrew the midterm numbers | Withdrawn-claims table (FIXES.md) | 1:59 – 2:20 |
| 6 | Hour boundaries make the tokenizer incremental at hour-aligned origins | `boundary_mismatch_and_compression.png` + callouts 0% / 57% / 32% | 2:20 – 2:48 |
| 7 | Negative control: only the normalization leak beats the oracle floor | `negative_control_oracle_floor.png` + callouts 0.1268 / 0.0694 | 2:48 – 3:15 |
| 8 | Ch1 dominates; every other channel is within a few percent | `inflation_all_folds_h12.png` (seed-0 labels) + callouts +63.39% / +62.03% | 3:15 – 3:44 |
| 9 | 3-seed robust analysis: the small channels are bounded | 3-seed robust table (`robust_stats_h12.md`) | 3:44 – 4:13 |
| 10 | Boundary leakage can hide partial-token bias | Trade-off table (0.3137 vs 0.3128; 0.4691 vs 0.4603) | 4:13 – 4:45 |
| 11 | At the next-day horizon every channel grows | `inflation_all_folds_h288.png` + summary row | 4:45 – 5:07 |
| 12 | 2025 holdout, scored once: Ch1 still inflates by ~70% | `holdout_inflation_h12.png` + MCS table | 5:07 – 5:32 |
| 13 | Exploratory extensions (folds only) | `extensions_h12.png` + callouts +2.70% / +3.06% / +1.70% | 5:32 – 6:02 |
| 14 | Conclusions | Conclusions + limitations | 6:02 – 6:33 |

---

## Slide 1 · Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series  `[0:00 – 0:21]`

*On screen:* Title slide

> Hi, this is our DA4022 project on look-ahead bias inside motif tokenizers for financial time series. We built a leak-free volatility forecasting pipeline, then switched on one source of future information at a time and measured how much better each one makes the forecasts look.

## Slide 2 · Which parts of a tokenizer leak the future — and by how much?  `[0:21 – 0:56]`

*On screen:* Two-column text: leak types and contributions

> BPE motif tokenizers are usually tested zero-shot on new datasets. Used for walk-forward forecasting on one market, they can leak in two ways. At fit time, statistics or tables are estimated on data that overlaps the test period. At encode time, applying Zhang et al.'s incremental-tokenization definition, BPE is not incremental at an arbitrary origin: a stream tokenized then split carries future information, while tokenizing only the context leaves a partial final token. We audit each channel and measure that trade-off.

## Slide 3 · A causal reference pipeline with five leakage switches  `[0:56 – 1:26]`

*On screen:* `pipeline_leak_switches.png` + arm/control table

> Here is the causal reference pipeline. Five-minute returns are normalized by trailing EWMA volatility, each hour gets a scale token, BPE merges never cross an hour boundary, and a causal RoV times SEI gate, borrowed from PATK, shortens tokens in volatile hours. A small GPT forecasts realized volatility. Each red switch is one leak, differing from its matched control in exactly that change, in training and testing.

## Slide 4 · Data, target, model and statistics  `[1:26 – 1:59]`

*On screen:* Setup bullets + folds table (26,290 pooled test origins; 8,760 holdout)

> We use Binance BTCUSDT five-minute data from 2021 to 2025; the proposal said 2019, but the study as run starts in 2021. The target is the sum of squared returns over the next 12 or 288 bars, using only data before the origin. The GPT has 855,169 parameters, with a direct head and a generative head. We have six expanding folds with 26,290 test origins in total, and 2025 as a holdout scored exactly once.

## Slide 5 · An audit withdrew the midterm numbers  `[1:59 – 2:20]`

*On screen:* Withdrawn-claims table (FIXES.md)

> Before the results: an October audit found bugs behind our midterm numbers. The plus 16.6 percent Channel 4 leak, GARCH looking 21.7 percent worse than HAR, the 84 percent boundary mismatch and the 3.99 times compression are all withdrawn. Everything today comes from the corrected pipeline.

## Slide 6 · Hour boundaries make the tokenizer incremental at hour-aligned origins  `[2:20 – 2:48]`

*On screen:* `boundary_mismatch_and_compression.png` + callouts 0% / 57% / 32%

> First, the token-level audit on Fold 1. With hour boundaries, the reference tokenizer gives 0 percent prefix mismatch at hour-aligned origins, and perturbing the future changes nothing. The centered gate of Channel 5 changes the pre-origin tokens 57 percent of the time, and unconstrained BPE straddles the origin at 32 percent of origins. Compression is modest: 1.40 bars per token at 64 merges.

## Slide 7 · Negative control: only the normalization leak beats the oracle floor  `[2:48 – 3:15]`

*On screen:* `negative_control_oracle_floor.png` + callouts 0.1268 / 0.0694

> Next, the negative control. On simulated GARCH data the analytic oracle reaches a QLIKE of 0.1268, and no causal model should beat it. The reference and the Channel 2 to 5 arms all stay above that floor. Only the Channel 1 normalization leak, at 0.0694, and the deliberate positive control go below it, so the controls behave as designed.

## Slide 8 · Ch1 dominates; every other channel is within a few percent  `[3:15 – 3:44]`

*On screen:* `inflation_all_folds_h12.png` (seed-0 labels) + callouts +63.39% / +62.03%

> Now the main result over all six folds at the next-hour horizon. Channel 1, where the normalization window reaches into the forecast horizon, cuts QLIKE by 63.39 percent on the direct head with three seeds, and the random-effects estimate across folds is 62.03 percent with zero heterogeneity. The figure itself shows seed-0 values, such as 63.67 percent, so we quote the three-seed numbers as the headline.

## Slide 9 · 3-seed robust analysis: the small channels are bounded  `[3:44 – 4:13]`

*On screen:* 3-seed robust table (`robust_stats_h12.md`)

> This table is the three-seed robust analysis. Beyond Channel 1, the effects are small: the 90 percent upper bounds limit Channel 2 to 1.44 percent, Channel 3 to 0.39, Channel 4 to 1.05 and Channel 5 to 2.84 percent on the direct head. Channel 5 direct and Channel 3 are significant after Holm correction, but tiny. Partial-token bias is not significant at this horizon.

## Slide 10 · Boundary leakage can hide partial-token bias  `[4:13 – 4:45]`

*On screen:* Trade-off table (0.3137 vs 0.3128; 0.4691 vs 0.4603)

> Here is the trade-off. At the next-hour horizon both effects are small on average. But where partial-token bias is large, at the next-day horizon and on the 2025 generative head, the boundary leak recovers almost all of it: 0.3137 against a reference of 0.3128, and 0.4691 against 0.4603. A leaky pipeline can look as good as a causal one. These two settings are single-seed, so we present this as an observed pattern.

## Slide 11 · At the next-day horizon every channel grows  `[4:45 – 5:07]`

*On screen:* `inflation_all_folds_h288.png` + summary row

> At the next-day horizon, on Fold 1, every channel grows. Channel 1 reaches 96.36 percent, Channel 4 16.10, Channel 5 7.02 and Channel 2 5.67 percent, all significant after Holm correction, while partial-token bias costs 19.53 percent. This is one fold, one seed and the direct head only.

## Slide 12 · 2025 holdout, scored once: Ch1 still inflates by ~70%  `[5:07 – 5:32]`

*On screen:* `holdout_inflation_h12.png` + MCS table

> Then the 2025 holdout, scored once on November 17 and never re-scored. Channel 1 still inflates accuracy by 73.90 percent on the direct head and 68.32 on the generative head. Channel 2 flips sign across heads. In the model confidence set, the leak-free GPT direct models lead on the folds, but on 2025 only GARCH remains.

## Slide 13 · Exploratory extensions (folds only)  `[5:32 – 6:02]`

*On screen:* `extensions_h12.png` + callouts +2.70% / +3.06% / +1.70%

> Finally, the exploratory extensions, which use the folds only. Even one future bar in the normalization window inflates accuracy by 2.70 percent, rising to 18.38 at six bars. Adding ETH data up to the split helps by 3.06 percent, while leaking ETH through the test period adds only 0.95, not significant after Holm. Token healing improves the same model by 1.70 percent, also not significant after Holm.

## Slide 14 · Conclusions  `[6:02 – 6:33]`

*On screen:* Conclusions + limitations

> To conclude: check normalization first, because a window that touches the horizon is the large, stable leak, and the only one that beats the oracle floor. Table and boundary leaks are small at one hour but grow at one day, and a boundary leak can hide partial-token bias. Limitations: one asset, a small model, and single seeds for some settings. Code, results and report are in our repository. Thank you.

---

**Recording tips**

- Record over the deck in presenter view. The same text is in each slide's notes.
- Pause about 1 s on each figure before reading its numbers.
- If time is short, trim slide 5's list of withdrawn numbers first. Keep the seed-0 caveat on slide 8, the single-seed caveat on slide 10, and the scored-once statement on slide 12.
