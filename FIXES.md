# Fixes after the 15 Oct 2026 audit

The proposal and the three papers (Götz et al. BPE, Zhang et al. UniTok, Liao et al. PATK)
were treated as the source of truth. Each item says what was wrong, what changed, and
which files.

## 1. Channel 4 and the partial-token arm

**Wrong:** the Channel 4 tokenizer was never fitted (0 merges). It fed 144 raw bins instead
of ~107 BPE tokens to a model trained on BPE tokens, and no token ever crossed the origin.
The reported +16.6% "significant leak" was input-format shift, not leakage. The
partial-token-bias arm promised in the proposal did not exist.

**Now:** the unconstrained tokenizer is fitted on train data. `ch4_boundary_leak` tokenizes
context + horizon once and keeps the token that straddles the origin (it does so at 31% of
Fold 1 test origins). `ch4_control_context_only` tokenizes the context alone: Ch4 leakage is
measured against it, and partial-token bias is measured as it vs `reference`.
Files: `tokenization/leak_tokenizer.py`.

## 2. Targets and origins aligned across every model

**Wrong:** GARCH forecasts were scored against HAR's target array, which started 1,440 bars
(5 days) later, so GARCH looked 21.7% worse than HAR. The negative control had the same
bug for HAR. HAR, GARCH and the GPT also used three different target definitions and
different origin sets, so DM tests between them were invalid. Hours were defined by array
index, which drifts after Binance outages (there are 6 gaps in Fold 1, up to 4.75 h).

**Now:**
* One convention everywhere: at origin t, use bars ≤ t−1, target RV_H[t] = Σ_{k<H} r²_{t+k}.
* Each fold is one contiguous frame, so EWMA does not restart at the test boundary. Train
  origins are purged (t + H ≤ split).
* Origins come from timestamps (minute == 0) and skip any window with a missing bar.
* HAR weekly lag is 2016 bars (7 days; crypto trades 24/7).
* GARCH is fitted with returns ×1000 (it was poorly scaled) and filtered causally.

Re-run on Fold 1: HAR 0.516, GARCH 0.483, GARCH better (DM p = 0.005).
Files: `data/preprocess.py`, `data/folds.py`, `models/baselines.py`,
`simulation/garch_simulator.py`, `experiments/run_baselines.py`.

## 3. Leaks in training and testing, one model per arm

**Wrong:** one model was trained on the reference and the leaks were swapped in at test time
only, so the experiment measured distribution shift (for Channel 2, the leaked merge ids
even mean different motifs). Conditional decoding was fitted but never used, so Channel 3
was identical to the reference by construction. Padding used id 0, which is also bin 0
(the most negative return), and the LM loss ignored real bin-0 tokens. GW tests and
GPT-vs-HAR/GARCH tests were missing.

**Now:**
* Each arm trains its own model with the switch on in both train and test. Ch3 reuses the
  reference model, because decoding tables never enter training.
* Each leak is compared with a matched control that differs in that one switch only.
  Ch1's control uses a context-only window std, so Ch1 no longer changes two things at once.
* Two forecast heads: the direct head, and a generative head that samples future tokens and
  decodes them with Götz conditional decoding tables (`E[x² | bin, prev bin]`). Ch3 acts
  through the generative head.
* There is a dedicated PAD id.
* Statistics: Diebold–Mariano with Holm correction across the leak family,
  Giacomini–White, TOST with a relative margin (flagged as trivial when forecasts are
  identical), moving-block bootstrap, and every arm vs HAR and GARCH on the same origins.
* The volatility loss is now QLIKE (matching the evaluation metric); `--vol-loss mse`
  restores the old behaviour.
* Options added: `--seeds`, `--horizon 288` (next-day RV), `--arms`.

The negative control now runs the same pipeline on GARCH data and checks every arm against
the analytic oracle floor. The old "gross leak" was the target times noise, so it could
not validate anything.
Files: `experiments/run_leak_experiments.py`, `experiments/run_oracle_experiment.py`,
`models/gpt_model.py`, `evaluation/hypothesis_tests.py`.

## 4. Boundary experiment and causality tests

**Wrong:**
* The data file the script looked for was gitignored, so it silently used simulated data.
* It tokenized raw returns, which fall in only 2 of 32 bins, so the "84%" was measured on
  a two-symbol tokenizer.
* The 0% causal result compared `encode(prefix)` with `encode(full[:t])`, the same input
  twice, so it could not fail.
* The perturbation test re-encoded the same prefix twice.

**Now:** real fold data on EWMA-normalized returns, with tests that can fail:
* Slice-vs-prefix: tokenize past the origin, cut by bars covered, compare with the prefix.
* Future perturbation: noise after t, then EWMA, normalized returns and gate flags are
  recomputed.
* Both run at hour-aligned origins and at origins 5 bars past the hour.

Results on Fold 1:

| Tokenizer | Mismatch, hour-aligned origins | Mismatch, off-hour origins | Other |
|---|---|---|---|
| Causal reference | 0% | 31% | – |
| Unconstrained BPE | 32% straddle | – | – |
| Centered gate | – | – | future perturbation changes the prefix 57% of the time |

Compression is only 1.40 bars/token with 64 merges (the old 3.99× was an artifact of raw returns).
Files: `tokenization/causality_tests.py`, `experiments/run_boundary_experiment.py`.

## 5. RoV × SEI gate and hourly scale tokens wired in

**Wrong:**
* The gate was never called.
* Its SEI measured high-frequency energy share, whereas PATK's SEI is energy around a
  dominant frequency, and it was combined as `rov·(1+sei)` instead of PATK's product.
* Channel 5 multiplied the scale by (1 + 0.5·FFT energy), which is about ×1.00005 on
  5-minute returns.
* Scale tokens were disabled in every call.

**Now:**
* SEI = share of non-DC spectral energy within ±1 bin of the dominant frequency (PATK
  Eq. 3–4), and score = RoV × SEI.
* The gate threshold is the 75th percentile of train-hour scores. Gated hours use only the
  first 16 of 64 merges, giving shorter tokens; about 23% of Fold 1 test hours are gated.
* Channel 5 centers the gate window on each hour, so for the last context hour it reaches
  6 bars past the origin; that is a real token-boundary leak (see the perturbation result
  above).
* Hourly scale tokens (absolute EWMA vol) are on by default (`--no-scale-tokens`,
  `--no-gate` to ablate).

Files: `tokenization/causal_gate.py`, `tokenization/bpe_tokenizer.py`,
`tokenization/leak_tokenizer.py`.

## Other

* `bpe_tokenizer.py` applies merges in rank order (Götz Alg. 2) via repeated lowest-rank
  merging, which gives identical output and is ~10× faster. Decomposition uses a lookup
  table, and the bogus `token_id % num_bins` fallback was removed.
* README claims were corrected, including the "cryptographically sealed" wording; the
  Office lock file was removed.
* The old midterm slide scripts now stop with a message, because their numbers were pre-fix.

## Still open (not in this patch)

* Folds start in 2021; the proposal says 2019–2025.
* Only Fold 1 has been re-run.
* The proposal cites Zhang et al. for "BPE is not incremental", but that paper defines the
  incremental property and never discusses BPE. Phrase it as applying their definition.
