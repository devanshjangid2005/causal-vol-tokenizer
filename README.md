# Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series

> **DA4022 Course Project · Team of 5**
> A harness that switches on one look-ahead leakage channel at a time inside a BPE
> motif tokenizer and measures how much each one inflates walk-forward realized
> volatility forecasts.

**Status (17 Nov 2026):** the pipeline was corrected after an audit (see [`FIXES.md`](FIXES.md)), and
the whole study now runs from one command (`experiments/run_all.py`) or one Colab notebook.
The earlier headline numbers for Channel 4 (+16.6%), GARCH vs HAR (−21.7%) and the
84% boundary-mismatch rate came from bugs and have been withdrawn. The token-level
audit and the baselines below were re-run on Fold 1 after the fixes; the per-channel
GPT audit is being re-run.

---

## Motivation

BPE motif tokenizers for time series (Götz et al., 2026) are validated zero-shot on
held-out datasets. Used for walk-forward forecasting on one market they can leak the
future in two ways:

1. **Fit-time leakage:** normalization statistics, the merge table or the conditional
   decoding tables are estimated on data overlapping the test period.
2. **Encode-time leakage:** rank-ordered BPE violates the incremental tokenization
   property (Zhang et al., 2026): a stream tokenized once and then cut at the forecast
   origin carries post-origin information in its token boundaries. Tokenizing the
   context alone instead leaves a partial final token (partial-token bias).

## Pipeline

* **Data:** Binance BTCUSDT 5-minute klines, one contiguous frame per fold (train + test),
  log returns, gaps flagged, returns across gaps set to zero.
* **Normalization:** trailing EWMA volatility (λ = 0.94), uses bars up to t−1 only.
* **Tokenizer (causal reference):** 32 uniform bins on [−4, 4], 64 Götz merges learned on
  train only, merges never cross a clock hour (UniTok-style hard boundary), an hourly
  scale token carrying absolute EWMA volatility, and a causal RoV × SEI gate (PATK-inspired)
  that applies only the first 16 merges in volatile hours (shorter tokens).
* **Model:** ~0.8M-parameter causal GPT; two forecast heads: a direct RV regression head,
  and a generative head (sample future tokens, decode with Götz conditional decoding
  tables `E[x² | bin, previous bin]`).
* **Target:** at origin t, RV_H[t] = Σ_{k<H} r²_{t+k} using information up to t−1.
  Every model (GPT arms, HAR-RV, GARCH, oracle) is scored on the same hour-aligned origins.

## Experimental arms (one switch each)

| Arm | Matched control | Switch |
|---|---|---|
| `reference` | – | leak-free pipeline above |
| `ch1_window_norm_leak` | `ch1_control_window` | window std computed over context **+ horizon** (control: context only) |
| `ch2_merge_table_leak` | `reference` | merge table fitted on train + test |
| `ch3_decode_table_leak` | `reference` | decoding tables fitted on train + test (generative head only) |
| `ch4_boundary_leak` | `ch4_control_context_only` | unconstrained BPE over context + horizon, then cut at t |
| `ch4_control_context_only` vs `reference` | – | **partial-token bias** (unconstrained vs hour-bounded) |
| `ch5_noncausal_fft_leak` | `reference` | gate's RoV × SEI window centered on the hour (reaches 6 bars past t) |
| `positive_control` | `reference` | context normalized by the true future RV |

Each arm trains its own model with the switch on in training **and** testing. Tests:
QLIKE with moving-block bootstrap CIs, Diebold–Mariano (HAC) with Holm correction across
the leak family, Giacomini–White, and TOST equivalence (±5% of the control's QLIKE).
Negative control: the same pipeline on simulated GARCH(1,1) data, where the analytic
oracle forecast is a floor no causal model can beat in expectation.

## Results re-run after the fixes (Fold 1: train 2021, test 2022-Q1)

**Token-level causality audit** (`run_boundary_experiment.py`, 500 test origins):

| Tokenizer | Slice ≠ prefix, hour-aligned origin | Slice ≠ prefix, origin 5 bars past the hour | Future perturbation changes pre-origin tokens |
|---|---|---|---|
| Hour boundaries, no gate | 0% | 35.6% | 0% |
| Hour boundaries + causal gate (reference) | 0% | 31.0% | 0% |
| Hour boundaries + centered gate (Ch5) | 0% | 30.4% | 57.0% |
| Unconstrained Götz BPE (Ch4) | 32.0% | 34.0% | 29.5% |

The causal reference is incremental only at hour-aligned origins, which is why every
experiment uses clock-hour origins. On normalized 5-minute returns, 64 merges compress
only 1.40 bars per token (128 merges: 1.54).

**Baselines** (`run_baselines.py`, 2,160 hourly test origins, next-hour RV):
HAR-RV QLIKE 0.516 [0.469, 0.575]; GARCH(1,1) QLIKE 0.483 [0.424, 0.556];
GARCH is better (DM p = 0.005).

**Per-channel GPT audit (all folds) and 2025 holdout:** produced by `experiments/run_all.py`; see
`results/inflation_table_h12.md` once it has run.

## Repository structure

```text
data/            download_binance.py, preprocess.py (EWMA, targets), folds.py (folds, holdout, origins)
simulation/      garch_simulator.py (GARCH DGP + analytic oracle)
tokenization/    bpe_tokenizer.py, causal_gate.py, leak_tokenizer.py (all arms), causality_tests.py
models/          gpt_model.py (GPT + generative RV), baselines.py (HAR-RV, GARCH)
evaluation/      metrics.py (QLIKE), hypothesis_tests.py (bootstrap, DM, GW, TOST, Holm)
experiments/     run_all.py (whole study, resumable), run_leak_experiments.py, run_baselines.py,
                 run_boundary_experiment.py, run_oracle_experiment.py, aggregate_folds.py, run_holdout.py
notebooks/       run_on_colab.ipynb (one-click GPU run of everything)
presentation/    generate_figures.py (old slide scripts are disabled: their numbers were pre-fix)
```

## Reproduce

**Easiest:** open `notebooks/run_on_colab.ipynb` in Colab with a T4 GPU and choose *Run all*.

**Locally:**

```bash
pip install torch numpy pandas scipy matplotlib requests pyarrow
python data/download_binance.py 2021 1 2025 12     # caches monthly CSVs in data/raw (~1 min)
python experiments/run_all.py                      # all folds, seeds, next-day, negative control, tables, figures
python experiments/run_all.py --holdout            # once, after reviewing the fold results
```

`run_all.py` skips any step whose output already exists, so it can be re-run after an interruption.
Its steps:

1. Token causality audit (Fold 1).
2. HAR-RV and GARCH(1,1) baselines (Folds 1–6).
3. GPT leak audit, Fold 1, seeds 0–2 (next-hour RV).
4. GPT leak audit, Folds 2–6, seed 0.
5. Negative control on simulated GARCH.
6. Next-day RV (H = 288) on Fold 1, direct head only. The generative head would need 288 sampled steps per forecast.
7. `aggregate_folds.py`: the cross-fold inflation table in `results/inflation_table_h12.md` and `.json`.
8. Figures.

Each leak-audit run also saves per-origin forecasts to `results/losses/*.npz`, so every table can
be recomputed without retraining.

## Exploratory extensions (folds only)

```bash
python experiments/run_extensions.py      # dose-response, token healing, cross-asset (ETH), extra seeds
```

It writes:

- `results/robust_stats_h12.md`: equivalence bounds, random-effects pooling across folds and a
  Model Confidence Set;
- `results/figures/extensions_h12.png`;
- per-run forecasts in `results/losses/fold_<k>_h12_<part>.npz`.

It never touches the 2025 holdout. See `FIXES.md` (20 Nov) for the design of each part.

## Holdout

The 2025 data is scored once, by `experiments/run_holdout.py`, which trains on 2021–2024 and tests on 2025:

* Running it writes `results/HOLDOUT_SCORED.json` with the time, the git commit and the settings.
* A second run refuses to score again unless `--i-know-this-rescores` is passed, and that re-score is
  recorded in the output.
* `--dry-run` exercises the same code on the Fold 6 split without loading 2025.

(The earlier "cryptographically sealed" wording was inaccurate: `data/SEALED_2025_HOLDOUT.txt` is a
written commitment, not a hash.)
