# Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series

> **DA4022 Course Project · Devansh Jangid, Aman Gupta, Arnav Tiku, Sanidhya Srivastava, Karan Malik**
> A harness that switches on one look-ahead leakage channel at a time inside a BPE
> motif tokenizer and measures how much each one inflates walk-forward realized
> volatility forecasts.

**Status (26 Nov 2026): complete.**

* The pipeline was corrected after the 15 Oct audit (see [`FIXES.md`](FIXES.md)).
* The full study has run: six walk-forward folds with three seeds each, the next-day horizon on Fold 1, and
  the synthetic GARCH negative control.
* The 2025 holdout was **scored once** on 17 Nov 2026 (`results/HOLDOUT_SCORED.json`, `"rescored": false`).
* The exploratory extensions use Folds 1–6 only: Ch1 dose-response, cross-asset ETH leakage, token healing,
  and robust statistics.
* The final report, slide deck and video script are in the repository root and `presentation/`.

The midterm headline numbers are **withdrawn** because they came from bugs: Channel 4 +16.6%, GARCH vs HAR
−21.7%, 84% boundary mismatch, and 3.99× compression. Do not cite them.

---

## Final results at a glance

Inflation is the % reduction in QLIKE versus the arm's matched control. A positive value means the leak
makes the forecast look better. Headline numbers are 3-seed pooled values over Folds 1–6 (26,290 hourly test
origins, H = 12) from [`results/robust_stats_h12.md`](results/robust_stats_h12.md).

| Channel (arm vs matched control) | Direct head | Generative head | 90% upper bound (direct / generative) |
|---|---:|---:|---|
| **Ch1** window normalization reaches the horizon | **+63.39%** | **+62.17%** | +65.04% / +63.71% |
| Ch2 merge table fitted on train + test | +0.64% | −2.58% | +1.44% / −0.60% |
| Ch3 decoding tables fitted on train + test | – | +0.24% | – / +0.39% |
| Ch4 tokenize the stream, then split | +0.47% | −0.95% | +1.05% / +0.29% |
| Ch5 centered (non-causal) gate FFT | +1.78% | +0.26% | +2.84% / +1.73% |
| Positive control (true future RV normalization) | +99.98% | +72.22% | – |
| Partial-token bias (unconstrained context-only vs hour-aligned) | +0.42% | +1.89% | +1.70% / +3.45% |

**Ch1 across folds:**

* Random-effects pooling across folds gives +62.03% [95% CI +58.80, +65.26] for the direct head, with I² = 0%.
* On simulated GARCH data, Ch1 is the only leakage channel whose forecasts beat the analytic oracle floor
  (QLIKE 0.0694 vs the floor of 0.1268). The other one that does is the deliberate positive control.

**Where the effects grow:**

* **Next-day horizon** (H = 288; Fold 1, seed 0, direct head):
  * Ch1 +96.36%, Ch4 +16.10%, Ch5 +7.02%, Ch2 +5.67%.
  * Partial-token bias −19.53%.
* **2025 holdout** (scored once, seed 0):
  * Ch1 +73.90% direct and +68.32% generative.
  * Generative head: Ch4 +15.30% against a partial-token bias of −20.32%. The boundary leak almost exactly
    cancels the partial-token cost.

**Model Confidence Set** (α = 0.10):

* Folds 1–6: {GPT unconstrained BPE direct 0.5556, GPT reference direct 0.5580}. GARCH (0.5747) and HAR-RV
  (0.6811) are excluded.
* 2025 holdout: {GARCH(1,1) 0.4267}.

**Exploratory extensions** (folds only):

* Ch1 dose-response: k = 1 future bar +2.70%, k = 3 +6.95%, k = 6 +18.38%.
* ETH data up to the split ("more data, no leak"): +3.06% direct. ETH through the test period adds a further
  +0.95%, not significant after Holm correction (p = 0.188).
* Token healing vs plain decoding, same model: +1.70% (90% CI +0.07% to +3.38%; Holm p = 0.295).

Full tables, figures and discussion are in [`FINAL_REPORT.md`](FINAL_REPORT.md).

> Two tables are kept apart on purpose:
> * [`results/inflation_table_h12.md`](results/inflation_table_h12.md) pools **seed 0** only (reference direct
>   QLIKE 0.5651). Its numbers are the labels on `results/figures/inflation_all_folds_h12.png`.
> * [`results/robust_stats_h12.md`](results/robust_stats_h12.md) averages **seeds 0–2** per origin
>   (reference 0.5580).
>
> `results/leakage_inflation_table.json` and `results/figures/leakage_inflation_comparison.png` hold
> **Fold 6**, not Fold 1. For Fold 1, use `results/fold_1_h12_leak_audit.json`.

## Deliverables

| File | What it is |
|---|---|
| [`FINAL_REPORT.md`](FINAL_REPORT.md) | Final report: method, audit fixes, all verified results, limitations, references |
| `presentation/build_final_presentation.py` | Builds `presentation/final_presentation.pptx` (14 slides, 16:9). It checks every quoted number against `results/` first |
| `presentation/final_presentation.pptx` | The generated deck, with speaker notes = the narration |
| [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) | Slide-by-slide narration with timings (about 6.5 minutes) |
| [`FIXES.md`](FIXES.md) | Audit log: what was wrong before 15 Oct, what changed, the 17 Nov study and the 20 Nov extensions |
| `results/` | All tables (`*.md`, `*.json`), per-origin forecasts (`losses/*.npz`) and figures (`figures/*.png`) |

The midterm scripts `presentation/make_midterm_slide.py` and `build_verified_midterm_presentation.py` are kept
for history. Their numbers predate the full study and are superseded by the final deck.

---

## Motivation

BPE motif tokenizers for time series (Götz et al., 2026) are validated zero-shot on held-out datasets. Used
for walk-forward forecasting on one market, they can leak the future in two ways:

1. **Fit-time leakage:** normalization statistics, the merge table or the conditional decoding tables are
   estimated on data overlapping the test period.
2. **Encode-time leakage:** applying the incremental-tokenization definition of Zhang et al. (2026),
   rank-ordered BPE is not incremental at an arbitrary origin. A stream tokenized once and then cut at the
   forecast origin carries post-origin information in its token boundaries. Tokenizing the context alone
   instead leaves a partial final token (partial-token bias).

## Pipeline

* **Data:**
  * Binance BTCUSDT 5-minute klines, **2021-01 to 2025-12**. The proposal said 2019–2025.
  * One contiguous frame per fold (train + test). Log returns; gaps are flagged and returns across gaps are
    set to zero.
* **Normalization:** trailing EWMA volatility (λ = 0.94), using bars up to t−1 only.
* **Tokenizer (causal reference):**
  * 32 uniform bins on [−4, 4] and 64 Götz merges learned on train data only.
  * Merges never cross a clock hour (UniTok-style hard boundary).
  * An hourly scale token carrying absolute EWMA volatility.
  * A causal RoV × SEI gate (mechanism borrowed from PATK) that applies only the first 16 merges in volatile
    hours, giving shorter tokens.
* **Model:** an 855,169-parameter causal GPT (4 layers, d_model 128) with two forecast heads:
  * a direct RV regression head;
  * a generative head that samples future tokens and decodes them with Götz conditional decoding tables
    `E[x² | bin, previous bin]`.
* **Target:** at origin t, RV_H[t] = Σ_{k<H} r²_{t+k}, using information up to t−1. Every model (GPT arms,
  HAR-RV, GARCH, oracle) is scored on the same hour-aligned origins.

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

Extension arms (folds only):

* `ch1_leak_k{1,3,6}` (off-by-k normalization window).
* `ext_ch4_control` / `ext_ch4_healed` (token healing on the same trained model).
* `xa_control` / `xa_leak` (ETH training data up to the split vs through the test period).

Each arm trains its own model with the switch on in training **and** testing. The tests are:

* QLIKE with moving-block bootstrap CIs.
* Diebold–Mariano (HAC) with Holm correction.
* Giacomini–White.
* TOST equivalence (±5% of the control's QLIKE).
* DerSimonian–Laird random-effects pooling across folds.
* A Model Confidence Set.

Negative control: the same pipeline on simulated GARCH(1,1) data, where the analytic oracle forecast is a
floor no causal model can beat in expectation.

## Repository structure

```text
data/            download_binance.py, preprocess.py (EWMA, targets), folds.py (folds, holdout, origins)
simulation/      garch_simulator.py (GARCH DGP + analytic oracle)
tokenization/    bpe_tokenizer.py, causal_gate.py, leak_tokenizer.py (all arms), causality_tests.py
models/          gpt_model.py (GPT + generative RV), baselines.py (HAR-RV, GARCH)
evaluation/      metrics.py (QLIKE), hypothesis_tests.py (bootstrap, DM, GW, TOST, Holm)
experiments/     run_all.py (whole study, resumable), run_extensions.py, robust_stats.py, run_leak_experiments.py,
                 run_baselines.py, run_boundary_experiment.py, run_oracle_experiment.py, run_cross_asset.py,
                 aggregate_folds.py, forecast_store.py, run_holdout.py
notebooks/       run_on_colab.ipynb (one-click GPU run of the main study)
presentation/    build_final_presentation.py (final deck), generate_figures.py; midterm scripts kept for history
results/         tables, per-origin forecasts (losses/), figures/
FINAL_REPORT.md  VIDEO_SCRIPT.md  FIXES.md
```

## Reproduce

**Easiest:** open `notebooks/run_on_colab.ipynb` in Colab with a T4 GPU and choose *Run all*.

**Locally:**

```bash
pip install torch numpy pandas scipy matplotlib requests pyarrow python-pptx pillow
python data/download_binance.py 2021 1 2025 12     # caches monthly CSVs in data/raw (~1 min)
python experiments/run_all.py                      # all folds, seeds, next-day, negative control, tables, figures
python experiments/run_extensions.py               # dose-response, token healing, cross-asset (ETH), extra seeds
python experiments/robust_stats.py                 # robust tables from the saved forecasts (no retraining)
python presentation/build_final_presentation.py    # final deck (checks every quoted number against results/)
```

`run_all.py` and `run_extensions.py` skip any step whose output already exists, so they can be re-run after
an interruption. Each run saves per-origin forecasts to `results/losses/*.npz`, so every table can be
recomputed without retraining. The run logs in `results/logs/` are gitignored.

## Holdout

The 2025 data is scored **once**, by `experiments/run_holdout.py`, which trains on 2021–2024 and tests on 2025.
That has already happened (17 Nov 2026, commit `f4ad204`). **Do not run it again**, and do not run
`run_all.py --holdout`.

* Running it writes `results/HOLDOUT_SCORED.json` with the time, the git commit and the settings.
* A second run refuses to score again unless `--i-know-this-rescores` is passed, and that re-score is
  recorded in the output.
* `--dry-run` exercises the same code on the Fold 6 split without loading 2025.
* The extensions never touch the holdout. `robust_stats.py` reads the already-saved holdout forecasts only
  for the Model Confidence Set.

(The earlier "cryptographically sealed" wording was inaccurate: `data/SEALED_2025_HOLDOUT.txt` is a
written commitment, not a hash.)
