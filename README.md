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
* Deliverables:
  * detailed technical notes, [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md): a longer companion to the report;
  * the recorded video `presentation/final_video.mp4` (about 10 minutes);
  * the 18-slide deck `presentation/final_presentation.pptx`, used for the recorded video;
  * the video script [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md), plus a per-speaker version,
    `presentation/PERSON_WISE_SCRIPT.pdf`;
  * the final 5-page report in the NeurIPS 2026 template, [`report/final_report.pdf`](report/final_report.pdf),
    with its Overleaf-ready LaTeX source in `report/` (see `report/README.md`).

The numbers in the initial 15 Oct commits (`7f9f6a6`–`dea2cb5`) are **withdrawn**. They came from bugs found
in the 15 Oct audit: Channel 4 +16.6%, GARCH −21.7% against the reference GPT (43% worse than HAR), 84% boundary mismatch, 3.99× compression, and the
rest of that README's results table. The submitted midterm slide (15 Oct) already withdrew the first three
and reported the corrected token audit and baselines. The fixes were committed on 17 Nov (`24d9c8f`). Do not
cite the pre-audit numbers.

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

The submitted report is [`report/final_report.pdf`](report/final_report.pdf). Full tables and discussion are in [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md).

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
| [`TECHNICAL_NOTES.md`](TECHNICAL_NOTES.md) | Detailed technical notes (Markdown), a longer companion to the 5-page report: method, audit fixes, all verified results, limitations, references |
| `report/` | Overleaf-ready LaTeX draft of the 5-page NeurIPS 2026 report, with tables and figures generated from `results/` (see `report/README.md`) |
| `presentation/build_final_presentation.py` | Builds `presentation/final_presentation.pptx` and `VIDEO_SCRIPT.md`. `--check` verifies every number on the slides and in the narration (112 checks, 166 quoted values) against `results/`, git and the code, and writes nothing |
| `presentation/final_presentation.pptx` | The final deck (18 slides, 16:9) used for the recorded video; its speaker notes are the narration |
| [`VIDEO_SCRIPT.md`](VIDEO_SCRIPT.md) | Slide-by-slide narration with timings, identical to the deck's speaker notes |
| `presentation/PERSON_WISE_SCRIPT.pdf` | The same narration split by speaker (who presents which slides), with hand-offs and a recording checklist |
| [`FIXES.md`](FIXES.md) | Audit log: what was wrong before 15 Oct, what changed, the 17 Nov study and the 20 Nov extensions |
| [`requirements.txt`](requirements.txt) | Python packages and the versions used for the study |
| `results/` | All tables (`*.md`, `*.json`), per-origin forecasts (`losses/*.npz`) and figures (`figures/*.png`) |

These files are kept for history only and are **not** the submitted midterm:

* `presentation/midterm_report_slide*.{pptx,png}`: a 17 Nov Fold 1 rebuild, even though they are labelled
  "October 15, 2026 Submission".
* `build_verified_midterm_presentation.py`: the script that built them.
* `make_midterm_slide.py` and `render_slide_image.py`: they hold pre-audit numbers.

All three scripts stop with a message on purpose, and everything here is superseded by the final deck. The
submitted midterm was a one-slide deck (15 Oct) covering the token audit, the Fold 1 baselines and the GPT
audit, which was then still running. Its numbers reappear unchanged (up to rounding) in the final results.

---

## Team and contributions

The work is split into five parts of roughly equal size. Each part pairs one area of the study (code and the
results it produced) with one share of the final deliverables.

| Member | Area of the study (code) | Results produced | Share of the final deliverables |
|---|---|---|---|
| **Sanidhya Srivastava** | **Data, folds and econometric baselines:** `data/download_binance.py` (Binance klines, 2025 timestamp fix), `data/preprocess.py` (log returns, trailing EWMA, RV targets), `data/folds.py` (expanding folds, clock-hour origins, purging, holdout spec), `models/baselines.py` (HAR-RV, constrained GARCH MLE), `experiments/run_baselines.py`, `experiments/forecast_store.py` | `results/fold_<k>_baselines.json` | Repository and reproducibility (`README.md`, `requirements.txt`, `FIXES.md`); report section on data and setup |
| **Arnav Tiku** | **Tokenizer and token-level causality:** `tokenization/bpe_tokenizer.py` (hour-bounded Götz BPE, scale tokens, conditional decoding), `tokenization/causal_gate.py` (RoV × SEI gate), `tokenization/causality_tests.py`, `experiments/run_boundary_experiment.py` | `results/boundary_mismatch_results.json` | Result figures (`presentation/generate_figures.py`, `results/figures/`); report section on the tokenizer and gate |
| **Devansh Jangid** | **Model, leak arms and main audit:** `models/gpt_model.py` (GPT with direct and generative heads, token healing), `tokenization/leak_tokenizer.py` (every arm and its matched control), `experiments/run_leak_experiments.py` (per-arm training, DM, GW, TOST, Holm) | `results/fold_<k>_h12_leak_audit.json`, `results/fold_1_h288_leak_audit.json` | Final deck and its number check (`presentation/build_final_presentation.py`); report section on the main results |
| **Karan Malik** | **Controls, holdout and extensions:** `simulation/garch_simulator.py`, `experiments/run_oracle_experiment.py` (negative control), `experiments/run_holdout.py` (score-once lock), `experiments/run_all.py`, `experiments/run_extensions.py`, `experiments/run_cross_asset.py`, `notebooks/run_on_colab.ipynb` | `results/negative_control_results.json`, `results/holdout_h12_leak_audit.json`, `results/HOLDOUT_SCORED.json`, `results/fold_<k>_h12_{dose,heal,xasset}.json` | Video script and recording (`VIDEO_SCRIPT.md`); report section on the controls, the holdout and the extensions |
| **Aman Gupta** | **Statistics and robust analysis:** `evaluation/metrics.py` (QLIKE), `evaluation/hypothesis_tests.py` (block bootstrap, DM, GW, TOST, Holm), `experiments/aggregate_folds.py`, `experiments/robust_stats.py` (equivalence bounds, random-effects pooling, Model Confidence Set) | `results/inflation_table_h{12,288}.*`, `results/robust_stats_h12.*` | NeurIPS report assembly (`report/`: LaTeX, `make_tables.py`, `make_figures.py`); report sections on statistics, discussion and limitations |

**Shared by all five:** reviewing the results, writing the report section for their own part in their own
words, and presenting that part in the video.

**Who presents what in the video** (`presentation/final_video.mp4`; script in
`presentation/PERSON_WISE_SCRIPT.pdf`):

| Order | Presenter | Slides | Topics |
|---|---|---|---|
| 1 | Arnav Tiku | 1–4 | Opening, the problem, research question, tokenizer pipeline |
| 2 | Sanidhya Srivastava | 5–7 | Experiment protocol, data and folds, continuity since the midterm |
| 3 | Aman Gupta | 8–11 | Token audit, negative control, main result, equivalence bounds |
| 4 | Karan Malik | 12–15 | Trade-off, next-day horizon, 2025 holdout, extensions |
| 5 | Devansh Jangid | 16–18 | Code walkthrough, takeaways, team and closing |

All commits were pushed from a single account, so `git log` does not show who wrote what.

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
notebooks/       run_on_colab.ipynb (GPU run of the main study; does not cover the extensions)
presentation/    build_final_presentation.py (final deck), generate_figures.py; midterm scripts kept for history
report/          LaTeX scaffold for the 5-page NeurIPS 2026 report (Overleaf)
results/         tables, per-origin forecasts (losses/), figures/
TECHNICAL_NOTES.md  VIDEO_SCRIPT.md  FIXES.md  requirements.txt
```

## Reproduce

**Results are already committed.** Because every output in `results/` exists, `run_all.py`, `run_extensions.py`
and the Colab notebook skip all training steps and only rebuild tables and figures. Pass `--force` to retrain.
Never run the holdout (see below).

**Locally:**

```bash
pip install -r requirements.txt                   # for CUDA, see the note at the top of requirements.txt
python data/download_binance.py 2021 1 2025 12     # caches monthly CSVs in data/raw (~1 min)
python experiments/run_all.py                      # all folds, seeds, next-day, negative control, tables, figures
python experiments/run_extensions.py               # dose-response, token healing, cross-asset (ETH), extra seeds
python experiments/robust_stats.py                 # robust tables from the saved forecasts (no retraining)
python presentation/build_final_presentation.py --check   # verify every quoted number against results/
python presentation/build_final_presentation.py    # rebuild the final deck
```

**Colab:** `notebooks/run_on_colab.ipynb` runs the main study (`run_all.py`) on a T4 GPU. It does not run the
extensions (`run_extensions.py`) or `robust_stats.py`. Its holdout cell refuses to score, because
`results/HOLDOUT_SCORED.json` already exists.

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

**Code freeze.** `data/SEALED_2025_HOLDOUT.txt` names 14 Nov as the code freeze. In fact the code was frozen at
`f4ad204` on 17 Nov, after two changes: the 2025 timestamp fix (without it the 2025 dates would be wrong) and
the constrained GARCH estimator. The holdout was then scored once at that commit. The 20 Nov extension commit
(`09e386a`) later added arms and options to the experiment code, but the holdout files
(`HOLDOUT_SCORED.json`, `holdout_h12_leak_audit.json`, `losses/holdout_h12.npz`) have not changed since
`156090a`.
