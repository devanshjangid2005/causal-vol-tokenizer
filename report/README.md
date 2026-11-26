# Final report: NeurIPS 2026 template (Overleaf)

**This text is a DRAFT.** The DA4022 brief requires a *human-written* report. Every team member must
rewrite the prose of `main.tex` in their own words before submission. The numbers, tables, figures
and citations can stay as they are: they come from `results/` and are checked automatically.

## Files

| File | What it is |
|---|---|
| `main.tex` | **Strict 5-page version** (the default): abstract, 5 sections, acknowledgements and references fit in 5 pages in total. No appendix. Main-text floats are Fig. 1 (pipeline), Fig. 2 (inflation per channel) and Table 1 (trade-off) |
| `main_with_appendix.tex` | Longer version: the same main text fits 5 pages, and an appendix follows the references (full robust table, negative-control figure, holdout figure, extensions table). Use it only if the Prof allows appendices. In Overleaf, switch it on with Menu → Main document |
| `neurips_2026.sty` | The official NeurIPS 2026 style file (unchanged), so the project compiles straight after upload |
| `refs.bib` | 13 references (from `FINAL_REPORT.md`, section "References") |
| `tables/tab_tradeoff.tex` | Main-text Table 1: boundary leakage vs partial-token bias |
| `tables/tab_robust.tex` | Appendix: full 3-seed robust statistics |
| `tables/tab_extensions.tex` | Appendix: exploratory extensions |
| `figures/pipeline_leak_switches.png` | Fig. 1, copied from `results/figures/` |
| `figures/fig_negative_control.pdf` | Fig. 2, negative control |
| `figures/fig_inflation_folds.pdf` | Fig. 3, inflation per channel, Folds 1–6 |
| `figures/fig_inflation_holdout.pdf` | Appendix figure, 2025 holdout |
| `make_tables.py`, `make_figures.py` | Regenerate the tables and figures from `results/` |

## Upload to Overleaf

1. Open the official template ("Formatting Instructions For NeurIPS 2026", by the NeurIPS 2026 Program Chairs) and choose **Open as Template**.
2. Keep the template's `neurips_2026.sty`, and its `checklist.tex` if you want the checklist.
3. Upload `main.tex`, `refs.bib`, `tables/` and `figures/`. You can upload `report_overleaf.zip` from the repo root and unzip it there.
4. Set `main.tex` as the main document. The default pdfLaTeX compiler works; BibTeX runs automatically.
5. Delete the template's own example `.tex` file, or just ignore it.

The document uses `\usepackage[preprint]{neurips_2026}`, which shows the author names with the footer "Preprint. Work in progress." The default option is the anonymous review version, and `final` is only for papers NeurIPS has accepted.

## Page budget (5 pages of main content)

References, acknowledgements and the appendix fall outside the 5 pages under NeurIPS rules. **Ask the instructor whether the course counts them the same way.**

| Section | Target | Draft words |
|---|---|---|
| Abstract | 0.25 page | 190 |
| 1 Introduction | 0.5 page | 319 |
| 2 Method (with Fig. 1) | 0.9 page | 525 |
| 3 Experimental setup and statistics | 0.5 page | 399 |
| 4 Results (with Figs. 2–3 and Table 1) | 2.0 pages | 790 |
| 5 Discussion and limitations | 0.6 page | 252 |
| **Total** | **~4.75** | **~2,475** (+180 in captions) |

**Float heights at full width (5.5 in):**

| Float | Height |
|---|---|
| Fig. 1 (pipeline) | 2.24 in |
| Fig. 2 (negative control) | 2.1 in |
| Fig. 3 (fold inflation) | 2.6 in |
| Table 1 (trade-off) | about 0.85 in |

**Estimated length:** 4.9–5.1 pages, depending on words per line (14–15 at 10 pt Times). It could not be compiled here, because there is no TeX on this machine.

**If Overleaf shows more than 5 pages**, in this order:
1. Move Fig. 2 to the appendix, keeping its numbers in the text (saves about 0.3 page).
2. Shorten the "Relation to the midterm" paragraph to one sentence.
3. Tighten the extension paragraph.

## What must stay when you rewrite

**Numbers.** Keep every number exactly as written, with its scope:
- 3 seeds on Folds 1–6;
- seed 0 for the next-day (H=288, Fold 1) results and the 2025 holdout;
- folds only for the extensions.

**Mandatory caveats.** The draft already states each of these; keep them all:
1. **The H=288 p-values are anti-conservative.** Hourly origins with 24-hour targets overlap, but the Newey–West lag is about 7.
2. **The partial-token-bias comparison is not a single switch.** It also drops the gate and uses one scale token instead of twelve.
3. **GARCH's exclusion from the fold-level Model Confidence Set depends on seeds 1–2.** With seed 0 alone, GARCH stays in (p = 0.127).
4. **The cross-asset "more data" gain also includes about twice the optimizer steps.**
5. **The gate and scale-token ablations planned at the midterm were not run.**
6. **The code freeze was `f4ad204` on 17 Nov**, not the planned 14 Nov.
7. **Midterm continuity:**
   - the pre-audit numbers came from the 15 Oct initial commits;
   - the midterm had already withdrawn the headline figures;
   - Ch1 rose from +7.09% to about +63% because each arm now trains its own model against a matched control.
8. **The GPT is not claimed to be the better forecaster.** In 2025 only GARCH remains in the Model Confidence Set.

**Author order.** The order is set once, in `\author{...}`. The midterm slide listed Sanidhya Srivastava, Arnav Tiku, Devansh Jangid, Karan Malik, Aman Gupta; the draft uses the order of the final repository. Agree on one order.

**Acknowledgements.** Add a funding or competing-interest statement in the `ack` block if the instructor wants one.

## Regenerate tables and figures (no experiment is re-run)

```
python report/make_tables.py    # checks every cell against results/*.md and the JSONs; stops on any mismatch
python report/make_figures.py   # draws from results/*.json only; PDF with embedded TrueType fonts
```
