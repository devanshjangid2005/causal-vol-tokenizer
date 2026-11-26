"""
presentation/build_final_presentation.py
Final DA4022 presentation (16:9, 18 slides) and its video script, for the causal-vol-tokenizer study.

Supersedes the text-heavy 14-slide final deck (commit 1695431) and the Fold-1-only midterm build
(presentation/build_verified_midterm_presentation.py).

Every number shown on a slide, in a chart, or spoken in the narration is read from results/ at build time
and compared with EXPECTED below; the build stops and writes nothing if any value differs.
  results/robust_stats_h12.json        3-seed pooled inflation, 90% CIs, Holm p, random effects, MCS
  results/inflation_table_h12.json     seed-0 pooled table (only used for the labelled seed-0 note)
  results/inflation_table_h288.json    next-day horizon, Fold 1, seed 0, direct head
  results/negative_control_results.json, results/boundary_mismatch_results.json
  results/holdout_h12_leak_audit.json, results/HOLDOUT_SCORED.json   (read only; nothing is re-scored)
  results/fold_<k>_baselines.json, results/fold_<k>_h12_leak_audit.json, results/leakage_inflation_table.json
The withdrawn pre-audit figures on the continuity slide are checked against the README of the initial 15 Oct
commits 7f9f6a6–dea2cb5 (`git show dea2cb5:README.md`), so the build must run inside a git clone.

results/figures/leakage_inflation_comparison.png is deliberately NOT used: it holds Fold 6
(results/leakage_inflation_table.json["fold"] == 6), not Fold 1.

Usage (from the repository root):
    pip install python-pptx pillow
    python presentation/build_final_presentation.py            # writes presentation/final_presentation.pptx
                                                               #   and VIDEO_SCRIPT.md (from NOTES)
    python presentation/build_final_presentation.py --check    # verify all numbers only; writes nothing
    python presentation/build_final_presentation.py --out other.pptx --script-out other.md
"""

import argparse
import json
import os
import re
import subprocess
import sys

from PIL import Image
from pptx import Presentation
from pptx.chart.data import CategoryChartData
from pptx.dml.color import RGBColor
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION, XL_LEGEND_POSITION, XL_TICK_LABEL_POSITION
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
REPO_URL = "github.com/devanshjangid2005/causal-vol-tokenizer"

# One place to change the author order (title slide). The team slide follows the README table order.
AUTHORS = ["Devansh Jangid", "Aman Gupta", "Arnav Tiku", "Sanidhya Srivastava", "Karan Malik"]
TEAM = [  # (name, area of the study, deliverable share) -- README "Team and contributions"
    ("Sanidhya Srivastava", "Data, folds and baselines", "Repository and reproducibility"),
    ("Arnav Tiku", "Tokenizer and causality audit", "Result figures"),
    ("Devansh Jangid", "Model, leak arms and main audit", "Final deck and number check"),
    ("Karan Malik", "Controls, 2025 holdout and extensions", "Video script and recording"),
    ("Aman Gupta", "Statistics and robust analysis", "NeurIPS report assembly"),
]
assert sorted(AUTHORS) == sorted(t[0] for t in TEAM)

# ----------------------------------------------------------------------------- palette
NAVY = RGBColor(0x0F, 0x1E, 0x3C)       # dominant: dark slides, titles, structure
INK = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)
LINE = RGBColor(0xCB, 0xD5, 0xE1)
PANEL = RGBColor(0xF1, 0xF5, 0xF9)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
ICE = RGBColor(0xCB, 0xD5, 0xE1)
LEAK = RGBColor(0xE8, 0x59, 0x0C)       # orange: anything that leaks
LEAK_T = RGBColor(0xFD, 0xEB, 0xDF)
SAFE = RGBColor(0x0C, 0xA6, 0x78)       # green: causal / leak-free
SAFE_T = RGBColor(0xE0, 0xF5, 0xEE)
DIRECT = RGBColor(0x3B, 0x5B, 0xDB)     # blue: direct head / neutral series
DIRECT_T = RGBColor(0xE5, 0xEB, 0xFB)
GRAYBAR = RGBColor(0x94, 0xA3, 0xB8)
# chart palette validated with the dataviz skill's validate_palette.js (3B5BDB, 0CA678, E8590C: all checks pass)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)
FONT = "Calibri"


# ============================================================================= numbers
def _load(name):
    with open(os.path.join(RES, name), encoding="utf-8") as f:
        return json.load(f)


def pct(v, nd=2):
    return f"{v:+.{nd}f}%"


def pval(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def q4(v):
    return f"{v:.4f}"


def load_numbers():
    """Formatted strings (and chart series) for everything on the slides, straight from results/."""
    N = {}
    rs = _load("robust_stats_h12.json")
    comps = [c for fam in rs["comparisons"].values() for c in fam]

    def robust(arm, control, head):
        for c in comps:
            if c["arm"] == arm and c["control"] == control and c["head"] == head:
                return c
        raise KeyError((arm, control, head))

    rows = [  # key, arm, control, head
        ("ch1_d", "ch1_window_norm_leak", "ch1_control_window", "direct"),
        ("ch1_g", "ch1_window_norm_leak", "ch1_control_window", "generative"),
        ("ch2_d", "ch2_merge_table_leak", "reference", "direct"),
        ("ch2_g", "ch2_merge_table_leak", "reference", "generative"),
        ("ch3_g", "ch3_decode_table_leak", "reference", "generative"),
        ("ch4_d", "ch4_boundary_leak", "ch4_control_context_only", "direct"),
        ("ch4_g", "ch4_boundary_leak", "ch4_control_context_only", "generative"),
        ("ch5_d", "ch5_noncausal_fft_leak", "reference", "direct"),
        ("ch5_g", "ch5_noncausal_fft_leak", "reference", "generative"),
        ("pos_d", "positive_control", "reference", "direct"),
        ("pos_g", "positive_control", "reference", "generative"),
        ("ptb_d", "ch4_control_context_only", "reference", "direct"),
        ("ptb_g", "ch4_control_context_only", "reference", "generative"),
        ("k1", "ch1_leak_k1", "ch1_control_window", "direct"),
        ("k3", "ch1_leak_k3", "ch1_control_window", "direct"),
        ("k6", "ch1_leak_k6", "ch1_control_window", "direct"),
        ("xa_leak_d", "xa_leak", "xa_control", "direct"),
        ("xa_more_d", "xa_control", "reference", "direct"),
        ("heal", "ext_ch4_healed", "ext_ch4_control", "generative"),
        ("heal_plain", "ext_ch4_control", "reference", "generative"),
        ("heal_vs_ref", "ext_ch4_healed", "reference", "generative"),
        ("xa_leak_g", "xa_leak", "xa_control", "generative"),
        ("xa_more_g", "xa_control", "reference", "generative"),
    ]
    for key, arm, ctrl, head in rows:
        c = robust(arm, ctrl, head)
        re_ = c["random_effects"]
        N[key + "_v"] = c["inflation_pct"]
        N[key + "_hi_v"] = c["ci90"][1]
        N[key] = pct(c["inflation_pct"])
        N[key + "_lo"] = pct(c["ci90"][0])
        N[key + "_hi"] = pct(c["ci90"][1])
        N[key + "_holm"] = pval(c["dm_p_holm"])
        N[key + "_re"] = pct(re_["estimate"])
        N[key + "_re_ci"] = f"[{re_['ci95'][0]:+.2f}, {re_['ci95'][1]:+.2f}]"
        N[key + "_i2"] = f"{100 * re_['I2']:.0f}%"
        N[key + "_n"] = f"{c['n_origins']:,}"
        N[key + "_seeds"] = str(len(c["seeds_arm"]))
        N[key + "_i2_v"] = 100 * re_["I2"]

    N["i2_max_small"] = f"{max(N[k + '_i2_v'] for k in ('ch2_d', 'ch2_g', 'ch3_g', 'ch4_d', 'ch4_g', 'ch5_d', 'ch5_g')):.0f}%"
    N["ext_labels"] = [N[k] for k in ("k1", "k3", "k6", "ch1_d", "xa_leak_d", "xa_more_d", "heal", "heal_plain",
                                       "heal_vs_ref")]
    mcs = {}
    for name, m in rs["mcs"].items():
        tag = "folds" if name.startswith("pooled") else "hold"
        mcs[tag] = m
        N[f"mcs_{tag}_in"] = sorted(m["in_mcs"])
    N["mcs_alpha"] = f"{mcs['folds']['alpha']:.2f}"
    mf = mcs["folds"]["mean_qlike"]
    N["ref3_d"] = q4(mf["GPT reference, direct"])
    N["unc3_d"] = q4(mf["GPT unconstrained BPE, direct"])
    N["garch_folds"] = q4(mf["GARCH(1,1)"])
    N["garch_folds_p"] = f"{mcs['folds']['mcs_pvalues']['GARCH(1,1)']:.3f}"
    N["garch_hold"] = q4(mcs["hold"]["mean_qlike"]["GARCH(1,1)"])

    s0 = _load("inflation_table_h12.json")
    N["s0_ch1_d"] = pct(s0["heads"]["direct"]["leak_tests"]["ch1_window_norm_leak"]["inflation_pct"])

    # folds and origins
    counts = []
    for k in range(1, 7):
        counts.append(_load(f"fold_{k}_baselines.json")["n_test_origins"])
    N["fold_test"] = [f"{c:,}" for c in counts]
    N["n_test_folds"] = f"{sum(counts):,}"

    # token causality audit + compression
    bm = _load("boundary_mismatch_results.json")
    v = bm["variants"]
    ref_v = v["hour_boundaries_causal_gate (reference)"]
    c5_v = v["hour_boundaries_centered_gate (ch5)"]
    c4_v = v["unconstrained_bpe (ch4)"]
    tok = lambda d, t, k: 100 * d[t][k]
    N["tok_straddle"] = [tok(ref_v, "slice_vs_prefix_hour_aligned", "mismatch_rate"),
                         tok(c5_v, "slice_vs_prefix_hour_aligned", "mismatch_rate"),
                         tok(c4_v, "slice_vs_prefix_hour_aligned", "mismatch_rate")]
    N["tok_perturb"] = [tok(ref_v, "future_perturbation_hour_aligned", "prefix_changed_rate"),
                        tok(c5_v, "future_perturbation_hour_aligned", "prefix_changed_rate"),
                        tok(c4_v, "future_perturbation_hour_aligned", "prefix_changed_rate")]
    N["tok_labels"] = [f"{x:.1f}%" for x in N["tok_straddle"] + N["tok_perturb"]]
    N["bm_origins"] = f"{bm['n_origins']}"
    N["bm_ch5_pert"] = f"{N['tok_perturb'][1]:.0f}%"
    N["bm_ch4_aligned"] = f"{N['tok_straddle'][2]:.0f}%"
    N["bm_ref_aligned"] = f"{N['tok_straddle'][0]:.0f}%"
    cc = {c["num_merges"]: c["compression_ratio"] for c in bm["compression_curve"]}
    N["comp64"] = f"{cc[64]:.2f}"

    # negative control
    nc = _load("negative_control_results.json")
    bf = nc["beats_oracle_floor"]
    N["nc_floor_v"] = bf["direct"]["reference"]["oracle_floor"]
    N["nc_floor"] = q4(N["nc_floor_v"])
    order = [("Positive control", "positive_control"), ("Ch1 window-norm leak", "ch1_window_norm_leak"),
             ("Oracle floor (analytic)", None), ("Ch5 centered gate", "ch5_noncausal_fft_leak"), ("Ch2 merge table", "ch2_merge_table_leak"),
             ("Ch4 stream-then-split", "ch4_boundary_leak"), ("Causal reference", "reference"),
             ("Ch4 control (context only)", "ch4_control_context_only"),
             ("Ch1 control (context window)", "ch1_control_window")]
    N["nc_rows"] = [(lab, bf["direct"][arm]["QLIKE"], bf["direct"][arm]["violates_floor"]) if arm else
                    (lab, N["nc_floor_v"], None) for lab, arm in order]
    small = [nc["heads"]["direct"]["leak_tests"][a]["inflation_pct"]
             for a in ("ch2_merge_table_leak", "ch4_boundary_leak", "ch5_noncausal_fft_leak")]
    N["nc_small_lo"], N["nc_small_hi"] = pct(min(small)), pct(max(small))
    N["nc_labels"] = [q4(q) for _, q, _ in N["nc_rows"]]
    N["nc_ch1_d"] = q4(bf["direct"]["ch1_window_norm_leak"]["QLIKE"])
    N["nc_ch1_g"] = q4(bf["generative"]["ch1_window_norm_leak"]["QLIKE"])
    N["nc_violators"] = sorted({a for h in bf.values() for a, r in h.items() if r["violates_floor"]})
    N["nc_n_test"] = f"{nc['n_test_origins']:,}"

    # next-day horizon
    h288 = _load("inflation_table_h288.json")["heads"]["direct"]
    for key, arm in [("ch1", "ch1_window_norm_leak"), ("ch2", "ch2_merge_table_leak"),
                     ("ch4", "ch4_boundary_leak"), ("ch5", "ch5_noncausal_fft_leak"),
                     ("pos", "positive_control")]:
        N["h288_" + key + "_v"] = h288["leak_tests"][arm]["inflation_pct"]
        N["h288_" + key] = pct(N["h288_" + key + "_v"])
    N["h288_ptb_v"] = h288["partial_token_bias"]["inflation_pct"]
    N["h288_ptb"] = pct(N["h288_ptb_v"])
    for key, arm in [("ref", "reference"), ("ch4c", "ch4_control_context_only"), ("ch4", "ch4_boundary_leak")]:
        N["h288_q_" + key + "_v"] = h288["arms"][arm]["QLIKE"]
        N["h288_q_" + key] = q4(N["h288_q_" + key + "_v"])

    # 2025 holdout (read only)
    hs = _load("HOLDOUT_SCORED.json")
    N["hold_rescored"] = hs["rescored"]
    N["hold_commit"] = hs["git_commit"][:7]
    N["hold_date"] = hs["scored_at_utc"][:10]
    ho = _load("holdout_h12_leak_audit.json")
    N["hold_n"] = f"{ho['n_test_origins']:,}"
    for head, hk in (("direct", "d"), ("generative", "g")):
        h = ho["heads"][head]
        for key, arm in [("ch1", "ch1_window_norm_leak"), ("ch2", "ch2_merge_table_leak"),
                         ("ch3", "ch3_decode_table_leak"), ("ch4", "ch4_boundary_leak"),
                         ("ch5", "ch5_noncausal_fft_leak")]:
            if arm in h["leak_tests"]:
                N[f"hold_{key}_{hk}_v"] = h["leak_tests"][arm]["inflation_pct"]
                N[f"hold_{key}_{hk}"] = pct(N[f"hold_{key}_{hk}_v"])
            else:
                N[f"hold_{key}_{hk}_v"] = None
        N[f"hold_ptb_{hk}"] = pct(h["partial_token_bias"]["inflation_pct"])
        for key, arm in [("ref", "reference"), ("ch4c", "ch4_control_context_only"), ("ch4", "ch4_boundary_leak")]:
            N[f"hold_q_{key}_{hk}_v"] = h["arms"][arm]["QLIKE"]
            N[f"hold_q_{key}_{hk}"] = q4(N[f"hold_q_{key}_{hk}_v"])
    N["hold_ref_vs_garch"] = pct(ho["heads"]["direct"]["vs_baselines"]["reference"]["garch_11"]["inflation_pct"])

    N["leakage_table_fold"] = _load("leakage_inflation_table.json")["fold"]

    # chart label strings (what PowerPoint prints with the number formats used below)
    N["main_labels"] = [N[k] for k in ("ch1_d", "ch2_d", "ch4_d", "ch5_d", "ch1_g", "ch2_g", "ch3_g", "ch4_g", "ch5_g")]
    N["bound_labels"] = [N[k + "_hi"] for k in ("ch2_d", "ch4_d", "ch5_d", "ch2_g", "ch3_g", "ch4_g", "ch5_g")]
    N["h288_labels"] = [N["h288_" + k] for k in ("ch1", "ch4", "ch5", "ch2", "ptb")]
    N["hold_labels"] = [N[f"hold_{k}_{h}"] for h in "dg" for k in ("ch1", "ch2", "ch3", "ch4", "ch5")
                        if N[f"hold_{k}_{h}_v"] is not None]
    return N


def git_preaudit():
    """Withdrawn pre-audit figures, read from the initial 15 Oct commit (dea2cb5)."""
    try:
        txt = subprocess.run(["git", "-C", ROOT, "show", "dea2cb5:README.md"], capture_output=True,
                             text=True, check=True, encoding="utf-8").stdout
    except (OSError, subprocess.CalledProcessError) as e:
        raise SystemExit(f"git show dea2cb5:README.md failed ({e}); run inside a git clone of the repository.")
    P = {}

    def grab(key, pattern):
        m = re.search(pattern, txt)
        P[key] = m.group(1) if m else None

    grab("pre_garch", r"GARCH\(1,1\) Baseline\*\* \| (0\.\d+)")
    grab("pre_ref", r"Causal Reference GPT \(Our Model\)\*\* \| \*\*(0\.\d+)")
    grab("pre_garch_infl", r"GARCH\(1,1\) Baseline\*\* \| 0\.\d+ \| \$(-?\d+\.\d+)")
    grab("pre_ch4", r"Channel 4 \(Boundary Leak\)\*\* \| \*\*0\.\d+\*\* \| \*\*\$\+(\d+\.\d+)")
    grab("pre_ch1", r"Channel 1 \(Window Norm Leak\)\*\* \| 0\.\d+ \| \*\*\$\+(\d+\.\d+)")
    grab("pre_mismatch", r"(\d+\.\d+)% boundary straddling")
    r1 = lambda k: f"{float(P[k]):.1f}" if P.get(k) else None
    P["pre_ch4_r"], P["pre_garch_infl_r"] = r1("pre_ch4"), r1("pre_garch_infl")
    P["pre_mismatch_r"] = f"{float(P['pre_mismatch']):.0f}" if P.get("pre_mismatch") else None
    return P


# Values quoted on the slides, in the charts and in the narration. The build stops if results/ disagrees.
EXPECTED = {
    # headline H = 12, 3 seeds (robust_stats_h12)
    "ch1_d": "+63.39%", "ch1_g": "+62.17%", "ch1_d_lo": "+61.77%", "ch1_d_hi": "+65.04%",
    "ch1_d_re": "+62.03%", "ch1_d_re_ci": "[+58.80, +65.26]", "ch1_d_i2": "0%", "ch1_d_n": "26,290",
    "ch2_d": "+0.64%", "ch2_g": "-2.58%", "ch3_g": "+0.24%", "ch4_d": "+0.47%", "ch4_g": "-0.95%",
    "ch5_d": "+1.78%", "ch5_g": "+0.26%", "ch5_d_holm": "0.034", "ch3_g_holm": "0.038",
    "ch2_d_hi": "+1.44%", "ch4_d_hi": "+1.05%", "ch5_d_hi": "+2.84%", "ch2_g_hi": "-0.60%",
    "ch3_g_hi": "+0.39%", "ch4_g_hi": "+0.29%", "ch5_g_hi": "+1.73%",
    "pos_d": "+99.98%", "ptb_d": "+0.42%", "ptb_g": "+1.89%", "ptb_d_holm": "0.586", "ptb_g_holm": "0.115",
    "s0_ch1_d": "+63.67%",
    # extensions (folds only)
    "k1": "+2.70%", "k3": "+6.95%", "k6": "+18.38%", "k1_holm": "0.040",
    "xa_leak_d": "+0.95%", "xa_leak_d_holm": "0.188", "xa_more_d": "+3.06%",
    "heal": "+1.70%", "heal_lo": "+0.07%", "heal_hi": "+3.38%", "heal_holm": "0.295",
    # MCS
    "mcs_alpha": "0.10", "ref3_d": "0.5580", "unc3_d": "0.5556", "garch_folds": "0.5747", "garch_folds_p": "0.042",
    "garch_hold": "0.4267", "mcs_folds_in": ["GPT reference, direct", "GPT unconstrained BPE, direct"],
    "mcs_hold_in": ["GARCH(1,1)"],
    # data
    "fold_test": ["2,160", "2,184", "4,416", "4,330", "4,416", "8,784"], "n_test_folds": "26,290",
    # token audit
    "bm_origins": "500", "tok_labels": ["0.0%", "0.0%", "32.0%", "0.0%", "57.0%", "29.5%"],
    "bm_ch5_pert": "57%", "bm_ch4_aligned": "32%", "bm_ref_aligned": "0%", "comp64": "1.40",
    # negative control
    "nc_floor": "0.1268", "nc_ch1_d": "0.0694", "nc_ch1_g": "0.1182", "nc_n_test": "2,159",
    "nc_labels": ["0.0002", "0.0694", "0.1268", "0.1293", "0.1297", "0.1304", "0.1316", "0.1332", "0.1389"],
    "nc_small_lo": "+1.41%", "nc_small_hi": "+2.12%", "i2_max_small": "95%",
    "k1_seeds": "1", "k3_seeds": "1", "k6_seeds": "1", "ch1_d_seeds": "3", "heal_seeds": "1",
    "xa_more_d_seeds": "2", "xa_leak_d_seeds": "2",
    "ext_labels": ["+2.70%", "+6.95%", "+18.38%", "+63.39%", "+0.95%", "+3.06%", "+1.70%", "+0.12%", "+1.82%"],
    "nc_violators": ["ch1_window_norm_leak", "positive_control"],
    # next-day horizon
    "h288_ch1": "+96.36%", "h288_ch2": "+5.67%", "h288_ch4": "+16.10%", "h288_ch5": "+7.02%",
    "h288_pos": "+99.99%", "h288_ptb": "-19.53%",
    "h288_q_ref": "0.3128", "h288_q_ch4c": "0.3739", "h288_q_ch4": "0.3137",
    # holdout
    "hold_rescored": False, "hold_commit": "f4ad204", "hold_date": "2026-11-17", "hold_n": "8,760",
    "hold_ch1_d": "+73.90%", "hold_ch1_g": "+68.32%", "hold_ch2_d": "-5.26%", "hold_ch2_g": "+4.93%",
    "hold_ch4_g": "+15.30%", "hold_ptb_g": "-20.32%", "hold_ref_vs_garch": "-3.49%",
    "hold_q_ref_g": "0.4603", "hold_q_ch4c_g": "0.5538", "hold_q_ch4_g": "0.4691",
    "hold_labels": ["+73.90%", "-5.26%", "-0.95%", "+1.33%", "+68.32%", "+4.93%", "+0.28%", "+15.30%", "+1.19%"],
    # chart labels
    "main_labels": ["+63.39%", "+0.64%", "+0.47%", "+1.78%", "+62.17%", "-2.58%", "+0.24%", "-0.95%", "+0.26%"],
    "bound_labels": ["+1.44%", "+1.05%", "+2.84%", "-0.60%", "+0.39%", "+0.29%", "+1.73%"],
    "h288_labels": ["+96.36%", "+16.10%", "+7.02%", "+5.67%", "-19.53%"],
    "leakage_table_fold": 6,
}
EXPECTED_PREAUDIT = {  # initial 15 Oct commit dea2cb5 (withdrawn numbers; never results)
    "pre_garch": "0.7770", "pre_ref": "0.6383", "pre_garch_infl": "-21.74", "pre_ch4": "16.56",
    "pre_ch1": "7.09", "pre_mismatch": "84.00", "pre_ch4_r": "16.6", "pre_garch_infl_r": "-21.7",
    "pre_mismatch_r": "84",
}
FIXED_FACTS = {  # model / code facts quoted on slides (checked against code in --check, see verify())
    "params": "855,169",
}


def verify(N, P):
    bad = [(k, v, N.get(k)) for k, v in EXPECTED.items() if N.get(k) != v]
    bad += [(k, v, P.get(k)) for k, v in EXPECTED_PREAUDIT.items() if P.get(k) != v]
    try:  # parameter count from the real model (needs torch); skipped with a note if torch is missing
        sys.path.insert(0, ROOT)
        from models.gpt_model import TimeSeriesGPT
        got = f"{TimeSeriesGPT(vocab_size=113, pad_id=112, max_seq_len=192).count_parameters():,}"
        if got != FIXED_FACTS["params"]:
            bad.append(("params", FIXED_FACTS["params"], got))
        n_code = 1
    except ImportError:
        print("note: torch not installed, parameter count not re-derived (855,169 per models/gpt_model.py)")
        n_code = 0
    if bad:
        for k, want, got in bad:
            print(f"MISMATCH {k}: expected {want!r}, sources give {got!r}")
        raise SystemExit("Numbers in results/ (or git/code) differ from the deck text. Nothing written.")
    n = len(EXPECTED) + len(EXPECTED_PREAUDIT) + n_code
    n_vals = sum(len(v) if isinstance(v, list) else 1 for v in EXPECTED.values()) + len(EXPECTED_PREAUDIT) + n_code
    msg = f"Verified {n} checks ({n_vals} quoted values) against results/, git and code."
    print(msg)
    return n_vals, msg


# ============================================================================= drawing helpers
def _no_shadow(shape):
    """Remove the theme style (its effectRef draws shadows); fills, lines and colors are set explicitly."""
    try:
        shape.shadow.inherit = False
    except Exception:
        pass
    st = shape._element.find(qn("p:style"))
    if st is not None:
        shape._element.remove(st)


def text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
         anchor=MSO_ANCHOR.TOP, spacing=4, margin=0.04, italic=False):
    """paras: str or list of str / (str, dict) with optional size/color/bold/italic/align/bullet/spacing."""
    if isinstance(paras, str):
        paras = [paras]
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(margin)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    for i, para in enumerate(paras):
        txt, opt = (para, {}) if isinstance(para, str) else para
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("spacing", spacing))
        runs = txt if isinstance(txt, list) else [(txt, {})]
        if opt.get("bullet"):
            runs = [("•  ", {"color": opt.get("bullet_color", MUTED)})] + list(runs)
        for rt, ro in runs:
            r = p.add_run()
            r.text = rt
            r.font.name = FONT
            r.font.size = Pt(ro.get("size", opt.get("size", size)))
            r.font.bold = ro.get("bold", opt.get("bold", bold))
            r.font.italic = ro.get("italic", opt.get("italic", italic))
            r.font.color.rgb = ro.get("color", opt.get("color", color))
    return tb


def box(slide, x, y, w, h, fill, line=None, shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.08, line_w=1.0,
        label=None, size=14, color=INK, bold=False, align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE,
        dash=False):
    s = slide.shapes.add_shape(shape, x, y, w, h)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        s.adjustments[0] = radius
    if fill is None:
        s.fill.background()
    else:
        s.fill.solid()
        s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
        s.line.width = Pt(line_w)
        if dash:
            ln = s.line._get_or_add_ln()
            d = ln.makeelement(qn("a:prstDash"), {"val": "dash"})
            ln.append(d)
    _no_shadow(s)
    if label is not None:
        tf = s.text_frame
        tf.word_wrap = True
        tf.vertical_anchor = anchor
        tf.margin_left = tf.margin_right = Inches(0.08)
        tf.margin_top = tf.margin_bottom = Inches(0.03)
        lines = label if isinstance(label, list) else [label]
        for i, ln_ in enumerate(lines):
            txt, opt = (ln_, {}) if isinstance(ln_, str) else ln_
            p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
            p.alignment = opt.get("align", align)
            p.space_after = Pt(opt.get("spacing", 0))
            r = p.add_run()
            r.text = txt
            r.font.name = FONT
            r.font.size = Pt(opt.get("size", size))
            r.font.bold = opt.get("bold", bold)
            r.font.italic = opt.get("italic", False)
            r.font.color.rgb = opt.get("color", color)
    return s


def circle(slide, cx, cy, d, fill, label, size=14, color=WHITE):
    return box(slide, cx - d // 2, cy - d // 2, d, d, fill, shape=MSO_SHAPE.OVAL, label=label, size=size,
               color=color, bold=True)


def arrow(slide, x1, y1, x2, y2, color=MUTED, width=1.75, head=True, dash=False):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = color
    c.line.width = Pt(width)
    _no_shadow(c)
    ln = c.line._get_or_add_ln()
    if dash:
        ln.append(ln.makeelement(qn("a:prstDash"), {"val": "dash"}))
    if head:
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))
    return c


def chip(slide, x, y, w, label, fill=PANEL, color=INK, size=12, h=Inches(0.36), bold=False):
    return box(slide, x, y, w, h, fill, label=label, size=size, color=color, radius=0.5, bold=bold)


def stat(slide, x, y, w, h, big, small, color=LEAK, fill=PANEL, big_size=34, small_size=13):
    box(slide, x, y, w, h, fill, radius=0.06)
    text(slide, x + Inches(0.18), y + Inches(0.1), w - Inches(0.3), Inches(0.7), big, size=big_size,
         color=color, bold=True)
    text(slide, x + Inches(0.18), y + Inches(0.1) + Pt(big_size * 1.35), w - Inches(0.3),
         h - Inches(0.2) - Pt(big_size * 1.35), small, size=small_size, color=INK)


def image(slide, path, x, y, w, h):
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    dw, dh = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(path, x + (w - dw) // 2, y + (h - dh) // 2, Emu(dw), Emu(dh))


def bar_chart(slide, x, y, w, h, categories, series, fmt='+0.00"%";-0.00"%"', horizontal=True,
              colors=None, point_colors=None, legend=True, vmin=None, vmax=None, gap=60, overlap=-10,
              label_size=12, axis_size=12, show_value_axis=False):
    """series: list of (name, values). colors: one RGB per series. point_colors: {series_idx: [rgb per point]}."""
    multi = legend and len(series) > 1
    if multi:  # legend row inside the chart's box, drawn by hand so its order matches the bars
        lx = x + Inches(0.1)
        for i, (name, _) in enumerate(series):
            box(slide, lx, y + Inches(0.08), Inches(0.2), Inches(0.2), (colors or [DIRECT, SAFE, LEAK])[i],
                radius=0.15)
            text(slide, lx + Inches(0.27), y, Inches(3.6), Inches(0.36), name, size=13, color=INK, margin=0,
                 anchor=MSO_ANCHOR.MIDDLE)
            lx += Inches(0.55) + Pt(len(name) * 6.6)
        y, h = y + Inches(0.45), h - Inches(0.45)
    cd = CategoryChartData()
    cd.categories = categories
    for name, vals in series:
        cd.add_series(name, vals, number_format=fmt)
    kind = XL_CHART_TYPE.BAR_CLUSTERED if horizontal else XL_CHART_TYPE.COLUMN_CLUSTERED
    gf = slide.shapes.add_chart(kind, x, y, w, h, cd)
    ch = gf.chart
    ch.font.name = FONT
    ch.font.size = Pt(axis_size)
    ch.font.color.rgb = INK
    ch.has_title = False
    ch.has_legend = False
    plot = ch.plots[0]
    plot.gap_width = gap
    plot.overlap = overlap
    plot.vary_by_categories = False
    plot.has_data_labels = True
    dl = plot.data_labels
    dl.number_format = fmt
    dl.number_format_is_linked = False
    dl.position = XL_LABEL_POSITION.OUTSIDE_END
    dl.font.size = Pt(label_size)
    dl.font.color.rgb = INK
    for i, s in enumerate(plot.series):
        s.format.fill.solid()
        s.format.fill.fore_color.rgb = (colors or [DIRECT, SAFE, LEAK])[i]
        s.invert_if_negative = False
        if point_colors and i in point_colors:
            for j, c in enumerate(point_colors[i]):
                pt = s.points[j]
                pt.format.fill.solid()
                pt.format.fill.fore_color.rgb = c
    ca, va = ch.category_axis, ch.value_axis
    ca.tick_labels.font.size = Pt(axis_size)
    ca.format.line.color.rgb = LINE
    ca.has_major_gridlines = False
    ca.tick_label_position = XL_TICK_LABEL_POSITION.LOW
    if horizontal:
        ca.reverse_order = True
    va.has_major_gridlines = False
    va.visible = show_value_axis
    va.tick_labels.font.size = Pt(axis_size - 1)
    va.format.line.color.rgb = LINE
    if vmin is not None:
        va.minimum_scale = vmin
    if vmax is not None:
        va.maximum_scale = vmax
    return ch


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SLIDE_W, SLIDE_H
        self.n = 0
        self.titles = []

    def _blank(self):
        self.n += 1
        return self.prs.slides.add_slide(self.prs.slide_layouts[6])

    def content(self, title, kicker, source, notes):
        s = self._blank()
        self.titles.append(title)
        text(s, Inches(0.6), Inches(0.32), Inches(11.5), Inches(0.32), kicker.upper(), size=12, color=LEAK,
             bold=True, margin=0)
        text(s, Inches(0.6), Inches(0.62), Inches(12.1), Inches(0.75), title, size=30, color=NAVY, bold=True,
             margin=0)
        if source:
            text(s, Inches(0.6), Inches(7.04), Inches(11.4), Inches(0.3), f"Source: {source}", size=10,
                 color=MUTED, margin=0)
        text(s, Inches(12.2), Inches(7.04), Inches(0.55), Inches(0.3), str(self.n), size=10, color=MUTED,
             align=PP_ALIGN.RIGHT, margin=0)
        s.notes_slide.notes_text_frame.text = notes
        return s

    def dark(self, title, notes, kicker=None, page=True):
        s = self._blank()
        self.titles.append(title)
        box(s, 0, 0, SLIDE_W, SLIDE_H, NAVY, shape=MSO_SHAPE.RECTANGLE)
        if kicker:
            text(s, Inches(0.6), Inches(0.32), Inches(11.5), Inches(0.32), kicker.upper(), size=12, color=LEAK,
                 bold=True, margin=0)
            text(s, Inches(0.6), Inches(0.62), Inches(12.1), Inches(0.75), title, size=30, color=WHITE,
                 bold=True, margin=0)
        if page:
            text(s, Inches(12.2), Inches(7.04), Inches(0.55), Inches(0.3), str(self.n), size=10, color=ICE,
                 align=PP_ALIGN.RIGHT, margin=0)
        s.notes_slide.notes_text_frame.text = notes
        return s


# ============================================================================= slides
NOTES = {}
ON_SCREEN = {}


def build(N, P, out_path, check_msg=""):
    d = Deck()
    fig = lambda name: os.path.join(FIG, name)
    IN = Inches

    # 1 ---------------------------------------------------------------- title (dark)
    s = d.dark("Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization", NOTES[1], page=False)
    text(s, IN(0.9), IN(0.85), IN(8), IN(0.4), "DA4022 COURSE PROJECT  ·  FINAL PRESENTATION", size=13,
         color=LEAK, bold=True, margin=0)
    text(s, IN(0.9), IN(1.45), IN(11.4), IN(2.1),
         "Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series",
         size=38, color=WHITE, bold=True, margin=0)
    text(s, IN(0.9), IN(3.65), IN(11), IN(0.5),
         "Which parts of a BPE tokenizer leak the future, and by how much?", size=20, color=ICE, margin=0,
         italic=True)
    # motif: context | origin | horizon strip
    x0, y0 = IN(0.9), IN(4.6)
    box(s, x0, y0, IN(6.2), IN(0.5), RGBColor(0x23, 0x36, 0x5C), radius=0.2,
        label="context  ·  bars ≤ t − 1", size=13, color=ICE)
    box(s, x0 + IN(6.35), y0, IN(2.6), IN(0.5), LEAK, radius=0.2, label="horizon", size=13, color=WHITE,
        bold=True)
    arrow(s, x0 + IN(6.27), y0 - IN(0.2), x0 + IN(6.27), y0 + IN(0.7), color=WHITE, width=2, head=False)
    text(s, x0 + IN(6.0), y0 + IN(0.72), IN(0.6), IN(0.3), "t", size=14, color=WHITE, bold=True,
         align=PP_ALIGN.CENTER, margin=0)
    text(s, IN(0.9), IN(5.75), IN(11.5), IN(0.45), "  ·  ".join(AUTHORS), size=17, color=WHITE, margin=0)
    text(s, IN(0.9), IN(6.2), IN(11.5), IN(0.4), "November 2026  ·  " + REPO_URL, size=13, color=ICE,
         margin=0)
    ON_SCREEN[1] = "Title, team, and the context / origin / horizon motif"

    # 2 ---------------------------------------------------------------- problem timeline
    s = d.content("A tokenizer can see past the forecast origin in two ways", "The problem",
                  "Götz et al. (2026) BPE motif tokenizer; Zhang et al. (2026) incremental-tokenization definition",
                  NOTES[2])
    L, R, Y = IN(0.9), IN(12.4), IN(3.55)
    tx = IN(8.35)  # origin t
    box(s, L, Y, tx - L - IN(0.05), IN(0.7), DIRECT_T, radius=0.15,
        label=[("Context  [t − 144, t)", {"bold": True, "color": DIRECT})], size=16)
    box(s, tx + IN(0.05), Y, R - tx - IN(0.05), IN(0.7), LEAK_T, radius=0.15,
        label=[("Horizon  [t, t + H)", {"bold": True, "color": LEAK})], size=16)
    arrow(s, tx, Y - IN(0.35), tx, Y + IN(1.05), color=NAVY, width=2.5, head=False)
    text(s, tx - IN(1.0), Y + IN(1.08), IN(2.0), IN(0.35), "forecast origin t", size=14, color=NAVY, bold=True,
         align=PP_ALIGN.CENTER, margin=0)
    # fit-time (above)
    text(s, L, IN(1.6), IN(5.5), IN(0.35), "FIT-TIME LEAKS", size=13, color=LEAK, bold=True, margin=0)
    box(s, IN(3.2), IN(2.05), IN(8.3), IN(0.5), None, line=LEAK, dash=True, radius=0.3,
        label="Ch1 · normalization window reaches into the horizon", size=14, color=INK)
    box(s, L, IN(2.7), R - L, IN(0.5), None, line=LEAK, dash=True, radius=0.3,
        label="Ch2 / Ch3 · merge or decoding tables fitted on data that overlaps the test period", size=14,
        color=INK)
    # encode-time (below)
    text(s, L, IN(4.85), IN(5.5), IN(0.35), "ENCODE-TIME LEAKS", size=13, color=LEAK, bold=True, margin=0)
    box(s, tx - IN(1.2), IN(5.3), IN(2.4), IN(0.55), LEAK, radius=0.3, label="token straddles t", size=14,
        color=WHITE, bold=True)
    text(s, tx + IN(1.35), IN(5.22), IN(3.0), IN(0.75),
         "Ch4 · tokenize the stream, then split: rank-ordered BPE is not incremental", size=14, color=INK,
         anchor=MSO_ANCHOR.MIDDLE, margin=0)
    box(s, tx - IN(2.6), IN(6.1), IN(3.4), IN(0.5), None, line=LEAK, dash=True, radius=0.3,
        label="Ch5 · centered gate window", size=14, color=INK)
    text(s, L, IN(5.3), IN(4.6), IN(1.3),
         [("The safe alternative, tokenizing the context alone, leaves a partial final token "
           "(partial-token bias).", {})], size=14, color=MUTED, margin=0)
    ON_SCREEN[2] = "Timeline of context / origin t / horizon, with fit-time and encode-time leak markers"

    # 3 ---------------------------------------------------------------- question + contributions
    s = d.content("Measure each leak on its own, against a matched control", "Research question & contributions",
                  "FINAL_REPORT.md §1", NOTES[3])
    text(s, IN(0.6), IN(1.6), IN(12), IN(0.9),
         "Which parts of a BPE motif-tokenization pipeline leak post-origin information, and how much does "
         "each inflate measured accuracy?", size=20, color=INK, italic=True, margin=0)
    cards = [("1", "Per-channel audit", "Five leaks, each switched on alone, in training and testing"),
             ("2", "Boundary vs partial token", "Two failure modes measured against one causal reference"),
             ("3", "Causal tokenizer", "Hour-bounded BPE, hourly scale tokens, causal RoV×SEI gate"),
             ("4", "Controls & holdout", "6 folds, GARCH oracle floor, positive control, 2025 scored once")]
    cw, cg, cy = IN(2.85), IN(0.27), IN(3.05)
    for i, (num, head, body) in enumerate(cards):
        cx = IN(0.6) + i * (cw + cg)
        box(s, cx, cy, cw, IN(3.0), PANEL, radius=0.06)
        circle(s, cx + IN(0.6), cy + IN(0.65), IN(0.7), NAVY if i < 3 else SAFE, num, size=20)
        text(s, cx + IN(0.25), cy + IN(1.25), cw - IN(0.4), IN(0.8), head, size=19, color=NAVY, bold=True,
             margin=0)
        text(s, cx + IN(0.25), cy + IN(2.05), cw - IN(0.4), IN(1.4), body, size=15, color=INK, margin=0)
    ON_SCREEN[3] = "Research question and four contribution cards"

    # 4 ---------------------------------------------------------------- pipeline flowchart
    s = d.content("The causal reference pipeline, and where each leak switches on", "Method",
                  "data/preprocess.py; tokenization/bpe_tokenizer.py, causal_gate.py, leak_tokenizer.py; "
                  "models/gpt_model.py", NOTES[4])
    steps = [("Returns", "5-min BTCUSDT", None),
             ("EWMA norm", "σ uses bars < t", "1"),
             ("32 bins", "edges [−4, 4]", None),
             ("BPE by hour", "64 merges + scale token", "2 4"),
             ("Causal gate", "RoV×SEI, trailing", "5"),
             ("GPT", "855,169 params", None)]
    bw, bh, by, gx = IN(1.45), IN(1.3), IN(2.25), IN(0.25)
    xs = [IN(0.6) + i * (bw + gx) for i in range(len(steps))]
    for i, (head, sub, chs) in enumerate(steps):
        box(s, xs[i], by, bw, bh, NAVY if head == "GPT" else DIRECT_T, radius=0.12,
            label=[(head, {"bold": True, "size": 16, "color": WHITE if head == "GPT" else NAVY}),
                   (sub, {"size": 12, "color": ICE if head == "GPT" else INK})])
        if i:
            arrow(s, xs[i - 1] + bw + IN(0.03), by + bh // 2, xs[i] - IN(0.03), by + bh // 2)
        if chs:
            for j, c in enumerate(chs.split()):
                circle(s, xs[i] + bw // 2 + IN(0.26) * (1 if j else -1) * (len(chs.split()) > 1),
                       by - IN(0.32), IN(0.44), LEAK, c, size=14)
    # heads
    hx = xs[-1] + bw + IN(0.5)
    hw = IN(12.73) - hx
    box(s, hx, IN(1.75), hw, IN(1.0), SAFE_T, radius=0.12,
        label=[("Direct head", {"bold": True, "size": 15, "color": NAVY}), ("regress RV", {"size": 12})])
    box(s, hx, IN(2.95), hw, IN(1.0), SAFE_T, radius=0.12,
        label=[("Generative head", {"bold": True, "size": 15, "color": NAVY}),
               ("sample tokens, decode E[x²]", {"size": 12})])
    circle(s, hx + hw // 2, IN(4.27), IN(0.4), LEAK, "3", size=13)
    arrow(s, xs[-1] + bw, by + IN(0.35), hx, IN(2.25))
    arrow(s, xs[-1] + bw, by + bh - IN(0.35), hx, IN(3.45))
    # target strip
    box(s, IN(0.6), IN(4.6), IN(12.13), IN(0.55), PANEL, radius=0.3,
        label="Target  RV_H[t] = Σ r²(t+k), k < H   ·   H = 12 (next hour) and H = 288 (next day)   ·   "
              "scored by QLIKE", size=14, color=INK)
    # legend for channels
    legend = [("1", "window norm uses the horizon"), ("2", "merge table fitted on train + test"),
              ("3", "decoding tables fitted on train + test"), ("4", "tokenize stream, then split"),
              ("5", "centered (non-causal) gate window")]
    lx, ly = IN(0.6), IN(5.6)
    for i, (c, lab) in enumerate(legend):
        col, row = i % 3, i // 3
        cx = lx + col * IN(4.1)
        cy = ly + row * IN(0.65)
        circle(s, cx + IN(0.22), cy + IN(0.22), IN(0.44), LEAK, c, size=14)
        text(s, cx + IN(0.55), cy, IN(3.5), IN(0.45), lab, size=15, color=INK, anchor=MSO_ANCHOR.MIDDLE,
             margin=0)
    ON_SCREEN[4] = "Pipeline flowchart with channel markers 1–5"

    # 5 ---------------------------------------------------------------- protocol
    s = d.content("One switch per arm, its own model, and two controls", "Experiment protocol",
                  "tokenization/leak_tokenizer.py (ARMS); experiments/run_leak_experiments.py; "
                  "simulation/garch_simulator.py", NOTES[5])
    lanes = [("Matched control", DIRECT_T, DIRECT, "same pipeline, switch off"),
             ("Leak arm", LEAK_T, LEAK, "exactly one switch on")]
    stages = ["Tokenize", "Train its own GPT", "Forecast test origins", "QLIKE"]
    for li, (name, fill, col, sub) in enumerate(lanes):
        ly = IN(1.75) + li * IN(1.45)
        box(s, IN(0.6), ly, IN(2.0), IN(1.05), fill, radius=0.12,
            label=[(name, {"bold": True, "size": 16, "color": col}), (sub, {"size": 12})])
        for j, st in enumerate(stages):
            sx = IN(2.95) + j * IN(1.55)
            box(s, sx, ly + IN(0.15), IN(1.35), IN(0.75), PANEL, radius=0.15, label=st, size=13, color=INK)
            arrow(s, sx - IN(0.33) if j else IN(2.62), ly + IN(0.525), sx - IN(0.02), ly + IN(0.525))
    text(s, IN(2.95), IN(4.2), IN(6.2), IN(0.5), "the leak is present in training and in testing (Ch3 reuses the "
         "reference model: decoding tables never enter training)", size=12, color=MUTED, italic=True, margin=0)
    # inflation
    box(s, IN(9.3), IN(2.25), IN(3.43), IN(1.5), NAVY, radius=0.1,
        label=[("Inflation", {"bold": True, "size": 18, "color": WHITE}),
               ("(QLIKE control − QLIKE leak) / QLIKE control", {"size": 13, "color": ICE}),
               ("positive = the leak flatters the forecast", {"size": 12, "color": ICE})])
    arrow(s, IN(9.08), IN(2.3), IN(9.28), IN(2.75))
    arrow(s, IN(9.08), IN(3.75), IN(9.28), IN(3.25))
    # controls
    box(s, IN(0.6), IN(4.85), IN(5.95), IN(1.6), SAFE_T, radius=0.08)
    text(s, IN(0.85), IN(4.95), IN(5.5), IN(0.4), "Negative control", size=17, color=SAFE, bold=True, margin=0)
    text(s, IN(0.85), IN(5.4), IN(5.5), IN(1.3),
         "Same pipeline on simulated GARCH(1,1). The analytic oracle sets a QLIKE floor no causal model can "
         "beat in expectation.", size=15, color=INK, margin=0)
    box(s, IN(6.78), IN(4.85), IN(5.95), IN(1.6), LEAK_T, radius=0.08)
    text(s, IN(7.03), IN(4.95), IN(5.5), IN(0.4), "Positive control", size=17, color=LEAK, bold=True, margin=0)
    text(s, IN(7.03), IN(5.4), IN(5.5), IN(1.3),
         "Normalize the context by the true future realized volatility: an extreme leak, as a reference "
         "for scale.", size=15, color=INK, margin=0)
    ON_SCREEN[5] = "Two-lane protocol diagram, inflation formula, negative and positive control cards"

    # 6 ---------------------------------------------------------------- data / folds gantt
    s = d.content("Six expanding walk-forward folds and one 2025 holdout", "Data & evaluation",
                  "data/folds.py; results/fold_<k>_baselines.json; results/holdout_h12_leak_audit.json", NOTES[6])
    gx0, gx1 = IN(2.0), IN(10.6)
    per_q = (gx1 - gx0) / 20  # 2021-Q1 .. 2025-Q4
    qx = lambda q: gx0 + int(per_q * q)
    for yr in range(5):
        text(s, qx(4 * yr), IN(1.55), int(per_q * 4), IN(0.3), str(2021 + yr), size=13, color=MUTED,
             align=PP_ALIGN.CENTER, margin=0)
        if yr:
            arrow(s, qx(4 * yr), IN(1.85), qx(4 * yr), IN(5.55), color=LINE, width=0.75, head=False)
    folds = [("Fold 1", 4, 5), ("Fold 2", 5, 6), ("Fold 3", 6, 8), ("Fold 4", 8, 10), ("Fold 5", 10, 12),
             ("Fold 6", 12, 16), ("Holdout", 16, 20)]
    text(s, IN(10.75), IN(1.55), IN(2.0), IN(0.3), "test origins", size=13, color=MUTED, margin=0)
    for i, (name, split, end) in enumerate(folds):
        ry = IN(1.95) + i * IN(0.5)
        hold = name == "Holdout"
        text(s, IN(0.6), ry, IN(1.3), IN(0.38), name, size=14, color=NAVY, bold=True, anchor=MSO_ANCHOR.MIDDLE,
             margin=0)
        box(s, qx(0), ry + IN(0.04), qx(split) - qx(0) - IN(0.03), IN(0.3), DIRECT_T, radius=0.3)
        box(s, qx(split), ry + IN(0.04), qx(end) - qx(split), IN(0.3), SAFE if hold else LEAK, radius=0.3)
        cnt = N["hold_n"] if hold else N["fold_test"][i]
        text(s, IN(10.75), ry, IN(2.0), IN(0.38), cnt + ("  · scored once" if hold else ""), size=14,
             color=INK, bold=hold, anchor=MSO_ANCHOR.MIDDLE, margin=0)
    text(s, IN(10.75), IN(5.5), IN(2.2), IN(0.38), f"{N['n_test_folds']} pooled", size=14, color=NAVY,
         bold=True, margin=0)
    # legend
    box(s, IN(2.0), IN(5.6), IN(0.4), IN(0.22), DIRECT_T, radius=0.3)
    text(s, IN(2.5), IN(5.52), IN(1.6), IN(0.35), "train", size=13, color=INK, margin=0)
    box(s, IN(3.4), IN(5.6), IN(0.4), IN(0.22), LEAK, radius=0.3)
    text(s, IN(3.9), IN(5.52), IN(1.6), IN(0.35), "test", size=13, color=INK, margin=0)
    box(s, IN(4.7), IN(5.6), IN(0.4), IN(0.22), SAFE, radius=0.3)
    text(s, IN(5.2), IN(5.52), IN(2.5), IN(0.35), "2025 holdout", size=13, color=INK, margin=0)
    stats = ["QLIKE", "block bootstrap CIs", "Diebold–Mariano + Holm", "TOST ±5%", "random-effects pooling",
             "Model Confidence Set"]
    sx = IN(0.6)
    for st in stats:
        w = IN(0.32) + Pt(len(st) * 7.0)
        chip(s, sx, IN(6.25), w, st, fill=NAVY, color=WHITE, size=13)
        sx += w + IN(0.15)
    ON_SCREEN[6] = "Gantt of 6 expanding folds + 2025 holdout with test-origin counts; statistics chips"

    # 7 ---------------------------------------------------------------- continuity
    s = d.content("Pre-audit numbers stay withdrawn", "Continuity since the midterm",
                  "git history (7f9f6a6–dea2cb5, 15 Oct; 24d9c8f, f4ad204, 156090a, 17 Nov); "
                  "submitted midterm deck (15 Oct)", NOTES[7])
    tl = [("15 Oct", "Initial repo", "7f9f6a6–dea2cb5: pre-audit numbers", LEAK),
          ("15 Oct", "Audit", "bugs traced and fixed", NAVY),
          ("15 Oct", "Midterm", "results re-run; Ch4, GARCH and 84% withdrawn", NAVY),
          ("17 Nov", "Fixes committed", "24d9c8f, fc12681", SAFE),
          ("17 Nov", "Holdout scored once", "at f4ad204 (156090a)", SAFE),
          ("20–21 Nov", "Extensions", "+ seeds 1–2, folds only", SAFE)]
    tx0, tw = IN(1.6), IN(10.13)
    step = tw / (len(tl) - 1)
    ty = IN(2.05)
    arrow(s, tx0, ty, tx0 + tw, ty, color=LINE, width=3, head=False)
    for i, (dt, head, sub, col) in enumerate(tl):
        cx = tx0 + int(step * i)
        circle(s, cx, ty, IN(0.32), col, "", size=8)
        text(s, cx - IN(1.0), ty - IN(0.55), IN(2.0), IN(0.3), dt, size=13, color=MUTED, bold=True,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx - IN(1.0), ty + IN(0.25), IN(2.0), IN(0.35), head, size=15, color=NAVY, bold=True,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx - IN(1.0), ty + IN(0.6), IN(2.0), IN(0.6), sub, size=12, color=INK, align=PP_ALIGN.CENTER,
             margin=0)
    rows = [("Withdrawn pre-audit figure (initial repo, 15 Oct)", "Cause", "Now"),
            (f"Ch4 boundary leak +{P['pre_ch4']}%", "tokenizer never fitted: input shift",
             f"{N['ch4_d']} (folds, 3 seeds)"),
            (f"GARCH baseline {P['pre_garch_infl']}% vs our reference GPT", "scored on a shifted target array",
             f"folds: GARCH {N['garch_folds']}, GPT {N['ref3_d']}"),
            (f"{P['pre_mismatch'][:2]}% boundary mismatch", "measured on raw returns (2 of 32 bins)",
             f"{N['bm_ref_aligned']} hour-aligned · {N['bm_ch4_aligned']} unconstrained"),
            (f"Ch1 window-norm leak +{P['pre_ch1']}%", "one model, leak swapped in at test time",
             f"{N['ch1_d']}: own model per arm")]
    tbl_y = IN(3.55)
    colw = [IN(5.0), IN(3.9), IN(3.23)]
    for r, row in enumerate(rows):
        ry = tbl_y + r * IN(0.62)
        cx = IN(0.6)
        for c, val in enumerate(row):
            hdr = r == 0
            fill = NAVY if hdr else (PANEL if r % 2 else WHITE)
            box(s, cx, ry, colw[c] - IN(0.05), IN(0.56), fill, radius=0.12, label=val,
                size=13 if hdr else 14, color=WHITE if hdr else (LEAK if c == 0 else INK),
                bold=hdr or c == 2, align=PP_ALIGN.LEFT)
            cx += colw[c]
    ON_SCREEN[7] = "Timeline 15 Oct → 21 Nov; table of withdrawn pre-audit figures and what replaced them"

    # 8 ---------------------------------------------------------------- token causality
    s = d.content("Hour boundaries make the tokenizer incremental at the origin", "Result 1 · token audit",
                  f"results/boundary_mismatch_results.json (Fold 1, {N['bm_origins']} origins; perturbation "
                  "test on 200)", NOTES[8])
    cats = ["Reference: hour-bounded + causal gate", "Ch5: centered gate", "Ch4: unconstrained BPE"]
    bar_chart(s, IN(0.5), IN(1.55), IN(8.3), IN(5.3), cats,
              [("Token straddles the origin (hour-aligned)", N["tok_straddle"]),
               ("Prefix changes when the future is perturbed", N["tok_perturb"])],
              fmt='0.0"%"', colors=[DIRECT, LEAK], vmin=0, vmax=70, gap=70, overlap=-5, label_size=14,
              axis_size=14)
    stat(s, IN(9.2), IN(1.7), IN(3.55), IN(1.55), N["bm_ref_aligned"],
         "reference: no straddle at hour-aligned origins, no change under future perturbation", color=SAFE)
    stat(s, IN(9.2), IN(3.4), IN(3.55), IN(1.55), N["bm_ch5_pert"],
         "Ch5: pre-origin tokens change when the future changes")
    stat(s, IN(9.2), IN(5.1), IN(3.55), IN(1.55), N["comp64"],
         "bars per token at 64 merges: motifs are short", color=DIRECT)
    ON_SCREEN[8] = "Native chart: straddle vs perturbation by tokenizer; callouts 0% / 57% / 1.40"

    # 9 ---------------------------------------------------------------- negative control
    s = d.content("On simulated data, only the normalization leak beats the floor", "Result 2 · negative control",
                  f"results/negative_control_results.json (simulated GARCH(1,1), {N['nc_n_test']} test origins, "
                  "seed 0, direct head)", NOTES[9])
    labs = [r[0] for r in N["nc_rows"]]
    vals = [r[1] for r in N["nc_rows"]]
    cols = [NAVY if r[2] is None else (LEAK if r[2] else SAFE) for r in N["nc_rows"]]
    bar_chart(s, IN(0.5), IN(1.55), IN(8.4), IN(5.35), labs, [("QLIKE", vals)], fmt="0.0000", colors=[SAFE],
              point_colors={0: cols}, legend=False, vmin=0, vmax=0.17, gap=45, label_size=13, axis_size=13)
    stat(s, IN(9.3), IN(1.7), IN(3.45), IN(1.6), N["nc_floor"],
         "analytic oracle floor: no causal model beats it in expectation", color=NAVY)
    stat(s, IN(9.3), IN(3.45), IN(3.45), IN(1.6), N["nc_ch1_d"],
         f"Ch1 leak, direct head (generative {N['nc_ch1_g']}): impossibly good")
    text(s, IN(9.3), IN(5.2), IN(3.45), IN(1.7),
         [([("■ ", {"color": LEAK}), ("below the floor  ", {}), ("■ ", {"color": SAFE}), ("above it", {})], {}),
          (f"Ch2, Ch4 and Ch5 still gain {N['nc_small_lo']} to {N['nc_small_hi']} over their controls here, but "
           "none crosses the floor.", {"size": 13, "color": MUTED})], size=14, color=INK, margin=0)
    ON_SCREEN[9] = "Native chart: QLIKE by arm vs the 0.1268 oracle floor; Ch1 and positive control below"

    # 10 --------------------------------------------------------------- main result
    s = d.content("Ch1 dominates: touching the horizon inflates accuracy by 63%",
                  "Result 3 · 6 folds, H = 12, 3 seeds",
                  f"results/robust_stats_h12.md ({N['ch1_d_n']} origins, seeds 0–2 averaged per origin)", NOTES[10])
    cats = ["Ch1 window norm", "Ch2 merge table", "Ch3 decode tables", "Ch4 stream, then split", "Ch5 centered gate"]
    bar_chart(s, IN(0.5), IN(1.6), IN(8.3), IN(5.3), cats,
              [("Direct head", [N["ch1_d_v"], N["ch2_d_v"], None, N["ch4_d_v"], N["ch5_d_v"]]),
               ("Generative head", [N["ch1_g_v"], N["ch2_g_v"], N["ch3_g_v"], N["ch4_g_v"], N["ch5_g_v"]])],
              colors=[DIRECT, SAFE], vmin=-10, vmax=80, gap=55, overlap=-5, label_size=13, axis_size=14)
    stat(s, IN(9.2), IN(1.7), IN(3.55), IN(1.6), N["ch1_d"],
         f"Ch1, direct head (90% CI {N['ch1_d_lo']} to {N['ch1_d_hi']})")
    stat(s, IN(9.2), IN(3.45), IN(3.55), IN(1.6), N["ch1_d_re"],
         f"random effects across folds {N['ch1_d_re_ci']}, I² = {N['ch1_d_i2']}", color=NAVY)
    text(s, IN(9.2), IN(5.3), IN(3.55), IN(1.4),
         f"Positive control for scale: {N['pos_d']}. Ch3 acts only through the generative head.", size=13,
         color=MUTED, margin=0)
    ON_SCREEN[10] = "Native chart of 3-seed inflation per channel and head; callouts +63.39% and +62.03%"

    # 11 --------------------------------------------------------------- small channels bounded
    s = d.content("The other four channels are bounded below 3%", "Result 3 · equivalence bounds",
                  "results/robust_stats_h12.md (90% block-bootstrap upper bounds; Holm within family)", NOTES[11])
    cats = ["Ch2 merge table", "Ch3 decode tables", "Ch4 stream, then split", "Ch5 centered gate"]
    bar_chart(s, IN(0.5), IN(1.6), IN(8.3), IN(5.3), cats,
              [("Direct head", [N["ch2_d_hi_v"], None, N["ch4_d_hi_v"], N["ch5_d_hi_v"]]),
               ("Generative head", [N["ch2_g_hi_v"], N["ch3_g_hi_v"], N["ch4_g_hi_v"], N["ch5_g_hi_v"]])],
              colors=[DIRECT, SAFE], vmin=-1.5, vmax=4, gap=55, overlap=-5, label_size=14, axis_size=14)
    text(s, IN(4.3), IN(1.6), IN(4.5), IN(0.36), "90% upper bound: “at most this much”", size=13,
         color=MUTED, italic=True, margin=0, align=PP_ALIGN.RIGHT, anchor=MSO_ANCHOR.MIDDLE)
    stat(s, IN(9.2), IN(1.7), IN(3.55), IN(1.6), N["ch5_d_hi"],
         f"largest bound: Ch5 direct. Point {N['ch5_d']}, Holm p {N['ch5_d_holm']}", color=DIRECT)
    stat(s, IN(9.2), IN(3.45), IN(3.55), IN(1.6), N["ch3_g"],
         f"Ch3 point estimate: significant (Holm p {N['ch3_g_holm']}) but tiny", color=SAFE)
    text(s, IN(9.2), IN(5.3), IN(3.55), IN(1.6),
         [(f"Partial-token bias: {N['ptb_d']} direct, {N['ptb_g']} generative, not significant.", {}),
          (f"Bounds are pooled; where folds disagree (I² up to {N['i2_max_small']}), random-effects intervals "
           "are wider.", {"size": 12, "color": MUTED})], size=14, color=INK, margin=0)
    ON_SCREEN[11] = "Native chart of 90% upper bounds for Ch2–Ch5; callouts +2.84% and +0.24%"

    # 12 --------------------------------------------------------------- trade-off
    s = d.content("A boundary leak can hide a large partial-token bias", "Result 4 · the trade-off",
                  "results/inflation_table_h288.md (Fold 1, seed 0); results/holdout_h12_leak_audit.json (seed 0)",
                  NOTES[12])
    cats = ["Next day · Fold 1 · direct", "2025 holdout · generative"]
    bar_chart(s, IN(0.5), IN(1.6), IN(8.0), IN(4.95), cats,
              [("Causal reference", [N["h288_q_ref_v"], N["hold_q_ref_g_v"]]),
               ("Context only", [N["h288_q_ch4c_v"], N["hold_q_ch4c_g_v"]]),
               ("Stream, then split (leak)", [N["h288_q_ch4_v"], N["hold_q_ch4_g_v"]])],
              fmt="0.0000", horizontal=False, colors=[SAFE, DIRECT, LEAK], vmin=0, vmax=0.62, gap=80,
              overlap=-8, label_size=14, axis_size=14)
    text(s, IN(0.6), IN(6.58), IN(7.8), IN(0.3), "QLIKE on the same origins: lower is better", size=13,
         color=MUTED, italic=True, margin=0, align=PP_ALIGN.CENTER)
    stat(s, IN(8.85), IN(1.7), IN(3.9), IN(1.65), N["h288_ptb"],
         f"partial-token cost at the next-day horizon (holdout generative {N['hold_ptb_g']})", color=DIRECT)
    stat(s, IN(8.85), IN(3.5), IN(3.9), IN(1.65), N["h288_ch4"],
         f"boundary leak wins it back (holdout {N['hold_ch4_g']}): leak ≈ reference")
    chip(s, IN(8.85), IN(5.45), IN(3.9), "single seed: an observed pattern", fill=PANEL, color=INK, size=14,
         h=IN(0.45))
    ON_SCREEN[12] = "Native chart: reference / context-only / stream-then-split QLIKE; callouts −19.53% and +16.10%"

    # 13 --------------------------------------------------------------- next-day horizon
    s = d.content("At the next-day horizon every measured channel grows", "Result 5 · H = 288",
                  "results/inflation_table_h288.md (Fold 1, seed 0, direct head)", NOTES[13])
    cats = ["Ch1 window norm", "Ch4 stream, then split", "Ch5 centered gate", "Ch2 merge table",
            "Partial-token bias"]
    vals = [N["h288_ch1_v"], N["h288_ch4_v"], N["h288_ch5_v"], N["h288_ch2_v"], N["h288_ptb_v"]]
    bar_chart(s, IN(0.5), IN(1.6), IN(8.3), IN(5.2), cats, [("Inflation", vals)], colors=[LEAK],
              point_colors={0: [LEAK, LEAK, LEAK, LEAK, DIRECT]}, legend=False, vmin=-50, vmax=115, gap=45,
              label_size=14, axis_size=14)
    chip(s, IN(9.2), IN(1.75), IN(3.55), "Fold 1 · seed 0 · direct head", fill=NAVY, color=WHITE, size=14,
         h=IN(0.45))
    stat(s, IN(9.2), IN(2.45), IN(3.55), IN(1.55), N["h288_pos"], "positive control, for scale", color=NAVY)
    box(s, IN(9.2), IN(4.2), IN(3.55), IN(1.6), LEAK_T, radius=0.06,
        label=[("p-values likely optimistic", {"bold": True, "size": 15, "color": LEAK}),
               ("hourly origins with 24-hour targets overlap; CIs use a 48-origin block bootstrap",
                {"size": 13, "color": INK})], align=PP_ALIGN.LEFT)
    ON_SCREEN[13] = "Native chart of H = 288 inflation; scope chip; optimistic-p chip; positive control callout"

    # 14 --------------------------------------------------------------- holdout
    s = d.content("2025 holdout, scored once: Ch1 still inflates accuracy by about 70%",
                  "Result 6 · out of sample",
                  f"results/holdout_h12_leak_audit.json; HOLDOUT_SCORED.json ({N['hold_date']}, commit "
                  f"{N['hold_commit']}, rescored = {str(N['hold_rescored']).lower()}); robust_stats_h12.md (MCS)",
                  NOTES[14])
    cats = ["Ch1 window norm", "Ch2 merge table", "Ch3 decode tables", "Ch4 stream, then split", "Ch5 centered gate"]
    bar_chart(s, IN(0.5), IN(1.6), IN(7.6), IN(5.25), cats,
              [("Direct head", [N["hold_ch1_d_v"], N["hold_ch2_d_v"], None, N["hold_ch4_d_v"], N["hold_ch5_d_v"]]),
               ("Generative head", [N["hold_ch1_g_v"], N["hold_ch2_g_v"], N["hold_ch3_g_v"], N["hold_ch4_g_v"],
                                    N["hold_ch5_g_v"]])],
              colors=[DIRECT, SAFE], vmin=-15, vmax=90, gap=55, overlap=-5, label_size=13, axis_size=14)
    text(s, IN(8.45), IN(1.6), IN(4.3), IN(0.4), f"Model Confidence Set (α = {N['mcs_alpha']})", size=16,
         color=NAVY, bold=True, margin=0)
    text(s, IN(8.45), IN(2.1), IN(2.1), IN(0.3), "Folds 1–6, 3 seeds", size=13, color=MUTED, margin=0)
    text(s, IN(10.65), IN(2.1), IN(2.1), IN(0.3), "2025 holdout", size=13, color=MUTED, margin=0)
    models = [("GPT unconstrained, direct", "GPT unconstrained BPE, direct"),
              ("GPT reference, direct", "GPT reference, direct"), ("GARCH(1,1)", "GARCH(1,1)")]
    for i, (lab, key) in enumerate(models):
        my = IN(2.5) + i * IN(0.62)
        for j, tag in enumerate(("folds", "hold")):
            inn = key in N[f"mcs_{tag}_in"]
            box(s, IN(8.45) + j * IN(2.2), my, IN(2.05), IN(0.5), SAFE if inn else PANEL, radius=0.3,
                label=lab, size=12, color=WHITE if inn else MUTED, bold=inn)
    text(s, IN(8.45), IN(4.45), IN(4.3), IN(0.4), "green = in the set", size=12, color=MUTED, margin=0)
    box(s, IN(8.45), IN(5.0), IN(4.3), IN(1.6), PANEL, radius=0.06,
        label=[(f"Ch2 flips sign: {N['hold_ch2_d']} direct, {N['hold_ch2_g']} generative", {"size": 14}),
               (f"Reference GPT (direct) vs GARCH on 2025: {N['hold_ref_vs_garch']}", {"size": 14})],
        align=PP_ALIGN.LEFT)
    ON_SCREEN[14] = "Native chart of holdout inflation; MCS chips (folds vs 2025)"

    # 15 --------------------------------------------------------------- extensions
    s = d.content("Extensions: one future bar is already enough to leak", "Result 7 · exploratory, folds only",
                  "results/robust_stats_h12.md (extension families; 2025 holdout not used)",
                  NOTES[15])
    panels = [
        ("Ch1: future bars in the window · direct", ["1 bar", "3 bars", "6 bars", "12 bars"],
         [N["k1_v"], N["k3_v"], N["k6_v"], N["ch1_d_v"]], [LEAK] * 4, 0, 78,
         f"1, 3, 6 bars: {N['k1_seeds']} seed · 12 bars: {N['ch1_d_seeds']} seeds"),
        ("Cross-asset ETH · direct", ["ETH leak through test", "ETH up to split"],
         [N["xa_leak_d_v"], N["xa_more_d_v"]], [LEAK, SAFE], 0, 4.2,
         f"{N['xa_more_d_seeds']} seeds; vs ETH-to-split and vs BTC only"),
        ("Token healing · generative", ["healed vs plain", "plain vs ref.", "healed vs ref."],
         [N["heal_v"], N["heal_plain_v"], N["heal_vs_ref_v"]], [DIRECT, GRAYBAR, GRAYBAR], 0, 2.6,
         f"{N['heal_seeds']} seed · ref. = hour-aligned reference"),
    ]
    pw = IN(3.95)
    for i, (head, cats, vals, cols, lo, hi, note) in enumerate(panels):
        px = IN(0.6) + i * (pw + IN(0.14))
        text(s, px, IN(1.5), pw, IN(0.35), head, size=15, color=NAVY, bold=True, margin=0)
        bar_chart(s, px - IN(0.1), IN(1.85), pw + IN(0.1), IN(2.85), cats, [("Inflation", vals)],
                  horizontal=False, colors=[cols[0]], point_colors={0: cols}, legend=False, vmin=lo, vmax=hi,
                  gap=60, label_size=13, axis_size=12)
        text(s, px, IN(4.72), pw, IN(0.3), note, size=12, color=MUTED, italic=True, margin=0)
    cw = IN(3.95)
    stat(s, IN(0.6), IN(5.15), cw, IN(1.7), N["k1"],
         f"one future bar in the window already leaks (Holm p {N['k1_holm']})")
    stat(s, IN(4.69), IN(5.15), cw, IN(1.7), N["xa_more_d"],
         f"ETH data up to the split helps; leaking ETH adds only {N['xa_leak_d']} (Holm p {N['xa_leak_d_holm']})",
         color=SAFE)
    stat(s, IN(8.78), IN(5.15), cw, IN(1.7), N["heal"],
         f"token healing, same model (Holm p {N['heal_holm']})", color=DIRECT)
    ON_SCREEN[15] = "Three native charts (dose-response, cross-asset, healing); callouts +2.70% / +3.06% / +1.70%"

    # 16 --------------------------------------------------------------- code walkthrough
    s = d.content("Code walkthrough: from raw bars to verified slides", "Repository tour",
                  REPO_URL + "  ·  main", NOTES[16])
    mods = [("data/", "download, EWMA,\nfolds, origins"), ("tokenization/", "BPE, causal gate,\nleak arms"),
            ("models/", "GPT, HAR-RV,\nGARCH(1,1)"), ("experiments/", "one model per arm,\nDM + Holm"),
            ("results/", "JSON tables,\nper-origin losses"), ("presentation/", "deck builder,\n--check")]
    mw, mg, my = IN(1.8), IN(0.27), IN(1.85)
    for i, (head, sub) in enumerate(mods):
        mx = IN(0.6) + i * (mw + mg)
        box(s, mx, my, mw, IN(1.35), NAVY if i in (1, 3) else DIRECT_T, radius=0.12,
            label=[(head, {"bold": True, "size": 16, "color": WHITE if i in (1, 3) else NAVY})] +
                  [(ln, {"size": 12, "color": ICE if i in (1, 3) else INK}) for ln in sub.split("\n")])
        if i:
            arrow(s, mx - mg + IN(0.02), my + IN(0.675), mx - IN(0.02), my + IN(0.675))
    stops = WALK_STOPS
    for i, (f, what) in enumerate(stops):
        col, row = i % 2, i // 2
        sx = IN(0.6) + col * IN(6.15)
        sy = IN(3.45) + row * IN(0.76)
        circle(s, sx + IN(0.24), sy + IN(0.3), IN(0.46), LEAK if i < 4 else SAFE, str(i + 1), size=14)
        text(s, sx + IN(0.6), sy, IN(5.45), IN(0.35), f, size=14, color=NAVY, bold=True, margin=0)
        text(s, sx + IN(0.6), sy + IN(0.33), IN(5.45), IN(0.4), what, size=13, color=INK, margin=0)
    term = box(s, IN(0.6), IN(5.95), IN(12.13), IN(0.85), NAVY, radius=0.08, align=PP_ALIGN.LEFT,
               label=[("$ python presentation/build_final_presentation.py --check", {"color": ICE, "size": 14}),
                      (check_msg, {"color": SAFE, "size": 14})])
    for para in term.text_frame.paragraphs:
        for r in para.runs:
            r.font.name = "Courier New"
    term.text_frame.margin_left = IN(0.25)
    ON_SCREEN[16] = "Repo map flowchart + numbered walkthrough stops + the live --check line (screen recording)"

    # 17 --------------------------------------------------------------- takeaways (dark)
    s = d.dark("Audit normalization first", NOTES[17], kicker="Takeaways & limitations")
    tk = [("1", f"{N['ch1_d']}", "a window touching the horizon is the one large, stable leak, and the only "
                                  "one that beats the oracle floor"),
          ("2", f"≤ {N['ch5_d_hi']}", "table and encode-time leaks at H = 12, but they grow at the next-day "
                                      "horizon"),
          ("3", "GARCH", "alone in the 2025 confidence set: we measure inflation, not a better forecaster")]
    for i, (num, big, body) in enumerate(tk):
        cx = IN(0.6) + i * IN(4.1)
        box(s, cx, IN(1.75), IN(3.85), IN(2.3), RGBColor(0x1A, 0x2D, 0x52), radius=0.06)
        circle(s, cx + IN(0.5), IN(2.25), IN(0.6), LEAK if i == 0 else SAFE, num, size=18)
        text(s, cx + IN(0.95), IN(2.0), IN(2.8), IN(0.55), big, size=26, color=WHITE, bold=True,
             anchor=MSO_ANCHOR.MIDDLE, margin=0)
        text(s, cx + IN(0.3), IN(2.75), IN(3.35), IN(1.6), body, size=15, color=ICE, margin=0)
    text(s, IN(0.6), IN(4.7), IN(5), IN(0.4), "Limitations", size=17, color=WHITE, bold=True, margin=0)
    lims = ["gate and scale-token ablation not run", "one asset; small model, 5 epochs",
            "single seed: next day, holdout, dose, healing", "next-day p-values likely optimistic"]
    for i, lim in enumerate(lims):
        cx = IN(0.6) + (i % 2) * IN(6.15)
        cy = IN(5.2) + (i // 2) * IN(0.6)
        chip(s, cx, cy, IN(5.95), lim, fill=RGBColor(0x1A, 0x2D, 0x52), color=WHITE, size=14, h=IN(0.48))
    text(s, IN(0.6), IN(6.65), IN(11), IN(0.35), REPO_URL, size=14, color=ICE, margin=0)
    ON_SCREEN[17] = "Three takeaway cards, four limitation chips, repository link"

    # 18 --------------------------------------------------------------- team (dark)
    s = d.dark("Team and contributions", NOTES[18], kicker="Team")
    cw, cg = IN(2.3), IN(0.16)
    for i, (name, area, share) in enumerate(TEAM):
        cx = IN(0.6) + i * (cw + cg)
        box(s, cx, IN(1.9), cw, IN(4.15), RGBColor(0x1A, 0x2D, 0x52), radius=0.06)
        initials = "".join(p[0] for p in name.split()[:2])
        circle(s, cx + cw // 2, IN(2.65), IN(0.95), LEAK if i % 2 == 0 else SAFE, initials, size=20)
        text(s, cx + IN(0.12), IN(3.3), cw - IN(0.24), IN(0.7), name, size=17, color=WHITE, bold=True,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx + IN(0.15), IN(4.05), cw - IN(0.3), IN(0.3), "STUDY", size=11, color=LEAK, bold=True,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx + IN(0.15), IN(4.32), cw - IN(0.3), IN(0.6), area, size=14, color=ICE,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx + IN(0.15), IN(4.98), cw - IN(0.3), IN(0.3), "DELIVERABLE", size=11, color=SAFE, bold=True,
             align=PP_ALIGN.CENTER, margin=0)
        text(s, cx + IN(0.15), IN(5.25), cw - IN(0.3), IN(0.55), share, size=14, color=ICE,
             align=PP_ALIGN.CENTER, margin=0)
    text(s, IN(0.6), IN(6.25), IN(12.1), IN(0.4),
         "All members reviewed the results, wrote their report section and present their part in the video.",
         size=15, color=WHITE, italic=True, margin=0)
    text(s, IN(0.6), IN(6.65), IN(11), IN(0.35), "Thank you  ·  " + REPO_URL, size=14, color=ICE, margin=0)
    ON_SCREEN[18] = "Five member cards: study area and deliverable share"

    d.prs.save(out_path)
    return d


# ============================================================================= narration
WALK_STOPS = [  # shown on slide 16 (line numbers verified against main @ fa2f120)
    ("tokenization/leak_tokenizer.py  :44-53", "arms table: every leak next to its matched control"),
    ("leak_tokenizer.py  :183 vs :185, :203-212", "Ch1 std over the horizon; Ch4 keeps the straddling token"),
    ("tokenization/bpe_tokenizer.py  :182-186", "merges stop at each clock hour, applied in rank order"),
    ("tokenization/causal_gate.py  :58 vs :60-61", "trailing 24-bar window vs the centered one (Ch5)"),
    ("experiments/run_leak_experiments.py  :188-190, :251-263", "one GPT per arm, leak in train and test; DM + Holm"),
    ("experiments/run_holdout.py  :55-59", "score-once lock: shown, never run; then --check live"),
]
WALK_REFS = [  # (slide-16 stop, file:lines, what) in screen-recording order; verified against main @ fa2f120
    (1, "tokenization/leak_tokenizer.py:44-53", "ARMS table: each leak arm with its matched control"),
    (2, "tokenization/leak_tokenizer.py:180-185", "Ch1: control std over the context (183) vs leak std over "
                                                  "context + horizon (185)"),
    (2, "tokenization/leak_tokenizer.py:203-212", "Ch4: encode context + horizon, keep tokens until C bars are "
                                                  "covered; the straddling token is kept (206-208)"),
    (2, "tokenization/leak_tokenizer.py:112, 118, 123", "optional, not narrated: train-only tables vs the Ch2 / "
                                                        "Ch3 tables fitted on train + test"),
    (3, "tokenization/bpe_tokenizer.py:182-186, 152-164", "hour-by-hour encoding; lowest-rank merge applied first"),
    (4, "tokenization/causal_gate.py:56-64, 73-77, 48", "trailing window (58) vs centered (60-61); threshold = "
                                                        "0.75 quantile of train hours"),
    (5, "experiments/run_leak_experiments.py:185-203", "per-arm loop: train and test inputs both from "
                                                       "mgr.extract(t, arm) (188-190); Ch3 reuses the reference "
                                                       "model (193-198)"),
    (5, "experiments/run_leak_experiments.py:156-163, 251-263", "compare(): DM, GW, TOST; Holm across the leak "
                                                                "family (260)"),
    (6, "experiments/run_holdout.py:55-59, 76-80", "refuses to re-score without --i-know-this-rescores; writes "
                                                   "the lock. Show, never run"),
    (6, "results/HOLDOUT_SCORED.json", "the lock: scored_at_utc, git_commit, seeds [0], rescored false"),
    (6, "terminal: python presentation/build_final_presentation.py --check", "run live from the repo root; "
                                                                             "writes nothing"),
]
WALK_NOTE = (
    "Now a short tour of the code. The repository runs from data through tokenization, models and experiments "
    "to results and this presentation. The leak tokenizer opens with a table pairing every leak arm with a "
    "control that differs in one switch. For Channel 1, the control takes the standard deviation over the "
    "context, and the leak stretches that window through the horizon. Channel 4 tokenizes past the origin, "
    "stops once the context is covered, and keeps the token that straddles the origin, so future bars ride "
    "inside it. The reference encoder works hour by hour, applying merges in rank order, so no token crosses an "
    "hour. The gate scores each hour on a trailing window; Channel 5 centers that window, so it peeks ahead. In "
    "the experiment runner, each arm trains its own model with the leak on in training and testing; only "
    "Channel 3 reuses the reference model, because decoding tables never enter training. Each leak faces its "
    "control in a Diebold-Mariano test with Holm correction. This guard refuses to re-score the 2025 holdout, "
    "and this block wrote the lock file; we show it, we never run it. Finally, live: the deck builder's check "
    "flag recomputes every quoted number from the results files and writes nothing. One mismatch stops the "
    "build.")
CONTINUITY_NOTE = (
    "Before the results, a note on continuity. Our first repository commit on 15 October held pre-audit "
    "numbers: a plus 16.6 percent Channel 4 leak, a GARCH baseline that looked 21.7 percent worse than our own "
    "model, and an 84 percent boundary mismatch. The audit that day traced them to bugs, and our midterm "
    "already withdrew them. Channel 1 rose from 7.09 to 63.39 percent because each arm now trains its own model "
    "against a matched control. Every number today comes from the corrected pipeline.")

NOTES.update({
    1: "Hello, we are presenting our DA4022 project on look-ahead bias inside motif tokenizers for financial "
       "time series. We built a leak-free volatility forecasting pipeline, then switched on one source of future "
       "information at a time and measured how much better each makes the forecasts look.",
    2: "A forecast at origin t may only use the context before t. A BPE motif tokenizer can break that in two "
       "ways. At fit time, a normalization window or a merge or decoding table can be estimated on data that "
       "overlaps the test period. At encode time, applying Zhang et al.'s definition of incremental "
       "tokenization, rank-ordered BPE is not incremental at an arbitrary origin, so a stream tokenized once and "
       "then split carries a token across t. Tokenizing only the context avoids that, but leaves a partial "
       "final token.",
    3: "So our question is which parts of the pipeline leak, and by how much each one inflates measured "
       "accuracy. We contribute a channel-by-channel audit, a measurement of the boundary versus partial-token "
       "trade-off, a causal tokenizer design, and an evaluation with controls and a holdout scored once.",
    4: "Here is the causal reference pipeline. Five-minute returns are normalized by trailing EWMA volatility, "
       "quantized into 32 bins, and merged by BPE that never crosses a clock hour, with a scale token for each "
       "hour. A causal RoV times SEI gate, a mechanism borrowed from PATK, shortens tokens in volatile hours. A "
       "GPT with 855,169 parameters forecasts realized volatility with a direct head and a generative head. The "
       "orange markers show where each of the five leaks switches on.",
    5: "Every leak arm differs from its matched control in exactly one switch, and each arm trains its own "
       "model, so the leak is present in training and in testing; only Channel 3 reuses the reference model. We compare their QLIKE losses on the same "
       "origins and report inflation: how much the leak flatters the forecast. A simulated GARCH negative "
       "control gives an oracle floor no causal model can beat, and a positive control shows an extreme leak "
       "for scale.",
    6: "We use Binance BTCUSDT five-minute data from 2021 to 2025, in six expanding walk-forward folds with "
       "26,290 test origins in total. 2025 is a holdout that was scored exactly once. Statistics include block "
       "bootstrap intervals, Diebold-Mariano tests with Holm correction, equivalence tests, random-effects "
       "pooling across folds and a model confidence set.",
    7: CONTINUITY_NOTE,
    8: "First, the token-level audit on Fold 1. With hour boundaries and the causal gate, no token straddles "
       "the origin and perturbing the future changes nothing. The centered gate of Channel 5 changes pre-origin "
       "tokens at 57 percent of origins, and unconstrained BPE straddles the origin at 32 percent. Motifs are "
       "short: 1.40 bars per token.",
    9: "Next, the negative control. On simulated GARCH data the oracle reaches a QLIKE of 0.1268. The "
       "reference and the other leak arms all stay above that floor. Only the Channel 1 normalization leak, at "
       "0.0694, and the deliberate positive control fall below it. Channels 2, 4 and 5 still gain up to 2.12 "
       "percent over their controls there, but none crosses the floor.",
    10: "Now the main result over six folds at the next-hour horizon, averaged over three seeds. Channel 1, "
        "where the normalization window reaches the horizon, cuts QLIKE by 63.39 percent on the direct head, "
        "with a random-effects estimate across folds of 62.03 percent and zero heterogeneity. Every other bar is "
        "within a few percent.",
    11: "For the other channels we report how large they could be. The 90 percent upper bounds stay below 3 "
        "percent: the largest is Channel 5 on the direct head, at 2.84. Channel 5 direct and Channel 3 are "
        "significant after Holm correction, but tiny. Partial-token bias is not significant at this horizon.",
    12: "This is the trade-off. At the next-day horizon, tokenizing only the context costs 19.53 percent, a "
        "partial-token bias. Tokenizing the stream and then splitting wins 16.10 percent back, landing at 0.3137 "
        "against a reference of 0.3128. The 2025 generative head shows the same pattern, while the 2025 direct head "
        "and the pooled folds show no such cost. So a leaky pipeline can look as good as a causal one. Both "
        "settings are single seed, so we call this an observed pattern.",
    13: "At the next-day horizon, on Fold 1, every measured channel grows: Channel 1 reaches 96.36 percent, Channel 4 "
        "16.10, Channel 5 7.02 and Channel 2 5.67. This is one fold, one seed and the direct head, and because "
        "hourly origins with 24-hour targets overlap, these p-values are likely optimistic.",
    14: "Then the 2025 holdout, scored once on 17 November and never re-scored. Channel 1 still inflates "
        "accuracy by 73.90 percent on the direct head and 68.32 on the generative head. Channel 2 flips sign "
        "across heads. In the model confidence set, leak-free GPT direct models lead on the folds, but on 2025 "
        "only GARCH remains.",
    15: "Finally, exploratory extensions on the folds only. One future bar in the normalization window already "
        "inflates accuracy by 2.70 percent, rising to 18.38 at six bars. Extra ETH data up to the split helps by "
        "3.06 percent, while leaking ETH through the test period adds only 0.95, not significant after Holm. "
        "Token healing improves the same model by 1.70 percent.",
    16: WALK_NOTE,
    17: "To conclude: audit normalization first. A window that touches the horizon is the one large, stable "
        "leak, and the only one that beats the oracle floor. Table and boundary leaks stay below 3 percent at "
        "one hour but grow at one day, and a boundary leak can hide partial-token bias. We did not run the "
        "promised gate and scale-token ablation, and several settings use a single seed.",
    18: "Thank you for watching. Each of us presented the part we led, and the code, results and report are in "
        "our repository.",
})


def write_script(path, d, N, n_checked):
    words = {i: len(NOTES[i].split()) for i in NOTES}
    total = sum(words.values())
    secs, t, rows = {}, 0, []
    for i in sorted(NOTES):
        sec = round(words[i] / 150 * 60) + 3
        rows.append((i, d.titles[i - 1], ON_SCREEN[i], t, t + sec))
        t += sec
    mmss = lambda s_: f"{s_ // 60}:{s_ % 60:02d}"
    L = ["# Video script: final presentation (DA4022, causal-vol-tokenizer)", "",
         "Generated by `presentation/build_final_presentation.py` from its `NOTES`. Each slide's narration is "
         "**identical** to that slide's speaker notes in `presentation/final_presentation.pptx`, and every number "
         f"in it is checked against `results/` before the deck is written ({n_checked} quoted values verified).", "",
         f"**Total:** about {mmss(t)} ({total} narration words at ~150 words per minute, plus about 3 s per slide "
         "transition). Brief: 10 minutes or less.", "",
         "**Reading guide for the numbers**",
         "- Next-hour (H = 12) headline numbers: Folds 1–6 pooled, seeds 0–2 averaged per origin "
         "(`results/robust_stats_h12.md`).",
         "- Next-day (H = 288): Fold 1, seed 0, direct head only (`results/inflation_table_h288.md`).",
         "- 2025 holdout: seed 0, scored once on 17 Nov 2026 and never re-scored (`results/HOLDOUT_SCORED.json`).",
         "- Extensions: Folds 1–6 only; the holdout is not used.",
         "- Withdrawn pre-audit figures (slide 7) come from the initial 15 Oct commits `7f9f6a6`–`dea2cb5` "
         "(README as of `dea2cb5`), not from results.",
         "", "## Running order", "", "| # | Slide | On screen | Start – end |", "|---|---|---|---|"]
    for i, ti, on, a, e in rows:
        L.append(f"| {i} | {ti} | {on} | {mmss(a)} – {mmss(e)} |")
    L += ["", "---", ""]
    for i, ti, on, a, e in rows:
        L += [f"## Slide {i} · {ti}  `[{mmss(a)} – {mmss(e)}]`", "", f"*On screen:* {on}", ""]
        if i == 16:
            L += ["*Screen recording, in order:*", ""]
            L += [f"{k + 1}. Stop {stop} · `{ref}`: {what}" for k, (stop, ref, what) in enumerate(WALK_REFS)]
            L += [""]
        L += [f"> {NOTES[i]}", ""]
    L += ["---", "", "## Recording tips", "",
          "- Record over the deck in presenter view; the same narration is in each slide's speaker notes.",
          "- Slide 16 is a screen recording: open each file at the lines listed above. Run "
          "`python presentation/build_final_presentation.py --check` live. Show `experiments/run_holdout.py` "
          "and `results/HOLDOUT_SCORED.json` on screen, but never run the holdout script.",
          "- Pause about a second on each chart before reading its numbers.",
          "- Slide 16 narration takes about 1:30; the screen recording can run to about 2:30 and the video still "
          "stays under 10 minutes. Rehearse the stops once.",
          "- Keep these lines even if you trim: the single-seed caveat (slide 12), the optimistic next-day "
          "p-values (slide 13), the holdout 'scored once' statement (slide 14) and the ablation not run (slide 17).",
          "- Each member narrates the slides of the area they led (slide 18)."]
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    return total, mmss(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "presentation", "final_presentation.pptx"))
    ap.add_argument("--script-out", default=os.path.join(ROOT, "VIDEO_SCRIPT.md"))
    ap.add_argument("--check", action="store_true", help="verify every number only; write nothing")
    args = ap.parse_args()
    N = load_numbers()
    P = git_preaudit()
    n_checked, check_msg = verify(N, P)
    for i in range(1, 19):
        if not NOTES.get(i):
            raise SystemExit(f"slide {i} has no narration")
    if args.check:
        return 0
    d = build(N, P, args.out, check_msg)
    total, dur = write_script(args.script_out, d, N, n_checked)
    print(f"Wrote {d.n} slides to {os.path.relpath(args.out, ROOT)}")
    print(f"Wrote {os.path.relpath(args.script_out, ROOT)}: {total} narration words, about {dur}")
    return 0


def fig(name):
    return os.path.join(FIG, name)


if __name__ == "__main__":
    sys.exit(main())
