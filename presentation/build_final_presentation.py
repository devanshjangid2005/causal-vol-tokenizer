"""
presentation/build_final_presentation.py
Final DA4022 presentation deck (16:9) for the causal-vol-tokenizer study.

Supersedes presentation/build_verified_midterm_presentation.py, the early Fold-1-only midterm build whose
numbers predate Folds 2-6, the 2025 holdout and the 20 Nov extensions.

Every number on the slides is read from results/ at build time:
  results/robust_stats_h12.json        3-seed pooled inflation, 90% CIs, Holm p, random effects, MCS
  results/inflation_table_h12.json     seed-0 pooled table (matches inflation_all_folds_h12.png)
  results/inflation_table_h288.json    next-day horizon, Fold 1, seed 0, direct head
  results/negative_control_results.json, results/boundary_mismatch_results.json
  results/holdout_h12_leak_audit.json, results/HOLDOUT_SCORED.json (read only; nothing is re-scored)
  results/fold_<k>_baselines.json, results/fold_<k>_h12_leak_audit.json
Before writing anything, the script checks the values quoted in the speaker notes and VIDEO_SCRIPT.md
(EXPECTED below) against those files and stops if any differ.

results/figures/leakage_inflation_comparison.png is deliberately NOT used: it holds Fold 6
(results/leakage_inflation_table.json["fold"] == 6), not Fold 1.

Usage (from the repository root):
    pip install python-pptx pillow
    python presentation/build_final_presentation.py            # -> presentation/final_presentation.pptx
    python presentation/build_final_presentation.py --check    # verify numbers only, write nothing
    python presentation/build_final_presentation.py --out other.pptx
"""

import argparse
import json
import os
import sys

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.util import Emu, Inches, Pt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")

TEAM_LINE = "DA4022 Course Project  ·  Team of 5"
DATE_LINE = "Final presentation  ·  November 2026"

# ----------------------------------------------------------------------------- palette
NAVY = RGBColor(0x0F, 0x1E, 0x3C)
INK = RGBColor(0x1E, 0x29, 0x3B)
MUTED = RGBColor(0x64, 0x74, 0x8B)
RULE = RGBColor(0xE2, 0xE8, 0xF0)
PANEL = RGBColor(0xF5, 0xF7, 0xFA)
ORANGE = RGBColor(0xC2, 0x41, 0x0C)   # matches the "significant" bars in the result figures
TEAL = RGBColor(0x0F, 0x8A, 0x7E)     # matches the "not significant" bars
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

SLIDE_W, SLIDE_H = Inches(13.333), Inches(7.5)


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
    """Formatted strings for everything shown on the slides, straight from results/."""
    N = {}
    rs = _load("robust_stats_h12.json")
    comps = [c for fam in rs["comparisons"].values() for c in fam]

    def robust(arm, control, head):
        for c in comps:
            if c["arm"] == arm and c["control"] == control and c["head"] == head:
                return c
        raise KeyError((arm, control, head))

    rows = [  # (key, label, arm, control, head)
        ("ch1_d", "Ch1 window norm uses horizon", "ch1_window_norm_leak", "ch1_control_window", "direct"),
        ("ch1_g", "Ch1 window norm uses horizon", "ch1_window_norm_leak", "ch1_control_window", "generative"),
        ("ch2_d", "Ch2 merge table on train+test", "ch2_merge_table_leak", "reference", "direct"),
        ("ch2_g", "Ch2 merge table on train+test", "ch2_merge_table_leak", "reference", "generative"),
        ("ch3_g", "Ch3 decode tables on train+test", "ch3_decode_table_leak", "reference", "generative"),
        ("ch4_d", "Ch4 tokenize stream, then split", "ch4_boundary_leak", "ch4_control_context_only", "direct"),
        ("ch4_g", "Ch4 tokenize stream, then split", "ch4_boundary_leak", "ch4_control_context_only", "generative"),
        ("ch5_d", "Ch5 centered gate FFT", "ch5_noncausal_fft_leak", "reference", "direct"),
        ("ch5_g", "Ch5 centered gate FFT", "ch5_noncausal_fft_leak", "reference", "generative"),
        ("pos_d", "Positive control", "positive_control", "reference", "direct"),
        ("pos_g", "Positive control", "positive_control", "reference", "generative"),
        ("ptb_d", "Partial-token bias", "ch4_control_context_only", "reference", "direct"),
        ("ptb_g", "Partial-token bias", "ch4_control_context_only", "reference", "generative"),
        ("k1", "k = 1", "ch1_leak_k1", "ch1_control_window", "direct"),
        ("k3", "k = 3", "ch1_leak_k3", "ch1_control_window", "direct"),
        ("k6", "k = 6", "ch1_leak_k6", "ch1_control_window", "direct"),
        ("xa_leak_d", "ETH leak", "xa_leak", "xa_control", "direct"),
        ("xa_leak_g", "ETH leak", "xa_leak", "xa_control", "generative"),
        ("xa_more_d", "ETH more data", "xa_control", "reference", "direct"),
        ("xa_more_g", "ETH more data", "xa_control", "reference", "generative"),
        ("heal", "Healed vs plain", "ext_ch4_healed", "ext_ch4_control", "generative"),
    ]
    N["robust_rows"] = []
    for key, label, arm, ctrl, head in rows:
        c = robust(arm, ctrl, head)
        re = c["random_effects"]
        N[key] = pct(c["inflation_pct"])
        N[key + "_lo"] = pct(c["ci90"][0])
        N[key + "_hi"] = pct(c["ci90"][1])
        N[key + "_holm"] = pval(c["dm_p_holm"])
        N[key + "_re"] = f"{pct(re['estimate'])} [{re['ci95'][0]:+.2f}, {re['ci95'][1]:+.2f}]"
        N[key + "_i2"] = f"{100 * re['I2']:.0f}%"
        N[key + "_seeds"] = f"{len(c['seeds_arm'])}/{len(c['seeds_control'])}"
        N[key + "_n"] = f"{c['n_origins']:,}"
        if not key.startswith(("k", "xa", "heal")):
            N["robust_rows"].append((label, head, N[key], f"{N[key + '_lo']} to {N[key + '_hi']}",
                                     N[key + "_holm"], N[key + "_re"], N[key + "_i2"]))

    for name, m in rs["mcs"].items():
        tag = "mcs_folds" if name.startswith("pooled") else "mcs_hold"
        N[tag] = [(mod, q4(m["mean_qlike"][mod]), f"{m['mcs_pvalues'][mod]:.3f}", mod in m["in_mcs"])
                  for mod in sorted(m["mean_qlike"], key=lambda k: m["mean_qlike"][k])]
        N[tag + "_in"] = sorted(m["in_mcs"])

    # seed-0 pooled table (the numbers printed on inflation_all_folds_h12.png)
    s0 = _load("inflation_table_h12.json")
    N["s0_seed"] = str(s0["seed"])
    N["s0_ch1_d"] = pct(s0["heads"]["direct"]["leak_tests"]["ch1_window_norm_leak"]["inflation_pct"])
    N["s0_ch1_g"] = pct(s0["heads"]["generative"]["leak_tests"]["ch1_window_norm_leak"]["inflation_pct"])
    N["s0_ref_d"] = q4(s0["heads"]["direct"]["arms"]["reference"]["QLIKE"])

    # pooled baselines and per-fold counts
    n_test = 0
    for k in range(1, 7):
        b = _load(f"fold_{k}_baselines.json")
        n_test += b["n_test_origins"]
    N["n_test_folds"] = f"{n_test:,}"
    mf = rs["mcs"]["pooled folds"]["mean_qlike"]
    N["har_pooled"] = q4(mf["HAR-RV"])
    N["garch_pooled"] = q4(mf["GARCH(1,1)"])
    N["ref3_d"] = q4(mf["GPT reference, direct"])
    N["unc3_d"] = q4(mf["GPT unconstrained BPE, direct"])
    N["ref3_g"] = q4(mf["GPT reference, generative"])

    # token causality audit + compression
    bm = _load("boundary_mismatch_results.json")
    v = bm["variants"]
    N["bm_origins"] = str(bm["n_origins"])
    N["bm_ref_aligned"] = f"{100 * v['hour_boundaries_causal_gate (reference)']['slice_vs_prefix_hour_aligned']['mismatch_rate']:.0f}%"
    N["bm_ref_off"] = f"{100 * v['hour_boundaries_causal_gate (reference)']['slice_vs_prefix_off_hour']['mismatch_rate']:.0f}%"
    N["bm_ref_pert"] = f"{100 * v['hour_boundaries_causal_gate (reference)']['future_perturbation_hour_aligned']['prefix_changed_rate']:.0f}%"
    N["bm_ch5_pert"] = f"{100 * v['hour_boundaries_centered_gate (ch5)']['future_perturbation_hour_aligned']['prefix_changed_rate']:.0f}%"
    N["bm_ch4_aligned"] = f"{100 * v['unconstrained_bpe (ch4)']['slice_vs_prefix_hour_aligned']['mismatch_rate']:.0f}%"
    N["bm_ch4_pert"] = f"{100 * v['unconstrained_bpe (ch4)']['future_perturbation_hour_aligned']['prefix_changed_rate']:.1f}%"
    cc = {c["num_merges"]: c["compression_ratio"] for c in bm["compression_curve"]}
    N["comp64"] = f"{cc[64]:.2f}"
    N["comp128"] = f"{cc[128]:.2f}"

    # negative control
    nc = _load("negative_control_results.json")
    bf = nc["beats_oracle_floor"]
    N["nc_floor"] = q4(bf["direct"]["reference"]["oracle_floor"])
    N["nc_ch1_d"] = q4(bf["direct"]["ch1_window_norm_leak"]["QLIKE"])
    N["nc_ch1_g"] = q4(bf["generative"]["ch1_window_norm_leak"]["QLIKE"])
    N["nc_ref_d"] = q4(bf["direct"]["reference"]["QLIKE"])
    viol = sorted(a for h in bf.values() for a, r in h.items() if r["violates_floor"])
    N["nc_violators"] = sorted(set(viol))
    ok_d = [r["QLIKE"] for a, r in bf["direct"].items() if not r["violates_floor"]]
    N["nc_others_d"] = f"{min(ok_d):.4f}–{max(ok_d):.4f}"

    # next-day horizon
    h288 = _load("inflation_table_h288.json")["heads"]["direct"]
    for key, arm in [("ch1", "ch1_window_norm_leak"), ("ch2", "ch2_merge_table_leak"),
                     ("ch4", "ch4_boundary_leak"), ("ch5", "ch5_noncausal_fft_leak")]:
        N["h288_" + key] = pct(h288["leak_tests"][arm]["inflation_pct"])
    N["h288_ptb"] = pct(h288["partial_token_bias"]["inflation_pct"])
    N["h288_q_ref"] = q4(h288["arms"]["reference"]["QLIKE"])
    N["h288_q_ch4c"] = q4(h288["arms"]["ch4_control_context_only"]["QLIKE"])
    N["h288_q_ch4"] = q4(h288["arms"]["ch4_boundary_leak"]["QLIKE"])

    # 2025 holdout (read only)
    hs = _load("HOLDOUT_SCORED.json")
    N["hold_rescored"] = hs["rescored"]
    N["hold_commit"] = hs["git_commit"][:7]
    N["hold_time"] = hs["scored_at_utc"][:10]
    ho = _load("holdout_h12_leak_audit.json")
    N["hold_n"] = f"{ho['n_test_origins']:,}"
    for head, hk in (("direct", "d"), ("generative", "g")):
        h = ho["heads"][head]
        for key, arm in [("ch1", "ch1_window_norm_leak"), ("ch2", "ch2_merge_table_leak"),
                         ("ch3", "ch3_decode_table_leak"), ("ch4", "ch4_boundary_leak"),
                         ("ch5", "ch5_noncausal_fft_leak"), ("pos", "positive_control")]:
            if arm in h["leak_tests"]:
                N[f"hold_{key}_{hk}"] = pct(h["leak_tests"][arm]["inflation_pct"])
        N[f"hold_ptb_{hk}"] = pct(h["partial_token_bias"]["inflation_pct"])
        N[f"hold_q_ref_{hk}"] = q4(h["arms"]["reference"]["QLIKE"])
        N[f"hold_q_ch4c_{hk}"] = q4(h["arms"]["ch4_control_context_only"]["QLIKE"])
        N[f"hold_q_ch4_{hk}"] = q4(h["arms"]["ch4_boundary_leak"]["QLIKE"])
    N["hold_har"] = q4(ho["baselines"]["har_rv"]["QLIKE"])
    N["hold_garch"] = q4(ho["baselines"]["garch_11"]["QLIKE"])
    N["hold_ref_vs_garch"] = pct(ho["heads"]["direct"]["vs_baselines"]["reference"]["garch_11"]["inflation_pct"])

    # leakage_inflation_table.json must be Fold 6 (that is why its figure is not used)
    N["leakage_table_fold"] = _load("leakage_inflation_table.json")["fold"]
    return N


# Values quoted in the speaker notes / VIDEO_SCRIPT.md. The build stops if results/ disagrees.
EXPECTED = {
    "ch1_d": "+63.39%", "ch1_g": "+62.17%", "ch1_d_lo": "+61.77%", "ch1_d_hi": "+65.04%",
    "ch1_d_re": "+62.03% [+58.80, +65.26]", "ch1_d_i2": "0%",
    "ch2_d": "+0.64%", "ch2_d_hi": "+1.44%", "ch2_g": "-2.58%", "ch3_g": "+0.24%", "ch3_g_hi": "+0.39%",
    "ch4_d": "+0.47%", "ch4_d_hi": "+1.05%", "ch4_g": "-0.95%", "ch5_d": "+1.78%", "ch5_d_hi": "+2.84%",
    "ch5_d_holm": "0.034", "pos_d": "+99.98%", "pos_g": "+72.22%",
    "ptb_d": "+0.42%", "ptb_g": "+1.89%",
    "k1": "+2.70%", "k3": "+6.95%", "k6": "+18.38%",
    "xa_leak_d": "+0.95%", "xa_leak_d_holm": "0.188", "xa_more_d": "+3.06%",
    "heal": "+1.70%", "heal_lo": "+0.07%", "heal_hi": "+3.38%", "heal_holm": "0.295",
    "s0_seed": "0", "s0_ch1_d": "+63.67%", "s0_ch1_g": "+65.72%", "s0_ref_d": "0.5651",
    "n_test_folds": "26,290", "har_pooled": "0.6811", "garch_pooled": "0.5747",
    "ref3_d": "0.5580", "unc3_d": "0.5556", "ref3_g": "0.6453",
    "bm_origins": "500", "bm_ref_aligned": "0%", "bm_ref_off": "31%", "bm_ref_pert": "0%",
    "bm_ch5_pert": "57%", "bm_ch4_aligned": "32%", "comp64": "1.40", "comp128": "1.54",
    "nc_floor": "0.1268", "nc_ch1_d": "0.0694", "nc_ch1_g": "0.1182",
    "nc_violators": ["ch1_window_norm_leak", "positive_control"],
    "h288_ch1": "+96.36%", "h288_ch2": "+5.67%", "h288_ch4": "+16.10%", "h288_ch5": "+7.02%",
    "h288_ptb": "-19.53%", "h288_q_ref": "0.3128", "h288_q_ch4c": "0.3739", "h288_q_ch4": "0.3137",
    "hold_rescored": False, "hold_commit": "f4ad204", "hold_n": "8,760",
    "hold_ch1_d": "+73.90%", "hold_ch1_g": "+68.32%", "hold_ch2_d": "-5.26%", "hold_ch2_g": "+4.93%",
    "hold_ch4_g": "+15.30%", "hold_ptb_g": "-20.32%",
    "hold_q_ref_g": "0.4603", "hold_q_ch4c_g": "0.5538", "hold_q_ch4_g": "0.4691", "hold_q_ref_d": "0.4416",
    "hold_har": "0.5667", "hold_garch": "0.4267", "hold_ref_vs_garch": "-3.49%",
    "mcs_folds_in": ["GPT reference, direct", "GPT unconstrained BPE, direct"],
    "mcs_hold_in": ["GARCH(1,1)"],
    "leakage_table_fold": 6,
}


def verify(N):
    bad = [(k, v, N.get(k)) for k, v in EXPECTED.items() if N.get(k) != v]
    if bad:
        for k, want, got in bad:
            print(f"MISMATCH {k}: expected {want!r}, results/ gives {got!r}")
        raise SystemExit("Numbers in results/ differ from the deck text. Nothing written.")
    print(f"Verified {len(EXPECTED)} quoted values against results/.")


# ============================================================================= drawing helpers
def _text(slide, x, y, w, h, paras, size=16, color=INK, bold=False, align=PP_ALIGN.LEFT,
          anchor=MSO_ANCHOR.TOP, spacing=6):
    """paras: list of str or (str, dict) where dict may set size/color/bold/bullet."""
    tb = slide.shapes.add_textbox(x, y, w, h)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.03)
    for i, para in enumerate(paras):
        txt, opt = (para, {}) if isinstance(para, str) else para
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = opt.get("align", align)
        p.space_after = Pt(opt.get("spacing", spacing))
        bullet = opt.get("bullet", False)
        r = p.add_run()
        r.text = ("•  " + txt) if bullet else txt
        r.font.size = Pt(opt.get("size", size))
        r.font.bold = opt.get("bold", bold)
        r.font.color.rgb = opt.get("color", color)
        r.font.name = "Calibri"
    return tb


def _rect(slide, x, y, w, h, fill, line=None):
    s = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    s.fill.solid()
    s.fill.fore_color.rgb = fill
    if line is None:
        s.line.fill.background()
    else:
        s.line.color.rgb = line
    s.shadow.inherit = False
    return s


def _image(slide, path, x, y, w, h):
    """Place an image inside the box (x, y, w, h), keeping its aspect ratio, centered."""
    with Image.open(path) as im:
        iw, ih = im.size
    scale = min(w / iw, h / ih)
    dw, dh = int(iw * scale), int(ih * scale)
    return slide.shapes.add_picture(path, x + (w - dw) // 2, y + (h - dh) // 2, Emu(dw), Emu(dh))


def _table(slide, x, y, w, rows, col_w, header_fill=NAVY, font=11, row_h=0.32, highlight=None, first_left=True):
    nr, nc = len(rows), len(rows[0])
    shape = slide.shapes.add_table(nr, nc, x, y, w, Inches(row_h * nr))
    tbl = shape.table
    for j, cw in enumerate(col_w):
        tbl.columns[j].width = Inches(cw)
    for i, row in enumerate(rows):
        tbl.rows[i].height = Inches(row_h)
        for j, val in enumerate(row):
            cell = tbl.cell(i, j)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
            tf = cell.text_frame
            tf.text = str(val)
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.LEFT if (j == 0 and first_left) else PP_ALIGN.CENTER
            r = p.runs[0]
            r.font.size = Pt(font)
            r.font.name = "Calibri"
            cell.fill.solid()
            if i == 0:
                cell.fill.fore_color.rgb = header_fill
                r.font.bold = True
                r.font.color.rgb = WHITE
            else:
                hl = highlight(i, row) if highlight else False
                cell.fill.fore_color.rgb = RGBColor(0xFD, 0xED, 0xE3) if hl else (PANEL if i % 2 == 0 else WHITE)
                r.font.color.rgb = ORANGE if (hl and j > 0) else INK
                r.font.bold = bool(hl and j > 0)
    return tbl


class Deck:
    def __init__(self):
        self.prs = Presentation()
        self.prs.slide_width, self.prs.slide_height = SLIDE_W, SLIDE_H
        self.n = 0

    def slide(self, title, kicker=None, source=None, notes=None):
        s = self.prs.slides.add_slide(self.prs.slide_layouts[6])
        self.n += 1
        _rect(s, 0, 0, SLIDE_W, Inches(0.08), ORANGE)
        if kicker:
            _text(s, Inches(0.55), Inches(0.28), Inches(12), Inches(0.35), [kicker.upper()], size=12,
                  color=ORANGE, bold=True)
        _text(s, Inches(0.55), Inches(0.55), Inches(12.2), Inches(0.75), [title], size=28, color=NAVY, bold=True)
        _rect(s, Inches(0.55), Inches(1.32), Inches(12.2), Emu(9525), RULE)
        foot = f"{self.n}"
        if source:
            _text(s, Inches(0.55), Inches(7.05), Inches(11.5), Inches(0.3), [f"Source: {source}"], size=10,
                  color=MUTED)
        _text(s, Inches(12.2), Inches(7.05), Inches(0.6), Inches(0.3), [foot], size=10, color=MUTED,
              align=PP_ALIGN.RIGHT)
        if notes:
            s.notes_slide.notes_text_frame.text = notes
        return s

    def callout(self, s, x, y, w, h, big, small, color=ORANGE):
        _rect(s, x, y, w, h, PANEL)
        _rect(s, x, y, Inches(0.07), h, color)
        _text(s, x + Inches(0.2), y + Inches(0.08), w - Inches(0.3), Inches(0.6), [big], size=26, color=color,
              bold=True)
        _text(s, x + Inches(0.2), y + Inches(0.68), w - Inches(0.3), h - Inches(0.75), [small], size=13,
              color=INK)


# ============================================================================= slides
NOTES = {}  # slide number -> narration (mirrored in VIDEO_SCRIPT.md)


def build(N, out_path):
    d = Deck()
    fig = lambda name: os.path.join(FIG, name)

    # 1 -------------------------------------------------------------- title
    s = d.prs.slides.add_slide(d.prs.slide_layouts[6])
    d.n += 1
    _rect(s, 0, 0, SLIDE_W, SLIDE_H, NAVY)
    _rect(s, Inches(0.8), Inches(2.0), Inches(0.12), Inches(2.3), ORANGE)
    _text(s, Inches(1.1), Inches(1.9), Inches(11.4), Inches(1.9),
          ["Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series"],
          size=36, color=WHITE, bold=True)
    _text(s, Inches(1.1), Inches(3.75), Inches(11.4), Inches(0.6),
          ["A per-channel leakage audit of BPE motif tokenizers for walk-forward volatility forecasting"],
          size=18, color=RGBColor(0xCB, 0xD5, 0xE1))
    _text(s, Inches(1.1), Inches(5.3), Inches(11.4), Inches(0.9), [TEAM_LINE, DATE_LINE], size=16,
          color=RGBColor(0xCB, 0xD5, 0xE1), spacing=4)
    s.notes_slide.notes_text_frame.text = NOTES[1]

    # 2 -------------------------------------------------------------- question & contributions
    s = d.slide("Which parts of a tokenizer leak the future — and by how much?", "Research question",
                notes=NOTES[2], source="DA4022 proposal; FINAL_REPORT.md §1")
    _text(s, Inches(0.55), Inches(1.6), Inches(6.0), Inches(5.2), [
        ("Two ways a BPE motif tokenizer can leak in walk-forward forecasting", {"bold": True, "size": 17}),
        ("Fit-time: normalization, merge table or decoding table estimated on data that overlaps the test period",
         {"bullet": True}),
        ("Encode-time: applying Zhang et al.'s incremental-tokenization definition, rank-ordered BPE is not "
         "incremental at an arbitrary origin — a stream tokenized then split carries post-origin information",
         {"bullet": True}),
        ("Tokenizing only the context instead leaves a partial final token (partial-token bias)", {"bullet": True}),
    ], size=17, spacing=10)
    _text(s, Inches(6.9), Inches(1.6), Inches(5.9), Inches(5.2), [
        ("Contributions", {"bold": True, "size": 17}),
        ("Channel-by-channel audit: five leaks, each switched on alone vs a matched control, in training and testing",
         {"bullet": True}),
        ("Boundary-leakage vs partial-token-bias trade-off measured against the same causal reference",
         {"bullet": True}),
        ("Causal design: hour-bounded BPE, hourly scale tokens, causal RoV×SEI gate (mechanism borrowed from PATK)",
         {"bullet": True}),
        ("6 walk-forward folds + GARCH oracle-floor negative control + 2025 holdout scored once + "
         "fold-only exploratory extensions", {"bullet": True}),
    ], size=17, spacing=10)

    # 3 -------------------------------------------------------------- pipeline
    s = d.slide("A causal reference pipeline with five leakage switches", "Method",
                notes=NOTES[3], source="results/figures/pipeline_leak_switches.png; tokenization/leak_tokenizer.py")
    _image(s, fig("pipeline_leak_switches.png"), Inches(0.55), Inches(1.5), Inches(12.2), Inches(3.6))
    _table(s, Inches(0.55), Inches(5.2), Inches(12.2), [
        ["Switch", "Leak arm vs matched control (one change each)"],
        ["Ch1 window norm", "scale = std(r[t−C : t+H]) vs std(r[t−C : t])"],
        ["Ch2 / Ch3 tables", "merge table / decoding tables fitted on train+test vs train only"],
        ["Ch4 boundary", "unconstrained BPE over context+horizon, cut at t vs context only"],
        ["Ch5 FFT gate", "gate window centered on the hour (reaches 6 bars past t) vs trailing"],
    ], [2.6, 9.6], font=12, row_h=0.33)

    # 4 -------------------------------------------------------------- setup
    s = d.slide("Data, target, model and statistics", "Experimental setup", notes=NOTES[4],
                source="data/folds.py; data/preprocess.py; models/gpt_model.py; results/fold_<k>_baselines.json")
    _text(s, Inches(0.55), Inches(1.55), Inches(6.2), Inches(5.4), [
        ("Data & target", {"bold": True, "size": 16, "color": NAVY}),
        ("Binance BTCUSDT 5-min returns, 2021–2025 (proposal said 2019–2025)", {"bullet": True}),
        ("Trailing EWMA vol (λ = 0.94); origins on clock hours; context 144 bars", {"bullet": True}),
        ("Target RV_H[t] = Σ r²_{t+k}, k < H, from bars ≤ t−1; H = 12 (next hour), 288 (next day)", {"bullet": True}),
        ("Model", {"bold": True, "size": 16, "color": NAVY}),
        ("855,169-parameter GPT (4 layers, d = 128): direct RV head + generative head decoding sampled tokens "
         "with E[x² | bin, previous bin]", {"bullet": True}),
        ("QLIKE training loss; HAR-RV and stationarity-constrained GARCH(1,1) on the same origins", {"bullet": True}),
        ("Statistics", {"bold": True, "size": 16, "color": NAVY}),
        ("Inflation = % QLIKE reduction vs matched control; block bootstrap, DM + Holm, GW, TOST ±5%, "
         "random-effects pooling, Model Confidence Set", {"bullet": True}),
    ], size=14, spacing=4)
    _table(s, Inches(7.0), Inches(1.65), Inches(5.75), [
        ["Split", "Test period", "Test origins"],
        ["Fold 1", "2022-Q1", "2,160"], ["Fold 2", "2022-Q2", "2,184"], ["Fold 3", "2022-H2", "4,416"],
        ["Fold 4", "2023-H1", "4,330"], ["Fold 5", "2023-H2", "4,416"], ["Fold 6", "2024", "8,784"],
        ["Folds 1–6", "pooled", N["n_test_folds"]],
        ["Holdout", "2025 (scored once)", N["hold_n"]],
    ], [1.5, 2.5, 1.75], font=13, row_h=0.4,
        highlight=lambda i, r: r[0] in ("Folds 1–6", "Holdout"))
    _text(s, Inches(7.0), Inches(5.45), Inches(5.75), Inches(1.0), [
        "Expanding windows, all training from 2021-01. Train origins purged so targets never reach the test period."],
        size=12, color=MUTED)

    # 5 -------------------------------------------------------------- audit fixes
    s = d.slide("An audit withdrew the midterm numbers", "Corrections (FIXES.md)", notes=NOTES[5],
                source="FIXES.md (15 Oct audit, fixes committed 24d9c8f / fc12681)")
    _table(s, Inches(0.55), Inches(1.6), Inches(12.2), [
        ["Withdrawn midterm claim", "Cause", "Corrected"],
        ["Ch4 “+16.6% significant leak”", "Ch4 tokenizer never fitted (0 merges): input-format shift",
         "BPE fitted on train; context-only control added"],
        ["GARCH 21.7% worse than HAR", "GARCH scored on HAR's target array, shifted 1,440 bars",
         "One target and origin set for all models"],
        ["“84% boundary mismatch”", "Measured on raw returns (2 of 32 bins used)",
         f"{N['bm_ref_aligned']} at hour-aligned origins; {N['bm_ch4_aligned']} straddle for unconstrained BPE"],
        ["“3.99× compression”", "Same raw-return artifact", f"{N['comp64']} bars/token at 64 merges"],
    ], [3.4, 4.6, 4.2], font=13, row_h=0.55)
    _text(s, Inches(0.55), Inches(4.75), Inches(12.2), Inches(2.1), [
        ("Also fixed: one model per arm with the leak in training and testing · conditional decoding actually used "
         "(Ch3) · dedicated PAD id · QLIKE loss · gate and scale tokens wired in · causality tests that can fail · "
         "clock-hour origins robust to outages · HAR weekly lag 2,016 bars · GARCH by constrained MLE", {}),
        ("Every number in this presentation comes from the corrected pipeline.", {"bold": True, "color": ORANGE}),
    ], size=15)

    # 6 -------------------------------------------------------------- token causality
    s = d.slide("Hour boundaries make the tokenizer incremental at hour-aligned origins", "Result 1 · token audit",
                notes=NOTES[6], source="results/boundary_mismatch_results.json (Fold 1); "
                                       "results/figures/boundary_mismatch_and_compression.png")
    _image(s, fig("boundary_mismatch_and_compression.png"), Inches(0.55), Inches(1.5), Inches(8.6), Inches(5.4))
    d.callout(s, Inches(9.45), Inches(1.6), Inches(3.3), Inches(1.6), N["bm_ref_aligned"],
              f"reference: prefix mismatch at hour-aligned origins, and {N['bm_ref_pert']} under future perturbation",
              TEAL)
    d.callout(s, Inches(9.45), Inches(3.35), Inches(3.3), Inches(1.6), N["bm_ch5_pert"],
              "centered gate (Ch5): pre-origin tokens change when the future is perturbed")
    d.callout(s, Inches(9.45), Inches(5.1), Inches(3.3), Inches(1.6), N["bm_ch4_aligned"],
              "unconstrained BPE (Ch4): a token straddles the hour-aligned origin")

    # 7 -------------------------------------------------------------- negative control
    s = d.slide("Negative control: only the normalization leak beats the oracle floor", "Result 2 · simulated GARCH",
                notes=NOTES[7], source="results/negative_control_results.json; "
                                       "results/figures/negative_control_oracle_floor.png")
    _image(s, fig("negative_control_oracle_floor.png"), Inches(0.55), Inches(1.5), Inches(8.6), Inches(5.4))
    d.callout(s, Inches(9.45), Inches(1.6), Inches(3.3), Inches(1.6), N["nc_floor"],
              "analytic oracle QLIKE — no causal model can beat it in expectation", TEAL)
    d.callout(s, Inches(9.45), Inches(3.35), Inches(3.3), Inches(1.6), N["nc_ch1_d"],
              f"Ch1 window-norm leak, direct head (generative {N['nc_ch1_g']}): impossibly good")
    _text(s, Inches(9.45), Inches(5.1), Inches(3.3), Inches(1.7), [
        f"All other arms, direct head: QLIKE {N['nc_others_d']}, none significantly below the floor. "
        "Only Ch1 and the positive control violate it, on both heads."], size=13, color=INK)

    # 8 -------------------------------------------------------------- main figure
    s = d.slide("Ch1 dominates; every other channel is within a few percent", "Result 3 · 6 folds, H = 12",
                notes=NOTES[8], source="results/figures/inflation_all_folds_h12.png = seed-0 pooled values "
                                       "(results/inflation_table_h12.md)")
    _image(s, fig("inflation_all_folds_h12.png"), Inches(0.55), Inches(1.45), Inches(8.4), Inches(5.5))
    d.callout(s, Inches(9.25), Inches(1.6), Inches(3.5), Inches(1.75), N["ch1_d"],
              f"Ch1 direct, 3 seeds, {N['ch1_d_n']} origins (90% CI {N['ch1_d_lo']} to {N['ch1_d_hi']})")
    d.callout(s, Inches(9.25), Inches(3.5), Inches(3.5), Inches(1.75), N["ch1_d_re"].split(" ")[0],
              f"random effects across folds, I² = {N['ch1_d_i2']}: the same size in every period")
    _text(s, Inches(9.25), Inches(5.4), Inches(3.5), Inches(1.5), [
        f"Figure labels are seed 0 (Ch1 {N['s0_ch1_d']} direct, {N['s0_ch1_g']} generative). "
        "Headline numbers use 3 seeds."], size=12, color=MUTED)

    # 9 -------------------------------------------------------------- robust table
    s = d.slide("3-seed robust analysis: the small channels are bounded", "Result 3 · equivalence bounds",
                notes=NOTES[9], source="results/robust_stats_h12.md / .json (seeds 0–2 averaged per origin; "
                                       "Holm within family; DerSimonian–Laird RE)")
    rows = [["Comparison", "Head", "Inflation", "90% CI (upper = bound)", "Holm p", "RE across folds [95% CI]", "I²"]]
    rows += [list(r) for r in N["robust_rows"]]
    _table(s, Inches(0.55), Inches(1.5), Inches(12.2), rows, [3.3, 1.15, 1.1, 2.2, 0.95, 2.75, 0.75],
           font=11, row_h=0.36, highlight=lambda i, r: r[0].startswith("Ch1"))

    # 10 ------------------------------------------------------------- trade-off
    s = d.slide("Boundary leakage can hide partial-token bias", "Result 4 · the trade-off", notes=NOTES[10],
                source="robust_stats_h12.md (H = 12); inflation_table_h288.md; holdout_h12_leak_audit.json")
    _table(s, Inches(0.55), Inches(1.6), Inches(12.2), [
        ["Setting", "Reference", "Ch4 control (context only)", "Ch4 leak (stream → split)",
         "Partial-token bias", "Boundary leakage"],
        ["H = 12, Folds 1–6, 3 seeds, direct", N["ref3_d"], N["unc3_d"], "–", N["ptb_d"] + " (n.s.)",
         N["ch4_d"] + " (n.s.)"],
        ["H = 288, Fold 1, seed 0, direct", N["h288_q_ref"], N["h288_q_ch4c"], N["h288_q_ch4"], N["h288_ptb"],
         N["h288_ch4"]],
        ["2025 holdout, seed 0, generative", N["hold_q_ref_g"], N["hold_q_ch4c_g"], N["hold_q_ch4_g"],
         N["hold_ptb_g"], N["hold_ch4_g"]],
    ], [3.4, 1.4, 2.2, 2.2, 1.5, 1.5], font=13, row_h=0.5, highlight=lambda i, r: i >= 2)
    _text(s, Inches(0.55), Inches(4.0), Inches(12.2), Inches(2.9), [
        ("At H = 12 both effects are small on average (pooled, 3 seeds).", {"bullet": True}),
        ("Where partial-token bias is large, the boundary leak recovers almost all of it: the leaky arm lands "
         f"next to the causal reference ({N['h288_q_ch4']} vs {N['h288_q_ref']}; "
         f"{N['hold_q_ch4_g']} vs {N['hold_q_ref_g']}).", {"bullet": True}),
        ("So a tokenize-then-split pipeline can look as good as a causal one while hiding the partial-token cost. "
         "Both large-effect settings are single-seed: an observed pattern, not an established effect.",
         {"bullet": True}),
    ], size=17, spacing=10)

    # 11 ------------------------------------------------------------- H = 288
    s = d.slide("At the next-day horizon every channel grows", "Result 5 · H = 288 (Fold 1, seed 0, direct)",
                notes=NOTES[11], source="results/inflation_table_h288.md; results/figures/inflation_all_folds_h288.png")
    _image(s, fig("inflation_all_folds_h288.png"), Inches(0.55), Inches(1.5), Inches(12.2), Inches(3.9))
    _table(s, Inches(0.55), Inches(5.55), Inches(12.2), [
        ["Ch1 window norm", "Ch2 merge table", "Ch4 boundary", "Ch5 centered FFT", "Partial-token bias"],
        [N["h288_ch1"], N["h288_ch2"], N["h288_ch4"], N["h288_ch5"], N["h288_ptb"]],
    ], [2.44] * 5, font=14, row_h=0.45, first_left=False)

    # 12 ------------------------------------------------------------- holdout
    s = d.slide("2025 holdout, scored once: Ch1 still inflates by ~70%", "Result 6 · out of sample",
                notes=NOTES[12], source=f"results/holdout_h12_leak_audit.json; HOLDOUT_SCORED.json "
                                        f"(scored {N['hold_time']}, commit {N['hold_commit']}, rescored = "
                                        f"{str(N['hold_rescored']).lower()}); robust_stats_h12.md (MCS)")
    _image(s, fig("holdout_inflation_h12.png"), Inches(0.55), Inches(1.45), Inches(8.3), Inches(3.3))
    _text(s, Inches(0.55), Inches(4.85), Inches(8.3), Inches(2.1), [
        (f"Ch1: {N['hold_ch1_d']} direct, {N['hold_ch1_g']} generative (seed 0, {N['hold_n']} origins)",
         {"bullet": True}),
        (f"Ch2 changes sign across heads ({N['hold_ch2_d']} direct, {N['hold_ch2_g']} generative): "
         "no consistent advantage", {"bullet": True}),
        (f"Generative Ch4 {N['hold_ch4_g']} recovers a {N['hold_ptb_g']} partial-token bias", {"bullet": True}),
    ], size=14, spacing=4)
    mrows = [["Model Confidence Set (α = 0.10)", "Folds 1–6", "2025"]]
    fold_q = {m: (q, inn) for m, q, _, inn in N["mcs_folds"]}
    hold_q = {m: (q, inn) for m, q, _, inn in N["mcs_hold"]}
    for m, _, _, _ in N["mcs_folds"]:
        mark = lambda t: f"{t[0]}{'  ✓' if t[1] else ''}"
        mrows.append([m, mark(fold_q[m]), mark(hold_q[m])])
    _table(s, Inches(9.05), Inches(1.55), Inches(3.7), mrows, [1.9, 0.9, 0.9], font=10, row_h=0.36)
    _text(s, Inches(9.05), Inches(4.2), Inches(3.7), Inches(2.6), [
        "✓ = in the MCS. Leak-free GPT direct models lead on the folds; on 2025 GARCH(1,1) alone remains "
        f"(reference GPT {N['hold_ref_vs_garch']} vs GARCH)."], size=12, color=INK)

    # 13 ------------------------------------------------------------- extensions
    s = d.slide("Exploratory extensions (folds only)", "Result 7 · dose-response, cross-asset, token healing",
                notes=NOTES[13], source="results/robust_stats_h12.md; results/figures/extensions_h12.png "
                                        "(2025 holdout not used)")
    _image(s, fig("extensions_h12.png"), Inches(0.55), Inches(1.45), Inches(12.2), Inches(3.6))
    cw = Inches(3.95)
    d.callout(s, Inches(0.55), Inches(5.2), cw, Inches(1.65), N["k1"],
              f"one future bar in the window (k=3 {N['k3']}, k=6 {N['k6']}); 1 seed")
    d.callout(s, Inches(4.67), Inches(5.2), cw, Inches(1.65), N["xa_more_d"],
              f"ETH data up to the split (more data, no leak); the ETH leak adds {N['xa_leak_d']} "
              f"(Holm p {N['xa_leak_d_holm']})", TEAL)
    d.callout(s, Inches(8.8), Inches(5.2), cw, Inches(1.65), N["heal"],
              f"token healing vs plain, same model (90% CI {N['heal_lo']} to {N['heal_hi']}; "
              f"Holm p {N['heal_holm']}); 1 seed", TEAL)

    # 14 ------------------------------------------------------------- conclusions
    s = d.slide("Conclusions", "Takeaways and limitations", notes=NOTES[14],
                source="FINAL_REPORT.md §9–11 · code: github.com/devanshjangid2005/causal-vol-tokenizer")
    _text(s, Inches(0.55), Inches(1.6), Inches(7.3), Inches(5.3), [
        ("What we found", {"bold": True, "size": 17, "color": NAVY}),
        (f"Audit normalization first: a window touching the horizon inflates accuracy by {N['ch1_d']} "
         f"(H = 12), {N['h288_ch1']} (H = 288), {N['hold_ch1_d']} on 2025 — and is the only leak that beats the "
         "oracle floor", {"bullet": True}),
        (f"Table and encode-time leaks are small at H = 12 (90% bounds ≤ {N['ch5_d_hi']}) but grow at the "
         "next-day horizon", {"bullet": True}),
        ("Hour-bounded tokens remove the boundary leak at no measurable partial-token cost at H = 12; "
         "a boundary leak can mask a large partial-token bias", {"bullet": True}),
        ("GPT forecasters lead on the folds but not on 2025 (GARCH alone in the MCS)", {"bullet": True}),
    ], size=17, spacing=12)
    _rect(s, Inches(8.15), Inches(1.6), Inches(4.6), Inches(5.2), PANEL)
    _text(s, Inches(8.35), Inches(1.75), Inches(4.25), Inches(5.0), [
        ("Limitations", {"bold": True, "size": 16, "color": NAVY}),
        ("One asset, data from 2021", {"bullet": True}),
        ("Small model (855k parameters), 5 epochs", {"bullet": True}),
        ("H = 288: Fold 1, one seed, direct head only", {"bullet": True}),
        ("Holdout, dose-response and healing: one seed", {"bullet": True}),
        ("Generative head is noisy (I² up to 95%)", {"bullet": True}),
        ("Reproduce", {"bold": True, "size": 16, "color": NAVY}),
        ("experiments/run_all.py · run_extensions.py · robust_stats.py", {"size": 13}),
    ], size=14, spacing=5)

    d.prs.save(out_path)
    return d.n


NOTES.update({
    1: "Hi, this is our DA4022 project on look-ahead bias inside motif tokenizers for financial time series. "
       "We built a leak-free volatility forecasting pipeline, then switched on one source of future "
       "information at a time and measured how much better each one makes the forecasts look.",
    2: "BPE motif tokenizers are usually tested zero-shot on new datasets. Used for walk-forward forecasting on "
       "one market, they can leak in two ways. At fit time, statistics or tables are estimated on data that "
       "overlaps the test period. At encode time, applying Zhang et al.'s incremental-tokenization definition, "
       "BPE is not incremental at an arbitrary origin: a stream tokenized then split carries future "
       "information, while tokenizing only the context leaves a partial final token. We audit each channel "
       "and measure that trade-off.",
    3: "Here is the causal reference pipeline. Five-minute returns are normalized by trailing EWMA volatility, "
       "each hour gets a scale token, BPE merges never cross an hour boundary, and a causal RoV times SEI gate, "
       "borrowed from PATK, shortens tokens in volatile hours. A small GPT forecasts realized volatility. Each "
       "red switch is one leak, differing from its matched control in exactly that change, in training and "
       "testing.",
    4: "We use Binance BTCUSDT five-minute data from 2021 to 2025; the proposal said 2019, but the study as run "
       "starts in 2021. The target is the sum of squared returns over the next 12 or 288 bars, using only data "
       "before the origin. The GPT has 855,169 parameters, with a direct head and a generative head. We have "
       "six expanding folds with 26,290 test origins in total, and 2025 as a holdout scored exactly once.",
    5: "Before the results: an October audit found bugs behind our midterm numbers. The plus 16.6 percent "
       "Channel 4 leak, GARCH looking 21.7 percent worse than HAR, the 84 percent boundary mismatch and the "
       "3.99 times compression are all withdrawn. Everything today comes from the corrected pipeline.",
    6: "First, the token-level audit on Fold 1. With hour boundaries, the reference tokenizer gives 0 percent "
       "prefix mismatch at hour-aligned origins, and perturbing the future changes nothing. The centered gate "
       "of Channel 5 changes the pre-origin tokens 57 percent of the time, and unconstrained BPE straddles the "
       "origin at 32 percent of origins. Compression is modest: 1.40 bars per token at 64 merges.",
    7: "Next, the negative control. On simulated GARCH data the analytic oracle reaches a QLIKE of 0.1268, "
       "and no causal model should beat it. The reference and the Channel 2 to 5 arms all stay above that "
       "floor. Only the Channel 1 normalization leak, at 0.0694, and the deliberate positive control go below "
       "it, so the controls behave as designed.",
    8: "Now the main result over all six folds at the next-hour horizon. Channel 1, where the normalization "
       "window reaches into the forecast horizon, cuts QLIKE by 63.39 percent on the direct head with three "
       "seeds, and the random-effects estimate across folds is 62.03 percent with zero heterogeneity. The "
       "figure itself shows seed-0 values, such as 63.67 percent, so we quote the three-seed numbers as the "
       "headline.",
    9: "This table is the three-seed robust analysis. Beyond Channel 1, the effects are small: the 90 percent "
       "upper bounds limit Channel 2 to 1.44 percent, Channel 3 to 0.39, Channel 4 to 1.05 and Channel 5 to "
       "2.84 percent on the direct head. Channel 5 direct and Channel 3 are significant after Holm correction, "
       "but tiny. Partial-token bias is not significant at this horizon.",
    10: "Here is the trade-off. At the next-hour horizon both effects are small on average. But where "
        "partial-token bias is large, at the next-day horizon and on the 2025 generative head, the boundary "
        "leak recovers almost all of it: 0.3137 against a reference of 0.3128, and 0.4691 against 0.4603. A "
        "leaky pipeline can look as good as a causal one. These two settings are single-seed, so we present "
        "this as an observed pattern.",
    11: "At the next-day horizon, on Fold 1, every channel grows. Channel 1 reaches 96.36 percent, Channel 4 "
        "16.10, Channel 5 7.02 and Channel 2 5.67 percent, all significant after Holm correction, while "
        "partial-token bias costs 19.53 percent. This is one fold, one seed and the direct head only.",
    12: "Then the 2025 holdout, scored once on November 17 and never re-scored. Channel 1 still inflates "
        "accuracy by 73.90 percent on the direct head and 68.32 on the generative head. Channel 2 flips sign "
        "across heads. In the model confidence set, the leak-free GPT direct models lead on the folds, but on "
        "2025 only GARCH remains.",
    13: "Finally, the exploratory extensions, which use the folds only. Even one future bar in the "
        "normalization window inflates accuracy by 2.70 percent, rising to 18.38 at six bars. Adding ETH data "
        "up to the split helps by 3.06 percent, while leaking ETH through the test period adds only 0.95, not "
        "significant after Holm. Token healing improves the same model by 1.70 percent, also not significant "
        "after Holm.",
    14: "To conclude: check normalization first, because a window that touches the horizon is the large, stable "
        "leak, and the only one that beats the oracle floor. Table and boundary leaks are small at one hour "
        "but grow at one day, and a boundary leak can hide partial-token bias. Limitations: one asset, a small "
        "model, and single seeds for some settings. Code, results and report are in our repository. "
        "Thank you.",
})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "presentation", "final_presentation.pptx"))
    ap.add_argument("--check", action="store_true", help="verify the numbers only; write nothing")
    args = ap.parse_args()
    for name in ["pipeline_leak_switches.png", "boundary_mismatch_and_compression.png",
                 "negative_control_oracle_floor.png", "inflation_all_folds_h12.png", "inflation_all_folds_h288.png",
                 "holdout_inflation_h12.png", "extensions_h12.png"]:
        if not os.path.exists(os.path.join(FIG, name)):
            raise SystemExit(f"missing figure results/figures/{name}")
    N = load_numbers()
    verify(N)
    if args.check:
        return
    n = build(N, args.out)
    print(f"Wrote {n} slides to {os.path.relpath(args.out, ROOT)}")


if __name__ == "__main__":
    sys.exit(main())
