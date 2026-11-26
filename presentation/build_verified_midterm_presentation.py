"""
presentation/build_verified_midterm_presentation.py
Generates the Single-Slide Midterm Report for DA4022 Project Presentation
with COMPLETE finalized empirical results from the full 8-model leak audit:
1. Token-level causality tests (0% aligned leak vs 32% BPE straddling vs 57% centered gate leak)
2. Aligned BTCUSDT Fold 1 baselines (GARCH 0.4825 vs HAR 0.5164, DM p=0.0048)
3. 8-model audit results: Ch1 +63.40% inflation (p<0.0001), Ch2/3 TOST equivalent, Positive Control +99.97%
4. 6-week execution roadmap to Nov 26 & Team division of labor

Outputs:
- presentation/midterm_report_slide.pptx (and presentation/midterm_report_slide_v2.pptx)
- presentation/midterm_report_slide.png (and presentation/midterm_report_slide_v2.png)
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.enum.shapes import MSO_SHAPE


def build_pptx(output_paths=("presentation/midterm_report_slide.pptx", "presentation/midterm_report_slide_v2.pptx")):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    
    # Theme Colors
    DARK_NAVY = RGBColor(10, 25, 47)      # #0A192F
    WHITE = RGBColor(255, 255, 255)
    CARD_BORDER = RGBColor(226, 232, 240) # Slate-200
    ACCENT_BLUE = RGBColor(30, 64, 175)   # Blue-800
    ACCENT_TEAL = RGBColor(13, 148, 136)  # Teal-600
    ACCENT_RED = RGBColor(220, 38, 38)    # Red-600
    ACCENT_GREEN = RGBColor(22, 101, 52)  # Green-800
    GRAY_TEXT = RGBColor(100, 116, 139)   # Slate-500
    DARK_TEXT = RGBColor(15, 23, 42)      # Slate-900

    # 1. HEADER BANNER
    header_box = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(1.15))
    header_box.fill.solid()
    header_box.fill.fore_color.rgb = DARK_NAVY
    header_box.line.color.rgb = DARK_NAVY
    
    tf = header_box.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.4)
    tf.margin_top = Inches(0.12)
    tf.margin_right = Inches(0.4)
    
    p = tf.paragraphs[0]
    p.text = "Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series"
    p.font.size = Pt(16.5)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    p2 = tf.add_paragraph()
    p2.text = "DA4022 Course Project Midterm Progress Report  |  Team of 5  |  October 15, 2026 Submission  |  Scope: Sept 22 – Oct 15 Sprints"
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = RGBColor(148, 163, 184)

    # -------------------------------------------------------------
    # 2. LEFT COLUMN: CORE QUESTION & ARCHITECTURE WITH LEAK SWITCHES
    # -------------------------------------------------------------
    left_x = Inches(0.35)
    left_w = Inches(3.9)
    
    # Core Research Question Box
    q_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_x, Inches(1.3), left_w, Inches(1.1))
    q_box.fill.solid()
    q_box.fill.fore_color.rgb = RGBColor(239, 246, 255)
    q_box.line.color.rgb = RGBColor(191, 219, 254)
    q_tf = q_box.text_frame
    q_tf.word_wrap = True
    q_tf.margin_left = Inches(0.15)
    q_tf.margin_top = Inches(0.1)
    
    qp1 = q_tf.paragraphs[0]
    qp1.text = "CORE RESEARCH QUESTION"
    qp1.font.size = Pt(9)
    qp1.font.bold = True
    qp1.font.color.rgb = ACCENT_BLUE
    
    qp2 = q_tf.add_paragraph()
    qp2.text = "How much does each look-ahead leakage channel (windowing, merge/decode tables, non-incremental token boundaries, FFT) artificially inflate walk-forward realized volatility forecasts in BPE-based foundation models?"
    qp2.font.size = Pt(8.5)
    qp2.font.color.rgb = DARK_TEXT

    # Pipeline Box
    pipe_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left_x, Inches(2.5), left_w, Inches(4.7))
    pipe_box.fill.solid()
    pipe_box.fill.fore_color.rgb = WHITE
    pipe_box.line.color.rgb = CARD_BORDER
    
    p_tf = pipe_box.text_frame
    p_tf.word_wrap = True
    p_tf.margin_left = Inches(0.15)
    p_tf.margin_top = Inches(0.1)
    
    pp1 = p_tf.paragraphs[0]
    pp1.text = "SYSTEM ARCHITECTURE & THE 5 LEAK SWITCHES"
    pp1.font.size = Pt(9.5)
    pp1.font.bold = True
    pp1.font.color.rgb = DARK_NAVY
    
    diag_path = "results/figures/pipeline_leak_switches.png"
    if os.path.exists(diag_path):
        slide.shapes.add_picture(diag_path, left_x + Inches(0.1), Inches(2.85), width=Inches(3.7))
        
    notes_box = slide.shapes.add_textbox(left_x + Inches(0.1), Inches(4.70), Inches(3.7), Inches(2.45))
    ntf = notes_box.text_frame
    ntf.word_wrap = True
    
    bullets = [
        ("• Causal Reference (Strictly Leak-Free Design):", True, ACCENT_GREEN),
        ("  Trailing EWMA vol normalization + UniTok hourly scale tokens (hard boundaries) + Götz BPE within-hour + Trailing RoV×SEI gate (PATK).", False, DARK_TEXT),
        ("• 5 Audited Leakage Channels (switched vs matched controls):", True, ACCENT_RED),
        ("  Ch 1: Per-window normalization using future horizon std", False, DARK_TEXT),
        ("  Ch 2: Merge table fit on full data (test set overlap)", False, DARK_TEXT),
        ("  Ch 3: Conditional decoding table fit on test data realizations", False, DARK_TEXT),
        ("  Ch 4: Unconstrained BPE token boundary straddling", False, DARK_TEXT),
        ("  Ch 5: Non-causal centered FFT spectral energy (+6 bars future)", False, DARK_TEXT),
        ("• Negative & Positive Controls:", True, RGBColor(185, 28, 28)),
        ("  GARCH analytic oracle floor lower-bound test & gross normalization target injection ceiling test.", False, DARK_TEXT)
    ]
    for text, bold, color in bullets:
        bp = ntf.add_paragraph()
        bp.text = text
        bp.font.size = Pt(7.3)
        bp.font.bold = bold
        bp.font.color.rgb = color

    # -------------------------------------------------------------
    # 3. CENTER COLUMN: WORKING EMPIRICAL EVIDENCE SO FAR
    # -------------------------------------------------------------
    center_x = Inches(4.45)
    center_w = Inches(4.75)
    
    center_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, center_x, Inches(1.3), center_w, Inches(5.9))
    center_box.fill.solid()
    center_box.fill.fore_color.rgb = WHITE
    center_box.line.color.rgb = CARD_BORDER
    
    c_tf = center_box.text_frame
    c_tf.word_wrap = True
    c_tf.margin_left = Inches(0.15)
    c_tf.margin_top = Inches(0.1)
    
    cp1 = c_tf.paragraphs[0]
    cp1.text = "PROGRESS SO FAR: WORKING EMPIRICAL EVIDENCE"
    cp1.font.size = Pt(10.5)
    cp1.font.bold = True
    cp1.font.color.rgb = DARK_NAVY
    
    cp2 = c_tf.add_paragraph()
    cp2.text = "Full 8-model leak audit executed on Binance BTCUSDT 5-min bars (130,843 bars, 2021-2022)."
    cp2.font.size = Pt(8)
    cp2.font.color.rgb = GRAY_TEXT

    # Sub-block 1: Causality Tests
    bm_y = Inches(1.85)
    slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, center_x + Inches(0.1), bm_y, center_w - Inches(0.2), Inches(1.28))
    bm_text = slide.shapes.add_textbox(center_x + Inches(0.15), bm_y + Inches(0.04), center_w - Inches(0.3), Inches(1.20))
    btf = bm_text.text_frame
    btf.word_wrap = True
    bp = btf.paragraphs[0]
    bp.text = "1. Token-Level Causality Audit (Real BTCUSDT Data — 500 Test Origins)"
    bp.font.size = Pt(8.5)
    bp.font.bold = True
    bp.font.color.rgb = ACCENT_BLUE
    
    bp_sub = btf.add_paragraph()
    bp_sub.text = (
        "• Slice-vs-Prefix & Future Perturbation Tests (tests that can fail):\n"
        "  - UniTok Hard Boundaries (Our Causal Ref): 0.0% aligned mismatch, 0.0% perturbation leak\n"
        "  - Centered Gate (Channel 5): Future noise changes past tokens in 57.0% of origins!\n"
        "  - Unconstrained Götz BPE (Channel 4): Straddles origin boundary in 32.0% of origins\n"
        "• BPE Compression on EWMA Returns: 1.14x (16 merges), 1.25x (32 merges), 1.40x (64 merges)"
    )
    bp_sub.font.size = Pt(7.4)
    bp_sub.font.color.rgb = DARK_TEXT

    # Sub-block 2: Baseline Comparison
    bs_y = Inches(3.20)
    slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, center_x + Inches(0.1), bs_y, center_w - Inches(0.2), Inches(1.20))
    bs_text = slide.shapes.add_textbox(center_x + Inches(0.15), bs_y + Inches(0.04), center_w - Inches(0.3), Inches(1.14))
    bstf = bs_text.text_frame
    bstf.word_wrap = True
    bsp = bstf.paragraphs[0]
    bsp.text = "2. Aligned Volatility Benchmarks on Fold 1 (2,160 Test Origins, H=12 Bars)"
    bsp.font.size = Pt(8.5)
    bsp.font.bold = True
    bsp.font.color.rgb = ACCENT_GREEN
    
    bsp_sub = bstf.add_paragraph()
    bsp_sub.text = (
        "• Evaluated on identical hour-aligned origins (Jan 1, 2022 to Mar 31, 2022) with identical RV target:\n"
        "  - GARCH(1,1): QLIKE = 0.4825  (95% bootstrap CI: [0.4242, 0.5557])\n"
        "  - HAR-RV:      QLIKE = 0.5164  (95% bootstrap CI: [0.4693, 0.5747])\n"
        "  - Hypothesis Tests: Diebold-Mariano stat = 2.818, p = 0.0048 (GARCH significantly beats HAR)\n"
        "  - Giacomini-White conditional predictive ability: stat = 27.218, p < 0.0001"
    )
    bsp_sub.font.size = Pt(7.4)
    bsp_sub.font.color.rgb = DARK_TEXT

    # Sub-block 3: Finalized Empirical Results Table
    tb_y = Inches(4.48)
    tbl_title = slide.shapes.add_textbox(center_x + Inches(0.1), tb_y, center_w - Inches(0.2), Inches(0.28))
    tbtf = tbl_title.text_frame
    tp = tbtf.paragraphs[0]
    tp.text = "3. Full 8-Model Look-Ahead Inflation Audit (Fold 1 Test Set, 2,160 Origins)"
    tp.font.size = Pt(8.5)
    tp.font.bold = True
    tp.font.color.rgb = DARK_NAVY
    
    rows = 8
    cols = 5
    table_shape = slide.shapes.add_table(rows, cols, center_x + Inches(0.1), tb_y + Inches(0.28), center_w - Inches(0.2), Inches(2.35))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.65)
    tbl.columns[1].width = Inches(0.70)
    tbl.columns[2].width = Inches(0.85)
    tbl.columns[3].width = Inches(0.80)
    tbl.columns[4].width = Inches(0.55)
    
    headers = ["Model / Audit Arm", "QLIKE", "Inflation vs Control", "DM p-val (Holm)", "TOST"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(7.3)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
            
    table_rows = [
        ("GARCH(1,1) Benchmark", "0.4825", "---", "DM p=0.005 vs HAR", "---"),
        ("HAR-RV Benchmark", "0.5164", "---", "Baseline target", "---"),
        ("Causal Reference GPT", "0.5010", "0.00%", "Reference arm", "REF"),
        ("Ch 1: Norm Leak", "0.1969", "+63.40%*", "p < 0.0001 (0.0000)*", "Reject"),
        ("Ch 2: Merge Table Leak", "0.5006", "+0.08%", "p = 0.9538 (0.9538)", "Equiv"),
        ("Ch 4: Boundary Leak", "0.5108", "-2.05%", "p = 0.2731 (0.8194)", "Reject"),
        ("Positive Control (Gross)", "0.0002", "+99.97%*", "p < 0.0001 (0.0000)*", "Reject")
    ]
    for i, rdata in enumerate(table_rows):
        for j, val in enumerate(rdata):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if i % 2 == 1 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(7.1)
                p.font.color.rgb = DARK_TEXT
                if j > 0:
                    p.alignment = PP_ALIGN.CENTER
                if "0.1969" in val or "+63.40%" in val or "+99.97%" in val:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_RED
                elif "Causal Reference" in val or "0.5010" in val:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_GREEN

    # -------------------------------------------------------------
    # 4. RIGHT COLUMN: ROADMAP TO NOV 26 & TEAM DIVISION OF WORK
    # -------------------------------------------------------------
    right_x = Inches(9.35)
    right_w = Inches(3.63)
    
    right_box = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, right_x, Inches(1.3), right_w, Inches(5.9))
    right_box.fill.solid()
    right_box.fill.fore_color.rgb = WHITE
    right_box.line.color.rgb = CARD_BORDER
    
    r_tf = right_box.text_frame
    r_tf.word_wrap = True
    r_tf.margin_left = Inches(0.15)
    r_tf.margin_top = Inches(0.1)
    
    rp1 = r_tf.paragraphs[0]
    rp1.text = "ROADMAP TO NOV 26 & TEAM EXECUTION"
    rp1.font.size = Pt(10)
    rp1.font.bold = True
    rp1.font.color.rgb = DARK_NAVY
    
    r_sub = r_tf.add_paragraph()
    r_sub.text = "6-Week Systematic Execution Schedule to Final Deliverable"
    r_sub.font.size = Pt(8)
    r_sub.font.color.rgb = GRAY_TEXT
    
    schedule = [
        ("Oct 16–22", "Multi-Fold Harness & Baselines", "Run HAR-RV, GARCH, and Ref GPT on Folds 1–4; causality audit pass."),
        ("Oct 23–29", "Causal RoV×SEI Gate & Multi-Horizon", "Ablate PATK gate; evaluate next-hour (H=12) and next-day (H=288) RV."),
        ("Oct 30–Nov 5", "Systematic 5 Leak Sweeps", "Evaluate all 5 channels across 6 folds against matched controls."),
        ("Nov 6–12", "3 Seeds & Statistical Verification", "Full 42 runs (6 folds × 7 variants × 3 seeds); block bootstrap, Holm & TOST."),
        ("Nov 13–18", "Code Freeze (Nov 14) & Holdout", "Freeze code; unseal and evaluate 2025 holdout exactly ONCE."),
        ("Nov 19–26", "Final Deliverables & Video", "Write final IEEE report, record 5-min demo video, clean GitHub repo.")
    ]
    
    timeline_box = slide.shapes.add_textbox(right_x + Inches(0.1), Inches(1.85), right_w - Inches(0.2), Inches(2.6))
    ttf = timeline_box.text_frame
    ttf.word_wrap = True
    
    tp0 = ttf.paragraphs[0]
    tp0.text = "Timeline Strip (Oct 16 - Nov 26 Final):"
    tp0.font.size = Pt(8.5)
    tp0.font.bold = True
    tp0.font.color.rgb = ACCENT_BLUE
    
    for dates, title, done in schedule:
        p = ttf.add_paragraph()
        p.text = f"• {dates}: {title} — {done}"
        p.font.size = Pt(7.3)
        p.font.color.rgb = DARK_TEXT
        
    roles_box = slide.shapes.add_textbox(right_x + Inches(0.1), Inches(4.5), right_w - Inches(0.2), Inches(2.5))
    rtf = roles_box.text_frame
    rtf.word_wrap = True
    
    rp_head = rtf.paragraphs[0]
    rp_head.text = "Team Roles & Division of Labor (Team of 5):"
    rp_head.font.size = Pt(8.5)
    rp_head.font.bold = True
    rp_head.font.color.rgb = ACCENT_TEAL
    
    roles = [
        ("Member A (Data & Baselines):", "Binance 5-min archive ingestion, EWMA normalization, RV targets, HAR-RV & GARCH baselines."),
        ("Member B (Tokenizer & Causality):", "Götz BPE, UniTok hour-boundary constraint, future-perturbation & mismatch rate audit."),
        ("Member C (GPT & Decoding):", "~0.8M-param causal GPT decoder, conditional decoding table, training & rollout harness."),
        ("Member D (Simulator & Leaks):", "GARCH(1,1) simulator, analytic oracle QLIKE floor, 5 leakage channel switches."),
        ("Member E (Stats & Deliverables):", "Block bootstrap CIs, Holm-Bonferroni, TOST tests, Diebold-Mariano, slide & report.")
    ]
    for member, duty in roles:
        p = rtf.add_paragraph()
        p.text = f"{member} {duty}"
        p.font.size = Pt(7.0)
        p.font.color.rgb = DARK_TEXT

    saved = []
    for path in output_paths:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            prs.save(path)
            saved.append(path)
            print(f"Successfully generated PowerPoint presentation: {path}")
        except Exception as e:
            print(f"Could not save to {path} (possibly open in another program): {e}")
    return saved


def render_slide_png(output_paths=("presentation/midterm_report_slide.png", "presentation/midterm_report_slide_v2.png")):
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    
    # 1. TOP HEADER BANNER
    header = patches.Rectangle((0, 0.88), 1.0, 0.12, facecolor="#0A192F", edgecolor="none")
    ax.add_patch(header)
    ax.text(0.025, 0.965, "Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series",
            fontsize=15.0, fontweight="bold", color="#FFFFFF", va="top")
    ax.text(0.025, 0.915, "DA4022 Course Project Midterm Progress Report  |  Team of 5  |  October 15, 2026 Submission  |  Scope: Sept 22 – Oct 15 Sprints",
            fontsize=9.8, color="#94A3B8", va="top")
    
    # 2. COLUMN 1: QUESTION & SYSTEM ARCHITECTURE
    c1_x, c1_w = 0.020, 0.295
    card_q = patches.FancyBboxPatch((c1_x, 0.745), c1_w, 0.115, boxstyle="round,pad=0.012,rounding_size=0.015",
                                    facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=1.5)
    ax.add_patch(card_q)
    ax.text(c1_x + 0.012, 0.845, "CORE RESEARCH QUESTION", fontsize=9.2, fontweight="bold", color="#1E40AF", va="top")
    q_str = (
        "How much does each look-ahead leakage channel (windowing, merge/\n"
        "decode tables, non-incremental token boundaries, FFT) artificially inflate\n"
        "walk-forward realized volatility forecasts in BPE foundation models?"
    )
    ax.text(c1_x + 0.012, 0.815, q_str, fontsize=8.0, color="#0F172A", va="top", linespacing=1.2)
    
    card_pipe = patches.FancyBboxPatch((c1_x, 0.030), c1_w, 0.700, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_pipe)
    ax.text(c1_x + 0.012, 0.715, "SYSTEM ARCHITECTURE & THE 5 LEAK SWITCHES", fontsize=9.2, fontweight="bold", color="#0A192F", va="top")
    
    pipe_img_path = "results/figures/pipeline_leak_switches.png"
    if os.path.exists(pipe_img_path):
        img_pipe = mpimg.imread(pipe_img_path)
        ax_pipe = fig.add_axes([c1_x + 0.005, 0.39, c1_w - 0.01, 0.30])
        ax_pipe.imshow(img_pipe)
        ax_pipe.axis("off")
        
    pipe_notes = (
        "• Causal Reference (Strictly Leak-Free Design):\n"
        "   - Trailing EWMA vol normalization (scale from past only)\n"
        "   - UniTok hourly scale tokens enforce hard boundaries\n"
        "   - Götz BPE within-hour + Trailing RoV×SEI gate (PATK)\n\n"
        "• 5 Audited Leakage Channels (switched vs matched controls):\n"
        "   [1] Per-window norm using future horizon std\n"
        "   [2] Merge table fit on full data (test set overlap)\n"
        "   [3] Conditional decoding table fit on test data\n"
        "   [4] Unconstrained BPE token boundary straddling\n"
        "   [5] Non-causal centered FFT spectral energy (+6 bars)\n\n"
        "• Negative Control: Analytic GARCH oracle floor lower-bound.\n"
        "• Positive Control: Deliberately gross target RV injection."
    )
    ax.text(c1_x + 0.012, 0.365, pipe_notes, fontsize=7.4, color="#1E293B", linespacing=1.20, va="top")
    
    # 3. COLUMN 2: EMPIRICAL EVIDENCE SO FAR
    c2_x, c2_w = 0.325, 0.380
    card_prog = patches.FancyBboxPatch((c2_x, 0.030), c2_w, 0.830, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_prog)
    ax.text(c2_x + 0.012, 0.845, "PROGRESS SO FAR: WORKING EMPIRICAL EVIDENCE", fontsize=10.2, fontweight="bold", color="#0A192F", va="top")
    ax.text(c2_x + 0.012, 0.825, "Full 8-model leak audit on Binance BTCUSDT 5-min bars (130k bars, 2021-2022)", fontsize=7.8, color="#64748B", va="top")
    
    # Box 1: Token Causality
    box_b = patches.FancyBboxPatch((c2_x + 0.01, 0.690), c2_w - 0.02, 0.120, boxstyle="round,pad=0.008,rounding_size=0.01",
                                   facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=1.2)
    ax.add_patch(box_b)
    ax.text(c2_x + 0.018, 0.795, "1. Token-Level Causality Audit (500 Test Origins, Real Returns)", fontsize=8.4, fontweight="bold", color="#15803D", va="top")
    b_text = (
        "• Slice-vs-Prefix & Future Perturbation Audit on BTCUSDT 5-min klines:\n"
        "   - UniTok Hard Boundaries (Our Causal Ref):  0.0% aligned mismatch, 0.0% perturbation leak!\n"
        "   - Centered Gate (Channel 5): Future noise changes past tokens in 57.0% of origins (real leak!)\n"
        "   - Unconstrained Götz BPE (Channel 4): Straddles origin boundary in 32.0% of origins\n"
        "• BPE Compression on EWMA Returns: 1.14x (16 merges), 1.25x (32 merges), 1.40x (64 merges)"
    )
    ax.text(c2_x + 0.018, 0.770, b_text, fontsize=7.1, color="#0F172A", linespacing=1.16, va="top")
    
    # Box 2: Aligned Baselines
    box_n = patches.FancyBboxPatch((c2_x + 0.01, 0.545), c2_w - 0.02, 0.130, boxstyle="round,pad=0.008,rounding_size=0.01",
                                   facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=1.2)
    ax.add_patch(box_n)
    ax.text(c2_x + 0.018, 0.660, "2. Aligned Volatility Benchmarks on Fold 1 (2,160 Test Origins, H=12 Bars)", fontsize=8.4, fontweight="bold", color="#1D4ED8", va="top")
    n_text = (
        "• Evaluated on identical hour-aligned origins (Jan 1, 2022 to Mar 31, 2022) with target sum r_{t+k}^2:\n"
        "   - GARCH(1,1): QLIKE = 0.4825  (95% bootstrap CI: [0.4242, 0.5557])\n"
        "   - HAR-RV:      QLIKE = 0.5164  (95% bootstrap CI: [0.4693, 0.5747])\n"
        "   - Diebold-Mariano Test: Stat = 2.818, p = 0.0048 (GARCH significantly outperforms HAR)\n"
        "   - Giacomini-White Conditional Predictive Ability: Stat = 27.218, p < 0.0001"
    )
    ax.text(c2_x + 0.018, 0.635, n_text, fontsize=7.1, color="#0F172A", linespacing=1.16, va="top")
    
    # Section 3: Experimental Matrix Table
    ax.text(c2_x + 0.012, 0.525, "3. Full 8-Model Look-Ahead Inflation Audit (Fold 1 Test Set, 2,160 Origins)", fontsize=8.4, fontweight="bold", color="#0A192F", va="top")
    
    table_data = [
        ["Model / Audit Arm", "QLIKE", "Inflation vs Control", "DM p-val (Holm)", "TOST"],
        ["GARCH(1,1) Benchmark", "0.4825", "---", "DM p=0.005 vs HAR", "---"],
        ["HAR-RV Benchmark", "0.5164", "---", "Baseline target", "---"],
        ["Causal Reference GPT", "0.5010", "0.00%", "Reference arm", "REF"],
        ["Ch 1: Norm Leak", "0.1969", "+63.40%*", "p < 0.0001 (0.0000)*", "Reject"],
        ["Ch 2: Merge Table Leak", "0.5006", "+0.08%", "p = 0.9538 (0.9538)", "Equiv"],
        ["Ch 4: Boundary Leak", "0.5108", "-2.05%", "p = 0.2731 (0.8194)", "Reject"],
        ["Positive Control (Gross)", "0.0002", "+99.97%*", "p < 0.0001 (0.0000)*", "Reject"]
    ]
    
    col_w = [0.120, 0.055, 0.085, 0.080, 0.040]
    col_starts = [c2_x + 0.01]
    for w in col_w[:-1]:
        col_starts.append(col_starts[-1] + w)
        
    start_y = 0.505
    r_h = 0.052
    
    rect_th = patches.Rectangle((c2_x + 0.01, start_y - r_h), sum(col_w), r_h, facecolor="#0A192F", edgecolor="none")
    ax.add_patch(rect_th)
    for j, h_txt in enumerate(table_data[0]):
        ax.text(col_starts[j] + col_w[j]/2, start_y - r_h/2, h_txt, ha="center", va="center",
                fontsize=7.2, fontweight="bold", color="#FFFFFF")
        
    for i, row in enumerate(table_data[1:]):
        curr_y = start_y - (i + 2) * r_h
        bg_col = "#F8FAFC" if i % 2 == 1 else "#FFFFFF"
        if "Positive" in row[0]:
            bg_col = "#FEF2F2"
        elif "Ch 1" in row[0]:
            bg_col = "#FFF1F2"
        elif "Causal Ref" in row[0]:
            bg_col = "#F0FDF4"
            
        rect_row = patches.Rectangle((c2_x + 0.01, curr_y), sum(col_w), r_h, facecolor=bg_col, edgecolor="#E2E8F0", lw=0.5)
        ax.add_patch(rect_row)
        
        for j, val in enumerate(row):
            txt_col = "#0F172A"
            b_flag = False
            if "0.1969" in val or "+63.40%" in val or "+99.97%" in val:
                txt_col = "#DC2626"
                b_flag = True
            elif "Causal Ref" in val or "0.5010" in val:
                txt_col = "#15803D"
                b_flag = True
                
            ax.text(col_starts[j] + (col_w[j]/2 if j > 0 else 0.004), curr_y + r_h/2, val,
                    ha="center" if j > 0 else "left", va="center",
                    fontsize=6.7, fontweight="bold" if b_flag else "normal", color=txt_col)
            
    # 4. COLUMN 3: TIMELINE & TEAM EXECUTION
    c3_x, c3_w = 0.715, 0.265
    card_plan = patches.FancyBboxPatch((c3_x, 0.030), c3_w, 0.830, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_plan)
    ax.text(c3_x + 0.012, 0.845, "ROADMAP TO NOV 26 & TEAM EXECUTION", fontsize=9.8, fontweight="bold", color="#0A192F", va="top")
    ax.text(c3_x + 0.012, 0.825, "6-Week Systematic Execution to Final Deliverables", fontsize=7.5, color="#64748B", va="top")
    
    timeline_text = (
        "Timeline to Final Submission (Nov 26):\n\n"
        "• Oct 16–22: Multi-Fold Harness & Baselines\n"
        "   Run HAR-RV, GARCH, and Ref GPT across Folds 1–4;\n"
        "   verify future-perturbation test passes on all folds.\n\n"
        "• Oct 23–29: Causal RoV×SEI Gate & Multi-Horizon\n"
        "   Ablate trailing RoV×SEI gate; extend forecast from\n"
        "   next-hour (H=12) to next-day (H=288 bars).\n\n"
        "• Oct 30–Nov 5: Systematic 5 Leak Sweeps\n"
        "   Evaluate all 5 channels across 6 folds against\n"
        "   matched controls; verify negative control holds.\n\n"
        "• Nov 6–12: 3 Seeds & Statistical Verification\n"
        "   Full 42 runs (6 folds × 7 variants × 3 seeds);\n"
        "   block-bootstrap CIs, Holm correction, TOST.\n\n"
        "• Nov 13–18: Code Freeze (Nov 14) & 2025 Holdout\n"
        "   Freeze codebase; unseal and score 2025 holdout\n"
        "   exactly ONCE to prevent post-hoc overfitting.\n\n"
        "• Nov 19–26: Final Report, Video & Open-Source\n"
        "   Write final report, record 5-min presentation,\n"
        "   polish GitHub repo with 1-click reproduction."
    )
    ax.text(c3_x + 0.012, 0.795, timeline_text, fontsize=7.1, color="#1E293B", linespacing=1.18, va="top")
    
    roles_box = patches.FancyBboxPatch((c3_x + 0.008, 0.065), c3_w - 0.016, 0.220, boxstyle="round,pad=0.008,rounding_size=0.01",
                                       facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2)
    ax.add_patch(roles_box)
    ax.text(c3_x + 0.016, 0.272, "Team Roles & Division of Labor (Team of 5):", fontsize=7.6, fontweight="bold", color="#0F766E", va="top")
    
    roles_text = (
        "• Member A: Binance data archive, EWMA vol, HAR-RV.\n"
        "• Member B: BPE tokenizer, UniTok bounds, causality tests.\n"
        "• Member C: ~0.8M-param GPT, conditional decoding, training.\n"
        "• Member D: GARCH simulator, oracle floor, leak switches.\n"
        "• Member E: Bootstrap CIs, Holm correction, TOST, report."
    )
    ax.text(c3_x + 0.016, 0.245, roles_text, fontsize=6.8, color="#1E293B", linespacing=1.22, va="top")
    ax.text(c3_x + 0.010, 0.042, "★ 2025 Holdout sealed in data/SEALED_2025_HOLDOUT.txt",
            fontsize=6.6, fontweight="bold", color="#991B1B", va="top")

    saved = []
    for path in output_paths:
        try:
            os.makedirs(os.path.dirname(path), exist_ok=True)
            plt.savefig(path, bbox_inches="tight", dpi=300)
            saved.append(path)
            print(f"Rendered slide image to {path}")
        except Exception as e:
            print(f"Could not save image to {path}: {e}")
    plt.close()
    return saved


if __name__ == "__main__":
    # This is a 17 Nov Fold-1-only rebuild, not the midterm actually submitted on 15 Oct (a one-slide deck,
    # DA4022_Midterm_Progress.pptx). Its outputs are kept for history; the final deck supersedes them.
    raise SystemExit("Superseded: Fold 1 rebuild of 17 Nov, not the submitted midterm; "
                     "use presentation/build_final_presentation.py.")
    build_pptx()
    render_slide_png()
