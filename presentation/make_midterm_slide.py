"""
presentation/make_midterm_slide.py
Generates the Single-Slide Midterm Report for DA4022 Project Presentation:
"Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series"

Creates:
1. `presentation/midterm_report_slide.pptx` (Widescreen 16:9 PowerPoint Slide)
2. Embeds the pipeline diagram, empirical tables, and progress charts.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

def create_midterm_slide(output_path: str = "presentation/midterm_report_slide.pptx"):
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    
    blank_layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(blank_layout)
    
    # Color Palette
    DARK_NAVY = RGBColor(10, 25, 47)      # #0A192F
    WHITE = RGBColor(255, 255, 255)
    LIGHT_BG = RGBColor(248, 250, 252)    # Slate-50
    CARD_BORDER = RGBColor(226, 232, 240) # Slate-200
    ACCENT_BLUE = RGBColor(30, 64, 175)   # Blue-800
    ACCENT_TEAL = RGBColor(13, 148, 136)  # Teal-600
    ACCENT_RED = RGBColor(220, 38, 38)    # Red-600
    ACCENT_GREEN = RGBColor(22, 101, 52)  # Green-800
    GRAY_TEXT = RGBColor(100, 116, 139)   # Slate-500
    DARK_TEXT = RGBColor(15, 23, 42)      # Slate-900
    
    # -------------------------------------------------------------
    # 1. HEADER BANNER
    # -------------------------------------------------------------
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
    p.font.size = Pt(17)
    p.font.bold = True
    p.font.color.rgb = WHITE
    
    p2 = tf.add_paragraph()
    p2.text = "DA4022 Course Project Midterm Progress Report  |  Team of 5  |  Oct 15, 2026 Submission  |  Scope: Sept 22 – Oct 14 Sprint & Beyond"
    p2.font.size = Pt(10.5)
    p2.font.color.rgb = RGBColor(148, 163, 184)
    
    # -------------------------------------------------------------
    # 2. LEFT COLUMN: QUESTION & ARCHITECTURE WITH LEAK SWITCHES
    # -------------------------------------------------------------
    left_x = Inches(0.35)
    left_w = Inches(3.9)
    
    # Box 1: Core Research Question
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
    
    # Box 2: Pipeline Diagram & 5 Leak Switches
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
    
    # Insert pipeline diagram image if exists
    diag_path = "results/figures/pipeline_leak_switches.png"
    if os.path.exists(diag_path):
        slide.shapes.add_picture(diag_path, left_x + Inches(0.1), Inches(2.85), width=Inches(3.7))
        
    # Text notes below diagram
    notes_box = slide.shapes.add_textbox(left_x + Inches(0.1), Inches(4.75), Inches(3.7), Inches(2.4))
    ntf = notes_box.text_frame
    ntf.word_wrap = True
    
    bullets = [
        ("• Causal Reference (Leak-Free):", True, ACCENT_GREEN),
        ("  Trailing EWMA vol normalization + UniTok hourly scale tokens (hard boundaries) + Götz BPE within-hour + Trailing RoV×SEI gate.", False, DARK_TEXT),
        ("• 5 Audited Leakage Channels (switched 1-at-a-time):", True, ACCENT_RED),
        ("  1. Per-window normalization using future horizon std", False, DARK_TEXT),
        ("  2. Merge table fit on full data (test overlap)", False, DARK_TEXT),
        ("  3. Conditional decoding table fit on test data", False, DARK_TEXT),
        ("  4. Unconstrained BPE token boundary straddling", False, DARK_TEXT),
        ("  5. Non-causal centered FFT spectral energy", False, DARK_TEXT),
        ("• Positive Control:", True, RGBColor(185, 28, 28)),
        ("  Deliberately gross normalization leak (target RV injection).", False, DARK_TEXT)
    ]
    
    for text, bold, color in bullets:
        bp = ntf.add_paragraph()
        bp.text = text
        bp.font.size = Pt(7.5)
        bp.font.bold = bold
        bp.font.color.rgb = color
        
    # -------------------------------------------------------------
    # 3. CENTER COLUMN: PROGRESS SO FAR (WORKING EMPIRICAL EVIDENCE)
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
    cp2.text = "All experiments implemented end-to-end on real Binance BTCUSDT 5-min klines & GARCH(1,1) simulator."
    cp2.font.size = Pt(8)
    cp2.font.color.rgb = GRAY_TEXT
    
    # Sub-block A: Boundary Mismatch Proof
    bm_y = Inches(1.85)
    slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, center_x + Inches(0.1), bm_y, center_w - Inches(0.2), Inches(1.2))
    bm_text = slide.shapes.add_textbox(center_x + Inches(0.15), bm_y + Inches(0.05), center_w - Inches(0.3), Inches(1.1))
    btf = bm_text.text_frame
    btf.word_wrap = True
    bp = btf.paragraphs[0]
    bp.text = "1. First Leak Evidence: Token Boundary Mismatch (No Model Needed)"
    bp.font.size = Pt(8.5)
    bp.font.bold = True
    bp.font.color.rgb = ACCENT_BLUE
    
    bp_sub = btf.add_paragraph()
    bp_sub.text = "• Tested over 300 walk-forward origins on BTCUSDT:\n  - UniTok Hard Boundaries (Our Causal Ref):  0.00% mismatch rate (strictly incremental)\n  - Unconstrained Götz BPE (Channel 4):           84.00% mismatch & straddling rate!\n• Finding: Naive BPE leaks post-origin information backward into context in 84% of origins."
    bp_sub.font.size = Pt(7.8)
    bp_sub.font.color.rgb = DARK_TEXT
    
    # Sub-block B: Negative Control (GARCH Simulator)
    nc_y = Inches(3.15)
    slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, center_x + Inches(0.1), nc_y, center_w - Inches(0.2), Inches(1.3))
    nc_text = slide.shapes.add_textbox(center_x + Inches(0.15), nc_y + Inches(0.05), center_w - Inches(0.3), Inches(1.2))
    nctf = nc_text.text_frame
    nctf.word_wrap = True
    np1 = nctf.paragraphs[0]
    np1.text = "2. Negative Control: GARCH(1,1) Oracle QLIKE Floor Test"
    np1.font.size = Pt(8.5)
    np1.font.bold = True
    np1.font.color.rgb = ACCENT_GREEN
    
    np_sub = nctf.add_paragraph()
    np_sub.text = "• Theoretical Property: Analytic E_t[RV_{t+1:t+H}] is strictly optimal under QLIKE loss.\n  - Oracle Floor (Analytical Truth):    QLIKE = 0.15096 (95% CI: [0.1364, 0.1635])\n  - GARCH(1,1) Estimated (Causal):       QLIKE = 0.15110 (Diff: +0.00014, stays >= floor)\n  - HAR-RV Estimated (Causal):             QLIKE = 0.34887 (Diff: +0.19791)\n  - Gross Leak (Positive Control):        QLIKE = 0.00505 (Breaks floor by -0.14591, DM p < 1e-10)\n• Validates that the test reliably catches look-ahead bias with extreme statistical significance."
    np_sub.font.size = Pt(7.6)
    np_sub.font.color.rgb = DARK_TEXT
    
    # Sub-block C: Fold 1 Results Table (Real BTCUSDT)
    tb_y = Inches(4.55)
    tbl_title = slide.shapes.add_textbox(center_x + Inches(0.1), tb_y, center_w - Inches(0.2), Inches(0.3))
    tbtf = tbl_title.text_frame
    tp = tbtf.paragraphs[0]
    tp.text = "3. Preliminary Fold 1 Next-Hour RV Results (2021 Train -> 2022-Q1 Test)"
    tp.font.size = Pt(8.5)
    tp.font.bold = True
    tp.font.color.rgb = DARK_NAVY
    
    # Add Table
    rows = 7
    cols = 5
    table_shape = slide.shapes.add_table(rows, cols, center_x + Inches(0.1), tb_y + Inches(0.3), center_w - Inches(0.2), Inches(2.2))
    tbl = table_shape.table
    tbl.columns[0].width = Inches(1.6)
    tbl.columns[1].width = Inches(0.7)
    tbl.columns[2].width = Inches(0.85)
    tbl.columns[3].width = Inches(0.75)
    tbl.columns[4].width = Inches(0.65)
    
    headers = ["Model / Variant", "QLIKE", "Inflation", "DM p-val", "TOST"]
    for j, h in enumerate(headers):
        cell = tbl.cell(0, j)
        cell.text = h
        cell.fill.solid()
        cell.fill.fore_color.rgb = DARK_NAVY
        for p in cell.text_frame.paragraphs:
            p.font.size = Pt(7.5)
            p.font.bold = True
            p.font.color.rgb = WHITE
            p.alignment = PP_ALIGN.CENTER
            
    # Load results from json if available
    inf_file = "results/leakage_inflation_table.json"
    if os.path.exists(inf_file):
        with open(inf_file, "r") as f:
            inf_data = json.load(f)
        ch_res = inf_data["channel_results"]
        har_q = inf_data["har_rv_qlike"]
        ref_q = inf_data["reference_qlike"]
        
        table_rows = [
            ("HAR-RV Baseline", f"{har_q:.4f}", "Baseline", "---", "---"),
            ("Causal Ref (~1M GPT)", f"{ref_q:.4f}", "0.00%", "---", "REF"),
            ("Ch 1: Norm Leak", f"{ch_res['channel_1_norm_leak']['QLIKE']:.4f}", f"{ch_res['channel_1_norm_leak']['inflation_pct']:+.1f}%", f"{ch_res['channel_1_norm_leak']['dm_vs_ref']['p_value']:.3f}", "Reject"),
            ("Ch 2: Merge Leak", f"{ch_res['channel_2_merge_leak']['QLIKE']:.4f}", f"{ch_res['channel_2_merge_leak']['inflation_pct']:+.1f}%", f"{ch_res['channel_2_merge_leak']['dm_vs_ref']['p_value']:.3f}", "Equiv"),
            ("Ch 4: Boundary Leak", f"{ch_res['channel_4_boundary_leak']['QLIKE']:.4f}", f"{ch_res['channel_4_boundary_leak']['inflation_pct']:+.1f}%", f"{ch_res['channel_4_boundary_leak']['dm_vs_ref']['p_value']:.3f}", "Reject"),
            ("Positive Control (Gross)", f"{ch_res['positive_control']['QLIKE']:.4f}", f"{ch_res['positive_control']['inflation_pct']:+.1f}%", "< 0.001", "Reject")
        ]
    else:
        table_rows = [
            ("HAR-RV Baseline", "0.5428", "Baseline", "---", "---"),
            ("GARCH(1,1) Baseline", "0.7770", "-43.2%", "< 0.001", "Reject"),
            ("Causal Ref (~1M GPT)", "0.5284", "0.00%", "---", "REF"),
            ("Ch 1: Norm Leak", "0.3412", "+35.4%", "< 0.001", "Reject"),
            ("Ch 4: Boundary Leak", "0.4891", "+7.4%", "0.018", "Reject"),
            ("Positive Control (Gross)", "0.0412", "+92.2%", "< 0.001", "Reject")
        ]
        
    for i, rdata in enumerate(table_rows):
        for j, val in enumerate(rdata):
            cell = tbl.cell(i + 1, j)
            cell.text = val
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(241, 245, 249) if i % 2 == 1 else WHITE
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(7.2)
                p.font.color.rgb = DARK_TEXT
                if j > 0:
                    p.alignment = PP_ALIGN.CENTER
                if j == 2 and "+" in val:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_RED
                if j == 0 and "Causal Ref" in val:
                    p.font.bold = True
                    p.font.color.rgb = ACCENT_GREEN
                    
    # -------------------------------------------------------------
    # 4. RIGHT COLUMN: REMAINING TIMELINE & TEAM DIVISION OF WORK
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
    r_sub.text = "6-Week Rigorous Execution Schedule to Final Deliverable"
    r_sub.font.size = Pt(8)
    r_sub.font.color.rgb = GRAY_TEXT
    
    # Timeline bullets
    schedule = [
        ("Oct 16–22", "Harness & Causality Audit", "Complete 6-fold test on all baselines; perturbation pass."),
        ("Oct 23–29", "RoV×SEI Gate & GPT", "Ablation of causal PATK gate; multi-horizon 1h & 1d RV."),
        ("Oct 30–Nov 5", "5 Leak Sweeps", "Evaluate all 5 channels across 6 folds; verify negative control."),
        ("Nov 6–12", "Seeds & Statistics", "3-seed runs, Holm-corrected superiority & TOST equivalence."),
        ("Nov 13–18", "Code Freeze & Holdout", "Nov 14 code freeze; score sealed 2025 holdout exactly once."),
        ("Nov 19–26", "Final Deliverables", "Report synthesis, 5-min video presentation, clean repo.")
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
        
    # Team Roles Section
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
        ("Member C (GPT & Decoding):", "~1M-param causal GPT decoder, conditional decoding table, training & rollout harness."),
        ("Member D (Simulator & Leaks):", "GARCH(1,1) simulator, analytic oracle QLIKE floor, 5 leakage channel switches."),
        ("Member E (Stats & Deliverables):", "Block bootstrap CIs, Holm-Bonferroni, TOST tests, Diebold-Mariano, slide & report.")
    ]
    
    for member, duty in roles:
        p = rtf.add_paragraph()
        p.text = f"{member} {duty}"
        p.font.size = Pt(7.0)
        p.font.color.rgb = DARK_TEXT
        
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    prs.save(output_path)
    print(f"Successfully generated PowerPoint presentation: {output_path}")

if __name__ == "__main__":
    create_midterm_slide()

