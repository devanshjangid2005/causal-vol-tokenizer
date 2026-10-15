"""
presentation/render_slide_image.py
Pixel-perfect high-resolution (300 DPI) rendering of the Single-Slide Midterm Report.
Uses top-aligned text anchors (va="top") to completely eliminate text collisions.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.image as mpimg

def render_slide(output_png: str = "presentation/midterm_report_slide.png"):
    fig = plt.figure(figsize=(16, 9), dpi=300)
    fig.patch.set_facecolor("#F8FAFC")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    
    # -------------------------------------------------------------
    # 1. TOP HEADER BANNER (y: 0.88 to 1.00)
    # -------------------------------------------------------------
    header = patches.Rectangle((0, 0.88), 1.0, 0.12, facecolor="#0A192F", edgecolor="none")
    ax.add_patch(header)
    ax.text(0.025, 0.965, "Quantifying Look-Ahead Bias via Causal, Volatility-Aware Motif Tokenization in Financial Time Series",
            fontsize=15.0, fontweight="bold", color="#FFFFFF", va="top")
    ax.text(0.025, 0.915, "DA4022 Course Project Midterm Progress Report  |  Team of 5  |  October 15, 2026 Submission  |  Scope: Sept 22 – Oct 14 Sprint & Execution",
            fontsize=9.8, color="#94A3B8", va="top")
    
    # -------------------------------------------------------------
    # COLUMN 1: QUESTION & SYSTEM ARCHITECTURE (x: 0.020 to 0.315)
    # -------------------------------------------------------------
    c1_x, c1_w = 0.020, 0.295
    
    # Question Card (y: 0.745 to 0.860)
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
    
    # Architecture Card (y: 0.030 to 0.730)
    card_pipe = patches.FancyBboxPatch((c1_x, 0.030), c1_w, 0.700, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_pipe)
    ax.text(c1_x + 0.012, 0.715, "SYSTEM ARCHITECTURE & THE 5 LEAK SWITCHES", fontsize=9.2, fontweight="bold", color="#0A192F", va="top")
    
    # Pipeline Diagram Image
    pipe_img_path = "results/figures/pipeline_leak_switches.png"
    if os.path.exists(pipe_img_path):
        img_pipe = mpimg.imread(pipe_img_path)
        ax_pipe = fig.add_axes([c1_x + 0.005, 0.39, c1_w - 0.01, 0.30])
        ax_pipe.imshow(img_pipe)
        ax_pipe.axis("off")
        
    pipe_notes = (
        "• Causal Reference (Strictly Leak-Free Design):\n"
        "   - Trailing EWMA normalization (scale from past only)\n"
        "   - UniTok hourly scale tokens enforce hard boundaries\n"
        "   - Götz BPE within-hour + Trailing RoV×SEI gate (PATK)\n\n"
        "• 5 Audited Leakage Channels (switched 1-at-a-time):\n"
        "   [1] Per-window norm using future horizon std\n"
        "   [2] Merge table fit on full data (test set overlap)\n"
        "   [3] Conditional decoding table fit on test data\n"
        "   [4] Unconstrained BPE token boundary straddling\n"
        "   [5] Non-causal centered FFT spectral energy\n\n"
        "• Positive Control: Deliberately gross normalization leak\n"
        "   (hands model target RV to measure theoretical ceiling)."
    )
    ax.text(c1_x + 0.012, 0.365, pipe_notes, fontsize=7.5, color="#1E293B", linespacing=1.22, va="top")
    
    # -------------------------------------------------------------
    # COLUMN 2: EMPIRICAL EVIDENCE SO FAR (x: 0.325 to 0.705)
    # -------------------------------------------------------------
    c2_x, c2_w = 0.325, 0.380
    
    card_prog = patches.FancyBboxPatch((c2_x, 0.030), c2_w, 0.830, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_prog)
    ax.text(c2_x + 0.012, 0.845, "PROGRESS SO FAR: WORKING EMPIRICAL EVIDENCE", fontsize=10.2, fontweight="bold", color="#0A192F", va="top")
    ax.text(c2_x + 0.012, 0.825, "Executed on real Binance BTCUSDT 5-min bars (130k bars) & GARCH(1,1) simulator", fontsize=7.8, color="#64748B", va="top")
    
    # Box 1: Boundary Mismatch (y: 0.705 to 0.810)
    box_b = patches.FancyBboxPatch((c2_x + 0.01, 0.705), c2_w - 0.02, 0.105, boxstyle="round,pad=0.008,rounding_size=0.01",
                                   facecolor="#F0FDF4", edgecolor="#BBF7D0", lw=1.2)
    ax.add_patch(box_b)
    ax.text(c2_x + 0.018, 0.798, "1. First Leak Evidence: Token Boundary Mismatch (No Model Needed)", fontsize=8.4, fontweight="bold", color="#15803D", va="top")
    b_text = (
        "• Evaluated over 300 walk-forward origins on BTCUSDT 5-min klines:\n"
        "   - UniTok Hard Boundaries (Our Causal Ref):   0.00% mismatch rate (strictly incremental)\n"
        "   - Unconstrained Götz BPE (Channel 4):           84.00% mismatch & straddling rate!\n"
        "• Takeaway: Standard BPE creates motifs spanning into the future in 84% of origins."
    )
    ax.text(c2_x + 0.018, 0.776, b_text, fontsize=7.2, color="#0F172A", linespacing=1.18, va="top")
    
    # Box 2: Negative Control (Oracle Floor) (y: 0.550 to 0.690)
    box_n = patches.FancyBboxPatch((c2_x + 0.01, 0.550), c2_w - 0.02, 0.140, boxstyle="round,pad=0.008,rounding_size=0.01",
                                   facecolor="#EFF6FF", edgecolor="#BFDBFE", lw=1.2)
    ax.add_patch(box_n)
    ax.text(c2_x + 0.018, 0.678, "2. Negative Control: GARCH(1,1) Oracle QLIKE Floor Validation", fontsize=8.4, fontweight="bold", color="#1D4ED8", va="top")
    n_text = (
        "• Theoretical Property: Analytic conditional variance is the mathematical lower bound under QLIKE.\n"
        "   - Oracle Floor (Analytical Truth):    QLIKE = 0.15096 (95% CI: [0.1364, 0.1635])\n"
        "   - GARCH(1,1) Estimated (Causal):       QLIKE = 0.15110 (Diff: +0.00014, stays >= floor)\n"
        "   - HAR-RV Estimated (Causal):             QLIKE = 0.34887 (Diff: +0.19791)\n"
        "   - Gross Leak (Positive Control):        QLIKE = 0.00505 (Breaks floor by -0.14591, p < 1e-10)\n"
        "• Takeaway: Confirms that our statistical harness reliably detects look-ahead bias."
    )
    ax.text(c2_x + 0.018, 0.656, n_text, fontsize=7.1, color="#0F172A", linespacing=1.18, va="top")
    
    # Section 3: Fold 1 Results Table Title
    ax.text(c2_x + 0.012, 0.528, "3. Real BTCUSDT Fold 1 Next-Hour RV Results (2021 Train -> 2022-Q1 Test)", fontsize=8.4, fontweight="bold", color="#0A192F", va="top")
    
    table_data = [
        ["Model / Variant", "QLIKE (95% CI)", "Inflation", "DM vs Ref", "TOST"],
        ["HAR-RV Baseline", "0.5428 [0.502, 0.591]", "Baseline", "p = 0.001", "Reject"],
        ["GARCH(1,1) Baseline", "0.7770 [0.692, 0.869]", "-21.74%", "p < 0.001", "Reject"],
        ["Causal Ref (~1M GPT)", "0.6383 [0.518, 0.802]", "0.00%", "Reference", "REF"],
        ["Ch 1: Norm Leak", "0.5930 [0.525, 0.650]", "+7.09%", "p = 0.416", "Diff"],
        ["Ch 2: Merge Table Leak", "0.6384 [0.517, 0.814]", "-0.02%", "p = 0.993", "Equiv"],
        ["Ch 3: Decode Table Leak", "0.6383 [0.518, 0.802]", "+0.00%", "p = 1.000", "Equiv"],
        ["Ch 4: Boundary Leak", "0.5326 [0.458, 0.636]", "+16.56%*", "p = 0.004*", "Reject"],
        ["Ch 5: Non-Causal FFT", "0.6376 [0.517, 0.801]", "+0.10%", "p = 0.003*", "Equiv"],
        ["Positive Control (Gross)", "0.1963 [0.180, 0.210]", "+69.24%*", "p < 0.001*", "Reject"]
    ]
    
    col_w = [0.125, 0.100, 0.055, 0.045, 0.035]
    col_starts = [c2_x + 0.01]
    for w in col_w[:-1]:
        col_starts.append(col_starts[-1] + w)
        
    start_y = 0.505
    r_h = 0.043
    
    # Table header
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
        elif "Ch 4" in row[0]:
            bg_col = "#FFFBEB"
        elif "Causal Ref" in row[0]:
            bg_col = "#F0FDF4"
            
        rect_row = patches.Rectangle((c2_x + 0.01, curr_y), sum(col_w), r_h, facecolor=bg_col, edgecolor="#E2E8F0", lw=0.5)
        ax.add_patch(rect_row)
        
        for j, val in enumerate(row):
            txt_col = "#0F172A"
            b_flag = False
            if j == 2 and "+" in val:
                txt_col = "#DC2626"
                b_flag = True
            elif "Causal Ref" in row[0] and j == 0:
                txt_col = "#15803D"
                b_flag = True
            elif j == 0 and "Leak" in val:
                txt_col = "#B91C1C"
                
            ax.text(col_starts[j] + (col_w[j]/2 if j > 0 else 0.004), curr_y + r_h/2, val,
                    ha="center" if j > 0 else "left", va="center",
                    fontsize=6.7, fontweight="bold" if b_flag else "normal", color=txt_col)
            
    # -------------------------------------------------------------
    # COLUMN 3: TIMELINE & TEAM EXECUTION (x: 0.715 to 0.980)
    # -------------------------------------------------------------
    c3_x, c3_w = 0.715, 0.265
    
    card_plan = patches.FancyBboxPatch((c3_x, 0.030), c3_w, 0.830, boxstyle="round,pad=0.012,rounding_size=0.015",
                                       facecolor="#FFFFFF", edgecolor="#E2E8F0", lw=1.5)
    ax.add_patch(card_plan)
    ax.text(c3_x + 0.012, 0.845, "ROADMAP TO NOV 26 & TEAM EXECUTION", fontsize=9.8, fontweight="bold", color="#0A192F", va="top")
    ax.text(c3_x + 0.012, 0.825, "6-Week Systematic Execution to Final Deliverables", fontsize=7.5, color="#64748B", va="top")
    
    timeline_text = (
        "Timeline to Final Submission (Nov 26):\n\n"
        "• Oct 16–22: Multi-Fold Pipeline & Causality Audit\n"
        "   Run HAR-RV, GARCH, and Ref GPT across Folds 1–4;\n"
        "   verify future-perturbation test passes on all folds.\n\n"
        "• Oct 23–29: Causal RoV×SEI Gate & Horizon Scaling\n"
        "   Ablate trailing RoV×SEI gate; extend forecast from\n"
        "   next-hour (H=12) to next-day (H=288 bars).\n\n"
        "• Oct 30–Nov 5: Systematic 5 Leak Sweeps\n"
        "   Evaluate all 5 channels across 6 folds; confirm\n"
        "   negative-control floor holds on simulated series.\n\n"
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
    
    # Roles Box (y: 0.065 to 0.285)
    roles_box = patches.FancyBboxPatch((c3_x + 0.008, 0.065), c3_w - 0.016, 0.220, boxstyle="round,pad=0.008,rounding_size=0.01",
                                       facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=1.2)
    ax.add_patch(roles_box)
    ax.text(c3_x + 0.016, 0.272, "Team Roles & Division of Labor (Team of 5):", fontsize=7.6, fontweight="bold", color="#0F766E", va="top")
    
    roles_text = (
        "• Member A: Binance data archive, EWMA vol, HAR-RV.\n"
        "• Member B: BPE tokenizer, UniTok bounds, causality tests.\n"
        "• Member C: ~1M-param GPT, conditional decoding, training.\n"
        "• Member D: GARCH simulator, oracle floor, leak switches.\n"
        "• Member E: Bootstrap CIs, Holm correction, TOST, report."
    )
    ax.text(c3_x + 0.016, 0.245, roles_text, fontsize=6.8, color="#1E293B", linespacing=1.22, va="top")
    
    ax.text(c3_x + 0.010, 0.042, "★ 2025 Holdout sealed in data/SEALED_2025_HOLDOUT.txt",
            fontsize=6.6, fontweight="bold", color="#991B1B", va="top")

    plt.savefig(output_png, bbox_inches="tight", dpi=300)
    plt.close()
    print(f"Rendered perfected slide image to {output_png}")

if __name__ == "__main__":
    render_slide()

