"""
presentation/generate_figures.py
Generates Publication-Quality Visualizations for the Midterm Report:
1. Figure 1: Pipeline Architecture Diagram with 5 Leakage Switches
2. Figure 2: Boundary-Mismatch & Compression Trade-off (UniTok vs Götz)
3. Figure 3: Negative Control: Baseline vs Oracle QLIKE Floor
4. Figure 4: Look-Ahead Inflation Summary across Channels & Positive Control
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

os.makedirs("results/figures", exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

def plot_pipeline_diagram(save_path: str = "results/figures/pipeline_leak_switches.png"):
    """
    Renders the End-to-End Methodological Architecture with the 5 Leak Switches marked.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    
    # Define blocks
    boxes = [
        {"x": 0.03, "y": 0.55, "w": 0.12, "h": 0.32, "title": "5-min Returns\n(Binance)", "sub": "Raw BTCUSDT", "color": "#E3F2FD", "ec": "#1E88E5"},
        {"x": 0.19, "y": 0.55, "w": 0.13, "h": 0.32, "title": "EWMA Vol\nNormalization", "sub": "Trailing Scale", "color": "#E8F5E9", "ec": "#43A047"},
        {"x": 0.36, "y": 0.55, "w": 0.14, "h": 0.32, "title": "Hourly Scale +\nUniTok Boundary", "sub": "Hard Constraint", "color": "#FFF3E0", "ec": "#FB8C00"},
        {"x": 0.54, "y": 0.55, "w": 0.14, "h": 0.32, "title": "Götz BPE\nMotif Tokenizer", "sub": "Fixed Bins/Merges", "color": "#F3E5F5", "ec": "#8E24AA"},
        {"x": 0.72, "y": 0.55, "w": 0.12, "h": 0.32, "title": "~1M Param GPT\nDecoder", "sub": "Causal NTP + RV", "color": "#EDE7F6", "ec": "#5E35B1"},
        {"x": 0.88, "y": 0.55, "w": 0.10, "h": 0.32, "title": "Conditional\nDecoding", "sub": "RV Forecast", "color": "#E0F2F1", "ec": "#00897B"},
    ]
    
    for b in boxes:
        rect = patches.FancyBboxPatch((b["x"], b["y"]), b["w"], b["h"], boxstyle="round,pad=0.02,rounding_size=0.03",
                                     facecolor=b["color"], edgecolor=b["ec"], linewidth=2)
        ax.add_patch(rect)
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.62, b["title"], ha="center", va="center", fontweight="bold", fontsize=10, color="#1A237E")
        ax.text(b["x"] + b["w"]/2, b["y"] + b["h"]*0.22, b["sub"], ha="center", va="center", fontsize=8.5, color="#555555")
        
    # Draw arrows between boxes
    for i in range(len(boxes) - 1):
        x_start = boxes[i]["x"] + boxes[i]["w"] + 0.015
        x_end = boxes[i+1]["x"] - 0.015
        y = boxes[i]["y"] + boxes[i]["h"] / 2
        ax.annotate("", xy=(x_end, y), xytext=(x_start, y),
                    arrowprops=dict(arrowstyle="-|>", lw=2, color="#37474F", mutation_scale=14))
        
    # Draw the Causal RoV x SEI Gate below UniTok & Götz BPE
    gate_rect = patches.FancyBboxPatch((0.40, 0.12), 0.24, 0.25, boxstyle="round,pad=0.02,rounding_size=0.03",
                                       facecolor="#FFFDE7", edgecolor="#FBC02D", linewidth=2)
    ax.add_patch(gate_rect)
    ax.text(0.52, 0.27, "Causal RoV × SEI Gate (PATK)", ha="center", va="center", fontweight="bold", fontsize=9.5, color="#F57F17")
    ax.text(0.52, 0.18, "Modulates motif length in volatile hours\n(Strictly trailing lookback)", ha="center", va="center", fontsize=8, color="#5D4037")
    
    # Arrow from Gate to Götz BPE
    ax.annotate("", xy=(0.60, 0.54), xytext=(0.54, 0.38),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, linestyle="--", color="#F57F17", mutation_scale=12))
    
    # Leak switches (Red callouts)
    leak_switches = [
        {"x": 0.255, "y": 0.92, "text": "Switch 1: Window Norm Leak\n(uses future test horizon std)"},
        {"x": 0.61, "y": 0.92, "text": "Switch 2: Merge Table Leak\n(estimated across test set)"},
        {"x": 0.93, "y": 0.92, "text": "Switch 3: Decode Table Leak\n(fit on test realizations)"},
        {"x": 0.43, "y": 0.44, "text": "Switch 4: Boundary Leak\n(global unconstrained BPE)"},
        {"x": 0.52, "y": 0.02, "text": "Switch 5: Non-Causal FFT Leak\n(centered window into future)"}
    ]
    
    for lk in leak_switches:
        ax.text(lk["x"], lk["y"], lk["text"], ha="center", va="center", fontsize=8, fontweight="bold", color="#B71C1C",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FFEBEE", edgecolor="#E53935", lw=1.2))
        
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 1.05)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved pipeline diagram to {save_path}")

def plot_boundary_mismatch_figure(save_path: str = "results/figures/boundary_mismatch_and_compression.png"):
    """
    Plots the boundary mismatch comparison and BPE compression curve.
    """
    with open("results/boundary_mismatch_results.json", "r") as f:
        data = json.load(f)
        
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.2))
    
    # Left: Mismatch & Straddling Rate Bar Chart
    labels = ["UniTok\nHard Boundaries\n(Causal Ref)", "Götz BPE\nUnconstrained\n(Leak Channel 4)"]
    mismatch_rates = [
        data["mismatch_evaluation"]["mismatch_rate_with_boundaries"] * 100.0,
        data["mismatch_evaluation"]["mismatch_rate_without_boundaries"] * 100.0
    ]
    colors = ["#2E7D32", "#C62828"]
    bars = ax1.bar(labels, mismatch_rates, color=colors, width=0.48, edgecolor="black", linewidth=1.2)
    ax1.set_ylabel("Boundary Mismatch Rate (%)", fontweight="bold")
    ax1.set_ylim(0, 100)
    ax1.set_title("Look-Ahead Token Boundary Mismatch\n(Evaluated over 300 Forecast Origins)", fontweight="bold", fontsize=11)
    
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.1f}%", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=11, color="black")
                     
    ax1.axhline(0, color="gray", lw=1)
    
    # Right: Compression Curve
    comp_curve = data["compression_curve"]
    merges = [c["num_merges"] for c in comp_curve]
    ratios = [c["compression_ratio"] for c in comp_curve]
    
    ax2.plot(merges, ratios, marker="o", color="#1565C0", lw=2.2, markersize=7, label="Hour-Constrained BPE")
    ax2.set_xlabel("Number of BPE Merges", fontweight="bold")
    ax2.set_ylabel("Compression Ratio (x)", fontweight="bold")
    ax2.set_title("Sequence Compression vs BPE Vocabulary\n(100h BTCUSDT Benchmark)", fontweight="bold", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.6)
    
    for x, y in zip(merges, ratios):
        ax2.annotate(f"{y:.2f}x", xy=(x, y), xytext=(0, 6), textcoords="offset points",
                     ha="center", fontsize=9, fontweight="bold", color="#0D47A1")
                     
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved boundary mismatch figure to {save_path}")

def plot_oracle_floor_figure(save_path: str = "results/figures/negative_control_oracle_floor.png"):
    """
    Plots the negative control benchmark: Baseline vs Oracle QLIKE Floor.
    """
    with open("results/negative_control_results.json", "r") as f:
        data = json.load(f)
        
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    
    models = ["Oracle Floor\n(Theoretical Min)", "GARCH(1,1)\n(Causal)", "HAR-RV\n(Causal)", "Gross Leak\n(Positive Control)"]
    qlikes = [
        data["oracle_floor_qlike"],
        data["garch_qlike"],
        data["har_qlike"],
        data["positive_control_gross_leak_qlike"]
    ]
    colors = ["#1565C0", "#2E7D32", "#F57F17", "#C62828"]
    
    bars = ax.bar(models, qlikes, color=colors, width=0.52, edgecolor="black", linewidth=1.2)
    ax.axhline(data["oracle_floor_qlike"], color="#1565C0", linestyle="--", lw=1.8, label=f"Oracle Floor ({data['oracle_floor_qlike']:.4f})")
    
    ax.set_ylabel("QLIKE Loss (Lower is Better)", fontweight="bold")
    ax.set_title("Negative Control: GARCH(1,1) Synthetic Benchmark\n(No Leak-Free Model Can Beat the Oracle Floor)", fontweight="bold", fontsize=11)
    ax.legend(loc="upper right", frameon=True)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=9.5)
                     
    ax.set_ylim(0, max(qlikes) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved oracle floor figure to {save_path}")

def plot_leakage_inflation_figure(save_path: str = "results/figures/leakage_inflation_comparison.png"):
    """
    Plots the Look-Ahead Inflation Summary across Channels.
    """
    json_path = "results/leakage_inflation_table.json"
    if not os.path.exists(json_path):
        print(f"Waiting for {json_path}...")
        return
        
    with open(json_path, "r") as f:
        data = json.load(f)
        
    ch_data = data["channel_results"]
    names = [
        "HAR-RV Baseline",
        "Causal Reference",
        "Ch 1: Norm Leak",
        "Ch 2: Merge Leak",
        "Ch 3: Decode Leak",
        "Ch 4: Boundary Leak",
        "Ch 5: FFT Leak",
        "Positive Control"
    ]
    
    qlikes = [
        data["har_rv_qlike"],
        ch_data["reference"]["QLIKE"],
        ch_data["channel_1_norm_leak"]["QLIKE"],
        ch_data["channel_2_merge_leak"]["QLIKE"],
        ch_data["channel_3_decode_leak"]["QLIKE"],
        ch_data["channel_4_boundary_leak"]["QLIKE"],
        ch_data["channel_5_fft_leak"]["QLIKE"],
        ch_data["positive_control"]["QLIKE"]
    ]
    
    colors = ["#78909C", "#2E7D32", "#E53935", "#FB8C00", "#8E24AA", "#D81B60", "#F4511E", "#B71C1C"]
    
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.bar(names, qlikes, color=colors, width=0.55, edgecolor="black", linewidth=1.1)
    
    # Reference baseline line
    ref_q = ch_data["reference"]["QLIKE"]
    ax.axhline(ref_q, color="#2E7D32", linestyle=":", lw=1.8, label=f"Causal Reference QLIKE ({ref_q:.4f})")
    
    ax.set_ylabel("Out-of-Sample QLIKE (Fold 1 Test)", fontweight="bold")
    ax.set_title("Per-Channel Look-Ahead Bias Audit on BTCUSDT Realized Volatility\n(Preliminary Fold 1 Single-Seed Results)", fontweight="bold", fontsize=11.5)
    plt.xticks(rotation=25, ha="right", fontsize=9.5, fontweight="bold")
    ax.legend(loc="upper right", frameon=True)
    
    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.4f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                     fontweight="bold", fontsize=9)
                     
    ax.set_ylim(0, max(qlikes) * 1.15)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved leakage inflation figure to {save_path}")

if __name__ == "__main__":
    plot_pipeline_diagram()
    plot_boundary_mismatch_figure()
    plot_oracle_floor_figure()
    plot_leakage_inflation_figure()

