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

def _load(path):
    if not os.path.exists(path):
        print(f"Skipping: {path} not found (run the matching experiment first)")
        return None
    with open(path) as f:
        return json.load(f)


def _bar_labels(ax, bars, fmt):
    for bar in bars:
        h = bar.get_height()
        ax.annotate(fmt.format(h), xy=(bar.get_x() + bar.get_width() / 2, h), xytext=(0, 3),
                    textcoords="offset points", ha="center", va="bottom", fontsize=9)


def plot_boundary_mismatch_figure(save_path: str = "results/figures/boundary_mismatch_and_compression.png"):
    """Slice-vs-prefix mismatch (hour-aligned and off-hour origins), future-perturbation
    sensitivity, and compression vs merge count."""
    data = _load("results/boundary_mismatch_results.json")
    if data is None:
        return
    names = list(data["variants"])
    short = ["Hour bounds\nno gate", "Hour bounds\ncausal gate\n(reference)",
             "Hour bounds\ncentered gate\n(Ch5)", "Unconstrained\nBPE (Ch4)"][:len(names)]
    aligned = [100 * data["variants"][n]["slice_vs_prefix_hour_aligned"]["mismatch_rate"] for n in names]
    off = [100 * data["variants"][n]["slice_vs_prefix_off_hour"]["mismatch_rate"] for n in names]
    pert = [100 * data["variants"][n]["future_perturbation_hour_aligned"]["prefix_changed_rate"] for n in names]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    x = np.arange(len(names))
    w = 0.26
    b1 = ax1.bar(x - w, aligned, w, label="Slice vs prefix, hour-aligned origin", color="#2E7D32")
    b2 = ax1.bar(x, off, w, label="Slice vs prefix, origin 5 bars past hour", color="#F9A825")
    b3 = ax1.bar(x + w, pert, w, label="Future perturbation changes prefix", color="#C62828")
    for b in (b1, b2, b3):
        _bar_labels(ax1, b, "{:.0f}%")
    ax1.set_xticks(x)
    ax1.set_xticklabels(short, fontsize=9)
    ax1.set_ylabel("Origins affected (%)")
    ax1.set_ylim(0, 100)
    ax1.set_title(f"Token-level causality tests ({data['n_origins']} test origins)", fontweight="bold", fontsize=11)
    ax1.legend(fontsize=8, loc="upper left")

    curve = data["compression_curve"]
    ks = [c["num_merges"] for c in curve]
    rs = [c["compression_ratio"] for c in curve]
    ax2.plot(ks, rs, marker="o", color="#1565C0", lw=2)
    for k, r in zip(ks, rs):
        ax2.annotate(f"{r:.2f}x", (k, r), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=9)
    ax2.set_xlabel("BPE merges")
    ax2.set_ylabel("Bars per token")
    ax2.set_title("Compression on normalized returns\n(hour-bounded, first 30 test days)", fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_oracle_floor_figure(save_path: str = "results/figures/negative_control_oracle_floor.png"):
    """Every arm's QLIKE on simulated GARCH data against the analytic oracle floor."""
    data = _load("results/negative_control_results.json")
    if data is None or "heads" not in data:
        return
    head = data["heads"]["direct"]
    floor = data["baselines"]["oracle"]["QLIKE"]
    arms = list(head["arms"])
    q = [head["arms"][a]["QLIKE"] for a in arms]
    viol = [data["beats_oracle_floor"]["direct"][a]["violates_floor"] for a in arms]
    fig, ax = plt.subplots(figsize=(10, 4.2))
    bars = ax.bar(arms, q, color=["#C62828" if v else "#78909C" for v in viol], edgecolor="black", lw=0.8)
    ax.axhline(floor, color="#1565C0", ls="--", lw=1.8, label=f"Oracle floor {floor:.4f}")
    _bar_labels(ax, bars, "{:.3f}")
    ax.set_ylabel("QLIKE (direct head)")
    ax.set_title("Negative control on simulated GARCH(1,1): red = significantly below the oracle floor",
                 fontweight="bold", fontsize=11)
    plt.xticks(rotation=25, ha="right", fontsize=9)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


def plot_leakage_inflation_figure(save_path: str = "results/figures/leakage_inflation_comparison.png"):
    """Inflation of each leak arm relative to its matched control, with Holm significance."""
    data = _load("results/leakage_inflation_table.json")
    if data is None or "heads" not in data:
        return
    heads = [h for h in ("direct", "generative") if h in data["heads"]]
    fig, axes = plt.subplots(1, len(heads), figsize=(6 * len(heads), 4.2), squeeze=False)
    for ax, head in zip(axes[0], heads):
        lt = data["heads"][head]["leak_tests"]
        arms = list(lt)
        inf = [lt[a]["inflation_pct"] for a in arms]
        sig = [lt[a].get("significant_holm", False) for a in arms]
        bars = ax.bar([a.replace("_", "\n", 1) for a in arms], inf,
                      color=["#C62828" if s else "#B0BEC5" for s in sig], edgecolor="black", lw=0.8)
        _bar_labels(ax, bars, "{:+.1f}%")
        ax.axhline(0, color="black", lw=0.8)
        ax.set_ylabel("QLIKE reduction vs matched control (%)")
        ax.set_title(f"{head} head (red = significant after Holm)", fontweight="bold", fontsize=11)
        ax.tick_params(axis="x", labelsize=8)
    fig.suptitle(f"Fold {data.get('fold')}: look-ahead inflation per channel, seeds {data.get('seeds')}",
                 fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"Saved {save_path}")


if __name__ == "__main__":
    plot_pipeline_diagram()
    plot_boundary_mismatch_figure()
    plot_oracle_floor_figure()
    plot_leakage_inflation_figure()

