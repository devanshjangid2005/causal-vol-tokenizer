"""
report/make_figures.py
Paper-width figures for the 5-page report, drawn ONLY from saved results (nothing is re-run).

    python report/make_figures.py

Reads   results/robust_stats_h12.json       (3-seed pooled inflation, 90% bootstrap CI, Holm p)
        results/holdout_h12_leak_audit.json  (2025 holdout, seed 0, scored once)
Writes  report/figures/fig_inflation_folds.{pdf,png}    main-text figure (Folds 1-6, 3 seeds)
        report/figures/fig_inflation_holdout.{pdf,png}  optional / appendix (2025 holdout, seed 0)

Sized for the NeurIPS text width (5.5 in) so text prints at >= 8 pt with width=\\linewidth.
PDF keeps TrueType fonts embedded (pdf.fonttype 42), as the template requires.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "figures"

# validated two-slot categorical palette (dataviz reference slots 1-2; passes all checks on white)
DIRECT, GEN = "#2a78d6", "#eb6834"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
WIDTH_IN = 5.5

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 8, "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "xtick.color": INK2, "ytick.color": INK, "xtick.major.width": 0.6, "ytick.major.size": 0,
    "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 300,
})

LARGE = [("ch1_window_norm_leak", "Ch1 normalization window"), ("positive_control", "Positive control")]
SMALL = [("ch2_merge_table_leak", "Ch2 merge table"), ("ch3_decode_table_leak", "Ch3 decoding tables"),
         ("ch4_boundary_leak", "Ch4 stream-then-split"), ("ch5_noncausal_fft_leak", "Ch5 centered gate"),
         ("ch4_control_context_only", "Partial-token bias")]
HEADS = [("direct", DIRECT, "o", +0.16), ("generative", GEN, "s", -0.16)]


def load(name):
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


def folds_data():
    """{arm: {head: (inflation, ci_lo, ci_hi, holm_p)}} from the 3-seed robust analysis."""
    rob = load("robust_stats_h12.json")["comparisons"]
    out = {}
    for fam in ("Original leak channels", "Partial-token bias"):
        for r in rob[fam]:
            out.setdefault(r["arm"], {})[r["head"]] = (r["inflation_pct"], r["ci90"][0], r["ci90"][1], r["dm_p_holm"])
    return out


def holdout_data():
    """{arm: {head: (inflation, None, None, p)}}; leaks use Holm p, partial-token bias its own DM p."""
    hold = load("holdout_h12_leak_audit.json")
    assert hold["fold"] == "holdout_2025" and hold["seeds"] == [0]
    out = {}
    for head, h in hold["heads"].items():
        for arm, lt in h["leak_tests"].items():
            out.setdefault(arm, {})[head] = (lt["inflation_pct"], None, None, lt["dm_p_holm"])
        pt = h["partial_token_bias"]
        out.setdefault("ch4_control_context_only", {})[head] = (pt["inflation_pct"], None, None, pt["dm_p"])
    return out


def panel(ax, rows, data, xlim, label_values=False, band=None, spread=1.0, xticks=None):
    ax.set_xlim(*xlim)
    if xticks is not None:
        ax.set_xticks(xticks)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows][::-1])
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    if band:
        ax.axvspan(-band, band, color=BAND, zorder=0, linewidth=0)
    if xlim[0] < 0 < xlim[1]:
        ax.axvline(0, color=AXIS, linewidth=0.8, zorder=1)
    for i, (arm, _) in enumerate(rows[::-1]):
        for head, color, marker, dy in HEADS:
            v = data.get(arm, {}).get(head)
            if v is None:
                continue
            x, lo, hi, p = v
            y = i + dy * spread
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=color, linewidth=1.2, solid_capstyle="round", zorder=2)
            sig = p < 0.05
            ax.plot(x, y, marker=marker, markersize=5, color=color, markeredgecolor=color, markeredgewidth=1.0,
                    markerfacecolor=color if sig else "white", linestyle="none", zorder=3)
            if label_values:
                xr = hi if hi is not None else x
                ax.annotate(f"{x:+.2f}%", (xr, y), xytext=(4, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=7.5, color=INK2)


def legend(fig, note):
    handles = [Line2D([], [], marker="o", color=DIRECT, linestyle="none", markersize=5, label="Direct head"),
               Line2D([], [], marker="s", color=GEN, linestyle="none", markersize=5, label="Generative head"),
               Line2D([], [], marker="o", color=MUTED, linestyle="none", markersize=5, label=note),
               Line2D([], [], marker="o", color=MUTED, markerfacecolor="white", linestyle="none", markersize=5,
                      label="not significant")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, handletextpad=0.3,
               columnspacing=1.2, bbox_to_anchor=(0.5, 1.0))


def figure(data, name, large_xlim, small_xlim, xlabel, sig_note, band=None):
    fig, (top, bot) = plt.subplots(2, 1, figsize=(WIDTH_IN, 2.65), gridspec_kw={"height_ratios": [2.4, 5]})
    panel(top, LARGE, data, large_xlim, label_values=True, spread=1.45, xticks=range(0, 101, 20))
    panel(bot, SMALL, data, small_xlim, band=band)
    top.set_title("Large effects", loc="left", fontsize=8, color=INK2, pad=3)
    bot.set_title("Other channels (zoomed axis)" + ("; shaded: $\\pm$5% equivalence margin" if band else ""),
                  loc="left", fontsize=8, color=INK2, pad=3)
    bot.set_xlabel(xlabel, color=INK2)
    legend(fig, sig_note)
    fig.tight_layout(rect=(0, 0, 1, 0.93), h_pad=0.6)
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT / name}.pdf and .png")


NEG_ROWS = [("reference", "Reference (causal)"), ("ch1_control_window", "Ch1 control (context std)"),
            ("ch1_window_norm_leak", "Ch1 normalization window"), ("ch2_merge_table_leak", "Ch2 merge table"),
            ("ch3_decode_table_leak", "Ch3 decoding tables"), ("ch4_control_context_only", "Ch4 control (context only)"),
            ("ch4_boundary_leak", "Ch4 stream-then-split"), ("ch5_noncausal_fft_leak", "Ch5 centered gate"),
            ("positive_control", "Positive control")]


def negative_control():
    """QLIKE of every arm on simulated GARCH(1,1) vs the analytic oracle floor (seed 0)."""
    d = load("negative_control_results.json")
    assert d["fold"] == "synthetic_garch" and d["seeds"] == [0]
    floor = d["baselines"]["oracle"]["QLIKE"]
    fig, ax = plt.subplots(figsize=(WIDTH_IN, 2.2))
    rows = NEG_ROWS[::-1]
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([lab for _, lab in rows])
    ax.set_xlim(0, 0.17)
    ax.grid(axis="x", color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.axvline(floor, color=INK2, linewidth=1.0, zorder=1)
    ax.annotate(f"oracle floor {floor:.4f}", (floor, len(rows) - 0.45), xytext=(3, 0), textcoords="offset points",
                ha="left", va="bottom", fontsize=7.5, color=INK2, annotation_clip=False)
    for i, (arm, _) in enumerate(rows):
        for head, color, marker, dy in HEADS:
            v = d["beats_oracle_floor"][head][arm]
            below = v["violates_floor"]  # one-sided DM p < 0.05 that the arm beats the oracle
            ax.plot(v["QLIKE"], i + dy, marker=marker, markersize=5, color=color, markeredgecolor=color,
                    markeredgewidth=1.0, markerfacecolor=color if below else "white", linestyle="none", zorder=3)
            if below:  # label on the left unless the point sits at the axis origin
                left = v["QLIKE"] > 0.02
                ax.annotate(f"{v['QLIKE']:.4f}", (v["QLIKE"], i + dy), xytext=(-6 if left else 5, 0),
                            textcoords="offset points", va="center", ha="right" if left else "left",
                            fontsize=7.5, color=INK2)
    ax.set_xlabel("QLIKE on simulated GARCH(1,1) test origins (lower is better)", color=INK2)
    legend(fig, "significantly below floor")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_negative_control.{ext}", bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Wrote {OUT / 'fig_negative_control'}.pdf and .png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    negative_control()
    figure(folds_data(), "fig_inflation_folds", (0, 120), (-6, 6),
           "Inflation vs matched control (%), 90% block-bootstrap CI", "Holm p < 0.05", band=5)
    figure(holdout_data(), "fig_inflation_holdout", (0, 120), (-25, 20),
           "Inflation vs matched control (%), 2025 holdout, seed 0", "p < 0.05")


if __name__ == "__main__":
    main()
