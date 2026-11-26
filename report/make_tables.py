"""
report/make_tables.py
Writes the LaTeX tables of the 5-page report from the saved results (no experiment is re-run).

    python report/make_tables.py

Reads   results/robust_stats_h12.json, results/inflation_table_h288.json,
        results/holdout_h12_leak_audit.json
Checks  every Table 1 cell against results/robust_stats_h12.md and every Table 2 value against
        results/inflation_table_h288.md / the holdout JSON (the script stops on any mismatch)
Writes  report/tables/tab_tradeoff.tex    (main text: boundary leakage vs partial-token bias)
        report/tables/tab_robust.tex      (appendix: full 3-seed robust statistics)
        report/tables/tab_extensions.tex  (appendix: exploratory extensions, folds only)
        (tabular only; the caption and \\label live in main.tex, title above the table)
"""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
OUT = Path(__file__).resolve().parent / "tables"

SHORT = {  # JSON label -> short label used in the paper
    "Ch1 window norm uses horizon": "Ch1 normalization window",
    "Ch2 merge table on train+test": "Ch2 merge table",
    "Ch3 decode tables on train+test": "Ch3 decoding tables",
    "Ch4 tokenize stream, then split": "Ch4 stream-then-split",
    "Ch5 centered gate FFT": "Ch5 centered gate",
    "Positive control": "Positive control",
    "Unconstrained BPE (context only) vs hour-aligned": "Partial-token bias",
    "k = 1 future bar": "Ch1 dose: $k{=}1$ future bar",
    "k = 3 future bars": "Ch1 dose: $k{=}3$ future bars",
    "k = 6 future bars": "Ch1 dose: $k{=}6$ future bars",
    "k = 12 future bars (= Ch1)": "Ch1 dose: $k{=}12$ (= Ch1)",
    "ETH through test period vs ETH to split": "ETH through test vs ETH to split (leak)",
    "ETH to split vs BTC only (more data, no leak)": "ETH to split vs BTC only (more data)",
    "Plain unconstrained vs hour-aligned (healing run)": "Plain unconstrained vs reference",
    "Healed vs plain (same model)": "Healed vs plain (same model)",
    "Healed unconstrained vs hour-aligned": "Healed unconstrained vs reference",
}


def check_md(r: dict, md: str, what: str):
    seeds = f"{len(r['seeds_arm'])}/{len(r['seeds_control'])}"
    want = f"| {r['label']} | {r['head']} | {seeds} | " + " | ".join(md_cells(r)) + " |"
    if want not in md:
        raise SystemExit(f"{what} mismatch with robust_stats_h12.md:\n  {want}")


def load(name: str) -> dict:
    with open(RES / name, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- plain formats (same as the .md)
def pct(x: float) -> str:
    return f"{x:+.2f}%"


def pval(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def md_cells(r: dict) -> list:
    """The 5 numeric cells exactly as experiments/robust_stats.py prints them in the .md."""
    re_ = r["random_effects"]
    return [pct(r["inflation_pct"]),
            f"{pct(r['ci90'][0])} to {pct(r['ci90'][1])}",
            pval(r["dm_p_holm"]),
            f"{re_['estimate']:+.2f}% [{re_['ci95'][0]:+.2f}, {re_['ci95'][1]:+.2f}]",
            f"{100 * re_['I2']:.0f}%"]


# ----------------------------------------------------------------- LaTeX formats
def tnum(x: float, nd: int = 2) -> str:
    """Signed number in math mode so the minus sign is a real minus."""
    return f"${x:+.{nd}f}$"


def tp(p: float) -> str:
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


# ----------------------------------------------------------------- Table 1
def table1() -> str:
    rob = load("robust_stats_h12.json")["comparisons"]
    md = (RES / "robust_stats_h12.md").read_text(encoding="utf-8")
    fams = [("Original leak channels", rob["Original leak channels"]),
            ("Partial-token bias", rob["Partial-token bias"])]

    lines = [r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{llrcrcr}",
             r"\toprule",
             r"Comparison & Head & Inflation (\%) & 90\% CI (\%) & Holm $p$ & RE across folds [95\% CI] (\%) & $I^2$ (\%) \\",
             r"\midrule"]
    n_checked = 0
    for fi, (fam, rows) in enumerate(fams):
        if fi:
            lines.append(r"\midrule")
        prev = None
        for r in rows:
            check_md(r, md, "Robust table")  # same cells as the markdown table written by robust_stats.py
            n_checked += 1
            re_ = r["random_effects"]
            label = SHORT[r["label"]] if r["label"] != prev else ""
            prev = r["label"]
            cells = [label, r["head"], tnum(r["inflation_pct"]),
                     f"[{tnum(r['ci90'][0])}, {tnum(r['ci90'][1])}]",
                     tp(r["dm_p_holm"]),
                     f"{tnum(re_['estimate'])} [{tnum(re_['ci95'][0])}, {tnum(re_['ci95'][1])}]",
                     f"{100 * re_['I2']:.0f}"]
            lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    print(f"Robust table: {n_checked} rows match results/robust_stats_h12.md")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- extensions (appendix)
def table_ext() -> str:
    rob = load("robust_stats_h12.json")["comparisons"]
    md = (RES / "robust_stats_h12.md").read_text(encoding="utf-8")
    fams = ["Extension: Ch1 dose-response (future bars in the window)",
            "Extension: cross-asset leakage (ETH)", "Extension: token healing"]
    lines = [r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{llcrcr}",
             r"\toprule",
             r"Comparison & Head & Seeds & Inflation (\%) & 90\% CI (\%) & Holm $p$ \\",
             r"\midrule"]
    n = 0
    for fi, fam in enumerate(fams):
        if fi:
            lines.append(r"\midrule")
        for r in rob[fam]:
            check_md(r, md, "Extensions table")
            n += 1
            seeds = f"{len(r['seeds_arm'])}/{len(r['seeds_control'])}"
            lines.append(" & ".join([SHORT[r["label"]], r["head"], seeds, tnum(r["inflation_pct"]),
                                     f"[{tnum(r['ci90'][0])}, {tnum(r['ci90'][1])}]", tp(r["dm_p_holm"])]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    print(f"Extensions table: {n} rows match results/robust_stats_h12.md")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------- Table 2
def tradeoff_row(head: dict) -> dict:
    a = head["arms"]
    return {"ref": a["reference"]["QLIKE"], "ctx": a["ch4_control_context_only"]["QLIKE"],
            "leak": a["ch4_boundary_leak"]["QLIKE"],
            "ptb": head["partial_token_bias"]["inflation_pct"], "ptb_p": head["partial_token_bias"]["dm_p"],
            "bl": head["leak_tests"]["ch4_boundary_leak"]["inflation_pct"],
            "bl_p": head["leak_tests"]["ch4_boundary_leak"]["dm_p_holm"]}


def table2() -> str:
    h288 = load("inflation_table_h288.json")
    hold = load("holdout_h12_leak_audit.json")
    assert h288["horizon"] == 288 and h288["folds"] == [1] and str(h288["seed"]) == "0", "H=288 scope changed"
    assert hold["fold"] == "holdout_2025" and hold["seeds"] == [0], "holdout scope changed"
    rows = [(r"$H{=}288$, Fold 1, seed 0, direct", tradeoff_row(h288["heads"]["direct"])),
            (r"2025 holdout, seed 0, direct", tradeoff_row(hold["heads"]["direct"])),
            (r"2025 holdout, seed 0, generative", tradeoff_row(hold["heads"]["generative"]))]

    # cross-checks: H=288 against its markdown table; holdout against the values quoted in
    # FINAL_REPORT.md section 8.5 (which were checked against this same JSON)
    md = (RES / "inflation_table_h288.md").read_text(encoding="utf-8")
    r = rows[0][1]
    for s in (f"| GPT: Causal reference (leak-free) | {r['ref']:.4f} |",
              f"| GPT: Ch4 control: unconstrained BPE, context only | {r['ctx']:.4f} |",
              f"| GPT: Ch4: tokenize stream, then split | {r['leak']:.4f} |",
              f"| Partial-token bias (Ch4 control vs reference) | reference | {r['ptb']:+.2f}% |",
              f"| Ch4: tokenize stream, then split | ch4_control_context_only | {r['bl']:+.2f}% |"):
        if s not in md:
            raise SystemExit(f"Table 2 mismatch with inflation_table_h288.md: {s}")
    expected = {1: ("0.4416", "0.4452", "0.4494", "-0.82", "-0.95"),
                2: ("0.4603", "0.5538", "0.4691", "-20.32", "+15.30")}
    for i, exp in expected.items():
        r = rows[i][1]
        got = (f"{r['ref']:.4f}", f"{r['ctx']:.4f}", f"{r['leak']:.4f}", f"{r['ptb']:+.2f}", f"{r['bl']:+.2f}")
        if got != exp:
            raise SystemExit(f"Table 2 holdout row {i} mismatch: {got} vs {exp}")

    lines = [r"\begin{adjustbox}{max width=\linewidth}",
             r"\begin{tabular}{lrrrrr}",
             r"\toprule",
             r" & \multicolumn{3}{c}{QLIKE (lower is better)} & Partial-token & Boundary \\",
             r"\cmidrule(lr){2-4}",
             r"Setting & Reference & Context-only & Stream-then-split & bias (\%) & leakage (\%) \\",
             r"\midrule"]
    for name, r in rows:
        lines.append(" & ".join([name, f"{r['ref']:.4f}", f"{r['ctx']:.4f}", f"{r['leak']:.4f}",
                                 f"{tnum(r['ptb'])} ({tp(r['ptb_p'])})",
                                 f"{tnum(r['bl'])} ({tp(r['bl_p'])})"]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}"]
    print("Trade-off table: 3 rows match results/inflation_table_h288.md and the holdout JSON")
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for old in ("table1_robust.tex", "table2_tradeoff.tex"):  # names used by an earlier version
        (OUT / old).unlink(missing_ok=True)
    header = "% Generated by report/make_tables.py from results/ -- do not edit by hand; re-run the script.\n"
    outs = {"tab_tradeoff.tex": table2(), "tab_robust.tex": table1(), "tab_extensions.tex": table_ext()}
    for name, body in outs.items():
        (OUT / name).write_text(header + body, encoding="utf-8")
    print("Wrote " + ", ".join(str(OUT / n) for n in outs))


if __name__ == "__main__":
    sys.exit(main())
