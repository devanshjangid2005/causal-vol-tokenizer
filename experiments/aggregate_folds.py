"""
experiments/aggregate_folds.py
Builds the per-channel inflation table across walk-forward folds (the main deliverable).

Reads the per-origin forecasts written by run_leak_experiments.py
(results/losses/fold_<k>_h<H>.npz). The six test periods are disjoint and consecutive
(2022-Q1 ... 2024), so the pooled analysis concatenates them in time order:
  * pooled QLIKE per arm with moving-block bootstrap CIs (origin-weighted across folds);
  * each leak vs its matched control: inflation %, Diebold-Mariano (HAC) with Holm across
    the leak family, Giacomini-White, TOST (+-5% of the control's QLIKE);
  * partial-token bias (Ch4 control vs reference);
  * reference and controls vs HAR-RV and GARCH(1,1);
  * per-fold inflation and DM p, and how many folds agree in sign / are significant;
  * seed robustness from any fold run with several seeds (Fold 1 by default).

Usage:
    python experiments/aggregate_folds.py                 # H=12, seed 0, all folds found
    python experiments/aggregate_folds.py --horizon 288
Outputs: results/inflation_table_h<H>.json and results/inflation_table_h<H>.md
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import glob
import json
import re
from typing import Dict, List

import numpy as np

from evaluation.metrics import qlike_loss
from evaluation.hypothesis_tests import holm_bonferroni_correction, diebold_mariano_test
from experiments.run_leak_experiments import ARMS, LEAK_ARMS, compare, summarize
from tokenization.leak_tokenizer import ORIGINAL_ARMS
from experiments.forecast_store import load_fold_forecasts

HEADS = ("direct", "generative")


def load_npz(path: str) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def arm_forecast(d: dict, head: str, arm: str, seed: str):
    if seed == "mean":
        keys = sorted(k for k in d if k.startswith(f"fc/{head}/{arm}/seed"))
        return np.mean([d[k] for k in keys], axis=0) if keys else None
    return d.get(f"fc/{head}/{arm}/seed{seed}")


def leak_family(head: str, arms: List[str]) -> List[str]:
    fam = [a for a in LEAK_ARMS if a in arms and ARMS[a]["control"] in arms]
    if head == "direct":  # Ch3 has no path into the direct head (identical forecasts)
        fam = [a for a in fam if a != "ch3_decode_table_leak"]
    return fam


def analyse(losses: Dict[str, np.ndarray], base_losses: Dict[str, np.ndarray], head: str, margin: float) -> dict:
    arms = list(losses)
    out = {"arms": {a: summarize(losses[a]) for a in arms},
           "baselines": {b: summarize(v) for b, v in base_losses.items()},
           "leak_tests": {}, "partial_token_bias": None, "vs_baselines": {}}
    fam = leak_family(head, arms)
    for a in fam:
        out["leak_tests"][a] = {"control": ARMS[a]["control"], **compare(losses[ARMS[a]["control"]], losses[a], margin)}
    for a, (adj, sig) in zip(fam, holm_bonferroni_correction([out["leak_tests"][a]["dm_p"] for a in fam]) if fam else []):
        out["leak_tests"][a]["dm_p_holm"] = adj
        out["leak_tests"][a]["significant_holm"] = sig
    if "reference" in losses and "ch4_control_context_only" in losses:
        out["partial_token_bias"] = compare(losses["reference"], losses["ch4_control_context_only"], margin)
    for a in ("reference", "ch1_control_window", "ch4_control_context_only"):
        if a in losses:
            out["vs_baselines"][a] = {b: compare(v, losses[a], margin) for b, v in base_losses.items()}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--seed", default="0", help="seed to pool across folds, or 'mean'")
    ap.add_argument("--tost-margin", type=float, default=0.05)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()

    folds = load_fold_forecasts(args.results, args.horizon)  # merges main, extra-seed and extension files
    if not folds:
        raise SystemExit("No per-origin forecast files found; run run_leak_experiments.py first.")
    print("Folds found:", sorted(folds))

    report = {"horizon": args.horizon, "seed": args.seed, "folds": sorted(folds), "heads": {}, "per_fold": {},
              "seed_robustness": {}}
    for head in HEADS:
        arms = [a for a in ORIGINAL_ARMS if all(arm_forecast(d, head, a, args.seed) is not None for d in folds.values())]
        if not arms:
            continue
        base_names = [k[5:] for k in folds[min(folds)] if k.startswith("base/")]
        ys, L, B = [], {a: [] for a in arms}, {b: [] for b in base_names}
        per_fold = {}
        for k, d in sorted(folds.items()):
            y = d["y"]
            ys.append(y)
            fl = {a: qlike_loss(y, arm_forecast(d, head, a, args.seed)) for a in arms}
            fb = {b: qlike_loss(y, d[f"base/{b}"]) for b in base_names}
            for a in arms:
                L[a].append(fl[a])
            for b in base_names:
                B[b].append(fb[b])
            pf = {"n_test_origins": int(len(y)), "QLIKE": {a: float(fl[a].mean()) for a in arms},
                  "baselines": {b: float(fb[b].mean()) for b in base_names}, "leaks": {}}
            for a in leak_family(head, arms):
                c = fl[ARMS[a]["control"]]
                s, p = diebold_mariano_test(c, fl[a])
                pf["leaks"][a] = {"inflation_pct": 100 * (c.mean() - fl[a].mean()) / c.mean(), "dm_stat": s, "dm_p": p}
            if "reference" in fl and "ch4_control_context_only" in fl:
                c = fl["reference"]
                s, p = diebold_mariano_test(c, fl["ch4_control_context_only"])
                pf["partial_token_bias"] = {"inflation_pct": 100 * (c.mean() - fl["ch4_control_context_only"].mean()) / c.mean(),
                                            "dm_stat": s, "dm_p": p}
            per_fold[k] = pf
        pooled = analyse({a: np.concatenate(v) for a, v in L.items()},
                         {b: np.concatenate(v) for b, v in B.items()}, head, args.tost_margin)
        for a, lt in pooled["leak_tests"].items():
            infl = [per_fold[k]["leaks"][a]["inflation_pct"] for k in per_fold]
            sig = [per_fold[k]["leaks"][a]["dm_p"] < 0.05 and per_fold[k]["leaks"][a]["dm_stat"] > 0 for k in per_fold]
            lt["folds_with_gain"] = int(sum(i > 0 for i in infl))
            lt["folds_significant_gain"] = int(sum(sig))
            lt["n_folds"] = len(per_fold)
        report["heads"][head] = pooled
        report["per_fold"][head] = per_fold

    # seed robustness: any fold whose file holds several seeds
    for k, d in folds.items():
        seeds = sorted({int(m.group(1)) for key in d for m in [re.search(r"/seed(\d+)$", key)] if m})
        if len(seeds) < 2:
            continue
        rob = {}
        for head in HEADS:
            arms = [a for a in ORIGINAL_ARMS if all(f"fc/{head}/{a}/seed{s}" in d for s in seeds)]
            if not arms:
                continue
            rob[head] = {}
            for a in leak_family(head, arms) + (["ch4_control_context_only"] if "reference" in arms and "ch4_control_context_only" in arms else []):
                ctrl = ARMS[a]["control"] or "reference"
                vals = []
                for s in seeds:
                    c = qlike_loss(d["y"], d[f"fc/{head}/{ctrl}/seed{s}"])
                    x = qlike_loss(d["y"], d[f"fc/{head}/{a}/seed{s}"])
                    vals.append(100 * (c.mean() - x.mean()) / c.mean())
                rob[head][a] = {"control": ctrl, "inflation_pct_per_seed": vals,
                                "mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1))}
        report["seed_robustness"][f"fold_{k}"] = {"seeds": seeds, **rob}

    os.makedirs(args.results, exist_ok=True)
    jpath = os.path.join(args.results, f"inflation_table_h{args.horizon}.json")
    with open(jpath, "w") as f:
        json.dump(report, f, indent=2)
    mpath = os.path.join(args.results, f"inflation_table_h{args.horizon}.md")
    with open(mpath, "w") as f:
        f.write(markdown(report))
    print(markdown(report))
    print(f"Saved {jpath} and {mpath}")


def _p(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def markdown(r: dict) -> str:
    lines = [f"# Look-ahead inflation, H = {r['horizon']} bars, folds {r['folds']}, seed {r['seed']}", ""]
    lines.append("Inflation = % reduction in QLIKE versus the matched control (positive = the leak makes "
                 "the forecast look better). Pooled over all test origins of the listed folds.")
    for head, h in r["heads"].items():
        lines += ["", f"## {head.capitalize()} head", "",
                  "| Arm | Control | Inflation | DM p | Holm p | GW p | TOST equivalent (±5%) | Folds with gain | Folds significant |",
                  "|---|---|---|---|---|---|---|---|---|"]
        for a, lt in h["leak_tests"].items():
            lines.append(f"| {ARMS[a]['label']} | {lt['control']} | {lt['inflation_pct']:+.2f}% | {_p(lt['dm_p'])} | "
                         f"{_p(lt['dm_p_holm'])} | {_p(lt['gw_p'])} | {'yes' if lt['tost']['is_equivalent'] else 'no'} | "
                         f"{lt['folds_with_gain']}/{lt['n_folds']} | {lt['folds_significant_gain']}/{lt['n_folds']} |")
        if h["partial_token_bias"]:
            pt = h["partial_token_bias"]
            lines.append(f"| Partial-token bias (Ch4 control vs reference) | reference | {pt['inflation_pct']:+.2f}% | "
                         f"{_p(pt['dm_p'])} | – | {_p(pt['gw_p'])} | {'yes' if pt['tost']['is_equivalent'] else 'no'} | – | – |")
        lines += ["", "| Model | Pooled QLIKE | 95% CI |", "|---|---|---|"]
        for b, v in h["baselines"].items():
            lines.append(f"| {b} | {v['QLIKE']:.4f} | {v['QLIKE_95_CI'][0]:.4f} – {v['QLIKE_95_CI'][1]:.4f} |")
        for a, v in h["arms"].items():
            lines.append(f"| GPT: {ARMS[a]['label']} | {v['QLIKE']:.4f} | {v['QLIKE_95_CI'][0]:.4f} – {v['QLIKE_95_CI'][1]:.4f} |")
        for a, vb in h["vs_baselines"].items():
            for b, c in vb.items():
                better = "better" if c["inflation_pct"] > 0 else "worse"
                lines.append(f"\n- {ARMS[a]['label']} vs {b}: {c['inflation_pct']:+.2f}% ({better}), DM p {_p(c['dm_p'])}")
    if r["seed_robustness"]:
        lines += ["", "## Seed robustness", ""]
        for fold, sr in r["seed_robustness"].items():
            for head in HEADS:
                if head not in sr:
                    continue
                lines += [f"{fold}, {head} head, seeds {sr['seeds']}:", "",
                          "| Arm | Control | Inflation per seed | Mean ± sd |", "|---|---|---|---|"]
                for a, v in sr[head].items():
                    per = ", ".join(f"{x:+.2f}%" for x in v["inflation_pct_per_seed"])
                    lines.append(f"| {ARMS[a]['label']} | {v['control']} | {per} | {v['mean']:+.2f} ± {v['sd']:.2f}% |")
                lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
