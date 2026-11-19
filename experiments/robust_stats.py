"""
experiments/robust_stats.py
Robustness statistics on top of the per-origin forecasts (no retraining needed).

For every comparison "arm vs its matched control":
  * pooled inflation %, with seed-averaged per-origin losses when several training seeds exist
    (the estimand is the expected inflation over training randomness);
  * 90% and 95% moving-block bootstrap CIs. The 90% CI upper bound is the equivalence bound:
    "this leak inflates accuracy by at most X%";
  * Diebold-Mariano (HAC) p-value with Holm correction within each family;
  * DerSimonian-Laird random-effects meta-analysis across folds (each fold weighted by its
    precision plus between-fold variance tau^2), with I^2 heterogeneity, so the result is not
    dominated by the fold with the most test origins.
Plus a Model Confidence Set (Hansen, Lunde & Nason 2011, T_max statistic, block bootstrap)
over the leak-free forecasters, for the pooled folds and for the (already scored) 2025 holdout.

Usage: python experiments/robust_stats.py [--horizon 12]
Outputs: results/robust_stats_h<H>.json and results/robust_stats_h<H>.md
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
from typing import Dict, List, Tuple

import numpy as np
from scipy import stats

from evaluation.metrics import qlike_loss
from evaluation.hypothesis_tests import diebold_mariano_test, newey_west_hac, holm_bonferroni_correction
from experiments.forecast_store import load_fold_forecasts, load_single, common_seeds, seed_avg_loss

# (label, arm, control, heads)
FAMILIES: Dict[str, List[Tuple[str, str, str, Tuple[str, ...]]]] = {
    "Original leak channels": [
        ("Ch1 window norm uses horizon", "ch1_window_norm_leak", "ch1_control_window", ("direct", "generative")),
        ("Ch2 merge table on train+test", "ch2_merge_table_leak", "reference", ("direct", "generative")),
        ("Ch3 decode tables on train+test", "ch3_decode_table_leak", "reference", ("generative",)),
        ("Ch4 tokenize stream, then split", "ch4_boundary_leak", "ch4_control_context_only", ("direct", "generative")),
        ("Ch5 centered gate FFT", "ch5_noncausal_fft_leak", "reference", ("direct", "generative")),
        ("Positive control", "positive_control", "reference", ("direct", "generative")),
    ],
    "Partial-token bias": [
        ("Unconstrained BPE (context only) vs hour-aligned", "ch4_control_context_only", "reference", ("direct", "generative")),
    ],
    "Extension: cross-asset leakage (ETH)": [
        ("ETH through test period vs ETH to split", "xa_leak", "xa_control", ("direct", "generative")),
        ("ETH to split vs BTC only (more data, no leak)", "xa_control", "reference", ("direct", "generative")),
    ],
    "Extension: token healing": [
        ("Plain unconstrained vs hour-aligned (healing run)", "ext_ch4_control", "reference", ("generative",)),
        ("Healed vs plain (same model)", "ext_ch4_healed", "ext_ch4_control", ("generative",)),
        ("Healed unconstrained vs hour-aligned", "ext_ch4_healed", "reference", ("generative",)),
    ],
    "Extension: Ch1 dose-response (future bars in the window)": [
        ("k = 1 future bar", "ch1_leak_k1", "ch1_control_window", ("direct",)),
        ("k = 3 future bars", "ch1_leak_k3", "ch1_control_window", ("direct",)),
        ("k = 6 future bars", "ch1_leak_k6", "ch1_control_window", ("direct",)),
        ("k = 12 future bars (= Ch1)", "ch1_window_norm_leak", "ch1_control_window", ("direct",)),
    ],
}

MCS_MODELS = [("HAR-RV", "base", "har_rv"), ("GARCH(1,1)", "base", "garch_11"),
              ("GPT reference, direct", "direct", "reference"), ("GPT reference, generative", "generative", "reference"),
              ("GPT unconstrained BPE, direct", "direct", "ch4_control_context_only"),
              ("GPT window-norm control, direct", "direct", "ch1_control_window")]


# --------------------------------------------------------------------------- bootstrap helpers
def block_starts(n: int, block: int, n_boot: int, rng) -> np.ndarray:
    nb = int(np.ceil(n / block))
    return rng.integers(0, n - block + 1, size=(n_boot, nb))


def block_means(x: np.ndarray, block: int) -> np.ndarray:
    cs = np.concatenate([[0.0], np.cumsum(x)])
    return (cs[block:] - cs[:-block]) / block


def inflation_ci(lc: np.ndarray, la: np.ndarray, block: int = 48, n_boot: int = 2000, seed: int = 0):
    """Bootstrap CIs of 100*(mean lc - mean la)/mean lc with jointly resampled blocks."""
    rng = np.random.default_rng(seed)
    st = block_starts(len(lc), block, n_boot, rng)
    mc, ma = block_means(lc, block)[st].mean(axis=1), block_means(la, block)[st].mean(axis=1)
    boot = 100.0 * (mc - ma) / mc
    return {"ci90": [float(np.percentile(boot, 5)), float(np.percentile(boot, 95))],
            "ci95": [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]}


def fold_effect(lc: np.ndarray, la: np.ndarray) -> Tuple[float, float]:
    """Inflation % and its HAC standard error (delta method, denominator treated as fixed)."""
    d = lc - la
    return 100.0 * d.mean() / lc.mean(), 100.0 * np.sqrt(newey_west_hac(d)) / lc.mean()


def random_effects(theta: np.ndarray, se: np.ndarray) -> dict:
    """DerSimonian-Laird random-effects pooling."""
    k = len(theta)
    w = 1.0 / se ** 2
    fe = np.sum(w * theta) / w.sum()
    q = float(np.sum(w * (theta - fe) ** 2))
    c = w.sum() - np.sum(w ** 2) / w.sum()
    tau2 = max(0.0, (q - (k - 1)) / c) if k > 1 else 0.0
    ws = 1.0 / (se ** 2 + tau2)
    re = float(np.sum(ws * theta) / ws.sum())
    se_re = float(np.sqrt(1.0 / ws.sum()))
    i2 = max(0.0, (q - (k - 1)) / q) if q > 0 and k > 1 else 0.0
    p = float(2 * (1 - stats.norm.cdf(abs(re / se_re))))
    return {"estimate": re, "se": se_re, "ci95": [re - 1.96 * se_re, re + 1.96 * se_re], "p": p,
            "tau": float(np.sqrt(tau2)), "I2": i2, "Q": q, "k": k}


# --------------------------------------------------------------------------- MCS
def model_confidence_set(losses: Dict[str, np.ndarray], alpha: float = 0.10, block: int = 48,
                         n_boot: int = 2000, seed: int = 0) -> dict:
    """Hansen, Lunde & Nason (2011) MCS with the T_max statistic and a moving-block bootstrap."""
    names = list(losses)
    L = np.column_stack([losses[n] for n in names])
    n = L.shape[0]
    rng = np.random.default_rng(seed)
    st = block_starts(n, block, n_boot, rng)
    bm = np.column_stack([block_means(L[:, j], block) for j in range(L.shape[1])])
    boot_means = bm[st].mean(axis=1)                      # (n_boot, m)
    mean = L.mean(axis=0)
    alive = list(range(len(names)))
    pvals, order, running = {}, [], 0.0
    while len(alive) > 1:
        sub, bsub = mean[alive], boot_means[:, alive]
        d = sub - sub.mean()                               # loss relative to the average of alive models
        db = bsub - bsub.mean(axis=1, keepdims=True)
        var = np.var(db, axis=0, ddof=1)
        t = d / np.sqrt(var)
        tb = (db - d) / np.sqrt(var)
        p = float(np.mean(tb.max(axis=1) >= t.max()))
        running = max(running, p)
        worst = alive[int(np.argmax(t))]
        pvals[names[worst]] = running
        order.append(names[worst])
        alive.remove(worst)
    pvals[names[alive[0]]] = 1.0
    return {"alpha": alpha, "mcs_pvalues": pvals,
            "in_mcs": [n for n in names if pvals[n] >= alpha], "elimination_order": order,
            "mean_qlike": {n: float(mean[i]) for i, n in enumerate(names)}}


def mcs_losses(sets: List[dict]) -> Dict[str, np.ndarray]:
    """Concatenated (pooled) losses for every MCS model available in all sets, seed-averaged."""
    out = {}
    for label, head, arm in MCS_MODELS:
        parts = []
        for d in sets:
            if head == "base":
                if f"base/{arm}" not in d:
                    break
                parts.append(qlike_loss(d["y"], d[f"base/{arm}"]))
            else:
                seeds = common_seeds(dict(enumerate(sets)), head, [arm])  # same seeds in every fold
                if not seeds:
                    break
                parts.append(seed_avg_loss(d, head, arm, seeds))
        else:
            out[label] = np.concatenate(parts)
    return out


# --------------------------------------------------------------------------- main
def analyse(folds: Dict[int, dict]) -> dict:
    res = {}
    for fam, comps in FAMILIES.items():
        rows = []
        for label, arm, ctrl, heads in comps:
            for head in heads:
                seeds_a = common_seeds(folds, head, [arm])
                seeds_c = common_seeds(folds, head, [ctrl])
                if not seeds_a or not seeds_c:
                    continue
                lcs = {k: seed_avg_loss(d, head, ctrl, seeds_c) for k, d in folds.items()}
                las = {k: seed_avg_loss(d, head, arm, seeds_a) for k, d in folds.items()}
                lc, la = np.concatenate(list(lcs.values())), np.concatenate(list(las.values()))
                s, p = diebold_mariano_test(lc, la)
                eff = [fold_effect(lcs[k], las[k]) for k in folds]
                row = {"label": label, "arm": arm, "control": ctrl, "head": head,
                       "seeds_arm": seeds_a, "seeds_control": seeds_c, "n_origins": int(len(lc)),
                       "inflation_pct": float(100 * (lc.mean() - la.mean()) / lc.mean()),
                       **inflation_ci(lc, la), "dm_stat": float(s), "dm_p": float(p),
                       "per_fold": {int(k): {"inflation_pct": e[0], "se": e[1]} for k, e in zip(folds, eff)}}
                if len(folds) > 1:
                    row["random_effects"] = random_effects(np.array([e[0] for e in eff]), np.array([e[1] for e in eff]))
                rows.append(row)
        if rows:
            for r, (adj, sig) in zip(rows, holm_bonferroni_correction([r["dm_p"] for r in rows])):
                r["dm_p_holm"], r["significant_holm"] = adj, sig
            res[fam] = rows
    return res


def _p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def markdown(rep: dict) -> str:
    L = [f"# Robustness statistics, H = {rep['horizon']}, folds {rep['folds']}", "",
         "Inflation = % QLIKE reduction of the arm versus its matched control (positive = looks better). "
         "Losses are averaged over the training seeds listed. CI = moving-block bootstrap (block 48 origins). "
         "RE = DerSimonian-Laird random-effects pooling of the per-fold estimates. "
         "Holm correction is applied within each family.", ""]
    for fam, rows in rep["comparisons"].items():
        L += [f"## {fam}", "",
              "| Comparison | Head | Seeds | Pooled | 90% CI (equivalence bound) | DM p (Holm) | RE across folds [95% CI] | I² |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            re_ = r.get("random_effects")
            re_s = f"{re_['estimate']:+.2f}% [{re_['ci95'][0]:+.2f}, {re_['ci95'][1]:+.2f}]" if re_ else "–"
            i2 = f"{100 * re_['I2']:.0f}%" if re_ else "–"
            L.append(f"| {r['label']} | {r['head']} | {len(r['seeds_arm'])}/{len(r['seeds_control'])} | "
                     f"{r['inflation_pct']:+.2f}% | {r['ci90'][0]:+.2f}% to {r['ci90'][1]:+.2f}% | "
                     f"{_p(r['dm_p_holm'])} | {re_s} | {i2} |")
        L.append("")
    for name, m in rep["mcs"].items():
        L += [f"## Model Confidence Set ({name}, α = {m['alpha']})", "",
              "| Model | Mean QLIKE | MCS p-value | In MCS |", "|---|---|---|---|"]
        for mod, q in sorted(m["mean_qlike"].items(), key=lambda kv: kv[1]):
            L.append(f"| {mod} | {q:.4f} | {m['mcs_pvalues'][mod]:.3f} | {'yes' if mod in m['in_mcs'] else 'no'} |")
        L.append("")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--results", default="results")
    args = ap.parse_args()
    folds = load_fold_forecasts(args.results, args.horizon)
    if not folds:
        raise SystemExit("No forecast files found in results/losses/")
    rep = {"horizon": args.horizon, "folds": sorted(folds), "comparisons": analyse(folds), "mcs": {}}
    pooled = mcs_losses(list(folds.values()))
    if len(pooled) > 1:
        rep["mcs"]["pooled folds"] = model_confidence_set(pooled)
    hold = load_single(args.results, f"holdout_h{args.horizon}.npz")
    if hold is not None:
        hl = mcs_losses([hold])
        if len(hl) > 1:
            rep["mcs"]["2025 holdout (already-scored forecasts)"] = model_confidence_set(hl)
    jp = os.path.join(args.results, f"robust_stats_h{args.horizon}.json")
    with open(jp, "w") as f:
        json.dump(rep, f, indent=2)
    md = markdown(rep)
    with open(os.path.join(args.results, f"robust_stats_h{args.horizon}.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"Saved {jp}")


if __name__ == "__main__":
    main()
