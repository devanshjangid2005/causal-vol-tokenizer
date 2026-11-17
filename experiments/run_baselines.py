"""
experiments/run_baselines.py
HAR-RV and GARCH(1,1) on a real fold, scored on the SAME hour-aligned test origins and
the SAME target RV_H[t] = sum_{k<H} r_{t+k}^2 that the GPT arms use.

(The first version scored GARCH against HAR's target array, which started 1,440 bars
later; that is why GARCH looked 21.7% worse than HAR.)

Usage: python experiments/run_baselines.py --fold 1 --horizon 12
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np

from data.folds import load_fold, select_origins
from data.preprocess import future_rv
from models.baselines import HARRVModel, GARCH11Model
from evaluation.metrics import compute_all_metrics, qlike_loss
from evaluation.hypothesis_tests import block_bootstrap_ci, diebold_mariano_test, giacomini_white_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=12)
    ap.add_argument("--context", type=int, default=144)
    args = ap.parse_args()

    df, split = load_fold(args.fold)
    r = df["return"].values
    y = future_rv(r, args.horizon)
    tr = select_origins(df, 0, split, args.horizon, args.context, min_history=2016)
    te = select_origins(df, split, len(df), args.horizon, args.context, min_history=2016)
    print(f"train origins {len(tr)} | test origins {len(te)} "
          f"({df['open_time'].iloc[te[0]]} -> {df['open_time'].iloc[te[-1]]})")

    har = HARRVModel().fit(r, tr, args.horizon)
    garch = GARCH11Model().fit(r[:split])
    print(f"GARCH params: omega={garch.omega:.3e} alpha={garch.alpha:.4f} beta={garch.beta:.4f}")
    preds = {"har_rv": har.predict(r, te, args.horizon), "garch_11": garch.predict(r, te, args.horizon)}

    out = {"fold": args.fold, "horizon": args.horizon, "n_test_origins": int(len(te))}
    losses = {}
    for name, p in preds.items():
        m = compute_all_metrics(y[te], p)
        losses[name] = qlike_loss(y[te], p)
        _, lo, hi = block_bootstrap_ci(losses[name])
        m["QLIKE_95_CI"] = [lo, hi]
        out[name] = m
        print(f"[{name:8s}] QLIKE {m['QLIKE']:.4f}  95% CI [{lo:.4f}, {hi:.4f}]")

    s, p = diebold_mariano_test(losses["har_rv"], losses["garch_11"])
    g, gp = giacomini_white_test(losses["har_rv"], losses["garch_11"])
    out["dm_har_minus_garch"] = {"stat": s, "p_value": p, "note": "positive stat = GARCH has lower loss"}
    out["gw_har_vs_garch"] = {"stat": g, "p_value": gp}
    print(f"DM (HAR - GARCH): stat {s:.3f}, p {p:.4f} | GW: stat {g:.3f}, p {gp:.4f}")

    os.makedirs("results", exist_ok=True)
    path = f"results/fold_{args.fold}_baselines.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
