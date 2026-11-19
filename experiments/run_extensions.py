"""
experiments/run_extensions.py
Exploratory extension study (folds 1-6 only; the 2025 holdout was scored once and is not touched).
Skips any step whose output already exists, so it can be re-run after an interruption.

    python experiments/run_extensions.py                       # everything
    python experiments/run_extensions.py --parts dose heal     # a subset
    python experiments/run_extensions.py --epochs 1 --smoke    # quick code check (writes to results_smoke/)

Parts, in run order (new findings first, extra seeds last):
  dose    Ch1 off-by-k dose-response: normalization window reaches k = 1, 3, 6 future bars
          (direct head).                                    -> results/losses/fold_<k>_h12_dose.npz
  heal    token healing for the generative head: retrains the unconstrained context-only arm and
          forecasts with and without healing from the same model.  -> fold_<k>_h12_heal.npz
  xasset  cross-asset leakage: BTC + ETH up to the split vs BTC + ETH through the test period
          (seeds 0 1).                                      -> fold_<k>_h12_xasset.npz
  seeds   seeds 1 and 2 of every original arm on folds 2-6 (Fold 1 already has 3 seeds)
                                                            -> fold_<k>_h12_seeds12.npz
Then: experiments/robust_stats.py, experiments/aggregate_folds.py, figures.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import shutil

from experiments.run_all import step, PY

ALL_PARTS = ["dose", "heal", "xasset", "seeds"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=ALL_PARTS, choices=ALL_PARTS)
    ap.add_argument("--folds", nargs="+", type=int, default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--epochs", default="5")
    ap.add_argument("--xa-seeds", nargs="+", default=["0", "1"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--smoke", action="store_true", help="tiny subsample, separate output folder")
    args = ap.parse_args()
    f, ep = args.force, ["--epochs", args.epochs]
    if any(k not in range(1, 7) for k in args.folds):
        raise SystemExit("Extensions run on folds 1-6 only.")

    R = "results_smoke" if args.smoke else "results"
    smoke = ["--max-train", "256", "--max-test", "150", "--n-samples", "2"] if args.smoke else []
    if args.smoke:
        os.makedirs(f"{R}/losses", exist_ok=True)
        for k in args.folds:  # the extension stats compare against the main run's forecasts
            src = f"results/losses/fold_{k}_h12.npz"
            if not os.path.exists(f"{R}/losses/fold_{k}_h12.npz"):
                step(f"smoke main fold {k}", [PY, "experiments/run_leak_experiments.py", "--fold", str(k), "--seeds", "0",
                                              *ep, *smoke, "--losses-out", f"{R}/losses/fold_{k}_h12.npz",
                                              "--out", f"{R}/fold_{k}_h12_main.json"], None, True, f"{R}/logs")
    log = f"{R}/logs"

    def lo(k, tag):
        return f"{R}/losses/fold_{k}_h12_{tag}.npz"

    if "xasset" in args.parts:
        last = args.folds and max(args.folds)
        end_year = 2024 if last >= 4 else (2023 if last >= 3 else 2022)
        step("download ETHUSDT", [PY, "data/download_binance.py", "2021", "1", str(end_year), "12", "ETHUSDT"],
             f"data/raw/ETHUSDT-5m-{end_year}-12.csv", f, log)
    for k in args.folds:
        if "dose" in args.parts:
            step(f"dose fold {k}", [PY, "experiments/run_leak_experiments.py", "--fold", str(k), "--arms",
                                    "ch1_leak_k1", "ch1_leak_k3", "ch1_leak_k6", "--seeds", "0", "--no-generative", *ep, *smoke,
                                    "--losses-out", lo(k, "dose"), "--out", f"{R}/fold_{k}_h12_dose.json"], lo(k, "dose"), f, log)
        if "heal" in args.parts:
            step(f"healing fold {k}", [PY, "experiments/run_leak_experiments.py", "--fold", str(k), "--arms",
                                       "ext_ch4_control", "--seeds", "0", *ep, *smoke,
                                       "--losses-out", lo(k, "heal"), "--out", f"{R}/fold_{k}_h12_heal.json"], lo(k, "heal"), f, log)
        if "xasset" in args.parts:
            step(f"cross-asset fold {k}", [PY, "experiments/run_cross_asset.py", "--fold", str(k), "--seeds", *args.xa_seeds,
                                           *ep, *smoke, "--losses-out", lo(k, "xasset"),
                                           "--out", f"{R}/fold_{k}_h12_xasset.json"], lo(k, "xasset"), f, log)
    if "seeds" in args.parts:
        for k in [k for k in args.folds if k != 1]:
            step(f"extra seeds fold {k}", [PY, "experiments/run_leak_experiments.py", "--fold", str(k), "--seeds", "1", "2",
                                           *ep, *smoke, "--losses-out", lo(k, "seeds12"),
                                           "--out", f"{R}/fold_{k}_h12_seeds12.json"], lo(k, "seeds12"), f, log)

    step("robust stats", [PY, "experiments/robust_stats.py", "--results", R], None, True, log)
    step("aggregate h12", [PY, "experiments/aggregate_folds.py", "--horizon", "12", "--results", R], None, True, log)
    step("extension figures", [PY, "presentation/generate_figures.py", "--results", R, "--only-extensions"], None, True, log)
    print(f"\nExtension study done. Main outputs: {R}/robust_stats_h12.md and {R}/figures/extensions_h12.png")


if __name__ == "__main__":
    main()
