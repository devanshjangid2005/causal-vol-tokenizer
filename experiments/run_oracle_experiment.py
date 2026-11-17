"""
experiments/run_oracle_experiment.py
Negative control on simulated GARCH(1,1) data, run through the SAME pipeline as the
real-data audit (tokenizers, gate, per-arm GPT training, both forecast heads).

On GARCH data the analytic oracle forecast is the best any causal forecaster can do in
expectation. So:
  * the causal reference and the controls should NOT beat the oracle floor;
  * an arm that beats it significantly must be using future information.
The first version of this script never ran the tokenizer or GPT, and its "positive
control" was the target itself times noise, so it could not validate the harness.

Usage: python experiments/run_oracle_experiment.py [--arms ...] [--seeds 0] [--epochs 5]
       (accepts the same options as run_leak_experiments.py)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np

from data.folds import load_fold
from simulation.garch_simulator import oracle_rv_forecast
from experiments.run_leak_experiments import parse_args, run_audit, print_table
from evaluation.hypothesis_tests import diebold_mariano_test


def main():
    args = parse_args()
    args.fold = "synthetic_garch"
    df, split, params = load_fold(synthetic=True)
    oracle = np.full(len(df), np.nan)
    ok = np.isfinite(df["true_sigma_sq"].values)
    oracle[ok] = oracle_rv_forecast(df["true_sigma_sq"].values[ok], params["omega"], params["alpha"],
                                    params["beta"], args.horizon)
    res = run_audit(df, split, args, extra_forecasts={"oracle": oracle})
    res["garch_params"] = params

    # does any arm beat the oracle floor? (one-sided: arm loss < oracle loss)
    floor = res["baselines"]["oracle"]["QLIKE"]
    res["beats_oracle_floor"] = {}
    for head, hr in res["heads"].items():
        res["beats_oracle_floor"][head] = {}
        for arm, v in hr["vs_baselines"].items():
            cmp = v["oracle"]
            one_sided_p = cmp["dm_p"] / 2 if cmp["dm_stat"] > 0 else 1 - cmp["dm_p"] / 2
            res["beats_oracle_floor"][head][arm] = {
                "QLIKE": hr["arms"][arm]["QLIKE"], "oracle_floor": floor,
                "one_sided_p_arm_better": one_sided_p, "violates_floor": bool(one_sided_p < 0.05)}

    print_table(res)
    print(f"\nOracle floor QLIKE: {floor:.4f}")
    for head, d in res["beats_oracle_floor"].items():
        for arm, v in d.items():
            print(f"  [{head}] {arm:26s} {v['QLIKE']:.4f}  beats floor: {v['violates_floor']} (p={v['one_sided_p_arm_better']:.4f})")

    os.makedirs("results", exist_ok=True)
    with open("results/negative_control_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print("Saved results/negative_control_results.json")


if __name__ == "__main__":
    main()
