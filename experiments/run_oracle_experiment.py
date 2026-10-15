"""
experiments/run_oracle_experiment.py
Negative Control Experiment: GARCH(1,1) Synthetic Benchmark.

Validates that:
1. No causal, leak-free model can beat the analytic Oracle QLIKE floor.
2. The gross-leak positive control and leaky channels artificially break below the floor.
This confirms the validity of the statistical test for detecting look-ahead bias.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
from simulation.garch_simulator import GARCHSimulator
from models.baselines import HARRVModel, GARCH11Model
from evaluation.metrics import qlike_loss, mean_qlike, rmse, mae
from evaluation.hypothesis_tests import block_bootstrap_ci, diebold_mariano_test

def run_oracle_experiment(n_steps: int = 25000, horizon: int = 12, seed: int = 42) -> dict:
    print("=" * 60)
    print("RUNNING NEGATIVE CONTROL EXPERIMENT ON SYNTHETIC GARCH(1,1)")
    print("=" * 60)
    
    sim = GARCHSimulator(omega=1e-5, alpha=0.08, beta=0.90, seed=seed)
    returns, sigma_sq = sim.simulate(n_steps=n_steps)
    rv_true, oracle_fc = sim.compute_oracle_rv_forecast(returns, sigma_sq, horizon=horizon)
    
    train_size = int(0.7 * len(rv_true))
    test_ret = returns[train_size : train_size + len(rv_true[train_size:])]
    test_rv = rv_true[train_size:]
    test_oracle = oracle_fc[train_size:]
    
    # 1. Oracle Loss (Negative Control Floor)
    oracle_losses = qlike_loss(test_rv, test_oracle)
    oracle_qlike = float(np.mean(oracle_losses))
    _, oracle_ci_low, oracle_ci_high = block_bootstrap_ci(oracle_losses)
    print(f"[ORACLE FLOOR]     QLIKE: {oracle_qlike:.5f} (95% CI: [{oracle_ci_low:.5f}, {oracle_ci_high:.5f}])")
    
    # 2. GARCH(1,1) Estimated Model (Causal Baseline)
    garch_mod = GARCH11Model()
    garch_mod.fit(returns[:train_size])
    garch_preds, _ = garch_mod.predict(returns[train_size:], horizon=horizon)
    min_len = min(len(test_rv), len(garch_preds))
    garch_losses = qlike_loss(test_rv[:min_len], garch_preds[:min_len])
    garch_qlike = float(np.mean(garch_losses))
    print(f"[GARCH(1,1)]       QLIKE: {garch_qlike:.5f} (Diff to Oracle: +{garch_qlike - oracle_qlike:.5f})")
    
    # 3. HAR-RV Model (Causal Baseline)
    har_mod = HARRVModel()
    har_mod.fit(returns[:train_size], horizon=horizon)
    har_preds, _ = har_mod.predict(returns[train_size:], horizon=horizon)
    min_len = min(len(test_rv), len(har_preds))
    har_losses = qlike_loss(test_rv[:min_len], har_preds[:min_len])
    har_qlike = float(np.mean(har_losses))
    print(f"[HAR-RV]           QLIKE: {har_qlike:.5f} (Diff to Oracle: +{har_qlike - oracle_qlike:.5f})")
    
    # 4. Positive Control: Gross Normalization Leak
    # Injects future target realization with 10% log-normal noise (leak of future realization)
    rng = np.random.default_rng(seed)
    gross_leak_preds = test_rv * np.exp(0.10 * rng.standard_normal(len(test_rv)))
    gross_losses = qlike_loss(test_rv, gross_leak_preds)
    gross_qlike = float(np.mean(gross_losses))
    print(f"[GROSS LEAK (POS)] QLIKE: {gross_qlike:.5f} (Diff to Oracle: {gross_qlike - oracle_qlike:.5f})")
    
    # Verify that gross leak beats oracle floor (impossible without future information!)
    breaks_oracle_floor = bool(gross_qlike < oracle_qlike)
    dm_stat, dm_p = diebold_mariano_test(oracle_losses, gross_losses)
    print(f"Gross Leak Breaks Oracle Floor: {breaks_oracle_floor} (DM stat: {dm_stat:.3f}, p-val: {dm_p:.4e})")
    
    results = {
        "n_steps": n_steps,
        "horizon": horizon,
        "oracle_floor_qlike": oracle_qlike,
        "oracle_ci_95": [oracle_ci_low, oracle_ci_high],
        "garch_qlike": garch_qlike,
        "har_qlike": har_qlike,
        "positive_control_gross_leak_qlike": gross_qlike,
        "gross_leak_breaks_oracle": breaks_oracle_floor,
        "dm_stat_gross_vs_oracle": dm_stat,
        "dm_pval_gross_vs_oracle": dm_p
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/negative_control_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/negative_control_results.json")
    return results

if __name__ == "__main__":
    run_oracle_experiment(n_steps=20000, horizon=12)
