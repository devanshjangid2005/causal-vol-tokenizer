"""
experiments/run_baselines.py
Evaluates Econometric Baselines (HAR-RV and GARCH(1,1)) on Real BTCUSDT Fold 1:
Train: 2021 (105k 5-minute bars)
Test:  2022-Q1 (26k 5-minute bars)
Evaluates Next-Hour (H = 12) Realized Variance Forecasting with QLIKE and Block-Bootstrap CIs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from data.folds import load_or_fetch_fold_data
from models.baselines import HARRVModel, GARCH11Model
from evaluation.metrics import compute_all_metrics, qlike_loss
from evaluation.hypothesis_tests import block_bootstrap_ci, diebold_mariano_test

def run_fold_1_baselines(horizon: int = 12) -> dict:
    print("=" * 60)
    print(f"RUNNING BASELINES ON FOLD 1 (BTCUSDT 5-MIN, HORIZON H={horizon})")
    print("=" * 60)
    
    df_train, df_test = load_or_fetch_fold_data(fold_idx=1)
    train_returns = df_train["return"].values
    test_returns = df_test["return"].values
    
    print(f"Train bars: {len(train_returns)} ({df_train['open_time'].min()} to {df_train['open_time'].max()})")
    print(f"Test bars:  {len(test_returns)} ({df_test['open_time'].min()} to {df_test['open_time'].max()})")
    
    # 1. HAR-RV Baseline
    print("\nFitting HAR-RV on Fold 1 Train...")
    har = HARRVModel()
    har.fit(train_returns, horizon=horizon)
    har_preds, y_true = har.predict(test_returns, horizon=horizon)
    har_metrics = compute_all_metrics(y_true, har_preds)
    har_qlike_losses = qlike_loss(y_true, har_preds)
    _, har_ci_low, har_ci_high = block_bootstrap_ci(har_qlike_losses)
    har_metrics["QLIKE_95_CI"] = [har_ci_low, har_ci_high]
    print(f"[HAR-RV]   QLIKE: {har_metrics['QLIKE']:.5f} (95% CI: [{har_ci_low:.5f}, {har_ci_high:.5f}]) RMSE: {har_metrics['RMSE']:.6f}")
    
    # 2. GARCH(1,1) Baseline
    print("\nFitting GARCH(1,1) on Fold 1 Train...")
    garch = GARCH11Model()
    garch.fit(train_returns)
    garch_preds, y_garch_true = garch.predict(test_returns, horizon=horizon)
    min_len = min(len(y_true), len(garch_preds))
    garch_metrics = compute_all_metrics(y_true[:min_len], garch_preds[:min_len])
    garch_qlike_losses = qlike_loss(y_true[:min_len], garch_preds[:min_len])
    _, garch_ci_low, garch_ci_high = block_bootstrap_ci(garch_qlike_losses)
    garch_metrics["QLIKE_95_CI"] = [garch_ci_low, garch_ci_high]
    print(f"[GARCH]    QLIKE: {garch_metrics['QLIKE']:.5f} (95% CI: [{garch_ci_low:.5f}, {garch_ci_high:.5f}]) RMSE: {garch_metrics['RMSE']:.6f}")
    
    # Diebold-Mariano test: HAR-RV vs GARCH(1,1)
    dm_stat, dm_p = diebold_mariano_test(garch_qlike_losses, har_qlike_losses[:min_len])
    print(f"\nDM Test (GARCH vs HAR-RV): Stat={dm_stat:.3f}, p-value={dm_p:.4f}")
    
    results = {
        "fold": 1,
        "horizon": horizon,
        "train_samples": len(train_returns),
        "test_samples": len(test_returns),
        "har_rv": har_metrics,
        "garch_11": garch_metrics,
        "dm_garch_vs_har": {"stat": dm_stat, "p_value": dm_p}
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/fold_1_baselines.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved baseline results to results/fold_1_baselines.json")
    return results

if __name__ == "__main__":
    run_fold_1_baselines(horizon=12)

