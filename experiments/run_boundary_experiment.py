"""
experiments/run_boundary_experiment.py
Boundary-Mismatch Experiment and Tokenizer Compression Analysis.

Quantifies:
1. Boundary Mismatch Rate & Straddling Rate across 500 origins (UniTok Hard Boundaries vs Götz Unconstrained BPE).
2. Tokenizer Compression Ratios across vocabulary merge counts (K = 16, 32, 64, 128).
3. Empirical proof of look-ahead boundary leakage without requiring model training.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import numpy as np
import pandas as pd
from tokenization.bpe_tokenizer import TimeSeriesBPETokenizer
from tokenization.causality_tests import evaluate_boundary_mismatch_rate, run_future_perturbation_test

def run_boundary_experiment(data_path: str = "data/btcusdt_5m_sample.csv") -> dict:
    print("=" * 60)
    print("RUNNING BOUNDARY MISMATCH & TOKENIZER COMPRESSION EXPERIMENT")
    print("=" * 60)
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        close = df["close"].values
        returns = np.zeros(len(close))
        returns[1:] = np.log(close[1:] / close[:-1])
    else:
        from simulation.garch_simulator import GARCHSimulator
        sim = GARCHSimulator()
        returns, _ = sim.simulate(n_steps=25000)
        
    train_size = int(0.7 * len(returns))
    train_ret = returns[:train_size]
    test_ret = returns[train_size:]
    
    # 1. Fit Causal vs Unconstrained Tokenizers
    tok_causal = TimeSeriesBPETokenizer(num_bins=32, num_merges=64, enforce_hour_boundaries=True)
    tok_causal.fit_vocabulary(train_ret)
    
    tok_unconstrained = TimeSeriesBPETokenizer(num_bins=32, num_merges=64, enforce_hour_boundaries=False)
    tok_unconstrained.fit_vocabulary(train_ret)
    
    # 2. Evaluate Boundary Mismatch Rate across 300 test origins
    mismatch_res = evaluate_boundary_mismatch_rate(
        tok_causal,
        tok_unconstrained,
        test_ret,
        num_eval_origins=300,
        eval_step=12
    )
    print(f"Total origins evaluated: {mismatch_res['total_origins_evaluated']}")
    print(f"UniTok Hard Boundaries Mismatch Rate:     {mismatch_res['mismatch_rate_with_boundaries'] * 100.0:.2f}%")
    print(f"Unconstrained BPE Mismatch Rate:          {mismatch_res['mismatch_rate_without_boundaries'] * 100.0:.2f}%")
    print(f"Unconstrained BPE Straddling Tokens Rate: {mismatch_res['straddling_boundary_rate'] * 100.0:.2f}%")
    
    # 3. Compression Ratio vs Number of Merges
    compression_stats = []
    test_slice = test_ret[:1200] # 100 hours
    for k in [16, 32, 64, 96, 128]:
        tok = TimeSeriesBPETokenizer(num_bins=32, num_merges=k, enforce_hour_boundaries=True)
        tok.fit_vocabulary(train_ret)
        encoded = tok.encode(test_slice, include_scale_tokens=False)
        comp_ratio = float(len(test_slice) / max(1, len(encoded)))
        compression_stats.append({
            "num_merges": k,
            "vocab_size": tok.vocab_size,
            "encoded_tokens": len(encoded),
            "compression_ratio": comp_ratio
        })
        print(f"Merges: {k:3d} | Vocab: {tok.vocab_size:3d} | Compression: {comp_ratio:.2f}x")
        
    # 4. Future Perturbation Test
    perturb_res = run_future_perturbation_test(tok_causal, test_ret[:300], origin_t=144)
    print("Future-Perturbation Invariance Test:", "PASSED" if perturb_res["is_prefix_invariant"] else "FAILED")
    
    output_data = {
        "mismatch_evaluation": mismatch_res,
        "compression_curve": compression_stats,
        "perturbation_test": perturb_res
    }
    
    os.makedirs("results", exist_ok=True)
    with open("results/boundary_mismatch_results.json", "w") as f:
        json.dump(output_data, f, indent=2)
    print("Saved boundary mismatch results to results/boundary_mismatch_results.json")
    return output_data

if __name__ == "__main__":
    run_boundary_experiment()

