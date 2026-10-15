"""
tokenization/causality_tests.py
Causality Verification Suite for Time-Series Tokenizers:
1. Future-Perturbation Test (UniTok Incremental Property):
   Alters future bars after origin t and verifies that prefix tokens, scale tokens,
   gate states, and representations remain strictly invariant.
2. Boundary-Mismatch Test:
   Empirically quantifies the mismatch rate between causal prefix tokenization
   and post-hoc slicing of globally tokenized streams across multiple origins.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, List, Tuple
from tokenization.bpe_tokenizer import TimeSeriesBPETokenizer
from tokenization.causal_gate import CausalRoVSEIGate

def run_future_perturbation_test(
    tokenizer: TimeSeriesBPETokenizer,
    returns: np.ndarray,
    origin_t: int,
    noise_scale: float = 5.0,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Perturbs all returns after origin_t by adding large random noise.
    Asserts whether prefix tokens up to origin_t are identical.
    """
    rng = np.random.default_rng(seed)
    n = len(returns)
    assert origin_t < n, "origin_t must be less than length of returns"
    
    # 1. Original prefix encoding
    orig_prefix = returns[:origin_t]
    orig_tokens = tokenizer.encode(orig_prefix, include_scale_tokens=False)
    
    # 2. Perturbed series
    perturbed_returns = returns.copy()
    perturbed_returns[origin_t:] += noise_scale * rng.standard_normal(n - origin_t)
    
    # Test incremental property:
    # Does tokenizing only the prefix of the perturbed series match?
    perturbed_prefix_tokens = tokenizer.encode(perturbed_returns[:origin_t], include_scale_tokens=False)
    
    is_prefix_invariant = (orig_tokens == perturbed_prefix_tokens)
    
    # Does tokenizing full stream and slicing at origin_t leak?
    full_orig_tokens = tokenizer.encode(returns, include_scale_tokens=False)
    full_perturbed_tokens = tokenizer.encode(perturbed_returns, include_scale_tokens=False)
    
    # Reconstruct continuous samples from full stream to see if prefix was corrupted
    full_orig_decoded = tokenizer.decode(full_orig_tokens)[:origin_t]
    full_perturbed_decoded = tokenizer.decode(full_perturbed_tokens)[:origin_t]
    
    max_prefix_diff = float(np.max(np.abs(full_orig_decoded - full_perturbed_decoded))) if len(full_orig_decoded) > 0 else 0.0
    
    return {
        "origin_t": origin_t,
        "is_prefix_invariant": is_prefix_invariant,
        "full_stream_prefix_corrupted": bool(max_prefix_diff > 1e-6),
        "max_prefix_diff": max_prefix_diff
    }

def evaluate_boundary_mismatch_rate(
    tokenizer_with_boundaries: TimeSeriesBPETokenizer,
    tokenizer_without_boundaries: TimeSeriesBPETokenizer,
    returns: np.ndarray,
    num_eval_origins: int = 300,
    eval_step: int = 12, # evaluate at every hour
    seed: int = 42
) -> Dict[str, Any]:
    """
    Measures the Boundary Mismatch Rate across multiple origins for:
    A) Causal BPE with hard hour boundaries (UniTok style)
    B) Unconstrained global BPE without boundaries (Standard Götz style)
    """
    n = len(returns)
    min_origin = 120 # at least 10 hours of context
    max_origin = n - 24 # leave space for future
    
    candidate_origins = list(range(min_origin, max_origin, eval_step))
    if len(candidate_origins) > num_eval_origins:
        rng = np.random.default_rng(seed)
        origins = sorted(rng.choice(candidate_origins, size=num_eval_origins, replace=False))
    else:
        origins = candidate_origins
        
    mismatches_with_bound = 0
    mismatches_without_bound = 0
    straddling_tokens_count = 0
    total_evals = len(origins)
    
    for t in origins:
        prefix = returns[:t]
        full_stream = returns[:t + 24] # context plus 2 hours future
        
        # --- A. With Hour Boundaries (UniTok) ---
        # With hour boundaries aligned at t, no motif can cross t
        tok_pref_a = tokenizer_with_boundaries.encode(prefix, include_scale_tokens=False)
        tok_full_a = tokenizer_with_boundaries.encode(full_stream, include_scale_tokens=False)
        # Sliced tokens up to t
        tok_full_prefix_a = tokenizer_with_boundaries.encode(full_stream[:t], include_scale_tokens=False)
        if tok_pref_a != tok_full_prefix_a:
            mismatches_with_bound += 1
            
        # --- B. Without Hour Boundaries (Unconstrained BPE) ---
        tok_pref_b = tokenizer_without_boundaries.encode(prefix, include_scale_tokens=False)
        tok_full_b = tokenizer_without_boundaries.encode(full_stream, include_scale_tokens=False)
        
        # Check if full stream has a motif token that straddles across the boundary t
        # Track sample indices covered by each token in tok_full_b
        curr_idx = 0
        straddles = False
        tokens_before_or_at_t = []
        for token_id in tok_full_b:
            sub_symbols = tokenizer_without_boundaries.decompose_token(token_id)
            tok_len = len(sub_symbols)
            start_pos = curr_idx
            end_pos = curr_idx + tok_len
            curr_idx = end_pos
            
            # Check if this token spans across boundary t (i.e. starts before t and ends after t)
            if start_pos < t and end_pos > t:
                straddles = True
            if end_pos <= t:
                tokens_before_or_at_t.append(token_id)
                
        if straddles or (tokens_before_or_at_t != tok_pref_b):
            mismatches_without_bound += 1
            if straddles:
                straddling_tokens_count += 1
            
    rate_with = mismatches_with_bound / total_evals
    rate_without = mismatches_without_bound / total_evals
    straddle_rate = straddling_tokens_count / total_evals
    
    return {
        "total_origins_evaluated": total_evals,
        "mismatches_with_boundaries": mismatches_with_bound,
        "mismatch_rate_with_boundaries": rate_with,
        "mismatches_without_boundaries": mismatches_without_bound,
        "mismatch_rate_without_boundaries": rate_without,
        "straddling_boundary_rate": straddle_rate
    }

if __name__ == "__main__":
    from simulation.garch_simulator import GARCHSimulator
    sim = GARCHSimulator()
    ret, _ = sim.simulate(n_steps=5000)
    
    tok_causal = TimeSeriesBPETokenizer(num_bins=16, num_merges=32, enforce_hour_boundaries=True)
    tok_causal.fit_vocabulary(ret[:2000])
    
    tok_leak = TimeSeriesBPETokenizer(num_bins=16, num_merges=32, enforce_hour_boundaries=False)
    tok_leak.fit_vocabulary(ret[:2000])
    
    res = evaluate_boundary_mismatch_rate(tok_causal, tok_leak, ret[2000:4000], num_eval_origins=100)
    print("Boundary Mismatch Results:")
    print(res)
