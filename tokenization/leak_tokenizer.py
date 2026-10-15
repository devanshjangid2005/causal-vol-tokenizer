"""
tokenization/leak_tokenizer.py
Implementation of the 5 Look-Ahead Leakage Channels & Controls:
1. Channel 1: Per-window normalization leak (uses target horizon in normalization)
2. Channel 2: Merge table leak (estimates BPE vocabulary on test period)
3. Channel 3: Decoding table leak (estimates conditional decoding table on test period)
4. Channel 4: Token boundary leak (non-incremental global BPE without hard boundaries)
5. Channel 5: Non-causal FFT leak (centered spectral window looking into future)
6. Positive Control: Deliberately gross normalization leak (normalizes by future RV)
7. Causal Reference: Completely leak-free baseline (trailing EWMA + hard boundaries + train-only tables)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from tokenization.bpe_tokenizer import TimeSeriesBPETokenizer

class LeakageManager:
    def __init__(
        self,
        num_bins: int = 32,
        num_merges: int = 64,
        bars_per_hour: int = 12
    ):
        self.num_bins = num_bins
        self.num_merges = num_merges
        self.bars_per_hour = bars_per_hour
        
        # Reference Causal Tokenizer
        self.causal_tok = TimeSeriesBPETokenizer(
            num_bins=num_bins,
            num_merges=num_merges,
            enforce_hour_boundaries=True,
            bars_per_hour=bars_per_hour
        )
        
        # Leak Channel 4 Tokenizer (Global BPE without boundaries)
        self.boundary_leak_tok = TimeSeriesBPETokenizer(
            num_bins=num_bins,
            num_merges=num_merges,
            enforce_hour_boundaries=False,
            bars_per_hour=bars_per_hour
        )

    def fit_reference(self, train_returns: np.ndarray, train_ewma_vol: np.ndarray):
        """Fits reference causal tokenizer strictly on training data."""
        train_norm = train_returns / np.maximum(train_ewma_vol, 1e-6)
        self.causal_tok.fit_vocabulary(train_norm)
        self.causal_tok.fit_conditional_decoding(train_norm)

    def fit_leaked_tables(self, full_returns: np.ndarray, full_ewma_vol: np.ndarray):
        """
        Fits merge table and decoding table on entire dataset (train + test),
        creating Channel 2 and Channel 3 leaked tokenizers.
        """
        full_norm = full_returns / np.maximum(full_ewma_vol, 1e-6)
        
        # Channel 2: Merge Table Leak
        self.merge_leaked_tok = TimeSeriesBPETokenizer(
            num_bins=self.num_bins,
            num_merges=self.num_merges,
            enforce_hour_boundaries=True,
            bars_per_hour=self.bars_per_hour
        )
        self.merge_leaked_tok.fit_vocabulary(full_norm)
        # Use causal decoding table
        self.merge_leaked_tok.conditional_decoding_table = self.causal_tok.conditional_decoding_table.copy()
        
        # Channel 3: Decoding Table Leak
        self.decoding_leaked_tok = TimeSeriesBPETokenizer(
            num_bins=self.num_bins,
            num_merges=self.num_merges,
            enforce_hour_boundaries=True,
            bars_per_hour=self.bars_per_hour
        )
        self.decoding_leaked_tok.merges = self.causal_tok.merges.copy()
        self.decoding_leaked_tok.merge_map = self.causal_tok.merge_map.copy()
        self.decoding_leaked_tok.vocab_size = self.causal_tok.vocab_size
        self.decoding_leaked_tok.fit_conditional_decoding(full_norm)

    def extract_features(
        self,
        returns: np.ndarray,
        ewma_vol: np.ndarray,
        origin_t: int,
        context_len: int = 144, # 12 hours
        horizon: int = 12,      # 1 hour ahead
        channel: str = "reference"
    ) -> Tuple[np.ndarray, float]:
        """
        Extracts tokenized input sequence and scale factor at forecast origin t.
        Channels supported:
        - 'reference': Causal trailing EWMA, hard boundaries, train-only tables
        - 'channel_1_norm_leak': Per-window normalization using future horizon
        - 'channel_2_merge_leak': Leaked merge table
        - 'channel_3_decode_leak': Leaked decoding table
        - 'channel_4_boundary_leak': Global BPE without hour boundaries
        - 'channel_5_fft_leak': Centered FFT looking into future
        - 'positive_control': Gross leak using true future target RV
        """
        start_idx = origin_t - context_len
        assert start_idx >= 0, "Insufficient context before origin_t"
        
        raw_context = returns[start_idx:origin_t]
        
        # 1. Normalization
        if channel == "reference":
            # Strictly causal: scale at origin t comes from past
            scale = float(ewma_vol[origin_t])
            norm_ctx = raw_context / max(scale, 1e-6)
            tok = self.causal_tok
            tokens = tok.encode(norm_ctx, include_scale_tokens=False)
            
        elif channel == "channel_1_norm_leak":
            # Window std computed over context PLUS future horizon!
            future_window = returns[start_idx : origin_t + horizon]
            leaked_scale = float(np.std(future_window))
            norm_ctx = raw_context / max(leaked_scale, 1e-6)
            tokens = self.causal_tok.encode(norm_ctx, include_scale_tokens=False)
            scale = leaked_scale
            
        elif channel == "channel_2_merge_leak":
            scale = float(ewma_vol[origin_t])
            norm_ctx = raw_context / max(scale, 1e-6)
            tokens = self.merge_leaked_tok.encode(norm_ctx, include_scale_tokens=False)
            
        elif channel == "channel_3_decode_leak":
            scale = float(ewma_vol[origin_t])
            norm_ctx = raw_context / max(scale, 1e-6)
            tokens = self.decoding_leaked_tok.encode(norm_ctx, include_scale_tokens=False)
            
        elif channel == "channel_4_boundary_leak":
            # Global BPE on context + future, sliced at origin t
            scale = float(ewma_vol[origin_t])
            full_window = returns[start_idx : origin_t + horizon]
            norm_full = full_window / max(scale, 1e-6)
            full_tokens = self.boundary_leak_tok.encode(norm_full, include_scale_tokens=False)
            # Find tokens up to context_len
            tokens = []
            curr = 0
            for t_id in full_tokens:
                sub = self.boundary_leak_tok.decompose_token(t_id)
                curr += len(sub)
                tokens.append(t_id)
                if curr >= context_len:
                    break
                    
        elif channel == "channel_5_fft_leak":
            # Centered FFT window [origin_t - 6, origin_t + 6]
            scale = float(ewma_vol[origin_t])
            norm_ctx = raw_context / max(scale, 1e-6)
            tokens = self.causal_tok.encode(norm_ctx, include_scale_tokens=False)
            # Future FFT energy leaked into scale factor
            centered_window = returns[origin_t - 6 : origin_t + 6]
            fft_energy = float(np.sum(np.abs(np.fft.rfft(centered_window)) ** 2))
            scale = scale * (1.0 + 0.5 * fft_energy)
            
        elif channel == "positive_control":
            # Deliberately gross leak: scale is proportional to true future RV!
            true_future_rv = float(np.sum(returns[origin_t : origin_t + horizon] ** 2))
            leaked_scale = float(np.sqrt(true_future_rv / horizon))
            norm_ctx = raw_context / max(leaked_scale, 1e-6)
            tokens = self.causal_tok.encode(norm_ctx, include_scale_tokens=False)
            scale = leaked_scale
            
        else:
            raise ValueError(f"Unknown channel: {channel}")
            
        return np.array(tokens, dtype=np.int64), scale

if __name__ == "__main__":
    from simulation.garch_simulator import GARCHSimulator
    sim = GARCHSimulator()
    ret, sig = sim.simulate(n_steps=5000)
    ewma = np.sqrt(sig)
    
    mgr = LeakageManager(num_bins=16, num_merges=32)
    mgr.fit_reference(ret[:2500], ewma[:2500])
    mgr.fit_leaked_tables(ret, ewma)
    
    for ch in ["reference", "channel_1_norm_leak", "channel_4_boundary_leak", "positive_control"]:
        toks, sc = mgr.extract_features(ret, ewma, origin_t=3000, channel=ch)
        print(f"Channel: {ch:25s} Tokens len: {len(toks)} Scale: {sc:.6f}")

