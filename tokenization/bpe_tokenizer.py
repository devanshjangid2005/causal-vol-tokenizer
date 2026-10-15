"""
tokenization/bpe_tokenizer.py
Götz-style BPE Motif Tokenizer for Time Series with UniTok-style Hard Hour Boundaries.

Key Features:
1. Equi-width / Quantile Discretization into M base symbols.
2. Hourly Scale Tokens: Quantize trailing EWMA volatility and insert at hour boundaries.
3. Hard Boundary Constraint: BPE merges are strictly blocked from crossing hour boundaries,
   guaranteeing the Incremental Tokenization Property at every hour-aligned origin.
4. Conditional Decoding: Inverts discrete symbols to continuous values using empirical
   conditional expectations E[x_i | symbol_i, symbol_{i-1}].
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional

class TimeSeriesBPETokenizer:
    def __init__(
        self,
        num_bins: int = 32,
        val_range: Tuple[float, float] = (-4.0, 4.0),
        num_merges: int = 64,
        enforce_hour_boundaries: bool = True,
        bars_per_hour: int = 12
    ):
        self.num_bins = num_bins
        self.val_range = val_range
        self.num_merges = num_merges
        self.enforce_hour_boundaries = enforce_hour_boundaries
        self.bars_per_hour = bars_per_hour
        
        # Bin edges for normalized returns
        self.bin_edges = np.linspace(val_range[0], val_range[1], num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])
        
        # Scale token bins (for hourly EWMA volatility)
        self.num_scale_bins = 16
        self.scale_edges = np.geomspace(1e-4, 0.05, self.num_scale_bins + 1)
        
        # Merge rules: list of ((token_a, token_b), new_token_id)
        self.merges: List[Tuple[Tuple[int, int], int]] = []
        self.merge_map: Dict[Tuple[int, int], int] = {}
        
        # Vocabulary size
        self.base_vocab_size = num_bins # 0 to M-1
        self.scale_token_offset = num_bins # M to M + num_scale_bins - 1
        self.motif_token_offset = num_bins + self.num_scale_bins # motif IDs start here
        self.vocab_size = self.motif_token_offset
        
        # Conditional decoding table: conditional_means[curr_base_bin, prev_base_bin]
        self.conditional_decoding_table = np.zeros((num_bins, num_bins), dtype=np.float64)
        # Default to bin centers
        for j in range(num_bins):
            self.conditional_decoding_table[j, :] = self.bin_centers[j]

    def quantize_sample(self, val: float) -> int:
        idx = int(np.digitize(val, self.bin_edges) - 1)
        return int(np.clip(idx, 0, self.num_bins - 1))

    def quantize_scale(self, scale_val: float) -> int:
        idx = int(np.digitize(scale_val, self.scale_edges) - 1)
        clamped = int(np.clip(idx, 0, self.num_scale_bins - 1))
        return self.scale_token_offset + clamped

    def quantize_series(self, returns_norm: np.ndarray) -> np.ndarray:
        indices = np.digitize(returns_norm, self.bin_edges) - 1
        return np.clip(indices, 0, self.num_bins - 1).astype(np.int32)

    def fit_vocabulary(self, returns_norm: np.ndarray, ewma_scales: Optional[np.ndarray] = None):
        """
        Learns BPE merge table from training data.
        If enforce_hour_boundaries is True, merges are only computed within each hour block.
        """
        raw_tokens = self.quantize_series(returns_norm).tolist()
        n = len(raw_tokens)
        
        # Partition into chunks (hours) if boundary enforced
        if self.enforce_hour_boundaries:
            chunks = [raw_tokens[i : i + self.bars_per_hour] for i in range(0, n, self.bars_per_hour)]
        else:
            chunks = [raw_tokens]
            
        current_token_id = self.motif_token_offset
        self.merges = []
        self.merge_map = {}
        
        for m in range(self.num_merges):
            pair_counts: Dict[Tuple[int, int], int] = {}
            for chunk in chunks:
                for a, b in zip(chunk[:-1], chunk[1:]):
                    pair = (a, b)
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
                    
            if not pair_counts:
                break
                
            best_pair = max(pair_counts.items(), key=lambda x: x[1])
            if best_pair[1] < 2:
                # No frequent pairs left
                break
                
            pair_to_merge = best_pair[0]
            new_id = current_token_id
            current_token_id += 1
            
            self.merges.append((pair_to_merge, new_id))
            self.merge_map[pair_to_merge] = new_id
            
            # Apply merge to all chunks
            new_chunks = []
            for chunk in chunks:
                merged_chunk = []
                i = 0
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair_to_merge:
                        merged_chunk.append(new_id)
                        i += 2
                    else:
                        merged_chunk.append(chunk[i])
                        i += 1
                new_chunks.append(merged_chunk)
            chunks = new_chunks
            
        self.vocab_size = current_token_id

    def fit_conditional_decoding(self, returns_norm: np.ndarray):
        """
        Fits Götz conditional decoding table E[x_i | symbol_i = j, symbol_{i-1} = k]
        """
        symbols = self.quantize_series(returns_norm)
        sums = np.zeros((self.num_bins, self.num_bins), dtype=np.float64)
        counts = np.zeros((self.num_bins, self.num_bins), dtype=np.int32)
        
        for i in range(1, len(symbols)):
            curr_s = symbols[i]
            prev_s = symbols[i - 1]
            sums[curr_s, prev_s] += returns_norm[i]
            counts[curr_s, prev_s] += 1
            
        for j in range(self.num_bins):
            for k in range(self.num_bins):
                if counts[j, k] > 0:
                    self.conditional_decoding_table[j, k] = sums[j, k] / counts[j, k]
                else:
                    self.conditional_decoding_table[j, k] = self.bin_centers[j]

    def encode(
        self,
        returns_norm: np.ndarray,
        ewma_scales: Optional[np.ndarray] = None,
        include_scale_tokens: bool = True
    ) -> List[int]:
        """
        Encodes a normalized return sequence into tokens.
        If enforce_hour_boundaries is True, inserts scale tokens at hour starts
        and applies BPE merges strictly inside each hour.
        """
        raw_symbols = self.quantize_series(returns_norm).tolist()
        n = len(raw_symbols)
        
        if not self.enforce_hour_boundaries:
            # Global BPE without boundaries (The Leak Channel 4)
            current = raw_symbols
            for pair, new_id in self.merges:
                merged = []
                i = 0
                while i < len(current):
                    if i < len(current) - 1 and (current[i], current[i + 1]) == pair:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(current[i])
                        i += 1
                current = merged
            return current

        # Causal / UniTok hour-aligned encoding
        token_stream = []
        for h_start in range(0, n, self.bars_per_hour):
            h_end = min(n, h_start + self.bars_per_hour)
            
            # 1. Scale token at hour start
            if include_scale_tokens and ewma_scales is not None:
                scale_val = ewma_scales[h_start]
                token_stream.append(self.quantize_scale(scale_val))
                
            # 2. BPE within hour
            chunk = raw_symbols[h_start:h_end]
            for pair, new_id in self.merges:
                merged = []
                i = 0
                while i < len(chunk):
                    if i < len(chunk) - 1 and (chunk[i], chunk[i + 1]) == pair:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(chunk[i])
                        i += 1
                chunk = merged
            token_stream.extend(chunk)
            
        return token_stream

    def decompose_token(self, token_id: int) -> List[int]:
        """
        Recursively unpacks a motif token back into its constituent base bin symbols.
        """
        if token_id < self.motif_token_offset:
            if token_id < self.scale_token_offset:
                return [token_id] # base bin symbol
            else:
                return [] # scale token has no return samples
                
        # Find the merge rule that generated this token
        for (a, b), tid in self.merges:
            if tid == token_id:
                return self.decompose_token(a) + self.decompose_token(b)
        return [token_id % self.num_bins]

    def decode(self, token_sequence: List[int], use_conditional: bool = True) -> np.ndarray:
        """
        Decodes a token sequence back into continuous normalized returns.
        """
        base_symbols = []
        for t in token_sequence:
            base_symbols.extend(self.decompose_token(t))
            
        if not base_symbols:
            return np.array([], dtype=np.float64)
            
        decoded = np.zeros(len(base_symbols), dtype=np.float64)
        decoded[0] = self.bin_centers[base_symbols[0]]
        
        for i in range(1, len(base_symbols)):
            curr_s = base_symbols[i]
            prev_s = base_symbols[i - 1]
            if use_conditional:
                decoded[i] = self.conditional_decoding_table[curr_s, prev_s]
            else:
                decoded[i] = self.bin_centers[curr_s]
                
        return decoded

if __name__ == "__main__":
    from simulation.garch_simulator import GARCHSimulator
    sim = GARCHSimulator()
    ret, sig = sim.simulate(n_steps=2000)
    tok = TimeSeriesBPETokenizer(num_bins=16, num_merges=32, enforce_hour_boundaries=True)
    tok.fit_vocabulary(ret[:1000])
    tok.fit_conditional_decoding(ret[:1000])
    encoded = tok.encode(ret[1000:1100], ewma_scales=np.sqrt(sig[1000:1100]))
    decoded = tok.decode(encoded)
    print(f"Original length: 100, Encoded length: {len(encoded)}, Decoded length: {len(decoded)}")
    print(f"Compression ratio: {100.0 / len(encoded):.2f}x")

