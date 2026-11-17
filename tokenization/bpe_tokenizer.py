"""
tokenization/bpe_tokenizer.py
Götz-style BPE motif tokenizer (Götz et al., 2026) with UniTok-style hard hour
boundaries (Zhang et al., 2026) and an optional per-hour gate (PATK-inspired).

1. Uniform bins on normalized returns (fixed edges, never fitted).
2. Merge table learned on TRAIN data only; merges never cross an hour boundary
   when enforce_hour_boundaries=True, so encoding is incremental at hour-aligned
   origins: encode(x[:t]) == encode(x)[:phi(t)] whenever t is on an hour boundary.
3. Hourly scale tokens: the absolute trailing EWMA volatility at each hour start is
   quantized and emitted before that hour's motifs (absolute level the normalized
   motifs cannot carry).
4. Per-hour gate: in hours flagged volatile, only the first `short_merges` merges are
   applied, so tokens are shorter (finer resolution where volatility is high).
5. Conditional decoding (Götz eq. 5-6): E[x_i | bin_i, bin_{i-1}] and, for variance
   forecasts, the matching conditional second moment E[x_i^2 | bin_i, bin_{i-1}].

Merges are applied in rank order (order of creation), which is exactly Götz Alg. 2.
The implementation repeatedly merges the lowest-rank adjacent pair everywhere it
occurs; because a merged token can only appear in pairs of higher rank, this gives
the same output as one left-to-right pass per merge, but much faster.

INPUT CONTRACT for hour-aligned encoding: the array passed to `encode` must start on
a clock hour and contain whole hours of `bars_per_hour` bars (select_origins in
data/folds.py guarantees this for every context window).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import List, Tuple, Dict, Optional, Sequence


class TimeSeriesBPETokenizer:
    def __init__(self, num_bins: int = 32, val_range: Tuple[float, float] = (-4.0, 4.0),
                 num_merges: int = 64, enforce_hour_boundaries: bool = True, bars_per_hour: int = 12,
                 num_scale_bins: int = 16, scale_range: Tuple[float, float] = (1e-4, 0.05)):
        self.num_bins = num_bins
        self.val_range = val_range
        self.num_merges = num_merges
        self.enforce_hour_boundaries = enforce_hour_boundaries
        self.bars_per_hour = bars_per_hour

        self.bin_edges = np.linspace(val_range[0], val_range[1], num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

        self.num_scale_bins = num_scale_bins
        self.scale_edges = np.geomspace(scale_range[0], scale_range[1], num_scale_bins + 1)

        self.scale_token_offset = num_bins
        self.motif_token_offset = num_bins + num_scale_bins
        self.vocab_size = self.motif_token_offset

        self.merges: List[Tuple[Tuple[int, int], int]] = []
        self.merge_rank: Dict[Tuple[int, int], int] = {}
        self._children: Dict[int, Tuple[int, int]] = {}
        self._expand_cache: Dict[int, Tuple[int, ...]] = {}

        # conditional decoding tables, indexed [current_bin, previous_bin]
        self.cond_mean = np.tile(self.bin_centers[:, None], (1, num_bins))
        self.cond_m2 = np.tile((self.bin_centers ** 2)[:, None], (1, num_bins))

    # ------------------------------------------------------------------ basics
    def quantize_series(self, x: np.ndarray) -> np.ndarray:
        idx = np.digitize(x, self.bin_edges) - 1
        return np.clip(idx, 0, self.num_bins - 1).astype(np.int64)

    def quantize_scale(self, s: float) -> int:
        idx = int(np.digitize(s, self.scale_edges) - 1)
        return self.scale_token_offset + int(np.clip(idx, 0, self.num_scale_bins - 1))

    def _chunks(self, symbols: List[int], chunk_starts: Optional[np.ndarray]) -> List[List[int]]:
        if not self.enforce_hour_boundaries:
            return [symbols]
        n = len(symbols)
        if chunk_starts is None:
            starts = list(range(0, n, self.bars_per_hour))
        else:
            starts = list(np.flatnonzero(chunk_starts))
            if not starts or starts[0] != 0:
                starts = [0] + starts
        ends = starts[1:] + [n]
        return [symbols[a:b] for a, b in zip(starts, ends) if b > a]

    # ------------------------------------------------------------------ fitting
    def fit_vocabulary(self, returns_norm: np.ndarray, chunk_starts: Optional[np.ndarray] = None):
        """
        Learns `num_merges` merges on training data. With hour boundaries, pairs are only
        counted inside hours; `chunk_starts` (bool per bar, True on a clock-hour bar or after
        a gap) lets the caller define hours by timestamp instead of by index.
        """
        chunks = self._chunks(self.quantize_series(returns_norm).tolist(), chunk_starts)
        next_id = self.motif_token_offset
        self.merges, self.merge_rank, self._children, self._expand_cache = [], {}, {}, {}
        for rank in range(self.num_merges):
            counts: Dict[Tuple[int, int], int] = {}
            for c in chunks:
                for a, b in zip(c[:-1], c[1:]):
                    counts[(a, b)] = counts.get((a, b), 0) + 1
            if not counts:
                break
            pair, cnt = max(counts.items(), key=lambda kv: kv[1])
            if cnt < 2:
                break
            self.merges.append((pair, next_id))
            self.merge_rank[pair] = rank
            self._children[next_id] = pair
            chunks = [self._replace_pair(c, pair, next_id) for c in chunks]
            next_id += 1
        self.vocab_size = next_id
        return self

    def fit_conditional_decoding(self, returns_norm: np.ndarray):
        """E[x_i | bin_i=j, bin_{i-1}=k] and E[x_i^2 | j, k] from data (Götz eq. 6)."""
        s = self.quantize_series(returns_norm)
        j, k, x = s[1:], s[:-1], returns_norm[1:]
        M = self.num_bins
        cnt = np.zeros((M, M))
        s1 = np.zeros((M, M))
        s2 = np.zeros((M, M))
        np.add.at(cnt, (j, k), 1.0)
        np.add.at(s1, (j, k), x)
        np.add.at(s2, (j, k), x * x)
        fallback_mean = np.tile(self.bin_centers[:, None], (1, M))
        fallback_m2 = np.tile((self.bin_centers ** 2)[:, None], (1, M))
        with np.errstate(invalid="ignore", divide="ignore"):
            self.cond_mean = np.where(cnt > 0, s1 / cnt, fallback_mean)
            self.cond_m2 = np.where(cnt > 0, s2 / cnt, fallback_m2)
        return self

    # backward-compatible name used by older scripts
    @property
    def conditional_decoding_table(self):
        return self.cond_mean

    # ------------------------------------------------------------------ encoding
    @staticmethod
    def _replace_pair(seq: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        out, i, n = [], 0, len(seq)
        a, b = pair
        while i < n:
            if i < n - 1 and seq[i] == a and seq[i + 1] == b:
                out.append(new_id)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        return out

    def _bpe(self, seq: List[int], max_rank: int) -> List[int]:
        """Applies merges with rank < max_rank in rank order (Götz Alg. 2)."""
        rank = self.merge_rank
        while len(seq) > 1:
            best, best_pair = max_rank, None
            for p in zip(seq[:-1], seq[1:]):
                r = rank.get(p)
                if r is not None and r < best:
                    best, best_pair = r, p
            if best_pair is None:
                break
            seq = self._replace_pair(seq, best_pair, self.merges[best][1])
        return seq

    def encode(self, returns_norm: np.ndarray, ewma_scales: Optional[np.ndarray] = None,
               include_scale_tokens: bool = True, gate_flags: Optional[Sequence[bool]] = None,
               short_merges: Optional[int] = None) -> List[int]:
        """
        returns_norm : normalized returns, starting on an hour boundary when hour-aligned.
        ewma_scales  : absolute EWMA vol per bar (same length); the value at each hour
                       start becomes that hour's scale token.
        gate_flags   : one bool per hour; True = volatile hour -> only `short_merges` merges.
        """
        symbols = self.quantize_series(returns_norm).tolist()
        full = len(self.merges)
        if not self.enforce_hour_boundaries:
            return self._bpe(symbols, full)

        short = full // 4 if short_merges is None else short_merges
        out: List[int] = []
        for h, start in enumerate(range(0, len(symbols), self.bars_per_hour)):
            if include_scale_tokens and ewma_scales is not None:
                out.append(self.quantize_scale(float(ewma_scales[start])))
            flag = bool(gate_flags[h]) if gate_flags is not None else False
            out.extend(self._bpe(symbols[start:start + self.bars_per_hour], short if flag else full))
        return out

    # ------------------------------------------------------------------ decoding
    def expand(self, token_id: int) -> Tuple[int, ...]:
        """Base-bin symbols covered by a token (scale tokens and unknown ids cover none)."""
        if token_id < self.scale_token_offset:
            return (token_id,)
        if token_id < self.motif_token_offset:
            return ()
        hit = self._expand_cache.get(token_id)
        if hit is None:
            pair = self._children.get(token_id)
            hit = () if pair is None else self.expand(pair[0]) + self.expand(pair[1])
            self._expand_cache[token_id] = hit
        return hit

    def decompose_token(self, token_id: int) -> List[int]:
        return list(self.expand(token_id))

    def token_lengths(self, size: int) -> np.ndarray:
        """Number of bars covered by each id in [0, size)."""
        return np.array([len(self.expand(i)) for i in range(size)], dtype=np.int64)

    def decode(self, token_sequence: Sequence[int], use_conditional: bool = True) -> np.ndarray:
        base = [s for t in token_sequence for s in self.expand(int(t))]
        if not base:
            return np.array([])
        out = np.empty(len(base))
        out[0] = self.bin_centers[base[0]]
        for i in range(1, len(base)):
            out[i] = self.cond_mean[base[i], base[i - 1]] if use_conditional else self.bin_centers[base[i]]
        return out


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.standard_t(4, 12 * 2000) * 0.7
    tok = TimeSeriesBPETokenizer().fit_vocabulary(x[:12 * 1500])
    tok.fit_conditional_decoding(x[:12 * 1500])
    enc = tok.encode(x[12 * 1500:12 * 1600], include_scale_tokens=False)
    print("merges:", len(tok.merges), "compression:", 1200 / len(enc))
