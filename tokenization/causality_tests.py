"""
tokenization/causality_tests.py
Causality checks that can actually fail.

1. Slice-vs-prefix test (UniTok incremental property, Zhang et al. Eq. 2):
   tokenize a stream that extends PAST the origin t, cut it at t (by the number of
   bars each token covers), and compare with tokenizing the prefix alone.
   A tokenizer is incremental at t iff the two agree and no token crosses t.

2. Future-perturbation test: replace every bar at or after t with noise, recompute
   anything derived from the series (normalized returns, gate flags), tokenize the
   extended stream again, cut at t, and check the pre-origin tokens are unchanged.

Encoders are passed in as callables encode(series, lo, hi) -> token list for bars
[lo, hi), so the same tests cover the hour-aligned tokenizer, the gate and the
unconstrained BPE.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Callable, Dict, List, Tuple

Encoder = Callable[[dict, int, int], List[int]]


def cut_at(tokens: List[int], lengths: Callable[[int], int], n_bars: int) -> Tuple[List[int], bool]:
    """First tokens covering exactly n_bars bars; flag=True if a token straddles the cut."""
    out, covered = [], 0
    for tid in tokens:
        if covered == n_bars:
            break
        L = lengths(tid)
        if covered + L > n_bars:
            return out, True
        out.append(tid)
        covered += L
    return out, False


def slice_vs_prefix(series: dict, encode: Encoder, lengths, origins, lo_of, future: int = 24) -> Dict[str, float]:
    mism = strad = 0
    for t in origins:
        lo = lo_of(t)
        full = encode(series, lo, t + future)
        cut, crossed = cut_at(full, lengths, t - lo)
        pref = encode(series, lo, t)
        strad += crossed
        mism += crossed or (cut != pref)
    n = len(origins)
    return {"origins": n, "mismatch_rate": mism / n, "straddle_rate": strad / n}


def future_perturbation(series: dict, rebuild, encode: Encoder, lengths, origins, lo_of,
                        future: int = 24, noise: float = 5.0, seed: int = 0) -> Dict[str, float]:
    """`rebuild(series, t, noise_rng)` returns a copy of `series` with bars >= t replaced and
    every derived quantity recomputed. Counts origins whose pre-origin tokens change."""
    rng = np.random.default_rng(seed)
    changed = 0
    for t in origins:
        lo = lo_of(t)
        a, _ = cut_at(encode(series, lo, t + future), lengths, t - lo)
        pert = rebuild(series, t, rng, noise)
        b, _ = cut_at(encode(pert, lo, t + future), lengths, t - lo)
        changed += a != b
    return {"origins": len(origins), "prefix_changed_rate": changed / len(origins)}
