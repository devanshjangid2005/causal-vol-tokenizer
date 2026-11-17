"""
tokenization/leak_tokenizer.py
Builds the model input for every experimental arm at a forecast origin t.

Every arm sees the context bars [t - C, t) and is scored on RV_H[t] = sum_{k<H} r_{t+k}^2.
Each leak arm differs from its matched control in ONE thing only:

  arm                        matched control            the single switch
  -------------------------  -------------------------  -----------------------------------------
  reference                  -                          leak-free: per-bar trailing EWMA norm,
                                                        hour boundaries, train-only tables,
                                                        causal RoV x SEI gate, hourly scale tokens
  ch1_control_window         -                          per-window std normalization, context only
  ch1_window_norm_leak       ch1_control_window         ... std computed over context + horizon
  ch2_merge_table_leak       reference                  merge table fitted on train + test
  ch3_decode_table_leak      reference                  decoding tables fitted on train + test
                                                        (only affects the generative forecast)
  ch4_control_context_only   -                          unconstrained BPE (no hour boundaries) on
                                                        the context alone -> partial-token bias
  ch4_boundary_leak          ch4_control_context_only   ... stream tokenized once over context +
                                                        horizon, then cut at t (the token that
                                                        straddles t is kept -> future info)
  ch5_noncausal_fft_leak     reference                  gate window centered on each hour
  positive_control           reference                  context normalized by the TRUE future RV

The partial-token trade-off of the proposal is read off two comparisons:
  boundary leakage   = ch4_boundary_leak vs ch4_control_context_only
  partial-token bias = ch4_control_context_only vs reference (hour-aligned, no partial token)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from tokenization.bpe_tokenizer import TimeSeriesBPETokenizer
from tokenization.causal_gate import CausalRoVSEIGate

ARMS: Dict[str, dict] = {
    "reference":                {"label": "Causal reference (leak-free)",     "control": None},
    "ch1_control_window":       {"label": "Ch1 control: window norm, causal", "control": None},
    "ch1_window_norm_leak":     {"label": "Ch1: window norm uses horizon",    "control": "ch1_control_window"},
    "ch2_merge_table_leak":     {"label": "Ch2: merge table on train+test",   "control": "reference"},
    "ch3_decode_table_leak":    {"label": "Ch3: decode tables on train+test", "control": "reference"},
    "ch4_control_context_only": {"label": "Ch4 control: unconstrained BPE, context only", "control": None},
    "ch4_boundary_leak":        {"label": "Ch4: tokenize stream, then split", "control": "ch4_control_context_only"},
    "ch5_noncausal_fft_leak":   {"label": "Ch5: centered (non-causal) gate FFT", "control": "reference"},
    "positive_control":         {"label": "Positive control: true future RV norm", "control": "reference"},
}


@dataclass
class ArmInput:
    tokens: List[int]
    scale: float              # forecasts are made for RV / scale^2 and rescaled by scale^2
    decode_tok_name: str      # which tokenizer's tables decode generated tokens
    known_future: List[int]   # base bins of post-origin bars already inside the input (ch4 leak only)


class LeakageManager:
    def __init__(self, num_bins: int = 32, num_merges: int = 64, bars_per_hour: int = 12,
                 context_len: int = 144, horizon: int = 12, use_gate: bool = True,
                 use_scale_tokens: bool = True, gate_lookback: int = 24, gate_quantile: float = 0.75,
                 short_merges: Optional[int] = None):
        assert context_len % bars_per_hour == 0
        self.C, self.H, self.bph = context_len, horizon, bars_per_hour
        self.use_gate, self.use_scale_tokens = use_gate, use_scale_tokens
        self.short_merges = num_merges // 4 if short_merges is None else short_merges
        kw = dict(num_bins=num_bins, num_merges=num_merges, bars_per_hour=bars_per_hour)
        self.toks: Dict[str, TimeSeriesBPETokenizer] = {
            "causal": TimeSeriesBPETokenizer(enforce_hour_boundaries=True, **kw),
            "unconstrained": TimeSeriesBPETokenizer(enforce_hour_boundaries=False, **kw),
        }
        self.gate_causal = CausalRoVSEIGate(gate_lookback, bars_per_hour, gate_quantile, mode="causal")
        self.gate_centered = CausalRoVSEIGate(gate_lookback, bars_per_hour, gate_quantile, mode="centered")
        self._cache: Dict[tuple, List[int]] = {}

    # ------------------------------------------------------------------ fitting
    def fit(self, df: pd.DataFrame, split: int):
        """Fits everything that must be train-only on df[:split]; leaked tables on all of df."""
        self.r = df["return"].values.astype(np.float64)
        self.ev = df["ewma_vol"].values.astype(np.float64)
        self.xn = df["norm_return"].values.astype(np.float64)
        times = pd.to_datetime(df["open_time"])
        on_hour = times.dt.minute.values == 0
        chunk_starts = on_hour | ~df["contiguous"].values.astype(bool)
        hour_starts = np.flatnonzero(on_hour)

        ct, ut = self.toks["causal"], self.toks["unconstrained"]
        ct.fit_vocabulary(self.xn[:split], chunk_starts[:split]).fit_conditional_decoding(self.xn[:split])
        ut.fit_vocabulary(self.xn[:split]).fit_conditional_decoding(self.xn[:split])

        # Channel 2: merge table learned on train + test; decoding tables stay train-only
        mt = TimeSeriesBPETokenizer(enforce_hour_boundaries=True, num_bins=ct.num_bins,
                                    num_merges=ct.num_merges, bars_per_hour=self.bph)
        mt.fit_vocabulary(self.xn, chunk_starts)
        mt.cond_mean, mt.cond_m2 = ct.cond_mean.copy(), ct.cond_m2.copy()
        self.toks["merge_leaked"] = mt
        # Channel 3: train-only merges, decoding tables learned on train + test
        dt = copy.deepcopy(ct)
        dt.fit_conditional_decoding(self.xn)
        self.toks["decode_leaked"] = dt

        # gates: thresholds from train hours only (windows restricted to the train span)
        train_hours = hour_starts[hour_starts < split]
        self.gate_causal.fit_threshold(self.xn[:split], train_hours)
        self.gate_centered.fit_threshold(self.xn[:split], train_hours)
        self.flags_causal = self.gate_causal.flags_per_bar(self.xn, hour_starts)
        self.flags_centered = self.gate_centered.flags_per_bar(self.xn, hour_starts)
        self._cache.clear()
        return self

    def vocab_size(self) -> int:
        return max(t.vocab_size for t in self.toks.values())

    # ------------------------------------------------------------------ helpers
    def _hour_flags(self, t: int, centered: bool = False) -> Optional[List[bool]]:
        if not self.use_gate:
            return None
        f = self.flags_centered if centered else self.flags_causal
        return [bool(f[s]) for s in range(t - self.C, t, self.bph)]

    def _hourly(self, tok_name: str, t: int, x: Optional[np.ndarray] = None, centered_gate: bool = False) -> List[int]:
        """Hour-aligned encoding of [t-C, t). If x is None the per-bar normalized series is used
        and each hour's encoding is cached (it does not depend on the origin)."""
        tok = self.toks[tok_name]
        flags = self._hour_flags(t, centered_gate)
        out: List[int] = []
        for h, s in enumerate(range(t - self.C, t, self.bph)):
            if self.use_scale_tokens:
                out.append(tok.quantize_scale(self.ev[s]))
            flag = flags[h] if flags is not None else False
            if x is None:
                key = (tok_name, s, flag)
                enc = self._cache.get(key)
                if enc is None:
                    enc = tok.encode(self.xn[s:s + self.bph], include_scale_tokens=False,
                                     gate_flags=[flag], short_merges=self.short_merges)
                    self._cache[key] = enc
            else:
                off = s - (t - self.C)
                enc = tok.encode(x[off:off + self.bph], include_scale_tokens=False,
                                 gate_flags=[flag], short_merges=self.short_merges)
            out.extend(enc)
        return out

    # ------------------------------------------------------------------ main entry
    def extract(self, t: int, arm: str) -> ArmInput:
        C, H, r = self.C, self.H, self.r
        assert t - C >= 0 and t + H <= len(r)
        if arm in ("reference", "ch2_merge_table_leak", "ch3_decode_table_leak", "ch5_noncausal_fft_leak"):
            tok_name = "merge_leaked" if arm == "ch2_merge_table_leak" else "causal"
            toks = self._hourly(tok_name, t, centered_gate=(arm == "ch5_noncausal_fft_leak"))
            dec = {"ch2_merge_table_leak": "merge_leaked", "ch3_decode_table_leak": "decode_leaked"}.get(arm, "causal")
            return ArmInput(toks, float(self.ev[t]), dec, [])

        if arm in ("ch1_control_window", "ch1_window_norm_leak", "positive_control"):
            ctx = r[t - C:t]
            if arm == "ch1_control_window":
                scale = float(np.std(ctx))
            elif arm == "ch1_window_norm_leak":
                scale = float(np.std(r[t - C:t + H]))
            else:
                scale = float(np.sqrt(np.sum(r[t:t + H] ** 2) / H))
            scale = max(scale, 1e-6)
            return ArmInput(self._hourly("causal", t, x=ctx / scale), scale, "causal", [])

        if arm in ("ch4_control_context_only", "ch4_boundary_leak"):
            tok = self.toks["unconstrained"]
            head = [tok.quantize_scale(self.ev[t])] if self.use_scale_tokens else []
            if arm == "ch4_control_context_only":
                return ArmInput(head + tok.encode(self.xn[t - C:t]), float(self.ev[t]), "unconstrained", [])
            stream = tok.encode(self.xn[t - C:t + H])
            kept, covered = [], 0
            for tid in stream:
                if covered >= C:
                    break
                kept.append(tid)
                covered += len(tok.expand(tid))
            overhang = covered - C
            known = list(tok.expand(kept[-1])[-overhang:]) if overhang > 0 else []
            return ArmInput(head + kept, float(self.ev[t]), "unconstrained", known)

        raise ValueError(f"unknown arm {arm}")
