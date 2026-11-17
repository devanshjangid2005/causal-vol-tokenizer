"""
tokenization/causal_gate.py
RoV x SEI gate inspired by PATK (Liao et al., AAAI-26).

PATK definitions (Eq. 1-4), adapted to a per-hour decision:
  * RoV (rate of variation): local gradient magnitude |x_{l+1} - x_l|, here averaged
    over the window -> how volatile the recent path is.
  * SEI (spectral energy intensity): energy of the spectrum in a band of radius eps
    around a dominant frequency w_k, here normalized by total (non-DC) energy -> how
    concentrated the recent path is around one periodic component.
  * PATK samples token start points with probability proportional to RoV * SEI
    (Eq. 4). We use the same product as a score and, instead of learned elastic
    lengths, apply a threshold: hours whose score exceeds the train-set quantile get
    shorter tokens (only the first quarter of the merge table).

Causal mode (reference): the score for the hour starting at bar s uses bars [s-L, s).
Centered mode (Channel 5 leak): the window is centered on the hour being tokenized,
[s + H/2 - L/2, s + H/2 + L/2), so for the last hour before a forecast origin it
reaches H/2 bars past the origin. This is the "non-causal FFT" of the proposal.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np


def rov(window: np.ndarray) -> float:
    return float(np.mean(np.abs(np.diff(window)))) if len(window) > 1 else 0.0


def sei(window: np.ndarray, band: int = 1) -> float:
    """Share of non-DC spectral energy within +-band bins of the dominant frequency."""
    if len(window) < 4:
        return 0.0
    power = np.abs(np.fft.rfft(window - window.mean())) ** 2
    power = power[1:]
    total = power.sum()
    if total <= 1e-18:
        return 0.0
    k = int(np.argmax(power))
    return float(power[max(0, k - band): k + band + 1].sum() / total)


class CausalRoVSEIGate:
    def __init__(self, lookback_window: int = 24, bars_per_hour: int = 12,
                 volatility_threshold_quantile: float = 0.75, mode: str = "causal"):
        assert mode in ("causal", "centered")
        self.L = lookback_window
        self.H = bars_per_hour
        self.q = volatility_threshold_quantile
        self.mode = mode
        self.threshold = np.inf

    def _window(self, x: np.ndarray, s: int) -> np.ndarray:
        if self.mode == "causal":
            a, b = s - self.L, s
        else:
            c = s + self.H // 2
            a, b = c - self.L // 2, c + self.L // 2
        if a < 0 or b > len(x):
            return np.empty(0)
        return x[a:b]

    def score(self, x: np.ndarray, s: int) -> float:
        w = self._window(x, s)
        return rov(w) * sei(w) if len(w) >= 4 else np.nan

    def scores(self, x: np.ndarray, hour_starts: np.ndarray) -> np.ndarray:
        return np.array([self.score(x, int(s)) for s in hour_starts])

    def fit_threshold(self, x: np.ndarray, hour_starts: np.ndarray):
        sc = self.scores(x, hour_starts)
        sc = sc[np.isfinite(sc)]
        self.threshold = float(np.quantile(sc, self.q)) if len(sc) else np.inf
        return self

    def flags_per_bar(self, x: np.ndarray, hour_starts: np.ndarray) -> np.ndarray:
        """Bool array over bars: True at hour starts whose hour is gated (volatile)."""
        out = np.zeros(len(x), dtype=bool)
        sc = self.scores(x, hour_starts)
        out[hour_starts] = np.nan_to_num(sc, nan=-np.inf) > self.threshold
        return out

    def evaluate_hour_gate(self, returns_up_to_t: np.ndarray) -> bool:
        """Single decision for the hour starting right after `returns_up_to_t` (causal mode)."""
        w = returns_up_to_t[-self.L:]
        return len(w) >= 4 and rov(w) * sei(w) > self.threshold
