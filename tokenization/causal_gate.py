"""
tokenization/causal_gate.py
Causal RoV x SEI Gate inspired by PATK (Liao et al., 2026).

Computes Rate-of-Variation (RoV) and Spectral Energy Intensity (SEI) on
STRICTLY TRAILING WINDOWS to adaptively modulate token length / motif granularity.
Ensures zero future leakage in tokenization boundaries.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple, Optional

class CausalRoVSEIGate:
    def __init__(
        self,
        lookback_window: int = 24, # trailing 2 hours of 5-min bars
        high_freq_cutoff: float = 0.25, # fraction of spectrum considered high-frequency
        volatility_threshold_quantile: float = 0.75
    ):
        self.lookback_window = lookback_window
        self.high_freq_cutoff = high_freq_cutoff
        self.volatility_threshold_quantile = volatility_threshold_quantile
        self.threshold = 1.0 # default adaptive threshold

    def compute_rov(self, trailing_returns: np.ndarray) -> float:
        """
        RoV (Rate-of-Variation): Mean absolute difference of successive returns
        quantifying local gradient dynamics in the trailing window.
        """
        if len(trailing_returns) < 2:
            return 1e-4
        diffs = np.abs(np.diff(trailing_returns))
        return float(np.mean(diffs))

    def compute_sei(self, trailing_returns: np.ndarray) -> float:
        """
        SEI (Spectral Energy Intensity): Proportion of spectral energy
        concentrated in high-frequency modes via FFT.
        """
        if len(trailing_returns) < 4:
            return 0.5
        fft_vals = np.fft.rfft(trailing_returns)
        power = np.abs(fft_vals) ** 2
        total_power = np.sum(power)
        if total_power < 1e-12:
            return 0.0
            
        n_freqs = len(power)
        cutoff_idx = int(np.floor(n_freqs * (1.0 - self.high_freq_cutoff)))
        high_freq_power = np.sum(power[cutoff_idx:])
        return float(high_freq_power / total_power)

    def fit_threshold(self, historical_returns: np.ndarray):
        """
        Calibrates the gate threshold on training data.
        """
        scores = []
        n = len(historical_returns)
        step = 12 # evaluate at each hour
        for t in range(self.lookback_window, n, step):
            trailing = historical_returns[t - self.lookback_window : t]
            rov = self.compute_rov(trailing)
            sei = self.compute_sei(trailing)
            scores.append(rov * (1.0 + sei))
            
        if scores:
            self.threshold = float(np.quantile(scores, self.volatility_threshold_quantile))

    def evaluate_hour_gate(self, returns_up_to_t: np.ndarray) -> bool:
        """
        Evaluates gate for the upcoming hour using strictly trailing data.
        Returns:
            True if high volatility (shorten tokens / restrict merges)
            False if normal volatility (standard compression)
        """
        if len(returns_up_to_t) < self.lookback_window:
            return False
            
        trailing = returns_up_to_t[-self.lookback_window:]
        rov = self.compute_rov(trailing)
        sei = self.compute_sei(trailing)
        score = rov * (1.0 + sei)
        return bool(score > self.threshold)

if __name__ == "__main__":
    from simulation.garch_simulator import GARCHSimulator
    sim = GARCHSimulator()
    ret, _ = sim.simulate(n_steps=2000)
    gate = CausalRoVSEIGate(lookback_window=24)
    gate.fit_threshold(ret[:1000])
    is_volatile = gate.evaluate_hour_gate(ret[900:1000])
    print(f"Gate threshold: {gate.threshold:.6f}, Is upcoming hour volatile: {is_volatile}")

