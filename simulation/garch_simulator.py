"""
simulation/garch_simulator.py
GARCH(1,1) data generator and analytic oracle forecast (negative control).

Timing convention matches data/preprocess.py: at origin t the oracle knows
returns up to t-1, hence sigma_sq[t] (the true conditional variance of r_t), and
forecasts RV_H[t] = sum_{k=0}^{H-1} r_{t+k}^2:
    E_{t-1}[RV_H[t]] = H * V + (sigma_sq[t] - V) * (1 - g^H) / (1 - g),   g = alpha + beta
No model that only uses the past can have lower expected QLIKE than this.
"""

import numpy as np
import pandas as pd
from typing import Tuple


class GARCHSimulator:
    def __init__(self, omega: float = 1e-5, alpha: float = 0.08, beta: float = 0.90, seed: int = 42):
        assert alpha + beta < 1.0, "Stationarity condition alpha + beta < 1 violated"
        self.omega = omega
        self.alpha = alpha
        self.beta = beta
        self.unconditional_var = omega / (1.0 - alpha - beta)
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def simulate(self, n_steps: int = 100000, burn_in: int = 5000) -> Tuple[np.ndarray, np.ndarray]:
        """Returns (returns, sigma_sq) where sigma_sq[t] is the true conditional variance of returns[t]."""
        total = n_steps + burn_in
        eps = self.rng.standard_normal(total)
        sigma_sq = np.zeros(total)
        returns = np.zeros(total)
        sigma_sq[0] = self.unconditional_var
        returns[0] = np.sqrt(sigma_sq[0]) * eps[0]
        for t in range(1, total):
            sigma_sq[t] = self.omega + self.alpha * returns[t - 1] ** 2 + self.beta * sigma_sq[t - 1]
            returns[t] = np.sqrt(sigma_sq[t]) * eps[t]
        return returns[burn_in:], sigma_sq[burn_in:]


def oracle_rv_forecast(sigma_sq_t: np.ndarray, omega: float, alpha: float, beta: float, horizon: int) -> np.ndarray:
    """Analytic E_{t-1}[sum_{k=0}^{H-1} r_{t+k}^2] given the true sigma_sq[t] at each origin."""
    g = alpha + beta
    v = omega / (1.0 - g)
    geom = (1.0 - g ** horizon) / (1.0 - g)
    return horizon * v + (np.asarray(sigma_sq_t) - v) * geom


if __name__ == "__main__":
    sim = GARCHSimulator(omega=5e-8, alpha=0.05, beta=0.93)
    r, s = sim.simulate(20000)
    print("per-bar sd:", r.std(), "unconditional sd:", np.sqrt(sim.unconditional_var))
