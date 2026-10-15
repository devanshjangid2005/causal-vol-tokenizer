"""
simulation/garch_simulator.py
GARCH(1,1) Data Generator and Oracle Volatility Forecaster.

Provides a synthetic ground-truth benchmark where true conditional volatility
is analytically known at every step. Serves as the negative control:
No causal, leak-free model can beat the oracle QLIKE floor.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any

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
        """
        Simulates returns and conditional variances.
        Returns:
            returns: shape (n_steps,)
            sigma_sq: shape (n_steps,) true conditional variance for each step t
        """
        total_steps = n_steps + burn_in
        eps = self.rng.standard_normal(total_steps)
        
        sigma_sq = np.zeros(total_steps, dtype=np.float64)
        returns = np.zeros(total_steps, dtype=np.float64)
        
        sigma_sq[0] = self.unconditional_var
        returns[0] = np.sqrt(sigma_sq[0]) * eps[0]
        
        for t in range(1, total_steps):
            sigma_sq[t] = self.omega + self.alpha * (returns[t - 1] ** 2) + self.beta * sigma_sq[t - 1]
            returns[t] = np.sqrt(sigma_sq[t]) * eps[t]
            
        return returns[burn_in:], sigma_sq[burn_in:]

    def compute_oracle_rv_forecast(self, returns: np.ndarray, sigma_sq: np.ndarray, horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes:
        1. True future realized variance: sum_{k=1}^H returns[t+k]^2
        2. Analytic oracle forecast at time t: E_t[ sum_{k=1}^H returns[t+k]^2 ]
        
        Formula:
            E_t[sigma_{t+k}^2] = V_inf + (alpha + beta)^{k-1} * (sigma_{t+1}^2 - V_inf)
            where sigma_{t+1}^2 = omega + alpha * returns[t]^2 + beta * sigma_sq[t]
        """
        n = len(returns)
        valid_len = n - horizon
        
        gamma = self.alpha + self.beta
        v_inf = self.unconditional_var
        
        if abs(gamma - 1.0) < 1e-12:
            geom_sum = float(horizon)
        else:
            geom_sum = (1.0 - gamma ** horizon) / (1.0 - gamma)
            
        realized_rv = np.zeros(valid_len, dtype=np.float64)
        oracle_forecast = np.zeros(valid_len, dtype=np.float64)
        
        # Vectorized or rolling sum for realized_rv
        returns_sq = returns ** 2
        for t in range(valid_len):
            realized_rv[t] = np.sum(returns_sq[t + 1 : t + 1 + horizon])
            
            # Information available at t: returns[t] and sigma_sq[t]
            sigma_next_sq = self.omega + self.alpha * (returns[t] ** 2) + self.beta * sigma_sq[t]
            oracle_forecast[t] = horizon * v_inf + (sigma_next_sq - v_inf) * geom_sum
            
        return realized_rv, oracle_forecast

def generate_garch_dataset(n_steps: int = 50000, horizon: int = 12, seed: int = 42) -> pd.DataFrame:
    """
    Generates a full synthetic benchmark dataframe with returns, true sigma_sq,
    realized variance, and oracle forecast.
    """
    sim = GARCHSimulator(seed=seed)
    returns, sigma_sq = sim.simulate(n_steps=n_steps)
    rv, oracle_fc = sim.compute_oracle_rv_forecast(returns, sigma_sq, horizon=horizon)
    
    df = pd.DataFrame({
        "return": returns[:len(rv)],
        "sigma_sq": sigma_sq[:len(rv)],
        "rv_target": rv,
        "oracle_forecast": oracle_fc
    })
    return df

if __name__ == "__main__":
    df = generate_garch_dataset(n_steps=10000, horizon=12)
    print(f"Generated synthetic GARCH dataset: {df.shape}")
    print(df.head())

