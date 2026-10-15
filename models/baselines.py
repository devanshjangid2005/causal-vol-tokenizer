"""
models/baselines.py
Econometric Baselines for Volatility Forecasting:
1. HAR-RV (Heterogeneous Autoregressive model of Realized Volatility - Corsi 2009)
2. GARCH(1,1) (Bollerslev 1986 via arch library)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from arch import arch_model
from typing import Tuple, Dict, Any, Optional

class HARRVModel:
    """
    Corsi (2009) HAR-RV model adapted for high-frequency 5-minute returns.
    Regresses future RV onto short-term (1h = 12 bars), medium-term (1d = 288 bars),
    and long-term (1w = 1440 or 2016 bars) trailing realized variances.
    """
    def __init__(self, lags: Tuple[int, int, int] = (12, 288, 1440)):
        self.lag_h, self.lag_d, self.lag_w = lags
        self.max_lag = max(lags)
        self.weights = None # [intercept, beta_h, beta_d, beta_w]
        
    def _create_features(self, returns_sq: np.ndarray, horizon: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = len(returns_sq)
        valid_indices = []
        X_list = []
        y_list = []
        
        # Precompute cumulative sums for fast rolling sums
        cumsum = np.zeros(n + 1, dtype=np.float64)
        cumsum[1:] = np.cumsum(returns_sq)
        
        def range_sum(start, end):
            return cumsum[end] - cumsum[start]
            
        start_t = self.max_lag
        end_t = n - horizon
        
        for t in range(start_t, end_t):
            rv_h = range_sum(t - self.lag_h, t)
            rv_d = range_sum(t - self.lag_d, t)
            rv_w = range_sum(t - self.lag_w, t)
            
            y = range_sum(t, t + horizon)
            
            X_list.append([1.0, rv_h, rv_d, rv_w])
            y_list.append(y)
            valid_indices.append(t)
            
        return np.array(X_list), np.array(y_list), np.array(valid_indices)

    def fit(self, returns: np.ndarray, horizon: int = 12):
        returns_sq = returns ** 2
        X, y, _ = self._create_features(returns_sq, horizon)
        # OLS with non-negativity regularization or standard pseudoinverse
        self.weights, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        # Ensure positive intercept to guarantee positive variance forecasts
        if self.weights[0] < 1e-8:
            self.weights[0] = 1e-8
        return self

    def predict(self, returns: np.ndarray, horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns: (predictions, true_targets)
        """
        returns_sq = returns ** 2
        X, y, indices = self._create_features(returns_sq, horizon)
        preds = np.maximum(X @ self.weights, 1e-8)
        return preds, y


class GARCH11Model:
    """
    Standard GARCH(1,1) model for returns.
    """
    def __init__(self, p: int = 1, q: int = 1):
        self.p = p
        self.q = q
        self.model_res = None
        self.omega = 1e-5
        self.alpha = 0.08
        self.beta = 0.90
        
    def fit(self, returns: np.ndarray):
        # Scale returns by 100 for numerical stability in optimizer
        scaled_returns = returns * 100.0
        try:
            am = arch_model(scaled_returns, vol='Garch', p=self.p, q=self.q, mean='Zero', dist='Normal')
            self.model_res = am.fit(disp='off', show_warning=False)
            params = self.model_res.params
            self.omega = float(params['omega']) / 10000.0
            self.alpha = float(params['alpha[1]'])
            self.beta = float(params['beta[1]'])
        except Exception as e:
            # Fallback to robust defaults if optimizer fails
            pass
        return self

    def predict(self, returns: np.ndarray, horizon: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """
        Multi-step ahead analytical cumulative variance forecast.
        """
        n = len(returns)
        returns_sq = returns ** 2
        gamma = self.alpha + self.beta
        v_inf = self.omega / max(1e-6, 1.0 - gamma) if gamma < 1.0 else 1e-4
        
        if abs(gamma - 1.0) < 1e-12:
            geom_sum = float(horizon)
        else:
            geom_sum = (1.0 - gamma ** horizon) / (1.0 - gamma)
            
        # Filter variances causally
        sigma_sq = np.zeros(n, dtype=np.float64)
        sigma_sq[0] = v_inf
        for t in range(1, n):
            sigma_sq[t] = self.omega + self.alpha * returns_sq[t - 1] + self.beta * sigma_sq[t - 1]
            
        valid_len = n - horizon
        preds = np.zeros(valid_len, dtype=np.float64)
        targets = np.zeros(valid_len, dtype=np.float64)
        
        for t in range(valid_len):
            targets[t] = np.sum(returns_sq[t + 1 : t + 1 + horizon])
            sigma_next_sq = self.omega + self.alpha * returns_sq[t] + self.beta * sigma_sq[t]
            preds[t] = max(1e-8, horizon * v_inf + (sigma_next_sq - v_inf) * geom_sum)
            
        return preds, targets

if __name__ == "__main__":
    from simulation.garch_simulator import GARCHSimulator
    sim = GARCHSimulator()
    ret, sig = sim.simulate(n_steps=5000)
    har = HARRVModel()
    har.fit(ret[:3000], horizon=12)
    preds_har, y_har = har.predict(ret[3000:], horizon=12)
    print("HAR test preds mean:", np.mean(preds_har), "target mean:", np.mean(y_har))
