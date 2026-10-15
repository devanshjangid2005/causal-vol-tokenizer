"""
data/preprocess.py
Feature Engineering & Target Generation for High-Frequency Crypto Volatility:
1. Log Returns: r_t = log(close_t / close_{t-1})
2. Trailing EWMA Volatility (Strictly Causal - RiskMetrics lambda=0.94)
3. Normalized Returns: r_norm_t = r_t / sigma_ewma_t
4. Realized Variance Targets:
   - Next-Hour RV (H = 12 bars): sum_{k=1}^{12} r_{t+k}^2
   - Next-Day RV (H = 288 bars): sum_{k=1}^{288} r_{t+k}^2
5. Causal RoV and SEI signals
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import Tuple

def compute_trailing_ewma_vol(returns: np.ndarray, decay: float = 0.94, initial_vol: float = 0.005) -> np.ndarray:
    """
    Computes strictly trailing EWMA variance:
    sigma_sq[t] = decay * sigma_sq[t-1] + (1 - decay) * returns[t-1]^2
    sigma_sq[t] is available at the START of bar t (no future data!).
    """
    n = len(returns)
    sigma_sq = np.zeros(n, dtype=np.float64)
    sigma_sq[0] = initial_vol ** 2
    
    for t in range(1, n):
        sigma_sq[t] = decay * sigma_sq[t - 1] + (1.0 - decay) * (returns[t - 1] ** 2)
        
    return np.sqrt(np.maximum(sigma_sq, 1e-12))

def build_dataset(df_raw: pd.DataFrame, decay: float = 0.94) -> pd.DataFrame:
    """
    Takes raw Binance dataframe with open_time and close prices,
    computes causal returns, EWMA vol, normalized returns, and RV targets.
    """
    df = df_raw.copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").reset_index(drop=True)
    
    close_prices = df["close"].values
    returns = np.zeros(len(close_prices), dtype=np.float64)
    returns[1:] = np.log(close_prices[1:] / np.maximum(close_prices[:-1], 1e-8))
    
    ewma_vol = compute_trailing_ewma_vol(returns, decay=decay)
    norm_returns = returns / np.maximum(ewma_vol, 1e-6)
    
    # Realized variance targets
    returns_sq = returns ** 2
    n = len(returns)
    
    # Precompute prefix sum for O(1) rolling sums
    cumsum_sq = np.zeros(n + 1, dtype=np.float64)
    cumsum_sq[1:] = np.cumsum(returns_sq)
    
    def future_rv(h: int) -> np.ndarray:
        rv = np.full(n, np.nan, dtype=np.float64)
        for t in range(n - h):
            rv[t] = cumsum_sq[t + 1 + h] - cumsum_sq[t + 1]
        return rv

    rv_1h = future_rv(12)   # Next 12 bars (1 hour)
    rv_1d = future_rv(288)  # Next 288 bars (1 day)
    
    out_df = pd.DataFrame({
        "open_time": df["open_time"],
        "close": close_prices,
        "return": returns,
        "ewma_vol": ewma_vol,
        "norm_return": norm_returns,
        "rv_target_1h": rv_1h,
        "rv_target_1d": rv_1d
    })
    
    return out_df

if __name__ == "__main__":
    raw_df = pd.read_csv("data/btcusdt_5m_sample.csv")
    processed = build_dataset(raw_df)
    print(f"Processed shape: {processed.shape}")
    print(processed.dropna().head())

