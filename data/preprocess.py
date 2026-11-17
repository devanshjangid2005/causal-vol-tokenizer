"""
data/preprocess.py
Feature engineering and target generation for 5-minute crypto volatility.

Timing convention (used by EVERY model in this repo):
    A forecast made at origin t may use bars 0 .. t-1 only.
    Its target is the realized variance of the next H bars:
        RV_H[t] = sum_{k=0}^{H-1} r_{t+k}^2
So the information set and the target never share a bar, and HAR-RV, GARCH,
the oracle and the GPT are all scored against the same RV_H[t].

Columns produced:
    open_time, close, return, ewma_vol, norm_return, contiguous,
    rv_target_1h (H=12), rv_target_1d (H=288)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

BAR = pd.Timedelta(minutes=5)


def compute_trailing_ewma_vol(returns: np.ndarray, decay: float = 0.94, initial_vol: float = 0.005) -> np.ndarray:
    """
    Strictly trailing EWMA volatility:
        sigma_sq[t] = decay * sigma_sq[t-1] + (1 - decay) * r[t-1]^2
    sigma[t] uses returns up to t-1 only, so it is known at origin t.
    """
    n = len(returns)
    sigma_sq = np.zeros(n, dtype=np.float64)
    sigma_sq[0] = initial_vol ** 2
    for t in range(1, n):
        sigma_sq[t] = decay * sigma_sq[t - 1] + (1.0 - decay) * (returns[t - 1] ** 2)
    return np.sqrt(np.maximum(sigma_sq, 1e-12))


def future_rv(returns: np.ndarray, horizon: int) -> np.ndarray:
    """RV_H[t] = sum_{k=0}^{H-1} r_{t+k}^2 ; NaN where the window runs past the end."""
    n = len(returns)
    cs = np.zeros(n + 1, dtype=np.float64)
    cs[1:] = np.cumsum(returns ** 2)
    rv = np.full(n, np.nan, dtype=np.float64)
    if n >= horizon:
        rv[: n - horizon + 1] = cs[horizon:] - cs[: n - horizon + 1]
    return rv


def build_dataset(df_raw: pd.DataFrame, decay: float = 0.94) -> pd.DataFrame:
    """
    Builds the feature frame from a CONTIGUOUS raw kline frame covering train and test.
    Computing EWMA once over the whole span keeps it continuous across the split
    (it only ever looks backwards, so this is causal).
    """
    df = df_raw.copy()
    df["open_time"] = pd.to_datetime(df["open_time"])
    df = df.sort_values("open_time").drop_duplicates("open_time").reset_index(drop=True)

    close = df["close"].values.astype(np.float64)
    returns = np.zeros(len(close), dtype=np.float64)
    returns[1:] = np.log(close[1:] / np.maximum(close[:-1], 1e-8))

    # contiguous[t] is False when bar t does not directly follow bar t-1 (exchange outage)
    gaps = df["open_time"].diff().values
    contiguous = np.ones(len(df), dtype=bool)
    contiguous[1:] = gaps[1:] == np.timedelta64(5, "m")
    returns[~contiguous] = 0.0  # a return across a gap is not a 5-minute return

    ewma_vol = compute_trailing_ewma_vol(returns, decay=decay)
    norm_returns = returns / np.maximum(ewma_vol, 1e-6)

    return pd.DataFrame({
        "open_time": df["open_time"],
        "close": close,
        "return": returns,
        "ewma_vol": ewma_vol,
        "norm_return": norm_returns,
        "contiguous": contiguous,
        "rv_target_1h": future_rv(returns, 12),
        "rv_target_1d": future_rv(returns, 288),
    })


if __name__ == "__main__":
    from data.folds import load_fold
    df, split = load_fold(1, synthetic=True)
    print(df.head())
    print("split index:", split)
