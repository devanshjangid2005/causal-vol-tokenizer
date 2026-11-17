"""
data/folds.py
Walk-forward folds, origin selection and the synthetic GARCH stand-in.

Each fold is loaded as ONE contiguous frame (train months followed by test months)
plus the index `split` of the first test bar. Keeping the frame contiguous means:
  * trailing EWMA does not restart at the test boundary,
  * test-time context windows and HAR lags can reach back into the train period
    (that is past data, so it is allowed),
  * train origins are purged so their targets never touch the test period
    (t + H <= split).

Origins are chosen by TIMESTAMP (minute == 0), not by array index, and an origin is
skipped if any bar in [t - context, t + H) is missing. This keeps the hour-boundary
tokenizer aligned to real clock hours even when Binance has outages.

The 2025 holdout is NOT part of any fold and is not downloaded by this code.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional

from data.preprocess import build_dataset

FOLDS_SPEC = [
    {"fold": 1, "name": "Fold 1 (2021 -> 2022-Q1)", "train_start": (2021, 1), "train_end": (2021, 12), "test_start": (2022, 1), "test_end": (2022, 3)},
    {"fold": 2, "name": "Fold 2 (2021-2022-Q1 -> 2022-Q2)", "train_start": (2021, 1), "train_end": (2022, 3), "test_start": (2022, 4), "test_end": (2022, 6)},
    {"fold": 3, "name": "Fold 3 (2021-2022-H1 -> 2022-H2)", "train_start": (2021, 1), "train_end": (2022, 6), "test_start": (2022, 7), "test_end": (2022, 12)},
    {"fold": 4, "name": "Fold 4 (2021-2022 -> 2023-H1)", "train_start": (2021, 1), "train_end": (2022, 12), "test_start": (2023, 1), "test_end": (2023, 6)},
    {"fold": 5, "name": "Fold 5 (2021-2023-H1 -> 2023-H2)", "train_start": (2021, 1), "train_end": (2023, 6), "test_start": (2023, 7), "test_end": (2023, 12)},
    {"fold": 6, "name": "Fold 6 (2021-2023 -> 2024)", "train_start": (2021, 1), "train_end": (2023, 12), "test_start": (2024, 1), "test_end": (2024, 12)},
]

# Final, score-once evaluation: train on every fold year, test on 2025.
HOLDOUT_SPEC = {"fold": "holdout", "name": "Holdout (2021-2024 -> 2025)", "train_start": (2021, 1), "train_end": (2024, 12),
                "test_start": (2025, 1), "test_end": (2025, 12)}
SEALED_HOLDOUT_SPEC = HOLDOUT_SPEC  # backward-compatible name


def _synthetic_fold(n_train: int, n_test: int, seed: int = 42) -> Tuple[pd.DataFrame, int, dict]:
    """GARCH(1,1) data with 5-minute timestamps, used for smoke tests and the negative control."""
    from simulation.garch_simulator import GARCHSimulator
    # 5-minute-scale parameters: per-bar sd ~ 1.6e-3, persistence 0.98
    sim = GARCHSimulator(omega=5e-8, alpha=0.05, beta=0.93, seed=seed)
    returns, sigma_sq = sim.simulate(n_steps=n_train + n_test)
    close = 30000.0 * np.exp(np.cumsum(returns))
    # first return in build_dataset is log(close[1]/close[0]) = returns[1], so prepend one bar
    close = np.concatenate([[30000.0], close])
    times = pd.date_range("2021-01-01", periods=len(close), freq="5min")
    df = build_dataset(pd.DataFrame({"open_time": times, "close": close}))
    sigma_full = np.concatenate([[np.nan], sigma_sq])  # sigma_full[t] is the true var of return[t]
    df["true_sigma_sq"] = sigma_full
    params = {"omega": sim.omega, "alpha": sim.alpha, "beta": sim.beta, "unconditional_var": sim.unconditional_var}
    return df, n_train + 1, params


def load_fold(fold_idx: int = 1, synthetic: bool = False, save_dir: str = "data/processed",
              n_train_synth: int = 105_120, n_test_synth: int = 25_920, seed: int = 42):
    """
    Returns (df, split) where df is contiguous and df.iloc[split] is the first test bar.
    With synthetic=True returns (df, split, garch_params) using simulated GARCH returns.
    """
    if synthetic:
        return _synthetic_fold(n_train_synth, n_test_synth, seed=seed)

    spec = FOLDS_SPEC[fold_idx - 1]
    return load_spec(spec, f"fold_{fold_idx}_full.parquet", save_dir)


def load_holdout(save_dir: str = "data/processed"):
    """(df, split) for the 2025 holdout. Only experiments/run_holdout.py should call this."""
    return load_spec(HOLDOUT_SPEC, "holdout_full.parquet", save_dir)


def load_spec(spec: dict, cache_name: str, save_dir: str = "data/processed"):
    from data.download_binance import download_range
    os.makedirs(save_dir, exist_ok=True)
    cache = os.path.join(save_dir, cache_name)
    df = None
    if os.path.exists(cache):
        try:
            df = pd.read_parquet(cache)
        except Exception:
            df = None
    if df is None:
        # download_range reuses any monthly CSVs already in data/raw
        print(f"Loading raw data for {spec['name']}...")
        raw = download_range(spec["train_start"][0], spec["train_start"][1],
                             spec["test_end"][0], spec["test_end"][1])
        df = build_dataset(raw)
        try:
            df.to_parquet(cache)
        except Exception as e:  # pyarrow missing: just skip the cache
            print(f"(not caching fold parquet: {e})")
    test_start = pd.Timestamp(year=spec["test_start"][0], month=spec["test_start"][1], day=1)
    split = int(np.searchsorted(df["open_time"].values, np.datetime64(test_start)))
    return df, split


def load_or_fetch_fold_data(fold_idx: int = 1, save_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Backward-compatible wrapper: (train_rows, test_rows) cut from the contiguous frame."""
    df, split = load_fold(fold_idx, save_dir=save_dir)
    return df.iloc[:split].reset_index(drop=True), df.iloc[split:].reset_index(drop=True)


def select_origins(df: pd.DataFrame, start: int, end: int, horizon: int, context_len: int,
                   step_hours: int = 1, min_history: int = 0) -> np.ndarray:
    """
    Forecast origins t with:
      * start <= t and t + horizon <= end   (end = split purges train targets from the test period)
      * t - max(context_len, min_history) >= 0
      * open_time[t] on a clock hour (minute == 0), every `step_hours` hours
      * no missing bar inside [t - context_len, t + horizon)
    """
    times = pd.to_datetime(df["open_time"])
    on_hour = (times.dt.minute.values == 0)
    cont = df["contiguous"].values.astype(np.int64)
    # bad_cum[i] = number of gap bars in [0, i)
    bad_cum = np.concatenate([[0], np.cumsum(1 - cont)])
    lo = max(start, context_len, min_history)
    cand = np.arange(lo, end - horizon + 1)
    cand = cand[on_hour[cand]]
    # gaps are flagged on the bar AFTER the gap, so check bars (t-context, t+horizon)
    a = cand - context_len + 1
    b = cand + horizon
    ok = (bad_cum[b] - bad_cum[a]) == 0
    cand = cand[ok]
    if step_hours > 1:
        hours = (times.values[cand] - times.values[cand[0]]) // np.timedelta64(1, "h") if len(cand) else cand
        cand = cand[(hours % step_hours) == 0]
    return cand


if __name__ == "__main__":
    for f in FOLDS_SPEC:
        print(f"Fold {f['fold']}: {f['name']}")
    df, split, _ = load_fold(synthetic=True, n_train_synth=5000, n_test_synth=2000)
    o = select_origins(df, split, len(df), 12, 144)
    print("synthetic test origins:", len(o), "first:", df["open_time"].iloc[o[0]])
