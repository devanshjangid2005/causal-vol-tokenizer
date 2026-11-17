"""
models/baselines.py
Econometric baselines, scored on exactly the same origins and target as the GPT.

At origin t every model may use returns up to t-1 and forecasts
RV_H[t] = sum_{k=0}^{H-1} r_{t+k}^2   (see data/preprocess.py).

1. HAR-RV (Corsi 2009) on 5-minute bars: lags of 1 hour (12), 1 day (288) and
   1 week (2016 = 7 x 288, crypto trades 24/7).
2. GARCH(1,1) (Bollerslev 1986) fitted with `arch` on the train period, then
   filtered causally over the whole series with the fitted parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from typing import Tuple


def _cumsum_sq(returns: np.ndarray) -> np.ndarray:
    cs = np.zeros(len(returns) + 1)
    cs[1:] = np.cumsum(returns ** 2)
    return cs


class HARRVModel:
    def __init__(self, lags: Tuple[int, int, int] = (12, 288, 2016)):
        self.lags = lags
        self.max_lag = max(lags)
        self.weights = None

    def _features(self, cs: np.ndarray, origins: np.ndarray) -> np.ndarray:
        cols = [np.ones(len(origins))]
        for lag in self.lags:
            cols.append(cs[origins] - cs[origins - lag])  # sum of r^2 over [t-lag, t)
        return np.column_stack(cols)

    def fit(self, returns: np.ndarray, origins: np.ndarray, horizon: int = 12):
        origins = origins[origins >= self.max_lag]
        cs = _cumsum_sq(returns)
        X = self._features(cs, origins)
        y = cs[origins + horizon] - cs[origins]
        self.weights, *_ = np.linalg.lstsq(X, y, rcond=None)
        return self

    def predict(self, returns: np.ndarray, origins: np.ndarray, horizon: int = 12) -> np.ndarray:
        assert np.all(origins >= self.max_lag), "HAR needs max_lag bars of history before each origin"
        cs = _cumsum_sq(returns)
        return np.maximum(self._features(cs, origins) @ self.weights, 1e-12)


class GARCH11Model:
    def __init__(self):
        self.omega, self.alpha, self.beta = 1e-7, 0.05, 0.93
        self.fitted = False

    def fit(self, returns_train: np.ndarray):
        from arch import arch_model
        try:
            # 5-minute returns are ~1e-3, so scale by 1000 to put the variance in arch's preferred range
            res = arch_model(returns_train * 1000.0, vol="Garch", p=1, q=1, mean="Zero", dist="Normal",
                             rescale=False).fit(disp="off", show_warning=False)
            if res.convergence_flag != 0:
                print(f"Warning: GARCH optimizer did not converge (flag {res.convergence_flag})")
            p = res.params
            self.omega = float(p["omega"]) / 1e6
            self.alpha = float(p["alpha[1]"])
            self.beta = float(p["beta[1]"])
            self.fitted = True
        except Exception as e:  # keep defaults but say so
            print(f"GARCH fit failed ({e}); using default parameters")
        return self

    def filter(self, returns: np.ndarray) -> np.ndarray:
        """sigma_sq[t] = omega + alpha r_{t-1}^2 + beta sigma_sq[t-1]: uses data up to t-1 only."""
        g = self.alpha + self.beta
        v = self.omega / (1.0 - g) if g < 1 else float(np.var(returns))
        s = np.empty(len(returns))
        s[0] = v
        for t in range(1, len(returns)):
            s[t] = self.omega + self.alpha * returns[t - 1] ** 2 + self.beta * s[t - 1]
        return s

    def predict(self, returns: np.ndarray, origins: np.ndarray, horizon: int = 12) -> np.ndarray:
        g = self.alpha + self.beta
        s = self.filter(returns)
        if g >= 1.0:
            return np.maximum(horizon * s[origins], 1e-12)
        v = self.omega / (1.0 - g)
        geom = (1.0 - g ** horizon) / (1.0 - g)
        return np.maximum(horizon * v + (s[origins] - v) * geom, 1e-12)


if __name__ == "__main__":
    from data.folds import load_fold, select_origins
    from evaluation.metrics import qlike_loss
    df, split, _ = load_fold(synthetic=True, n_train_synth=30000, n_test_synth=8000)
    r = df["return"].values
    tr = select_origins(df, 0, split, 12, 144, min_history=2016)
    te = select_origins(df, split, len(df), 12, 144, min_history=2016)
    y = df["rv_target_1h"].values[te]
    har = HARRVModel().fit(r, tr)
    g = GARCH11Model().fit(r[:split])
    print("HAR QLIKE  ", qlike_loss(y, har.predict(r, te)).mean())
    print("GARCH QLIKE", qlike_loss(y, g.predict(r, te)).mean())
