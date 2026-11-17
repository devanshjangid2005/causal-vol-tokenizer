"""
models/baselines.py
Econometric baselines, scored on exactly the same origins and target as the GPT.

At origin t every model may use returns up to t-1 and forecasts
RV_H[t] = sum_{k=0}^{H-1} r_{t+k}^2   (see data/preprocess.py).

1. HAR-RV (Corsi 2009) on 5-minute bars: lags of 1 hour (12), 1 day (288) and
   1 week (2016 = 7 x 288, crypto trades 24/7).
2. GARCH(1,1) (Bollerslev 1986) fitted by constrained maximum likelihood on the train
   period, then filtered causally over the whole series with the fitted parameters.
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
    """
    Gaussian GARCH(1,1) by maximum likelihood with stationarity enforced by construction:
    alpha = p * s, beta = p * (1 - s), p = 0.9999 * sigmoid(.), s = sigmoid(.), omega = exp(.).
    (arch's optimizer failed to converge on Fold 6 and returned alpha + beta = 1.01; this
    estimator matches arch where arch converges, e.g. Fold 1: alpha 0.1049, beta 0.8834.)
    The variance recursion is evaluated with scipy.signal.lfilter, so a fit takes ~1 s.
    """
    MAX_PERSISTENCE = 0.9999

    def __init__(self, scale: float = 1000.0):
        self.scale = scale
        self.omega, self.alpha, self.beta = 1e-7, 0.05, 0.93
        self.fitted = False
        self.at_persistence_cap = False

    def _unpack(self, t):
        p = self.MAX_PERSISTENCE / (1.0 + np.exp(-t[1]))
        share = 1.0 / (1.0 + np.exp(-t[2]))
        return np.exp(t[0]), p * share, p * (1.0 - share)

    def _nll(self, t, r2, v0):
        from scipy.signal import lfilter
        omega, a, b = self._unpack(t)
        x = omega + a * np.concatenate([[v0], r2[:-1]])
        s2, _ = lfilter([1.0], [1.0, -b], x, zi=[b * v0])
        s2 = np.maximum(s2, 1e-12)
        return 0.5 * np.sum(np.log(s2) + r2 / s2)

    def fit(self, returns_train: np.ndarray):
        from scipy.optimize import minimize
        r2 = (np.asarray(returns_train, dtype=np.float64) * self.scale) ** 2
        v0 = float(r2.mean())
        best = None
        for a, b in [(0.05, 0.93), (0.10, 0.88), (0.15, 0.80), (0.03, 0.96)]:
            p = (a + b) / self.MAX_PERSISTENCE
            t0 = np.array([np.log(v0 * (1 - a - b)), np.log(p / (1 - p)), np.log(a / b)])
            res = minimize(self._nll, t0, args=(r2, v0), method="L-BFGS-B")
            if best is None or res.fun < best.fun:
                best = res
        omega, self.alpha, self.beta = self._unpack(best.x)
        self.omega = float(omega / self.scale ** 2)
        self.fitted = bool(best.success)
        self.alpha, self.beta = float(self.alpha), float(self.beta)
        self.at_persistence_cap = bool((self.alpha + self.beta) > self.MAX_PERSISTENCE - 1e-4)
        if not best.success:
            print(f"Warning: GARCH MLE did not converge ({best.message})")
        if self.at_persistence_cap:
            print("Note: GARCH persistence is at the 0.9999 cap (near-integrated volatility in this training window)")
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
