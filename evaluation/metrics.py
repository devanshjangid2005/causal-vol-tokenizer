"""
evaluation/metrics.py
Volatility Forecasting Loss Functions and Evaluation Metrics.
Implements Patton (2011) QLIKE loss, MSE, RMSE, and MAE.
"""

import numpy as np

def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Computes elementwise QLIKE loss:
    QLIKE(y, y_hat) = y / y_hat - log(y / y_hat) - 1
    
    This is strictly non-negative and strictly minimized when y_hat = E[y].
    """
    y = np.maximum(y_true, eps)
    y_hat = np.maximum(y_pred, eps)
    ratio = y / y_hat
    return ratio - np.log(ratio) - 1.0

def mean_qlike(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean(qlike_loss(y_true, y_pred, eps=eps)))

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean((y_true - y_pred) ** 2))

def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))

def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "QLIKE": mean_qlike(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "MAE": mae(y_true, y_pred)
    }

if __name__ == "__main__":
    y = np.array([0.001, 0.002, 0.005])
    y_hat = np.array([0.0012, 0.0019, 0.0048])
    print("Metrics test:", compute_all_metrics(y, y_hat))

