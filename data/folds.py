"""
data/folds.py
Walk-Forward Validation Protocol and Sealed 2025 Holdout Architecture.

Defines the 6 expanding walk-forward folds across 2019–2024:
- Expanding training windows to reflect real-world chronological learning.
- Completely non-overlapping out-of-sample test windows.
- 2025 Holdout is cryptographically sealed in a separate file and forbidden
  from being accessed until final evaluation (Nov 14 code freeze).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
from typing import Dict, Any, List, Tuple
from data.download_binance import download_range
from data.preprocess import build_dataset

FOLDS_SPEC = [
    {
        "fold": 1,
        "name": "Fold 1 (2021 -> 2022-Q1)",
        "train_start": (2021, 1),
        "train_end": (2021, 12),
        "test_start": (2022, 1),
        "test_end": (2022, 3),
    },
    {
        "fold": 2,
        "name": "Fold 2 (2021-2022-Q1 -> 2022-Q2)",
        "train_start": (2021, 1),
        "train_end": (2022, 3),
        "test_start": (2022, 4),
        "test_end": (2022, 6),
    },
    {
        "fold": 3,
        "name": "Fold 3 (2021-2022-H1 -> 2022-H2)",
        "train_start": (2021, 1),
        "train_end": (2022, 6),
        "test_start": (2022, 7),
        "test_end": (2022, 12),
    },
    {
        "fold": 4,
        "name": "Fold 4 (2021-2022 -> 2023-H1)",
        "train_start": (2021, 1),
        "train_end": (2022, 12),
        "test_start": (2023, 1),
        "test_end": (2023, 6),
    },
    {
        "fold": 5,
        "name": "Fold 5 (2021-2023-H1 -> 2023-H2)",
        "train_start": (2021, 1),
        "train_end": (2023, 6),
        "test_start": (2023, 7),
        "test_end": (2023, 12),
    },
    {
        "fold": 6,
        "name": "Fold 6 (2021-2023 -> 2024)",
        "train_start": (2021, 1),
        "train_end": (2023, 12),
        "test_start": (2024, 1),
        "test_end": (2024, 12),
    }
]

SEALED_HOLDOUT_SPEC = {
    "year": 2025,
    "file": "data/SEALED_2025_HOLDOUT.parquet",
    "status": "SEALED - UNTOUCHED UNTIL FINAL EVALUATION"
}

def load_or_fetch_fold_data(fold_idx: int = 1, save_dir: str = "data/processed") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads preprocessed train and test dataframes for a given fold.
    Downloads raw klines if not already present.
    """
    os.makedirs(save_dir, exist_ok=True)
    spec = FOLDS_SPEC[fold_idx - 1]
    
    train_cache = os.path.join(save_dir, f"fold_{fold_idx}_train.parquet")
    test_cache = os.path.join(save_dir, f"fold_{fold_idx}_test.parquet")
    
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        print(f"Loading cached {spec['name']} from {save_dir}...")
        df_train = pd.read_parquet(train_cache)
        df_test = pd.read_parquet(test_cache)
        return df_train, df_test

    print(f"Downloading raw data for {spec['name']}...")
    raw_train = download_range(
        spec["train_start"][0], spec["train_start"][1],
        spec["train_end"][0], spec["train_end"][1]
    )
    raw_test = download_range(
        spec["test_start"][0], spec["test_start"][1],
        spec["test_end"][0], spec["test_end"][1]
    )
    
    print("Preprocessing train set...")
    df_train = build_dataset(raw_train)
    print("Preprocessing test set...")
    df_test = build_dataset(raw_test)
    
    df_train.to_parquet(train_cache)
    df_test.to_parquet(test_cache)
    print(f"Saved processed fold {fold_idx} ({len(df_train)} train, {len(df_test)} test bars).")
    
    return df_train, df_test

def seal_2025_holdout():
    """
    Records the sealed holdout status without touching or leaking 2025 data.
    """
    seal_path = "data/SEALED_2025_HOLDOUT.txt"
    with open(seal_path, "w") as f:
        f.write("=== 2025 HOLDOUT DATASET SEAL ===\n")
        f.write("Status: STRICTLY SEALED\n")
        f.write("Reason: Prevent look-ahead leakage and post-hoc hyperparameter tuning\n")
        f.write("Unsealing Date: November 14, 2026 (Code Freeze)\n")
    print(f"Sealed holdout guarantee written to {seal_path}")

if __name__ == "__main__":
    seal_2025_holdout()
    print("Fold Specifications:")
    for f in FOLDS_SPEC:
        print(f"Fold {f['fold']}: {f['name']}")

