"""
data/download_binance.py
Downloads public 5-minute klines (BTCUSDT by default; any spot symbol, e.g. ETHUSDT) from the
Binance Vision Data Archive.
Extracts open, high, low, close, volume, and timestamp.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import io
import zipfile
import requests
import pandas as pd
from typing import List, Optional

BASE_URL = "https://data.binance.vision/data/spot/monthly/klines/{symbol}/5m"

COLUMNS = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count", "taker_buy_volume",
    "taker_buy_quote_volume", "ignore"
]

def download_month(year: int, month: int, save_dir: str = "data/raw", symbol: str = "BTCUSDT") -> Optional[pd.DataFrame]:
    """
    Downloads and parses a single month's 5m kline zip file.
    """
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{symbol}-5m-{year:04d}-{month:02d}.zip"
    url = f"{BASE_URL.format(symbol=symbol)}/{filename}"
    local_csv = os.path.join(save_dir, f"{symbol}-5m-{year:04d}-{month:02d}.csv")
    
    if os.path.exists(local_csv):
        print(f"Loading cached {os.path.basename(local_csv)}...")
        df = pd.read_csv(local_csv)
        # Handle case where header was saved or missing
        if "open_time" not in df.columns:
            df.columns = COLUMNS
        # Filter out header string row if present
        df = df[pd.to_numeric(df["open_time"], errors="coerce").notnull()].copy()
        return df

    print(f"Downloading {filename} from {url}...")
    resp = None
    for attempt in range(3):
        try:
            resp = requests.get(url, timeout=60)
            break
        except requests.RequestException as e:
            print(f"  attempt {attempt + 1} failed: {e}")
    if resp is None or resp.status_code != 200:
        print(f"Warning: Failed to fetch {url} (HTTP {getattr(resp, 'status_code', 'no response')})")
        return None
        
    with zipfile.ZipFile(io.BytesIO(resp.content)) as z:
        for inner_name in z.namelist():
            if inner_name.endswith(".csv"):
                with z.open(inner_name) as f:
                    # Binance raw CSVs have no header
                    df = pd.read_csv(f, header=None)
                    if df.shape[1] == 12:
                        df.columns = COLUMNS
                    # Filter out non-numeric open_time if any
                    df = df[pd.to_numeric(df["open_time"], errors="coerce").notnull()].copy()
                    df.to_csv(local_csv, index=False)
                    return df
    return None

def download_range(start_year: int, start_month: int, end_year: int, end_month: int, save_dir: str = "data/raw",
                   symbol: str = "BTCUSDT") -> pd.DataFrame:
    """
    Downloads and concatenates klines across a date range.
    """
    frames = []
    curr_y, curr_m = start_year, start_month
    
    while (curr_y < end_year) or (curr_y == end_year and curr_m <= end_month):
        df_m = download_month(curr_y, curr_m, save_dir=save_dir, symbol=symbol)
        if df_m is not None:
            frames.append(df_m)
            print(f"Loaded {curr_y}-{curr_m:02d}: {len(df_m)} bars")
        curr_m += 1
        if curr_m > 12:
            curr_m = 1
            curr_y += 1
            
    if not frames:
        raise RuntimeError("No data was downloaded successfully.")
        
    combined = pd.concat(frames, ignore_index=True)
    # Binance spot archives use milliseconds up to 2024 and MICROseconds from 2025-01-01.
    t = pd.to_numeric(combined["open_time"]).astype("int64")
    t = t.where(t < 10**14, t // 1000)
    combined["open_time"] = pd.to_datetime(t, unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
        
    combined = combined.sort_values("open_time").drop_duplicates(subset=["open_time"]).reset_index(drop=True)
    print(f"Successfully combined {len(combined)} 5-minute bars from {combined['open_time'].min()} to {combined['open_time'].max()}")
    return combined

if __name__ == "__main__":
    # python data/download_binance.py 2021 1 2025 12 [SYMBOL]   -> caches every month in data/raw
    a = [int(x) for x in sys.argv[1:5]] if len(sys.argv) >= 5 else [2021, 1, 2025, 12]
    sym = sys.argv[5] if len(sys.argv) >= 6 else "BTCUSDT"
    df = download_range(*a, symbol=sym)
    print(f"Downloaded range: {df.shape}")

