"""
experiments/forecast_store.py
Loads the per-origin forecast files written by the experiment scripts and merges every file
that belongs to the same fold:

    results/losses/fold_<k>_h<H>.npz            main run (run_leak_experiments.py)
    results/losses/fold_<k>_h<H>_<tag>.npz      extra seeds or extension runs (run_extensions.py)

Merged files must share the same test origins (checked). Keys are
    y, origin, origin_time, base/<model>, fc/<head>/<arm>/seed<s>
"""

import os
import re
import glob
from typing import Dict, List, Optional

import numpy as np

from evaluation.metrics import qlike_loss


def _load(path: str) -> dict:
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def _merge(into: dict, new: dict, path: str):
    if not into:
        into.update(new)
        return
    if not np.array_equal(into["origin"], new["origin"]):
        raise ValueError(f"{path}: test origins differ from the other files of this fold")
    for k, v in new.items():
        if k in ("y", "origin", "origin_time") or k.startswith("base/"):
            continue
        if k in into and not np.allclose(into[k], v):
            raise ValueError(f"{path}: duplicate forecast key {k} with different values")
        into[k] = v


def load_fold_forecasts(results: str = "results", horizon: int = 12, prefix: str = "fold") -> Dict[int, dict]:
    """{fold number: merged dict} for every fold with at least one file."""
    rx = re.compile(rf"{prefix}_(\d+)_h{horizon}(?:_[A-Za-z0-9]+)?\.npz$")
    out: Dict[int, dict] = {}
    paths = sorted(glob.glob(os.path.join(results, "losses", f"{prefix}_*_h{horizon}*.npz")))
    # main file first so its baselines are the ones kept
    paths.sort(key=lambda p: (int(rx.search(os.path.basename(p)).group(1)) if rx.search(os.path.basename(p)) else 0,
                              "_" in os.path.basename(p).split(f"_h{horizon}")[1]))
    for p in paths:
        m = rx.search(os.path.basename(p))
        if not m:
            continue
        _merge(out.setdefault(int(m.group(1)), {}), _load(p), p)
    return dict(sorted(out.items()))


def load_single(results: str, name: str) -> Optional[dict]:
    p = os.path.join(results, "losses", name)
    return _load(p) if os.path.exists(p) else None


def seeds_of(d: dict, head: str, arm: str) -> List[int]:
    rx = re.compile(rf"^fc/{re.escape(head)}/{re.escape(arm)}/seed(\d+)$")
    return sorted(int(m.group(1)) for k in d for m in [rx.match(k)] if m)


def common_seeds(folds: Dict[int, dict], head: str, arms: List[str]) -> List[int]:
    sets = [set(seeds_of(d, head, a)) for d in folds.values() for a in arms]
    return sorted(set.intersection(*sets)) if sets else []


def seed_avg_loss(d: dict, head: str, arm: str, seeds: List[int]) -> np.ndarray:
    """Per-origin QLIKE averaged over training seeds (expected loss over training randomness)."""
    return np.mean([qlike_loss(d["y"], d[f"fc/{head}/{arm}/seed{s}"]) for s in seeds], axis=0)
