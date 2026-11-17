"""
experiments/run_boundary_experiment.py
Token-level causality audit and compression curve on real BTCUSDT data (no model needed).

For each tokenizer variant it runs the two tests in tokenization/causality_tests.py at
  (a) hour-aligned origins and (b) origins 5 bars after the hour:
  * hour boundaries, no gate
  * hour boundaries + causal RoV x SEI gate        (the causal reference)
  * hour boundaries + centered gate                (Channel 5)
  * unconstrained Götz BPE                         (Channel 4)
All tokenizers are fitted on the train period of the fold, on trailing-EWMA
normalized returns, and evaluated on the test period.

Usage: python experiments/run_boundary_experiment.py --fold 1 [--origins 500]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import numpy as np
import pandas as pd

from data.folds import load_fold, select_origins
from data.preprocess import compute_trailing_ewma_vol
from tokenization.leak_tokenizer import LeakageManager
from tokenization.bpe_tokenizer import TimeSeriesBPETokenizer
from tokenization.causality_tests import slice_vs_prefix, future_perturbation


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--origins", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df, split = load_fold(args.fold)
    mgr = LeakageManager().fit(df, split)
    ct, ut = mgr.toks["causal"], mgr.toks["unconstrained"]
    bph = 12
    hour_starts = np.flatnonzero(pd.to_datetime(df["open_time"]).dt.minute.values == 0)

    base = {"r": mgr.r, "xn": mgr.xn, "ev": mgr.ev,
            "flags_causal": mgr.flags_causal, "flags_centered": mgr.flags_centered}

    def rebuild(series, t, rng, noise):
        """Replace bars >= t with noise and recompute every derived array (EWMA, norm, gates)."""
        s = {k: v.copy() for k, v in series.items()}
        s["r"][t:] = noise * np.std(series["r"][:t]) * rng.standard_normal(len(s["r"]) - t)
        lo = max(0, t - 2000)  # EWMA memory is ~50 bars, 2000 is plenty
        ev = compute_trailing_ewma_vol(s["r"][lo:], initial_vol=float(series["ev"][lo]))
        s["ev"][lo:] = ev
        s["xn"][lo:] = s["r"][lo:] / np.maximum(ev, 1e-6)
        hs = hour_starts[(hour_starts >= t - 48) & (hour_starts <= t + 48)]
        s["flags_causal"][hs] = mgr.gate_causal.flags_per_bar(s["xn"], hs)[hs]
        s["flags_centered"][hs] = mgr.gate_centered.flags_per_bar(s["xn"], hs)[hs]
        return s

    def hourly_encoder(flag_key):
        def enc(s, lo, hi):
            out = []
            for a in range(lo, hi, bph):
                b = min(a + bph, hi)
                out.append(ct.quantize_scale(s["ev"][a]))
                flag = bool(s[flag_key][a]) if flag_key else False
                out.extend(ct.encode(s["xn"][a:b], include_scale_tokens=False,
                                     gate_flags=[flag], short_merges=mgr.short_merges))
            return out
        return enc

    variants = {
        "hour_boundaries_no_gate": (hourly_encoder(None), lambda i: len(ct.expand(i))),
        "hour_boundaries_causal_gate (reference)": (hourly_encoder("flags_causal"), lambda i: len(ct.expand(i))),
        "hour_boundaries_centered_gate (ch5)": (hourly_encoder("flags_centered"), lambda i: len(ct.expand(i))),
        "unconstrained_bpe (ch4)": (lambda s, lo, hi: ut.encode(s["xn"][lo:hi]), lambda i: len(ut.expand(i))),
    }

    rng = np.random.default_rng(args.seed)
    aligned = select_origins(df, split, len(df), 30, 150)
    aligned = np.sort(rng.choice(aligned, size=min(args.origins, len(aligned)), replace=False))
    off = aligned + 5  # 5 bars past the hour; context still starts on a clock hour
    lo_aligned = lambda t: t - 144
    lo_off = lambda t: t - 5 - 144

    results = {"fold": args.fold, "n_origins": int(len(aligned)), "variants": {}}
    for name, (enc, lens) in variants.items():
        res = {
            "slice_vs_prefix_hour_aligned": slice_vs_prefix(base, enc, lens, aligned, lo_aligned),
            "slice_vs_prefix_off_hour": slice_vs_prefix(base, enc, lens, off, lo_off),
            "future_perturbation_hour_aligned": future_perturbation(base, rebuild, enc, lens, aligned[:200], lo_aligned),
        }
        results["variants"][name] = res
        print(f"{name:42s} aligned mismatch {res['slice_vs_prefix_hour_aligned']['mismatch_rate']:6.1%} "
              f"(straddle {res['slice_vs_prefix_hour_aligned']['straddle_rate']:6.1%}) | "
              f"off-hour mismatch {res['slice_vs_prefix_off_hour']['mismatch_rate']:6.1%} | "
              f"perturbation changes prefix {res['future_perturbation_hour_aligned']['prefix_changed_rate']:6.1%}")

    # compression vs merge count, hour-aligned, normalized returns, test period
    xn_test = mgr.xn[split:split + 12 * 24 * 30]  # first 30 days of test
    curve = []
    for k in [16, 32, 64, 96, 128]:
        tok = TimeSeriesBPETokenizer(num_merges=k).fit_vocabulary(mgr.xn[:split])
        n_tok = len(tok.encode(xn_test, include_scale_tokens=False))
        curve.append({"num_merges": k, "vocab_size": tok.vocab_size, "compression_ratio": len(xn_test) / n_tok})
        print(f"merges {k:3d}: compression {len(xn_test) / n_tok:.2f}x")
    results["compression_curve"] = curve

    os.makedirs("results", exist_ok=True)
    with open("results/boundary_mismatch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("Saved results/boundary_mismatch_results.json")


if __name__ == "__main__":
    main()
