"""
experiments/run_cross_asset.py
Cross-asset ("indirect temporal") leakage channel, after Meyer et al. (2025/26),
"Rethinking Evaluation in the Era of Time Series Foundation Models: (Un)known Information
Leakage Challenges": a model trained on a *different but correlated* series that covers the
test period can benefit from it even though no test observation of the target series was seen.

ETHUSDT and BTCUSDT hourly realized variance correlate at about 0.93 (2021-06 to 2022-03).

Two arms, evaluated on the SAME BTC test origins as every other arm; the only switch is how far
the ETH training samples reach:
  xa_control : BTC train origins + ETH origins whose targets end before the split
  xa_leak    : BTC train origins + ETH origins through the end of the test period
Both use the BTC train-only tokenizer, gate thresholds and decoding tables (ETH is normalized by
its own trailing EWMA), and the same model and training settings as the reference arm.
xa_control vs the main-run reference measures "more data, no leak"; xa_leak vs xa_control is the
leak. This is an exploratory extension: folds only, never the 2025 holdout.

Usage: python experiments/run_cross_asset.py --fold 1 --seeds 0
       (accepts the training options of run_leak_experiments.py, plus --symbol)
Outputs: results/losses/fold_<k>_h<H>_xasset.npz and results/fold_<k>_h<H>_xasset.json
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import copy
import json
import time

import numpy as np
import pandas as pd
import torch

from data.folds import load_fold, select_origins
from data.preprocess import future_rv
from tokenization.leak_tokenizer import LeakageManager
from experiments.run_leak_experiments import (parse_args, train_model, predict_direct, predict_generative,
                                              fit_baselines, summarize, compare)
from evaluation.metrics import qlike_loss


def view_of(mgr: LeakageManager, df: pd.DataFrame) -> LeakageManager:
    """Same fitted tokenizers, gates and tables as `mgr`, but reading another asset's series."""
    m = copy.copy(mgr)
    m.r = df["return"].values.astype(np.float64)
    m.ev = df["ewma_vol"].values.astype(np.float64)
    m.xn = df["norm_return"].values.astype(np.float64)
    hs = np.flatnonzero(pd.to_datetime(df["open_time"]).dt.minute.values == 0)
    m.flags_causal = mgr.gate_causal.flags_per_bar(m.xn, hs)
    m.flags_centered = mgr.gate_centered.flags_per_bar(m.xn, hs)
    m._cache = {}
    return m


def main():
    extra = {"--symbol": "ETHUSDT"}
    argv = sys.argv[1:]
    if "--symbol" in argv:
        i = argv.index("--symbol")
        extra["--symbol"] = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    args = parse_args(argv)
    if not isinstance(args.fold, int) or not 1 <= args.fold <= 6:
        raise SystemExit("Cross-asset runs are for folds 1-6 only (never the 2025 holdout).")
    sym = extra["--symbol"]
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    H, C = args.horizon, args.context
    smoke = bool(args.max_train or args.max_test)

    df_b, split_b = load_fold(args.fold)
    df_e, split_e = load_fold(args.fold, symbol=sym)
    mgr = LeakageManager(context_len=C, horizon=H, use_gate=not args.no_gate,
                         use_scale_tokens=not args.no_scale_tokens).fit(df_b, split_b)
    eth = view_of(mgr, df_e)
    vocab = mgr.vocab_size()

    y_b, y_e = future_rv(df_b["return"].values, H), future_rv(df_e["return"].values, H)
    train_b = select_origins(df_b, 0, split_b, H, C)
    test_b = select_origins(df_b, split_b, len(df_b), H, C, min_history=2016)
    eth_pre = select_origins(df_e, 0, split_e, H, C)
    eth_all = select_origins(df_e, 0, len(df_e), H, C)
    if smoke:
        rng = np.random.default_rng(0)
        pick = lambda o, n: np.sort(rng.choice(o, min(n, len(o)), replace=False))
        n = args.max_train or len(train_b)
        train_b, eth_pre = pick(train_b, n), pick(eth_pre, n)
        eth_post = eth_all[eth_all >= split_e]
        eth_all = np.sort(np.concatenate([eth_pre, pick(eth_post, max(1, n // 4))]))
        if args.max_test:
            test_b = test_b[:args.max_test]
    eth_test_window = int((eth_all >= split_e - H).sum())
    print(f"device {device} | BTC train {len(train_b)} | ETH to split {len(eth_pre)} | "
          f"ETH through test {len(eth_all)} ({eth_test_window} in/after the test window) | BTC test {len(test_b)}")

    t0 = time.time()
    tr_b = [mgr.extract(int(t), "reference") for t in train_b]
    e_in = {int(t): eth.extract(int(t), "reference") for t in eth_all}
    for t in eth_pre:
        e_in.setdefault(int(t), eth.extract(int(t), "reference"))
    te_in = [mgr.extract(int(t), "reference") for t in test_b]
    print(f"tokens extracted ({time.time() - t0:.0f}s)")

    y_te = y_b[test_b]
    preds = {"direct": {}, "generative": {}}
    for arm, eo in (("xa_control", eth_pre), ("xa_leak", eth_all)):
        inputs = tr_b + [e_in[int(t)] for t in eo]
        y = np.concatenate([y_b[train_b], y_e[eo]])
        print(f"\n== {arm}: {len(inputs)} training samples")
        for seed in args.seeds:
            model = train_model(inputs, y, vocab, args, seed, device)
            preds["direct"].setdefault(arm, []).append(predict_direct(model, te_in, args, device))
            if not args.no_generative:
                preds["generative"].setdefault(arm, []).append(predict_generative(model, te_in, mgr, args, device, seed))
            print(f"   seed {seed}: direct QLIKE {qlike_loss(y_te, preds['direct'][arm][-1]).mean():.4f}")

    base = fit_baselines(df_b, split_b, test_b, args)
    res = {"fold": args.fold, "symbol": sym, "horizon": H, "seeds": args.seeds, "epochs": args.epochs,
           "n_train_btc": int(len(train_b)), "n_eth_to_split": int(len(eth_pre)), "n_eth_through_test": int(len(eth_all)),
           "n_test_origins": int(len(test_b)), "heads": {}}
    for head, by_arm in preds.items():
        if not by_arm:
            continue
        L = {a: np.mean([qlike_loss(y_te, p) for p in ps], axis=0) for a, ps in by_arm.items()}
        res["heads"][head] = {"arms": {a: summarize(v) for a, v in L.items()},
                              "xa_leak_vs_xa_control": compare(L["xa_control"], L["xa_leak"], args.tost_margin)}
        lt = res["heads"][head]["xa_leak_vs_xa_control"]
        print(f"[{head}] xa_control {L['xa_control'].mean():.4f} | xa_leak {L['xa_leak'].mean():.4f} | "
              f"inflation {lt['inflation_pct']:+.2f}% (DM p {lt['dm_p']:.4f})")

    if not smoke or args.losses_out:
        out_npz = args.losses_out or f"results/losses/fold_{args.fold}_h{H}_xasset.npz"
        arrays = {"y": y_te, "origin": test_b,
                  "origin_time": pd.to_datetime(df_b["open_time"]).values[test_b].astype("datetime64[ns]").astype(np.int64)}
        arrays.update({f"base/{k}": np.asarray(v, dtype=np.float64) for k, v in base.items()})
        for head, by_arm in preds.items():
            for a, ps in by_arm.items():
                for seed, p in zip(args.seeds, ps):
                    arrays[f"fc/{head}/{a}/seed{seed}"] = np.asarray(p, dtype=np.float64)
        os.makedirs(os.path.dirname(out_npz), exist_ok=True)
        np.savez_compressed(out_npz, **arrays)
        out_json = args.out or f"results/fold_{args.fold}_h{H}_xasset.json"
        with open(out_json, "w") as f:
            json.dump(res, f, indent=2)
        print(f"Saved {out_npz} and {out_json}")


if __name__ == "__main__":
    main()
