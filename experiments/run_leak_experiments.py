"""
experiments/run_leak_experiments.py
Per-channel look-ahead inflation audit (one fold).

What changed versus the first version (see FIXES.md):
  * every arm trains its OWN model with the leak switched on in training AND testing
    (Channel 3 reuses the reference model: the decoding tables are not used in training);
  * each leak is compared with a matched control that differs in that one switch only;
  * Channel 4 really tokenizes the stream past the origin, and the context-only
    unconstrained arm measures partial-token bias;
  * HAR-RV, GARCH(1,1) and every GPT arm are scored on the same origins and target;
  * two forecast heads: direct regression and generative + conditional decoding.

Usage (Fold 1, 1 seed, ~1-2 h on CPU, much faster on GPU):
    python experiments/run_leak_experiments.py --fold 1
    python experiments/run_leak_experiments.py --fold 1 --seeds 0 1 2 --epochs 5
    python experiments/run_leak_experiments.py --fold 1 --arms reference ch4_control_context_only ch4_boundary_leak
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from data.folds import load_fold, select_origins
from data.preprocess import future_rv
from tokenization.leak_tokenizer import LeakageManager, ARMS, ArmInput, PAIRED_INFERENCE, ORIGINAL_ARMS
from models.gpt_model import TimeSeriesGPT, pad_left, generative_rv
from models.baselines import HARRVModel, GARCH11Model
from evaluation.metrics import qlike_loss
from evaluation.hypothesis_tests import (block_bootstrap_ci, diebold_mariano_test, giacomini_white_test,
                                         tost_equivalence_test, holm_bonferroni_correction)

LEAK_ARMS = ["ch1_window_norm_leak", "ch2_merge_table_leak", "ch3_decode_table_leak",
             "ch4_boundary_leak", "ch5_noncausal_fft_leak", "positive_control"]


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--horizon", type=int, default=12, help="12 = next hour, 288 = next day")
    ap.add_argument("--context", type=int, default=144)
    ap.add_argument("--arms", nargs="+", default=list(ORIGINAL_ARMS), choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lm-weight", type=float, default=0.5)
    ap.add_argument("--vol-loss", choices=["qlike", "mse"], default="qlike")
    ap.add_argument("--max-len", type=int, default=192)
    ap.add_argument("--no-generative", action="store_true")
    ap.add_argument("--n-samples", type=int, default=8)
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--no-scale-tokens", action="store_true")
    ap.add_argument("--tost-margin", type=float, default=0.05, help="fraction of the control's mean QLIKE")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-train", type=int, default=None, help="smoke tests only: subsample train origins")
    ap.add_argument("--max-test", type=int, default=None, help="smoke tests only: first N test origins")
    ap.add_argument("--out", default=None)
    ap.add_argument("--losses-out", default=None,
                    help="npz with per-origin forecasts and targets (default results/losses/fold_<k>_h<H>.npz)")
    return ap.parse_args(argv)


# ---------------------------------------------------------------------- models
def train_model(inputs: List[ArmInput], y: np.ndarray, vocab: int, args, seed: int, device: str) -> TimeSeriesGPT:
    torch.manual_seed(seed)
    np.random.seed(seed)
    pad = vocab
    model = TimeSeriesGPT(vocab_size=vocab + 1, pad_id=pad, max_seq_len=args.max_len).to(device)
    X = pad_left([a.tokens for a in inputs], args.max_len, pad)
    scales = np.array([a.scale for a in inputs])
    yn = torch.tensor(np.maximum(y / scales ** 2, 1e-8), dtype=torch.float32)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    n = len(X)
    for ep in range(args.epochs):
        model.train()
        perm = torch.randperm(n)
        tot, t0 = 0.0, time.time()
        for b in range(0, n, args.batch_size):
            idx = perm[b:b + args.batch_size]
            ids, tgt = X[idx].to(device), yn[idx].to(device)
            logits, pred = model(ids)
            lm = F.cross_entropy(logits[:, :-1].reshape(-1, vocab + 1), ids[:, 1:].reshape(-1), ignore_index=pad)
            if args.vol_loss == "qlike":
                ratio = tgt / pred
                vol = (ratio - torch.log(ratio) - 1.0).mean()
            else:
                vol = F.mse_loss(pred, tgt)
            loss = vol + args.lm_weight * lm
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(idx)
        print(f"    epoch {ep + 1}/{args.epochs} loss {tot / n:.4f} ({time.time() - t0:.0f}s)")
    return model


@torch.no_grad()
def predict_direct(model, inputs: List[ArmInput], args, device) -> np.ndarray:
    model.eval()
    out = []
    for b in range(0, len(inputs), 256):
        chunk = inputs[b:b + 256]
        ids = pad_left([a.tokens for a in chunk], args.max_len, model.pad_id).to(device)
        _, p = model(ids)
        out.append(p.cpu().numpy() * np.array([a.scale for a in chunk]) ** 2)
    return np.concatenate(out)


def predict_generative(model, inputs: List[ArmInput], mgr: LeakageManager, args, device, seed) -> np.ndarray:
    tok = mgr.toks[inputs[0].decode_tok_name]
    vocab = mgr.vocab_size()
    token_bins = [tok.expand(i) for i in range(vocab)]
    token_len = np.array([len(b) for b in token_bins])
    last_bins = []
    for a in inputs:
        lb = 0
        for tid in reversed(a.tokens):
            if token_bins[tid]:
                lb = token_bins[tid][-1]
                break
        last_bins.append(lb)
    heal = [a.heal_bins for a in inputs] if any(a.heal_bins for a in inputs) else None
    m2 = generative_rv(model, [a.tokens for a in inputs], last_bins, [a.known_future for a in inputs],
                       token_len, token_bins, tok.cond_m2, args.horizon, n_samples=args.n_samples,
                       device=device, seed=seed, heal_bins=heal)
    return m2 * np.array([a.scale for a in inputs]) ** 2


def fit_baselines(df: pd.DataFrame, split: int, test_origins: np.ndarray, args) -> Dict[str, np.ndarray]:
    r = df["return"].values
    tr = select_origins(df, 0, split, args.horizon, args.context, min_history=2016)
    har = HARRVModel().fit(r, tr, args.horizon)
    garch = GARCH11Model().fit(r[:split])
    return {"har_rv": har.predict(r, test_origins, args.horizon),
            "garch_11": garch.predict(r, test_origins, args.horizon)}


# ---------------------------------------------------------------------- statistics
def summarize(losses: np.ndarray) -> dict:
    m, lo, hi = block_bootstrap_ci(losses)
    return {"QLIKE": m, "QLIKE_95_CI": [lo, hi]}


def compare(control: np.ndarray, arm: np.ndarray, margin: float) -> dict:
    """Positive dm_stat / inflation = the arm has LOWER loss than its control."""
    stat, p = diebold_mariano_test(control, arm)
    gw, gw_p = giacomini_white_test(control, arm)
    c = float(np.mean(control))
    return {"inflation_pct": 100.0 * (c - float(np.mean(arm))) / c,
            "dm_stat": stat, "dm_p": p, "gw_stat": gw, "gw_p": gw_p,
            "tost": tost_equivalence_test(arm, control, margin=margin, relative=True)}


def run_audit(df: pd.DataFrame, split: int, args, extra_forecasts: Optional[Dict[str, np.ndarray]] = None) -> dict:
    device = ("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device
    H, C = args.horizon, args.context
    y_all = future_rv(df["return"].values, H)

    mgr = LeakageManager(context_len=C, horizon=H, use_gate=not args.no_gate,
                         use_scale_tokens=not args.no_scale_tokens).fit(df, split)
    vocab = mgr.vocab_size()
    train_o = select_origins(df, 0, split, H, C)
    test_o = select_origins(df, split, len(df), H, C, min_history=2016)
    if args.max_train:
        train_o = np.sort(np.random.default_rng(0).choice(train_o, min(args.max_train, len(train_o)), replace=False))
    if args.max_test:
        test_o = test_o[:args.max_test]
    y_tr, y_te = y_all[train_o], y_all[test_o]
    print(f"device {device} | vocab {vocab} | train origins {len(train_o)} | test origins {len(test_o)}")

    preds: Dict[str, Dict[str, List[np.ndarray]]] = {"direct": {}, "generative": {}}
    ref_models: Dict[int, TimeSeriesGPT] = {}
    for arm in args.arms:
        print(f"\n== {arm}: {ARMS[arm]['label']}")
        t0 = time.time()
        te_in = [mgr.extract(int(t), arm) for t in test_o]
        if arm != "ch3_decode_table_leak":
            tr_in = [mgr.extract(int(t), arm) for t in train_o]
        print(f"   tokens extracted ({time.time() - t0:.0f}s)")
        for seed in args.seeds:
            if arm == "ch3_decode_table_leak":
                # decoding tables never enter training, so the reference model IS the ch3 model
                if seed not in ref_models:
                    ref_in = [mgr.extract(int(t), "reference") for t in train_o]
                    ref_models[seed] = train_model(ref_in, y_tr, vocab, args, seed, device)
                model = ref_models[seed]
            else:
                model = train_model(tr_in, y_tr, vocab, args, seed, device)
                if arm == "reference":
                    ref_models[seed] = model
            preds["direct"].setdefault(arm, []).append(predict_direct(model, te_in, args, device))
            if not args.no_generative:
                preds["generative"].setdefault(arm, []).append(
                    predict_generative(model, te_in, mgr, args, device, seed))
                # inference-time variants that reuse this arm's model (e.g. token healing)
                for pa in PAIRED_INFERENCE.get(arm, []):
                    pa_in = [mgr.extract(int(t), pa) for t in test_o]
                    preds["generative"].setdefault(pa, []).append(
                        predict_generative(model, pa_in, mgr, args, device, seed))
            q = qlike_loss(y_te, preds["direct"][arm][-1]).mean()
            print(f"   seed {seed}: direct QLIKE {q:.4f}")

    base = fit_baselines(df, split, test_o, args)
    if extra_forecasts:
        base.update({k: v[test_o] for k, v in extra_forecasts.items()})
    base_losses = {k: qlike_loss(y_te, v) for k, v in base.items()}

    if getattr(args, "losses_out", None):
        # per-origin forecasts so folds can be pooled later (experiments/aggregate_folds.py)
        arrays = {"y": y_te, "origin": test_o,
                  "origin_time": pd.to_datetime(df["open_time"]).values[test_o].astype("datetime64[ns]").astype(np.int64)}
        for k, v in base.items():
            arrays[f"base/{k}"] = np.asarray(v, dtype=np.float64)
        for head, by_arm in preds.items():
            for a, ps in by_arm.items():
                for seed, p in zip(args.seeds, ps):
                    arrays[f"fc/{head}/{a}/seed{seed}"] = np.asarray(p, dtype=np.float64)
        os.makedirs(os.path.dirname(args.losses_out) or ".", exist_ok=True)
        np.savez_compressed(args.losses_out, **arrays)
        print(f"Saved per-origin forecasts to {args.losses_out}")

    results = {"fold": getattr(args, "fold", None), "horizon": H, "context": C,
               "n_train_origins": int(len(train_o)), "n_test_origins": int(len(test_o)),
               "seeds": args.seeds, "epochs": args.epochs, "vol_loss": args.vol_loss,
               "gate": not args.no_gate, "scale_tokens": not args.no_scale_tokens,
               "baselines": {k: summarize(v) for k, v in base_losses.items()}, "heads": {}}
    if "garch_11" in base_losses and "har_rv" in base_losses:
        s, p = diebold_mariano_test(base_losses["har_rv"], base_losses["garch_11"])
        results["baselines"]["dm_har_minus_garch"] = {"stat": s, "p": p}

    for head, by_arm in preds.items():
        if not by_arm:
            continue
        losses = {a: np.mean([qlike_loss(y_te, p) for p in ps], axis=0) for a, ps in by_arm.items()}
        per_seed = {a: [float(qlike_loss(y_te, p).mean()) for p in ps] for a, ps in by_arm.items()}
        hr = {"arms": {}, "leak_tests": {}, "partial_token_bias": None, "vs_baselines": {}}
        for a in losses:
            hr["arms"][a] = {"label": ARMS[a]["label"], **summarize(losses[a]), "per_seed_QLIKE": per_seed[a]}
        # each leak vs its matched control, Holm across the leak family
        tested = [a for a in LEAK_ARMS if a in losses and ARMS[a]["control"] in losses]
        if head == "direct":
            # Ch3 only changes the decoding tables, which the direct head never uses: its direct
            # forecasts equal the reference by construction, so it is not a test there.
            tested = [a for a in tested if a != "ch3_decode_table_leak"]
        for a in tested:
            hr["leak_tests"][a] = {"control": ARMS[a]["control"],
                                   **compare(losses[ARMS[a]["control"]], losses[a], args.tost_margin)}
        holm = holm_bonferroni_correction([hr["leak_tests"][a]["dm_p"] for a in tested]) if tested else []
        for a, (adj, sig) in zip(tested, holm):
            hr["leak_tests"][a]["dm_p_holm"] = adj
            hr["leak_tests"][a]["significant_holm"] = sig
        if "ch4_control_context_only" in losses and "reference" in losses:
            hr["partial_token_bias"] = compare(losses["reference"], losses["ch4_control_context_only"], args.tost_margin)
        for a in losses:
            hr["vs_baselines"][a] = {b: compare(bl, losses[a], args.tost_margin) for b, bl in base_losses.items()}
        results["heads"][head] = hr
    return results


def print_table(res: dict):
    print("\nBaselines:", {k: round(v["QLIKE"], 4) for k, v in res["baselines"].items() if "QLIKE" in v})
    for head, hr in res["heads"].items():
        print(f"\n[{head} head]  arm | QLIKE | vs control: inflation, DM p (Holm), TOST-equivalent")
        for a, v in hr["arms"].items():
            lt = hr["leak_tests"].get(a)
            extra = ""
            if lt:
                extra = (f"| vs {lt['control']}: {lt['inflation_pct']:+.2f}%  p={lt['dm_p']:.4f} "
                         f"(Holm {lt['dm_p_holm']:.4f})  equiv={lt['tost']['is_equivalent']}")
            print(f"  {a:26s} {v['QLIKE']:.4f} {extra}")
        if hr["partial_token_bias"]:
            pt = hr["partial_token_bias"]
            print(f"  partial-token bias (ch4 control vs reference): {pt['inflation_pct']:+.2f}%  p={pt['dm_p']:.4f}")


def main():
    args = parse_args()
    smoke = bool(args.max_train or args.max_test)
    if args.losses_out is None and not smoke:
        args.losses_out = f"results/losses/fold_{args.fold}_h{args.horizon}.npz"
    df, split = load_fold(args.fold)
    res = run_audit(df, split, args)
    print_table(res)
    os.makedirs("results", exist_ok=True)
    out = args.out or f"results/fold_{args.fold}_h{args.horizon}_leak_audit.json"
    with open(out, "w") as f:
        json.dump(res, f, indent=2)
    # the single-fold figure table is Fold 1's main run only (it used to be overwritten by every fold)
    if args.horizon == 12 and args.fold == 1 and args.out is None and not smoke:
        with open("results/leakage_inflation_table.json", "w") as f:
            json.dump(res, f, indent=2)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
