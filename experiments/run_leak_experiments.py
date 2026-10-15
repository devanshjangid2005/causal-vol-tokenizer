"""
experiments/run_leak_experiments.py
Full End-to-End Leakage Audit and Look-Ahead Inflation Measurement:
1. Trains Causal Reference ~1M-Parameter GPT on Fold 1 Train (2021).
2. Evaluates Reference against the 5 Leak Channels and Positive Control on Fold 1 Test (2022-Q1).
3. Produces the Per-Channel Inflation Table with 95% Block-Bootstrap CIs,
   Diebold-Mariano tests against HAR-RV, TOST equivalence tests, and Holm correction.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import time
from typing import List, Tuple, Optional, Dict, Any
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from data.folds import load_or_fetch_fold_data
from tokenization.leak_tokenizer import LeakageManager
from models.gpt_model import TimeSeriesGPT
from evaluation.metrics import qlike_loss, mean_qlike, compute_all_metrics
from evaluation.hypothesis_tests import (
    block_bootstrap_ci,
    diebold_mariano_test,
    tost_equivalence_test,
    holm_bonferroni_correction
)

class TimeSeriesDataset(Dataset):
    def __init__(self, token_seqs: List[np.ndarray], scales: List[float], targets: List[float], max_len: int = 128):
        self.samples = []
        for seq, sc, tgt in zip(token_seqs, scales, targets):
            # Pad or truncate to max_len
            if len(seq) > max_len:
                seq = seq[-max_len:]
            pad_len = max_len - len(seq)
            padded = np.pad(seq, (pad_len, 0), mode='constant', constant_values=0)
            norm_tgt = tgt / max(sc ** 2, 1e-8)
            self.samples.append((
                torch.tensor(padded, dtype=torch.long),
                torch.tensor(sc, dtype=torch.float32),
                torch.tensor(norm_tgt, dtype=torch.float32),
                torch.tensor(tgt, dtype=torch.float32)
            ))
            
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        return self.samples[idx]

def extract_dataset_tokens(
    mgr: LeakageManager,
    df: pd.DataFrame,
    channel: str = "reference",
    context_len: int = 144,
    horizon: int = 12,
    step: int = 12,
    max_samples: Optional[int] = None
) -> Tuple[List[np.ndarray], List[float], List[float]]:
    returns = df["return"].values
    ewma_vol = df["ewma_vol"].values
    rv_target = df["rv_target_1h"].values
    
    n = len(returns)
    start_idx = context_len
    end_idx = n - horizon
    
    indices = list(range(start_idx, end_idx, step))
    if max_samples and len(indices) > max_samples:
        indices = indices[:max_samples]
        
    tokens_list = []
    scales_list = []
    targets_list = []
    
    for t in indices:
        toks, sc = mgr.extract_features(
            returns, ewma_vol, origin_t=t, context_len=context_len, horizon=horizon, channel=channel
        )
        tokens_list.append(toks)
        scales_list.append(sc)
        targets_list.append(float(rv_target[t]))
        
    return tokens_list, scales_list, targets_list

def run_leak_experiments():
    print("=" * 70)
    print("RUNNING PER-CHANNEL LOOK-AHEAD INFLATION EXPERIMENT (FOLD 1)")
    print("=" * 70)
    
    device = torch.device("cpu")
    
    # 1. Load Data
    df_train, df_test = load_or_fetch_fold_data(fold_idx=1)
    
    # 2. Setup Leakage Manager
    print("Fitting Tokenizers on Train vs Full Data...")
    mgr = LeakageManager(num_bins=32, num_merges=64)
    mgr.fit_reference(df_train["return"].values, df_train["ewma_vol"].values)
    mgr.fit_leaked_tables(
        np.concatenate([df_train["return"].values, df_test["return"].values]),
        np.concatenate([df_train["ewma_vol"].values, df_test["ewma_vol"].values])
    )
    
    vocab_size = max(mgr.causal_tok.vocab_size, mgr.merge_leaked_tok.vocab_size) + 10
    print(f"Vocab size: {vocab_size}")
    
    # 3. Prepare Train Data
    print("Extracting train sequences...")
    train_toks, train_sc, train_tgt = extract_dataset_tokens(
        mgr, df_train, channel="reference", context_len=144, horizon=12, step=12
    )
    train_ds = TimeSeriesDataset(train_toks, train_sc, train_tgt, max_len=128)
    train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
    print(f"Total train origins: {len(train_ds)}")
    
    # 4. Initialize and Train ~1M Parameter GPT
    model = TimeSeriesGPT(
        vocab_size=vocab_size,
        max_seq_len=128,
        d_model=128,
        nhead=4,
        num_layers=4,
        dim_feedforward=512,
        dropout=0.1
    ).to(device)
    
    print(f"Initialized TimeSeriesGPT: {model.count_parameters():,} parameters.")
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion_vol = nn.MSELoss()
    criterion_lm = nn.CrossEntropyLoss(ignore_index=0)
    
    print("\nTraining TimeSeriesGPT for 5 epochs on Fold 1...")
    model.train()
    for epoch in range(1, 6):
        start_ep = time.time()
        total_loss = 0.0
        for seqs, scs, norm_tgts, tgts in train_loader:
            seqs, norm_tgts = seqs.to(device), norm_tgts.to(device)
            optimizer.zero_grad()
            
            logits, rv_pred = model(seqs)
            
            # Loss on next-token prediction
            targets_lm = torch.roll(seqs, -1, dims=1)
            targets_lm[:, -1] = 0
            loss_lm = criterion_lm(logits.view(-1, vocab_size), targets_lm.view(-1))
            
            # Loss on normalized RV prediction
            loss_vol = criterion_vol(rv_pred.squeeze(-1), norm_tgts)
            
            loss = loss_vol + 0.1 * loss_lm
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item() * len(seqs)
            
        avg_loss = total_loss / len(train_ds)
        print(f"Epoch {epoch}/5 - Loss: {avg_loss:.4f} ({time.time() - start_ep:.1f}s)")
        
    # 5. Evaluate on Test Fold Across Channels
    print("\nEvaluating on Fold 1 Test Set Across Leakage Channels...")
    model.eval()
    
    channels = [
        ("reference", "Causal Reference (Leak-Free)"),
        ("channel_1_norm_leak", "Channel 1: Per-Window Norm Leak"),
        ("channel_2_merge_leak", "Channel 2: Merge Table Leak"),
        ("channel_3_decode_leak", "Channel 3: Decoding Table Leak"),
        ("channel_4_boundary_leak", "Channel 4: Token Boundary Leak"),
        ("channel_5_fft_leak", "Channel 5: Non-Causal FFT Leak"),
        ("positive_control", "Positive Control (Gross Leak)")
    ]
    
    channel_results = {}
    channel_losses_dict = {}
    
    # Load HAR baseline for comparison
    with open("results/fold_1_baselines.json", "r") as f:
        baseline_res = json.load(f)
    har_qlike = baseline_res["har_rv"]["QLIKE"]
    
    for ch_key, ch_name in channels:
        test_toks, test_sc, test_tgt = extract_dataset_tokens(
            mgr, df_test, channel=ch_key, context_len=144, horizon=12, step=12
        )
        test_ds = TimeSeriesDataset(test_toks, test_sc, test_tgt, max_len=128)
        test_loader = DataLoader(test_ds, batch_size=128, shuffle=False)
        
        all_preds = []
        all_trues = []
        with torch.no_grad():
            for seqs, scs, norm_tgts, tgts in test_loader:
                seqs = seqs.to(device)
                _, norm_rv_pred = model(seqs)
                # Rescale predicted normalized RV by scale^2
                pred_rv = norm_rv_pred.squeeze(-1).cpu().numpy() * (scs.numpy() ** 2)
                all_preds.extend(pred_rv.tolist())
                all_trues.extend(tgts.numpy().tolist())
                
        y_true = np.array(all_trues)
        y_pred = np.maximum(np.array(all_preds), 1e-8)
        
        losses = qlike_loss(y_true, y_pred)
        mean_q = float(np.mean(losses))
        _, ci_l, ci_u = block_bootstrap_ci(losses)
        all_mets = compute_all_metrics(y_true, y_pred)
        
        channel_losses_dict[ch_key] = losses
        channel_results[ch_key] = {
            "name": ch_name,
            "QLIKE": mean_q,
            "QLIKE_95_CI": [ci_l, ci_u],
            "RMSE": all_mets["RMSE"],
            "MAE": all_mets["MAE"]
        }
        print(f"[{ch_name:35s}] QLIKE: {mean_q:.5f} (95% CI: [{ci_l:.5f}, {ci_u:.5f}])")

    # 6. Compute Inflation Rates & Statistical Tests relative to Reference
    ref_losses = channel_losses_dict["reference"]
    ref_qlike = channel_results["reference"]["QLIKE"]
    
    print("\n" + "=" * 80)
    print("PER-CHANNEL LOOK-AHEAD INFLATION TABLE (Summary)")
    print("=" * 80)
    print(f"{'Channel / Model':36s} | {'QLIKE':7s} | {'Inflation (%)':14s} | {'DM vs Ref (p)':14s} | {'TOST Equiv':10s}")
    print("-" * 88)
    
    p_values_to_adjust = []
    keys_for_adjust = []
    
    for ch_key, ch_name in channels:
        q_val = channel_results[ch_key]["QLIKE"]
        inflation_pct = ((ref_qlike - q_val) / ref_qlike) * 100.0 if ch_key != "reference" else 0.0
        
        if ch_key == "reference":
            dm_str = "---"
            tost_str = "REF"
            channel_results[ch_key]["inflation_pct"] = 0.0
            channel_results[ch_key]["dm_vs_ref"] = {"stat": 0.0, "p_value": 1.0}
            channel_results[ch_key]["tost"] = {"is_equivalent": True}
        else:
            ch_losses = channel_losses_dict[ch_key]
            # DM test: H0: loss_ref == loss_ch
            dm_stat, dm_p = diebold_mariano_test(ref_losses, ch_losses)
            tost_res = tost_equivalence_test(ch_losses, ref_losses, margin=0.03)
            
            p_values_to_adjust.append(dm_p)
            keys_for_adjust.append(ch_key)
            
            dm_str = f"{dm_p:.4f} (z={dm_stat:.2f})"
            tost_str = "YES" if tost_res["is_equivalent"] else "NO (Diff)"
            
            channel_results[ch_key]["inflation_pct"] = float(inflation_pct)
            channel_results[ch_key]["dm_vs_ref"] = {"stat": float(dm_stat), "p_value": float(dm_p)}
            channel_results[ch_key]["tost"] = tost_res
            
        print(f"{ch_name:36s} | {q_val:.5f} | {inflation_pct:+12.2f}% | {dm_str:14s} | {tost_str:10s}")
        
    # Apply Holm-Bonferroni correction
    holm_results = holm_bonferroni_correction(p_values_to_adjust)
    for k, (adj_p, sig) in zip(keys_for_adjust, holm_results):
        channel_results[k]["dm_vs_ref"]["holm_adj_p"] = adj_p
        channel_results[k]["dm_vs_ref"]["is_significant_holm"] = sig

    summary_output = {
        "har_rv_qlike": har_qlike,
        "reference_qlike": ref_qlike,
        "channel_results": channel_results
    }
    
    with open("results/leakage_inflation_table.json", "w") as f:
        json.dump(summary_output, f, indent=2)
    print("\nSaved complete inflation results to results/leakage_inflation_table.json")
    return summary_output

if __name__ == "__main__":
    run_leak_experiments()
