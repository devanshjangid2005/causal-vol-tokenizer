"""
models/gpt_model.py
~0.8M-parameter causal GPT over motif tokens, with two ways to forecast RV:

1. Direct head: a regression head on the last hidden state predicts RV / scale^2.
2. Generative head (Götz-style): sample future tokens autoregressively, expand them to
   base bins, and decode each bin with the conditional decoding tables
   E[x_i^2 | bin_i, bin_{i-1}]; RV = scale^2 * mean over samples of the sum over the
   next H bars. This is the path through which the decoding table (Channel 3) matters.

A dedicated PAD id (never a real token) is used, so padding cannot be confused with
bin 0 (the most negative return bin), and the LM loss ignores only PAD targets.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Sequence


class TimeSeriesGPT(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, max_seq_len: int = 192, d_model: int = 128,
                 nhead: int = 4, num_layers: int = 4, dim_feedforward: int = 512, dropout: float = 0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.pad_id = pad_id
        self.max_seq_len = max_seq_len
        self.tok_emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.vol_head = nn.Sequential(nn.Linear(d_model, 64), nn.GELU(), nn.Linear(64, 1), nn.Softplus())
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, ids: torch.Tensor):
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        causal = torch.triu(torch.full((t, t), float("-inf"), device=ids.device), diagonal=1)
        # Sequences are LEFT-padded with a dedicated PAD id (zero embedding). We do not add a
        # key-padding mask: with left padding it creates all-masked rows (NaN). PAD is a
        # distinct, information-free token, so attending to it is harmless and identical
        # across arms; the important fix is that it no longer collides with bin 0.
        h = self.transformer(x, mask=causal)
        h = self.ln_f(h)
        return self.lm_head(h), self.vol_head(h[:, -1, :]).squeeze(-1) + 1e-6

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def pad_left(seqs: Sequence[Sequence[int]], max_len: int, pad_id: int) -> torch.Tensor:
    out = torch.full((len(seqs), max_len), pad_id, dtype=torch.long)
    for i, s in enumerate(seqs):
        s = list(s)[-max_len:]
        if s:
            out[i, max_len - len(s):] = torch.tensor(s, dtype=torch.long)
    return out


@torch.no_grad()
def generative_rv(model: TimeSeriesGPT, seqs: List[List[int]], last_bins: List[int],
                  known_future: List[List[int]], token_len: np.ndarray, token_bins: List[tuple],
                  cond_m2: np.ndarray, horizon: int, n_samples: int = 8, temperature: float = 1.0,
                  batch_size: int = 512, device: str = "cpu", seed: int = 0,
                  heal_bins: Optional[List[List[int]]] = None) -> np.ndarray:
    """
    Returns, per sequence, the mean over samples of sum_{k<H} E[x_{t+k}^2 | bins] in
    normalized units (multiply by scale^2 outside). token_len/token_bins describe the
    tokenizer whose decoding tables are being used.

    Token healing (heal_bins): the caller has dropped the final, possibly partial, token of the
    context and passes its bins here. Until those bins are re-produced, sampling is restricted to
    tokens whose bins are prefix-compatible with the remaining ones (the dropped token itself is
    always allowed). Re-produced bins are past data and are not counted; any bins of the same
    token beyond them are the first forecast bins. This is the standard token-healing heuristic
    (see Phan et al. 2024/2025 for the exact, more expensive correction).
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    model.eval()
    L = model.max_seq_len
    out = np.zeros(len(seqs))
    vocab_ok = torch.zeros(model.vocab_size, dtype=torch.bool)
    vocab_ok[: len(token_len)] = True
    vocab_ok[model.pad_id] = False
    # tokens that cover no bars but are not scale tokens (unused ids) are also blocked
    rep = [i for i in range(len(seqs)) for _ in range(n_samples)]
    mask_cache = {}

    def compatible(need: tuple) -> torch.Tensor:
        m = mask_cache.get(need)
        if m is None:
            m = vocab_ok.clone()
            for tid in range(len(token_bins)):
                tb = token_bins[tid]
                if not tb or tuple(tb[:len(need)]) != need[:len(tb)]:
                    m[tid] = False
            mask_cache[need] = m
        return m

    for b0 in range(0, len(rep), batch_size):
        idx = rep[b0:b0 + batch_size]
        ids = pad_left([seqs[i] for i in idx], L, model.pad_id).to(device)
        acc = np.zeros(len(idx))
        done = np.zeros(len(idx), dtype=np.int64)
        prev = np.array([last_bins[i] for i in idx], dtype=np.int64)
        need = [tuple(heal_bins[i]) if heal_bins else () for i in idx]
        for j, i in enumerate(idx):  # bars already known inside the input (ch4 leak)
            for b in known_future[i][:horizon]:
                acc[j] += cond_m2[b, prev[j]]
                prev[j] = b
                done[j] += 1
        for _ in range(3 * horizon + 8):
            if (done >= horizon).all():
                break
            logits, _ = model(ids)
            logits = logits[:, -1, :].float().cpu() / max(temperature, 1e-4)
            logits[:, ~vocab_ok] = float("-inf")
            for j in range(len(idx)):
                if need[j]:
                    logits[j, ~compatible(need[j])] = float("-inf")
            nxt = torch.multinomial(F.softmax(logits, dim=-1), 1, generator=g).squeeze(1).numpy()
            for j, tid in enumerate(nxt):
                bins = token_bins[tid]
                if need[j]:
                    m = min(len(bins), len(need[j]))
                    for b in bins[:m]:      # re-produced past bins: condition only
                        prev[j] = b
                    need[j] = need[j][m:]
                    bins = bins[m:]
                for b in bins:
                    if done[j] >= horizon:
                        break
                    acc[j] += cond_m2[b, prev[j]]
                    prev[j] = b
                    done[j] += 1
            ids = torch.cat([ids[:, 1:], torch.as_tensor(nxt, device=device).unsqueeze(1)], dim=1)
        # sequences that ran out of steps: scale up their partial sum
        acc = np.where(done > 0, acc * horizon / np.maximum(done, 1), np.nan)
        for j, i in enumerate(idx):
            out[i] += acc[j] / n_samples
    return out
