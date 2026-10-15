"""
models/gpt_model.py
Lightweight ~1M-Parameter Causal GPT with Conditional Decoding for Realized Volatility Forecasting.

Architecture:
- Discrete token embedding + Positional embedding
- Causal Transformer Decoder (4 layers, d_model=128, nhead=4, dim_feedforward=512)
- Multi-task Output:
  1. Next-token prediction head (for generative autoregressive rollout)
  2. Volatility prediction head (direct projection to normalized realized variance)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List

class TimeSeriesGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int = 150,
        max_seq_len: int = 256,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.max_seq_len = max_seq_len
        self.d_model = d_model
        
        # Embeddings
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)
        
        # Transformer Decoder Layers
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        
        # Heads
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        # Volatility regression head: predicts normalized RV
        self.vol_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus() # guarantees positive volatility forecast
        )
        
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
        return mask

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: (batch_size, seq_len)
        Returns:
            logits: (batch_size, seq_len, vocab_size)
            norm_rv_pred: (batch_size, 1) normalized realized variance forecast
        """
        b, t = input_ids.size()
        assert t <= self.max_seq_len, f"Sequence length {t} exceeds max {self.max_seq_len}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        x = self.drop(x)
        
        mask = self._generate_causal_mask(t, input_ids.device)
        h = self.transformer(x, mask=mask, is_causal=True)
        h = self.ln_f(h)
        
        logits = self.lm_head(h)
        # Volatility prediction from final hidden state
        last_hidden = h[:, -1, :]
        norm_rv_pred = self.vol_head(last_hidden)
        
        return logits, norm_rv_pred

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @torch.no_grad()
    def autoregressive_rollout(
        self,
        prefix_ids: torch.Tensor,
        rollout_steps: int = 12,
        temperature: float = 0.8
    ) -> torch.Tensor:
        """
        Generates next token sequence autoregressively.
        prefix_ids: shape (1, seq_len)
        Returns generated tokens: shape (1, rollout_steps)
        """
        curr = prefix_ids.clone()
        generated = []
        for _ in range(rollout_steps):
            if curr.size(1) > self.max_seq_len - 1:
                curr = curr[:, -(self.max_seq_len - 1):]
            logits, _ = self.forward(curr)
            next_token_logits = logits[:, -1, :] / max(temperature, 1e-4)
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated.append(next_token)
            curr = torch.cat([curr, next_token], dim=1)
        return torch.cat(generated, dim=1)

if __name__ == "__main__":
    model = TimeSeriesGPT(vocab_size=120, max_seq_len=256, d_model=128, num_layers=4)
    param_count = model.count_parameters()
    print(f"Model Parameter Count: {param_count:,} (~{param_count / 1e6:.2f}M params)")
    
    dummy_input = torch.randint(0, 120, (2, 64))
    logits, rv_pred = model(dummy_input)
    print(f"Logits shape: {logits.shape}, RV pred shape: {rv_pred.shape}")
    
    rollout = model.autoregressive_rollout(dummy_input[:1, :10], rollout_steps=12)
    print("Rollout shape:", rollout.shape)

