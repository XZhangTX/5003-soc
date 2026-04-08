import math
from typing import Optional

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


class SpectrumTransformerRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_freq: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.input_proj = nn.Linear(d_in, d_model)
        self.freq_embed = nn.Embedding(n_freq, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=n_freq + 1)
        self.freq_gate = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        self.last_freq_attn: Optional[torch.Tensor] = None

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, _ = x.size()
        if length != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequencies, got {length}")

        h = self.input_proj(x)
        freq_index = torch.arange(length, device=x.device)
        h = h + self.freq_embed(freq_index)[None, :, :]

        gate_logits = self.freq_gate(h).squeeze(-1)
        gate = torch.softmax(gate_logits, dim=1)
        self.last_freq_attn = gate.detach().cpu()
        h = h * gate.unsqueeze(-1)

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        h = self.pos_enc(h)
        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])
        return self.head(cls_out).squeeze(-1)


ITransformerEncoder = SpectrumTransformerRegressor
