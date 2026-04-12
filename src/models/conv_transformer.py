import math
from typing import Optional

import torch
import torch.nn as nn

from .soc_itransformer import SinusoidalPositionalEncoding


class ConvTransformerRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_freq: int,
        d_model: int = 64,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        conv_channels: int = 32,
        kernel_size: int = 9,
        patch_stride: int = 4,
        use_pos_enc: bool = True,
        use_token_embed: bool = True,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.d_in = d_in
        self.use_pos_enc = use_pos_enc
        self.use_token_embed = use_token_embed
        self.last_freq_attn: Optional[torch.Tensor] = None

        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(d_in, conv_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=kernel_size, stride=patch_stride, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )

        # Infer token length from the actual stem so the embedding stays aligned
        # for both odd and even kernel sizes.
        with torch.no_grad():
            dummy = torch.zeros(1, d_in, n_freq)
            token_len = self.stem(dummy).shape[-1]
        self.token_len = token_len
        self.token_embed = nn.Parameter(torch.zeros(1, token_len, d_model))
        nn.init.trunc_normal_(self.token_embed, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=token_len + 1)
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
        batch_size, length, channels = x.size()
        if length != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequencies, got {length}")
        if channels != self.d_in:
            raise ValueError(f"Expected feature dim {self.d_in}, got {channels}")

        h = x.transpose(1, 2)
        h = self.stem(h)
        h = h.transpose(1, 2)
        if self.use_token_embed:
            h = h + self.token_embed[:, : h.size(1), :]

        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)
        if self.use_pos_enc:
            h = self.pos_enc(h)
        h = self.encoder(h)
        cls_out = self.norm(h[:, 0, :])
        return self.head(cls_out).squeeze(-1)


class CNNOnlyRegressor(nn.Module):
    def __init__(
        self,
        d_in: int,
        n_freq: int,
        d_model: int = 64,
        dropout: float = 0.1,
        conv_channels: int = 32,
        kernel_size: int = 9,
        patch_stride: int = 4,
    ):
        super().__init__()
        self.n_freq = n_freq
        self.d_in = d_in
        self.last_freq_attn: Optional[torch.Tensor] = None

        padding = kernel_size // 2
        self.stem = nn.Sequential(
            nn.Conv1d(d_in, conv_channels, kernel_size=kernel_size, stride=1, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.GELU(),
            nn.Conv1d(conv_channels, d_model, kernel_size=kernel_size, stride=patch_stride, padding=padding),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, length, channels = x.size()
        if length != self.n_freq:
            raise ValueError(f"Expected {self.n_freq} frequencies, got {length}")
        if channels != self.d_in:
            raise ValueError(f"Expected feature dim {self.d_in}, got {channels}")

        h = x.transpose(1, 2)
        h = self.stem(h)
        h = self.pool(h).reshape(batch_size, -1)
        return self.head(h).squeeze(-1)

