from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .discovery import S11Record


META_COLUMNS = ("Cycle", "DC", "SOC")


@dataclass
class S11DatasetBundle:
    x: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    freqs: list[float]
    feature_dim: int
    scaler: tuple[np.ndarray, np.ndarray] | None


class S11AlignedDataset(Dataset):
    def __init__(self, bundle: S11DatasetBundle):
        self.X = bundle.x.astype(np.float32)
        self.y = bundle.y.astype(np.float32)
        self.groups = bundle.groups
        self.freqs = bundle.freqs
        self.feature_dim = bundle.feature_dim
        self.n_freq = len(bundle.freqs)
        self.scaler = bundle.scaler

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.from_numpy(self.X[idx]), torch.tensor(self.y[idx], dtype=torch.float32)


def _frequency_columns(df: pd.DataFrame) -> list[str]:
    cols = []
    for col in df.columns:
        if isinstance(col, str) and col.startswith("F"):
            cols.append(col)
    return cols


def _parse_freq(col: str) -> float:
    return float(str(col).replace("F", ""))


def _read_pair(record: S11Record, include_phase: bool) -> tuple[pd.DataFrame, pd.DataFrame | None]:
    mag_df = pd.read_csv(record.mag_path)
    pha_df = None
    if include_phase:
        if record.pha_path is None:
            raise ValueError(f"Missing phase file for {record.record_id}")
        pha_df = pd.read_csv(record.pha_path)
        if len(mag_df) != len(pha_df):
            raise ValueError(f"Length mismatch for {record.record_id}: mag={len(mag_df)} pha={len(pha_df)}")
        for col in META_COLUMNS:
            if col in mag_df.columns and col in pha_df.columns:
                mag_vals = mag_df[col].astype(str).to_numpy()
                pha_vals = pha_df[col].astype(str).to_numpy()
                if not np.array_equal(mag_vals, pha_vals):
                    raise ValueError(f"Metadata mismatch on {col} for {record.record_id}")
    return mag_df, pha_df


def build_bundle(
    records: list[S11Record],
    include_phase: bool = True,
    dc_mode: str = "all",
    amp_mode: str = "db_to_linear",
    freq_min: Optional[float] = None,
    freq_max: Optional[float] = None,
    scaler: tuple[np.ndarray, np.ndarray] | None = None,
) -> S11DatasetBundle:
    if not records:
        raise ValueError("No S11 records found for dataset construction")

    x_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    freq_template: list[float] | None = None

    for record in records:
        mag_df, pha_df = _read_pair(record, include_phase=include_phase)
        freq_cols = _frequency_columns(mag_df)
        if pha_df is not None:
            pha_freq_cols = _frequency_columns(pha_df)
            freq_cols = [col for col in freq_cols if col in pha_freq_cols]

        selected_cols = []
        selected_freqs = []
        for col in freq_cols:
            freq = _parse_freq(col)
            if freq_min is not None and freq < float(freq_min):
                continue
            if freq_max is not None and freq > float(freq_max):
                continue
            selected_cols.append(col)
            selected_freqs.append(freq)

        if not selected_cols:
            continue
        if freq_template is None:
            freq_template = selected_freqs
        elif freq_template != selected_freqs:
            raise ValueError(f"Frequency layout mismatch for {record.record_id}")

        mask = np.ones(len(mag_df), dtype=bool)
        if dc_mode in {"C", "D"} and "DC" in mag_df.columns:
            mask &= mag_df["DC"].astype(str).str.upper().eq(dc_mode).to_numpy()
        if not mask.any():
            continue

        mag_values = mag_df.loc[mask, selected_cols].astype(np.float32).to_numpy()
        if amp_mode == "db_to_linear":
            mag_values = np.power(10.0, mag_values / 20.0)

        if include_phase and pha_df is not None:
            pha_values = pha_df.loc[mask, selected_cols].astype(np.float32).to_numpy()
            pha_rad = np.deg2rad(pha_values)
            features = np.stack([mag_values, np.sin(pha_rad), np.cos(pha_rad)], axis=-1)
        else:
            features = mag_values[..., None]

        soc = mag_df.loc[mask, "SOC"].astype(np.float32).to_numpy() / 100.0
        groups = np.array([record.record_id] * len(soc), dtype=object)

        x_parts.append(features)
        y_parts.append(soc)
        group_parts.append(groups)

    if not x_parts or freq_template is None:
        raise ValueError("Dataset is empty after applying current filters")

    x = np.concatenate(x_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    groups = np.concatenate(group_parts, axis=0)

    fitted_scaler = scaler
    if amp_mode == "zscore":
        if scaler is None:
            mean = x[:, :, 0].mean(axis=0, keepdims=True)
            std = x[:, :, 0].std(axis=0, keepdims=True) + 1e-6
            fitted_scaler = (mean.astype(np.float32), std.astype(np.float32))
        mean, std = fitted_scaler
        x[:, :, 0] = (x[:, :, 0] - mean) / std

    return S11DatasetBundle(
        x=x,
        y=y,
        groups=groups,
        freqs=freq_template,
        feature_dim=x.shape[2],
        scaler=fitted_scaler,
    )
