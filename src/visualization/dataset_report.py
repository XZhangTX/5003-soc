import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.dataset import build_bundle
from src.data.dataset_soh_proxy import build_bundle_soh_proxy, inverse_cycle_scale
from src.data.discovery import discover_s11_records
from src.utils import ensure_dir, save_json, set_seed, timestamp



def _split_records(records, test_ratio: float, seed: int):
    groups = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(groups, groups=groups))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records



def _frequency_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if isinstance(col, str) and col.startswith("F")]



def _read_mag(record):
    return pd.read_csv(record.mag_path)



def _read_pha(record):
    if record.pha_path is None:
        return None
    return pd.read_csv(record.pha_path)



def _select_freq_cols(df: pd.DataFrame, freq_min=None, freq_max=None):
    cols = []
    freqs = []
    for col in _frequency_columns(df):
        freq = float(col.replace("F", ""))
        if freq_min is not None and freq < float(freq_min):
            continue
        if freq_max is not None and freq > float(freq_max):
            continue
        cols.append(col)
        freqs.append(freq)
    return cols, np.asarray(freqs, dtype=np.float32)



def _apply_amp_mode(values: np.ndarray, amp_mode: str):
    if amp_mode == "db_to_linear":
        return np.power(10.0, values / 20.0)
    return values



def _unwrap_phase_deg(phase_deg: np.ndarray) -> np.ndarray:
    phase_rad = np.deg2rad(np.asarray(phase_deg, dtype=np.float32))
    return np.rad2deg(np.unwrap(phase_rad))



def _pick_record_most_cycles(records):
    best = None
    best_cycles = -1
    for record in records:
        mag_df = _read_mag(record)
        if "Cycle" not in mag_df.columns:
            continue
        n_cycles = int(mag_df["Cycle"].nunique())
        if n_cycles > best_cycles:
            best = record
            best_cycles = n_cycles
    return best



def _pick_record_with_soc_coverage(records, target_socs=(20.0, 50.0, 80.0, 100.0), require_phase=False):
    best = None
    best_score = float("inf")
    for record in records:
        if require_phase and record.pha_path is None:
            continue
        mag_df = _read_mag(record)
        if "SOC" not in mag_df.columns:
            continue
        soc_values = mag_df["SOC"].astype(np.float32).to_numpy()
        if len(np.unique(soc_values)) < 4:
            continue
        score = 0.0
        for target in target_socs:
            score += float(np.min(np.abs(soc_values - target)))
        if score < best_score:
            best = record
            best_score = score
    return best



def _pick_records_same_soc(records, target_soc=50.0, top_k=3):
    scored = []
    for record in records:
        mag_df = _read_mag(record)
        if "SOC" not in mag_df.columns:
            continue
        median_soc = float(mag_df["SOC"].astype(np.float32).median())
        scored.append((abs(median_soc - target_soc), record, median_soc))
    scored.sort(key=lambda item: item[0])
    return [(record, soc) for _, record, soc in scored[:top_k]]



def _plot_same_record_different_soc(record, out_path: Path, freq_min=None, freq_max=None, target_socs=(20.0, 50.0, 80.0, 100.0)):
    mag_df = _read_mag(record)
    freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
    if len(freq_cols) == 0 or "SOC" not in mag_df.columns:
        return
    soc_values = mag_df["SOC"].astype(np.float32).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    used = []
    for target in target_socs:
        nearest = float(soc_values[np.argmin(np.abs(soc_values - target))])
        if any(abs(nearest - seen) < 1e-6 for seen in used):
            continue
        used.append(nearest)
        mask = np.isclose(soc_values, nearest)
        curve = mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        ax.plot(freqs, curve, linewidth=1.6, label=f"SOC {nearest:.1f}%")
    ax.set_title(f"Same Record, Different SOC: {record.record_id}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_same_record_different_phase_soc(record, out_path: Path, freq_min=None, freq_max=None, target_socs=(20.0, 50.0, 80.0, 100.0)):
    pha_df = _read_pha(record)
    if pha_df is None or "SOC" not in pha_df.columns:
        return
    freq_cols, freqs = _select_freq_cols(pha_df, freq_min=freq_min, freq_max=freq_max)
    if len(freq_cols) == 0:
        return
    soc_values = pha_df["SOC"].astype(np.float32).to_numpy()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    used = []
    for target in target_socs:
        nearest = float(soc_values[np.argmin(np.abs(soc_values - target))])
        if any(abs(nearest - seen) < 1e-6 for seen in used):
            continue
        used.append(nearest)
        mask = np.isclose(soc_values, nearest)
        curve = pha_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        curve = _unwrap_phase_deg(curve)
        ax.plot(freqs, curve, linewidth=1.6, label=f"SOC {nearest:.1f}%")
    ax.set_title(f"Same Record, Different SOC Phase: {record.record_id}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Unwrapped Phase (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_different_records_same_soc(records_with_soc, out_path: Path, freq_min=None, freq_max=None):
    if not records_with_soc:
        return
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for record, median_soc in records_with_soc:
        mag_df = _read_mag(record)
        freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
        soc_values = mag_df["SOC"].astype(np.float32).to_numpy()
        target = float(median_soc)
        nearest = float(mag_df.iloc[np.argmin(np.abs(soc_values - target))]["SOC"])
        mask = np.isclose(soc_values, nearest)
        curve = mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        ax.plot(freqs, curve, linewidth=1.6, label=f"{record.record_id} | SOC~{nearest:.1f}")
    ax.set_title("Different Records at Similar SOC")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_amp_mode_comparison(record, out_path: Path, freq_min=None, freq_max=None):
    mag_df = _read_mag(record)
    freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
    cycles = np.sort(mag_df["Cycle"].astype(np.float32).unique())
    if len(cycles) == 0 or len(freq_cols) == 0:
        return
    cycle = float(cycles[len(cycles) // 2])
    mask = mag_df["Cycle"].astype(np.float32).to_numpy() == cycle
    raw_db = mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
    db_to_linear = _apply_amp_mode(raw_db.copy(), "db_to_linear")
    zscore = (raw_db - raw_db.mean()) / (raw_db.std() + 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), sharex=True)
    curves = [(raw_db, "raw_db"), (db_to_linear, "db_to_linear"), (zscore, "zscore")]
    for ax, (curve, title) in zip(axes, curves):
        ax.plot(freqs, curve, linewidth=1.5)
        ax.set_title(title)
        ax.set_xlabel("Frequency")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Value")
    fig.suptitle(f"Preprocessing Comparison on {record.record_id} (Cycle {int(cycle)})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_phase_wrap_vs_unwrap(record, out_path: Path, freq_min=None, freq_max=None):
    pha_df = _read_pha(record)
    if pha_df is None:
        return
    freq_cols, freqs = _select_freq_cols(pha_df, freq_min=freq_min, freq_max=freq_max)
    if len(freq_cols) == 0:
        return
    curve = pha_df.loc[:, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
    unwrapped = _unwrap_phase_deg(curve)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(freqs, curve, linewidth=1.2, label="Wrapped Phase")
    ax.plot(freqs, unwrapped, linewidth=1.6, label="Unwrapped Phase")
    ax.set_title(f"Wrapped vs Unwrapped Phase: {record.record_id}")
    ax.set_xlabel("Frequency")
    ax.set_ylabel("Phase (deg)")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_complex_plane_by_soc(record, out_path: Path, freq_min=None, freq_max=None, target_socs=(20.0, 50.0, 80.0, 100.0)):
    mag_df = _read_mag(record)
    pha_df = _read_pha(record)
    if pha_df is None or "SOC" not in mag_df.columns:
        return
    freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
    if len(freq_cols) == 0:
        return
    soc_values = mag_df["SOC"].astype(np.float32).to_numpy()

    fig, ax = plt.subplots(figsize=(6, 6))
    used = []
    for target in target_socs:
        nearest = float(soc_values[np.argmin(np.abs(soc_values - target))])
        if any(abs(nearest - seen) < 1e-6 for seen in used):
            continue
        used.append(nearest)
        mask = np.isclose(soc_values, nearest)
        mag_curve = _apply_amp_mode(mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0), "db_to_linear")
        pha_curve = pha_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        pha_rad = np.deg2rad(_unwrap_phase_deg(pha_curve))
        real = mag_curve * np.cos(pha_rad)
        imag = mag_curve * np.sin(pha_rad)
        ax.plot(real, imag, linewidth=1.5, label=f"SOC {nearest:.1f}%")
    ax.set_title(f"Complex-Plane S11 Trajectories: {record.record_id}")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_single_distribution(values, title, xlabel, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.hist(values, bins=np.linspace(0.0, 100.0, 21), color="#1f77b4", alpha=0.8, edgecolor="white", linewidth=0.6)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_split_distribution(train_values, val_values, title, xlabel, out_path: Path):
    fig, ax = plt.subplots(figsize=(5.5, 4))
    bins = np.linspace(0.0, 100.0, 21)
    ax.hist(train_values, bins=bins, alpha=0.65, label="Train", color="#1f77b4", edgecolor="white", linewidth=0.5)
    ax.hist(val_values, bins=bins, alpha=0.55, label="Val", color="#ff7f0e", edgecolor="white", linewidth=0.5)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _extract_resonance_features(values: np.ndarray, freqs: np.ndarray):
    min_idx = int(np.argmin(values))
    return float(freqs[min_idx]), float(values[min_idx]), min_idx



def _collect_soc_resonance_points(records, freq_min=None, freq_max=None, dc_mode="all"):
    socs, res_freqs, min_mags = [], [], []
    for record in records:
        mag_df = _read_mag(record)
        freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
        if len(freq_cols) == 0:
            continue
        mask = np.ones(len(mag_df), dtype=bool)
        if dc_mode in {"C", "D"} and "DC" in mag_df.columns:
            mask &= mag_df["DC"].astype(str).str.upper().eq(dc_mode).to_numpy()
        rows = mag_df.loc[mask, ["SOC", *freq_cols]]
        for _, row in rows.iterrows():
            vals = row[freq_cols].astype(np.float32).to_numpy()
            rf, mm, _ = _extract_resonance_features(vals, freqs)
            socs.append(float(row["SOC"]))
            res_freqs.append(rf)
            min_mags.append(mm)
    return np.asarray(socs), np.asarray(res_freqs), np.asarray(min_mags)



def _collect_soh_resonance_points(records, freq_min=None, freq_max=None, dc_mode="all"):
    curve_features = []
    cycles = []
    for record in records:
        mag_df = _read_mag(record)
        freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
        if len(freq_cols) == 0 or "Cycle" not in mag_df.columns:
            continue
        mask = np.ones(len(mag_df), dtype=bool)
        if dc_mode in {"C", "D"} and "DC" in mag_df.columns:
            mask &= mag_df["DC"].astype(str).str.upper().eq(dc_mode).to_numpy()
        if not mask.any():
            continue
        curve = mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        rf, mm, _ = _extract_resonance_features(curve, freqs)
        cycle = float(mag_df.loc[mask, "Cycle"].astype(np.float32).iloc[0])
        curve_features.append((rf, mm))
        cycles.append(cycle)
    if not cycles:
        return np.asarray([]), np.asarray([]), np.asarray([])
    cycles = np.asarray(cycles, dtype=np.float32)
    cmin = float(cycles.min())
    cmax = float(cycles.max())
    soh_proxy = (cycles - cmin) / max(cmax - cmin, 1.0) * 100.0
    features = np.asarray(curve_features, dtype=np.float32)
    return soh_proxy, features[:, 0], features[:, 1]



def _collect_phase_slope_points(records, target="soc", freq_min=None, freq_max=None, dc_mode="all"):
    targets, slopes = [], []
    cycles = []
    curve_cache = []
    for record in records:
        mag_df = _read_mag(record)
        pha_df = _read_pha(record)
        if pha_df is None:
            continue
        freq_cols, freqs = _select_freq_cols(mag_df, freq_min=freq_min, freq_max=freq_max)
        if len(freq_cols) == 0:
            continue
        mask = np.ones(len(mag_df), dtype=bool)
        if dc_mode in {"C", "D"} and "DC" in mag_df.columns:
            mask &= mag_df["DC"].astype(str).str.upper().eq(dc_mode).to_numpy()
        if not mask.any():
            continue
        mag_rows = mag_df.loc[mask, ["SOC", "Cycle", *freq_cols]]
        pha_rows = pha_df.loc[mask, ["SOC", "Cycle", *freq_cols]]
        if target == "soc":
            for idx in range(len(mag_rows)):
                mag_curve = mag_rows.iloc[idx][freq_cols].astype(np.float32).to_numpy()
                pha_curve = pha_rows.iloc[idx][freq_cols].astype(np.float32).to_numpy()
                _, _, min_idx = _extract_resonance_features(mag_curve, freqs)
                unwrapped = _unwrap_phase_deg(pha_curve)
                grad = np.gradient(unwrapped, freqs)
                slopes.append(float(grad[min_idx]))
                targets.append(float(mag_rows.iloc[idx]["SOC"]))
        else:
            mag_curve = mag_rows[freq_cols].astype(np.float32).to_numpy().mean(axis=0)
            pha_curve = pha_rows[freq_cols].astype(np.float32).to_numpy().mean(axis=0)
            _, _, min_idx = _extract_resonance_features(mag_curve, freqs)
            unwrapped = _unwrap_phase_deg(pha_curve)
            grad = np.gradient(unwrapped, freqs)
            slopes.append(float(grad[min_idx]))
            cycles.append(float(mag_rows["Cycle"].astype(np.float32).iloc[0]))
    if target == "soc":
        return np.asarray(targets), np.asarray(slopes)
    cycles = np.asarray(cycles, dtype=np.float32)
    if len(cycles) == 0:
        return np.asarray([]), np.asarray([])
    cmin = float(cycles.min())
    cmax = float(cycles.max())
    soh_proxy = (cycles - cmin) / max(cmax - cmin, 1.0) * 100.0
    return soh_proxy, np.asarray(slopes)



def _plot_feature_vs_target(x, y, xlabel, ylabel, out_path: Path):
    if len(x) == 0 or len(y) == 0:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.scatter(x, y, s=12, alpha=0.45)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_soc_soh_joint_distribution(records, out_path: Path, dc_mode="all"):
    rows = []
    cycles = []
    for record in records:
        mag_df = _read_mag(record)
        if "SOC" not in mag_df.columns or "Cycle" not in mag_df.columns:
            continue
        mask = np.ones(len(mag_df), dtype=bool)
        if dc_mode in {"C", "D"} and "DC" in mag_df.columns:
            mask &= mag_df["DC"].astype(str).str.upper().eq(dc_mode).to_numpy()
        if not mask.any():
            continue
        cycle = float(mag_df.loc[mask, "Cycle"].astype(np.float32).iloc[0])
        soc_vals = mag_df.loc[mask, "SOC"].astype(np.float32).to_numpy()
        rows.extend([(float(soc), cycle) for soc in soc_vals])
        cycles.append(cycle)
    if not rows:
        return
    cycles_arr = np.asarray(cycles, dtype=np.float32)
    cmin = float(cycles_arr.min())
    cmax = float(cycles_arr.max())
    soc_vals = np.asarray([r[0] for r in rows], dtype=np.float32)
    cycle_vals = np.asarray([r[1] for r in rows], dtype=np.float32)
    soh_proxy = (cycle_vals - cmin) / max(cmax - cmin, 1.0) * 100.0

    fig, ax = plt.subplots(figsize=(6, 5))
    hb = ax.hexbin(soc_vals, soh_proxy, gridsize=25, cmap="viridis", mincnt=1)
    fig.colorbar(hb, ax=ax, label="Count")
    ax.set_xlabel("SOC (%)")
    ax.set_ylabel("SOH Proxy")
    ax.set_title("SOC-SOH Joint Distribution")
    ax.grid(True, alpha=0.15)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def main(args):
    set_seed(args.seed)
    records = discover_s11_records(
        args.input_root,
        name_contains=args.name_contains,
        require_phase=args.include_phase,
        data_mode=args.data_mode,
    )
    if not records:
        raise ValueError(f"No matching S11 records found under {args.input_root}")

    train_records, val_records = _split_records(records, args.test_ratio, args.seed)

    soc_train = build_bundle(
        train_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    soc_val = build_bundle(
        val_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        scaler=soc_train.scaler,
    )
    soh_train = build_bundle_soh_proxy(
        train_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    soh_val = build_bundle_soh_proxy(
        val_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        scaler=soh_train.scaler,
        label_scale=soh_train.label_scale,
    )

    out_dir = ensure_dir(Path(args.output_root) / "dataset_report" / (args.tag or timestamp()))
    figures_dir = ensure_dir(out_dir / "figures")

    soc_train_pct = soc_train.y * 100.0
    soc_val_pct = soc_val.y * 100.0
    soh_train_pct = inverse_cycle_scale(soh_train.y, soh_train.label_scale)
    soh_val_pct = inverse_cycle_scale(soh_val.y, soh_train.label_scale)

    cycle_counts = []
    train_ids = {r.record_id for r in train_records}
    for record in records:
        mag_df = _read_mag(record)
        cycle_counts.append(int(mag_df["Cycle"].nunique()) if "Cycle" in mag_df.columns else 0)

    summary_rows = [
        {
            "split": "all",
            "record_count": len(records),
            "train_record_count": len(train_records),
            "val_record_count": len(val_records),
            "soc_sample_count": int(len(soc_train.x) + len(soc_val.x)),
            "soh_sample_count": int(len(soh_train.x) + len(soh_val.x)),
            "n_freq": int(len(soc_train.freqs)),
            "feature_dim": int(soc_train.feature_dim),
            "mean_cycles_per_record": float(np.mean(cycle_counts)),
            "min_cycles_per_record": int(np.min(cycle_counts)),
            "max_cycles_per_record": int(np.max(cycle_counts)),
            "has_phase_required": bool(args.include_phase),
            "amp_mode": args.amp_mode,
            "data_mode": args.data_mode,
        },
        {
            "split": "train",
            "record_count": len(train_records),
            "soc_sample_count": int(len(soc_train.x)),
            "soh_sample_count": int(len(soh_train.x)),
            "soc_min": float(soc_train_pct.min()),
            "soc_max": float(soc_train_pct.max()),
            "soh_min": float(soh_train_pct.min()),
            "soh_max": float(soh_train_pct.max()),
        },
        {
            "split": "val",
            "record_count": len(val_records),
            "soc_sample_count": int(len(soc_val.x)),
            "soh_sample_count": int(len(soh_val.x)),
            "soc_min": float(soc_val_pct.min()),
            "soc_max": float(soc_val_pct.max()),
            "soh_min": float(soh_val_pct.min()),
            "soh_max": float(soh_val_pct.max()),
        },
    ]
    pd.DataFrame(summary_rows).to_csv(out_dir / "summary.csv", index=False)

    pd.DataFrame(
        {
            "record_id": [record.record_id for record in records],
            "day": [record.day for record in records],
            "series_name": [record.series_name for record in records],
            "has_phase": [record.has_phase for record in records],
            "cycle_count": cycle_counts,
            "split": ["train" if record.record_id in train_ids else "val" for record in records],
        }
    ).to_csv(out_dir / "record_summary.csv", index=False)

    save_json(
        out_dir / "record_lists.json",
        {
            "all_records": [record.record_id for record in records],
            "train_records": [record.record_id for record in train_records],
            "val_records": [record.record_id for record in val_records],
        },
    )

    _plot_single_distribution(soc_train_pct.tolist() + soc_val_pct.tolist(), "SOC Distribution", "SOC (%)", figures_dir / "soc_distribution.png")
    _plot_single_distribution(soh_train_pct.tolist() + soh_val_pct.tolist(), "SOH Proxy Distribution", "SOH Proxy", figures_dir / "soh_distribution.png")
    _plot_split_distribution(soc_train_pct, soc_val_pct, "SOC Train/Val Split", "SOC (%)", figures_dir / "soc_split_distribution.png")
    _plot_split_distribution(soh_train_pct, soh_val_pct, "SOH Proxy Train/Val Split", "SOH Proxy", figures_dir / "soh_split_distribution.png")

    rep_record = _pick_record_most_cycles(records)
    if rep_record is not None:
        _plot_amp_mode_comparison(rep_record, figures_dir / "amp_mode_comparison.png", freq_min=args.freq_min, freq_max=args.freq_max)
        if args.include_phase and rep_record.pha_path is not None:
            _plot_phase_wrap_vs_unwrap(rep_record, figures_dir / "phase_wrap_vs_unwrap.png", freq_min=args.freq_min, freq_max=args.freq_max)

    soc_record = _pick_record_with_soc_coverage(records, require_phase=False)
    if soc_record is not None:
        _plot_same_record_different_soc(soc_record, figures_dir / "same_record_different_soc.png", freq_min=args.freq_min, freq_max=args.freq_max)

    if args.include_phase:
        phase_record = _pick_record_with_soc_coverage(records, require_phase=True)
        if phase_record is not None:
            _plot_same_record_different_phase_soc(phase_record, figures_dir / "same_record_different_phase_soc.png", freq_min=args.freq_min, freq_max=args.freq_max)
            _plot_complex_plane_by_soc(phase_record, figures_dir / "complex_plane_by_soc.png", freq_min=args.freq_min, freq_max=args.freq_max)

    same_soc_records = _pick_records_same_soc(records, target_soc=args.reference_soc, top_k=3)
    _plot_different_records_same_soc(same_soc_records, figures_dir / "different_records_same_soc.png", freq_min=args.freq_min, freq_max=args.freq_max)

    soc_targets, soc_res_freqs, soc_min_mags = _collect_soc_resonance_points(records, freq_min=args.freq_min, freq_max=args.freq_max, dc_mode=args.dc_mode)
    _plot_feature_vs_target(soc_res_freqs, soc_targets, "Resonance Frequency", "SOC (%)", figures_dir / "resonance_frequency_vs_soc.png")
    _plot_feature_vs_target(soc_min_mags, soc_targets, "Minimum Magnitude (dB)", "SOC (%)", figures_dir / "minimum_magnitude_vs_soc.png")

    soh_targets, soh_res_freqs, soh_min_mags = _collect_soh_resonance_points(records, freq_min=args.freq_min, freq_max=args.freq_max, dc_mode=args.dc_mode)
    _plot_feature_vs_target(soh_res_freqs, soh_targets, "Resonance Frequency", "SOH Proxy", figures_dir / "resonance_frequency_vs_soh.png")
    _plot_feature_vs_target(soh_min_mags, soh_targets, "Minimum Magnitude (dB)", "SOH Proxy", figures_dir / "minimum_magnitude_vs_soh.png")

    if args.include_phase:
        soc_phase_targets, soc_phase_slopes = _collect_phase_slope_points(records, target="soc", freq_min=args.freq_min, freq_max=args.freq_max, dc_mode=args.dc_mode)
        _plot_feature_vs_target(soc_phase_slopes, soc_phase_targets, "Phase Slope at Resonance", "SOC (%)", figures_dir / "phase_slope_vs_soc.png")
        soh_phase_targets, soh_phase_slopes = _collect_phase_slope_points(records, target="soh", freq_min=args.freq_min, freq_max=args.freq_max, dc_mode=args.dc_mode)
        _plot_feature_vs_target(soh_phase_slopes, soh_phase_targets, "Phase Slope at Resonance", "SOH Proxy", figures_dir / "phase_slope_vs_soh.png")

    _plot_soc_soh_joint_distribution(records, figures_dir / "soc_soh_joint_distribution.png", dc_mode=args.dc_mode)

    print(f"Saved dataset report artifacts to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export dataset statistics and paper-ready visualizations")
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--include-phase", action="store_true")
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="zscore", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument("--reference-soc", type=float, default=50.0)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
