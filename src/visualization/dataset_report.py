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


META_COLUMNS = {"Cycle", "DC", "SOC"}


def _split_records(records, test_ratio: float, seed: int):
    groups = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(groups, groups=groups))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records



def _frequency_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if isinstance(col, str) and col.startswith("F")]



def _pick_representative_record(records):
    best = None
    best_cycles = -1
    for record in records:
        mag_df = pd.read_csv(record.mag_path)
        if "Cycle" not in mag_df.columns:
            continue
        n_cycles = mag_df["Cycle"].nunique()
        if n_cycles > best_cycles:
            best = record
            best_cycles = n_cycles
    return best



def _plot_label_distributions(soc_train, soc_val, soh_train, soh_val, out_path: Path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    bins_soc = np.linspace(0.0, 100.0, 21)
    axes[0].hist(soc_train, bins=bins_soc, alpha=0.65, label="Train", color="#1f77b4")
    axes[0].hist(soc_val, bins=bins_soc, alpha=0.55, label="Val", color="#ff7f0e")
    axes[0].set_title("SOC Distribution")
    axes[0].set_xlabel("SOC (%)")
    axes[0].set_ylabel("Count")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    bins_soh = np.linspace(0.0, 100.0, 21)
    axes[1].hist(soh_train, bins=bins_soh, alpha=0.65, label="Train", color="#1f77b4")
    axes[1].hist(soh_val, bins=bins_soh, alpha=0.55, label="Val", color="#ff7f0e")
    axes[1].set_title("SOH Proxy Distribution")
    axes[1].set_xlabel("SOH Proxy")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)



def _plot_representative_spectra(record, out_path: Path, include_phase: bool):
    mag_df = pd.read_csv(record.mag_path)
    freq_cols = _frequency_columns(mag_df)
    freqs = np.array([float(col.replace("F", "")) for col in freq_cols], dtype=np.float32)
    cycles = np.sort(mag_df["Cycle"].astype(np.float32).unique())
    if len(cycles) == 0:
        return
    selected_cycles = [cycles[0], cycles[len(cycles) // 2], cycles[-1]]
    selected_cycles = list(dict.fromkeys(float(c) for c in selected_cycles))

    nrows = 2 if include_phase and record.pha_path is not None else 1
    fig, axes = plt.subplots(nrows, 1, figsize=(10, 4 + 2.5 * (nrows - 1)), sharex=True)
    if nrows == 1:
        axes = [axes]

    for cycle in selected_cycles:
        mask = mag_df["Cycle"].astype(np.float32).to_numpy() == cycle
        mag_curve = mag_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
        axes[0].plot(freqs, mag_curve, linewidth=1.6, label=f"Cycle {int(cycle)}")

    axes[0].set_title(f"Representative Magnitude Spectra: {record.record_id}")
    axes[0].set_ylabel("Magnitude (dB)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    if nrows == 2:
        pha_df = pd.read_csv(record.pha_path)
        for cycle in selected_cycles:
            mask = pha_df["Cycle"].astype(np.float32).to_numpy() == cycle
            pha_curve = pha_df.loc[mask, freq_cols].astype(np.float32).to_numpy().mean(axis=0)
            axes[1].plot(freqs, pha_curve, linewidth=1.6, label=f"Cycle {int(cycle)}")
        axes[1].set_title(f"Representative Phase Spectra: {record.record_id}")
        axes[1].set_ylabel("Phase (deg)")
        axes[1].grid(True, alpha=0.25)

    axes[-1].set_xlabel("Frequency")
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

    stats_rows = [
        {"split": "all", "records": len(records), "soc_samples": len(soc_train.x) + len(soc_val.x), "soh_samples": len(soh_train.x) + len(soh_val.x), "n_freq": len(soc_train.freqs), "feature_dim": soc_train.feature_dim, "include_phase": args.include_phase, "amp_mode": args.amp_mode, "data_mode": args.data_mode},
        {"split": "train", "records": len(train_records), "soc_samples": len(soc_train.x), "soh_samples": len(soh_train.x), "soc_min": float((soc_train.y * 100.0).min()), "soc_max": float((soc_train.y * 100.0).max()), "soh_min": float(inverse_cycle_scale(soh_train.y, soh_train.label_scale).min()), "soh_max": float(inverse_cycle_scale(soh_train.y, soh_train.label_scale).max())},
        {"split": "val", "records": len(val_records), "soc_samples": len(soc_val.x), "soh_samples": len(soh_val.x), "soc_min": float((soc_val.y * 100.0).min()), "soc_max": float((soc_val.y * 100.0).max()), "soh_min": float(inverse_cycle_scale(soh_val.y, soh_train.label_scale).min()), "soh_max": float(inverse_cycle_scale(soh_val.y, soh_train.label_scale).max())},
    ]
    pd.DataFrame(stats_rows).to_csv(out_dir / "summary.csv", index=False)

    save_json(
        out_dir / "record_lists.json",
        {
            "all_records": [record.record_id for record in records],
            "train_records": [record.record_id for record in train_records],
            "val_records": [record.record_id for record in val_records],
        },
    )

    _plot_label_distributions(
        soc_train.y * 100.0,
        soc_val.y * 100.0,
        inverse_cycle_scale(soh_train.y, soh_train.label_scale),
        inverse_cycle_scale(soh_val.y, soh_train.label_scale),
        out_dir / "split_distribution.png",
    )

    rep_record = _pick_representative_record(records)
    if rep_record is not None:
        _plot_representative_spectra(rep_record, out_dir / "representative_spectra.png", include_phase=args.include_phase)

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
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
