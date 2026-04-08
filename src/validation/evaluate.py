import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import S11AlignedDataset, build_bundle, discover_s11_records
from src.models import SpectrumTransformerRegressor
from src.train.train import evaluate_model
from src.utils import ensure_dir, save_json, timestamp
from src.visualization import plot_attention_heatmap, plot_prediction_scatter


def main(args):
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {})
    records = discover_s11_records(
        args.input_root,
        name_contains=args.name_contains,
        require_phase=args.include_phase,
        data_mode=args.data_mode,
    )
    if not records:
        raise ValueError(f"No matching S11 records found under {args.input_root}")

    scaler = None
    if ckpt.get("scaler") is not None:
        mean, std = ckpt["scaler"]
        scaler = (np.asarray(mean, dtype=np.float32), np.asarray(std, dtype=np.float32))

    bundle = build_bundle(
        records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=cfg.get("amp_mode", args.amp_mode),
        freq_min=cfg.get("freq_min", args.freq_min),
        freq_max=cfg.get("freq_max", args.freq_max),
        scaler=scaler,
    )
    ds = S11AlignedDataset(bundle)
    pin_memory = args.pin_memory and torch.cuda.is_available()
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumTransformerRegressor(
        d_in=bundle.feature_dim,
        n_freq=len(bundle.freqs),
        d_model=cfg.get("d_model", args.d_model),
        nhead=cfg.get("nhead", args.nhead),
        num_layers=cfg.get("layers", args.layers),
        dim_feedforward=cfg.get("ffn", args.ffn),
        dropout=cfg.get("dropout", args.dropout),
    ).to(device)
    model.load_state_dict(ckpt["model"])

    print(
        f"Loaded {len(records)} records for eval | n_freq={len(bundle.freqs)} feature_dim={bundle.feature_dim} | "
        f"batch_size={args.batch_size} num_workers={args.num_workers} pin_memory={pin_memory}"
    )

    metrics, y_true, y_pred, attention = evaluate_model(model, loader, device, collect_attention=True)
    out_dir = ensure_dir(Path(args.output_root) / "eval" / (args.tag or timestamp()))
    pd.DataFrame({"y_true": y_true * 100.0, "y_pred": y_pred * 100.0}).to_csv(out_dir / "predictions.csv", index=False)
    save_json(out_dir / "metrics.json", metrics)
    plot_prediction_scatter(y_true * 100.0, y_pred * 100.0, out_dir / "scatter.png", title="Evaluation Scatter")
    if attention is not None:
        pd.DataFrame({"freq": bundle.freqs, "weight": attention}).to_csv(out_dir / "attention.csv", index=False)
        plot_attention_heatmap(bundle.freqs, attention, out_dir / "attention_heatmap.png")
    print(f"Saved evaluation outputs to {out_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Evaluate a trained checkpoint on aligned S11 CSV files")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--include-phase", action="store_true")
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="db_to_linear", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--freq-min", type=float, default=1900000)
    parser.add_argument("--freq-max", type=float, default=2150000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
