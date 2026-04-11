import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data import S11AlignedDataset, build_bundle, discover_s11_records
from src.models import ConvTransformerRegressor, SpectrumTransformerRegressor
from src.train.train_state import evaluate_model_state
from src.utils import ensure_dir, save_json, timestamp
from src.visualization import plot_attention_heatmap, plot_prediction_scatter


def build_model(cfg, bundle, model_arch: str):
    common = dict(
        d_in=bundle.feature_dim,
        n_freq=len(bundle.freqs),
        d_model=cfg.get("d_model", 64),
        nhead=cfg.get("nhead", 8),
        num_layers=cfg.get("layers", 4),
        dim_feedforward=cfg.get("ffn", 256),
        dropout=cfg.get("dropout", 0.1),
    )
    if model_arch == "transformer":
        return SpectrumTransformerRegressor(**common)
    if model_arch == "conv_transformer":
        return ConvTransformerRegressor(
            **common,
            conv_channels=cfg.get("conv_channels", 32),
            kernel_size=cfg.get("kernel_size", 9),
            patch_stride=cfg.get("patch_stride", 4),
        )
    raise ValueError(f"Unsupported model_arch={model_arch}")


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
    model_arch = cfg.get("model_arch", "transformer")
    model = build_model(cfg, bundle, model_arch).to(device)
    model.load_state_dict(ckpt["model"])

    print(
        f"Loaded {len(records)} records for eval | model={model_arch} | n_freq={len(bundle.freqs)} "
        f"feature_dim={bundle.feature_dim} | batch_size={args.batch_size} num_workers={args.num_workers} "
        f"pin_memory={pin_memory}"
    )

    metrics, y_true, y_pred, attention = evaluate_model_state(
        model,
        loader,
        device,
        inverse_target=lambda y: np.asarray(y, dtype=np.float32) * 100.0,
        collect_attention=True,
    )
    out_dir = ensure_dir(Path(args.output_root) / "eval" / (args.tag or timestamp()))
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    save_json(out_dir / "metrics.json", metrics)
    plot_prediction_scatter(y_true, y_pred, out_dir / "scatter.png", title="Evaluation Scatter")
    if attention is not None and attention.ndim == 1 and len(attention) == len(bundle.freqs):
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
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
