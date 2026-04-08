import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data.discovery import discover_s11_records
from src.data.dataset_new import S11AlignedDatasetNew, build_bundle_new
from src.models import SpectrumTransformerRegressor
from src.utils import ensure_dir, save_json, set_seed, timestamp
from src.visualization import plot_attention_heatmap, plot_prediction_scatter, plot_training_curve



@torch.no_grad()
def evaluate_model_new(model, loader, device, collect_attention: bool = False):
    model.eval()
    ys, ps = [], []
    attn_list = [] if collect_attention else None
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
        if collect_attention and getattr(model, "last_freq_attn", None) is not None:
            attn_list.append(model.last_freq_attn.numpy())

    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    err = y_pred - y_true
    metrics = {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "r2": float(1.0 - (np.sum(np.square(err)) / (np.sum(np.square(y_true - np.mean(y_true))) + 1e-12))),
    }
    attention = None
    if collect_attention and attn_list:
        attention = np.concatenate(attn_list, axis=0).mean(axis=0)
    return metrics, y_true, y_pred, attention
def _split_records(records, test_ratio: float, seed: int):
    groups = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    train_idx, val_idx = next(splitter.split(groups, groups=groups))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    return train_records, val_records


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

    train_records, val_records = _split_records(records, test_ratio=args.test_ratio, seed=args.seed)
    train_bundle = build_bundle_new(
        train_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    val_bundle = build_bundle_new(
        val_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        scaler=train_bundle.scaler,
    )

    train_ds = S11AlignedDatasetNew(train_bundle)
    val_ds = S11AlignedDatasetNew(val_bundle)
    pin_memory = args.pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.val_num_workers,
        pin_memory=pin_memory,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SpectrumTransformerRegressor(
        d_in=train_bundle.feature_dim,
        n_freq=len(train_bundle.freqs),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
    ).to(device)

    print(
        f"Loaded {len(records)} records | train={len(train_records)} val={len(val_records)} | "
        f"n_freq={len(train_bundle.freqs)} feature_dim={train_bundle.feature_dim} | "
        f"batch_size={args.batch_size} num_workers={args.num_workers}/{args.val_num_workers} pin_memory={pin_memory}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_l1 = nn.L1Loss()
    loss_l2 = nn.MSELoss()

    run_dir = ensure_dir(Path(args.output_root) / "train" / (args.tag or f"{timestamp()}_new"))
    best_path = run_dir / "best.pt"
    log_rows = []
    best_rmse = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_loader, desc=f"train-{epoch}", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb)
            loss = 0.5 * loss_l1(pred, yb) + 0.5 * loss_l2(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        metrics, y_true, y_pred, attention = evaluate_model_new(model, val_loader, device, collect_attention=True)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(train_losses)),
            **metrics,
        }
        log_rows.append(row)
        print(f"Epoch {epoch:03d} | train_loss={row['train_loss']:.4f} | val_rmse={metrics['rmse']:.4f} | val_mae={metrics['mae']:.4f}")

        pred_df = val_bundle.sample_meta.copy()
        pred_df["y_true"] = y_true * 100.0
        pred_df["y_pred"] = y_pred * 100.0
        pred_df["abs_err"] = np.abs(pred_df["y_pred"] - pred_df["y_true"])

        pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
        pred_df.to_csv(run_dir / "val_predictions_new.csv", index=False)
        plot_prediction_scatter(pred_df["y_true"], pred_df["y_pred"], run_dir / "val_scatter.png", title="Validation Prediction Scatter")
        if attention is not None:
            pd.DataFrame({"freq": train_bundle.freqs, "weight": attention}).to_csv(run_dir / "attention.csv", index=False)
            plot_attention_heatmap(train_bundle.freqs, attention, run_dir / "attention_heatmap.png")

        payload = {
            "model": model.state_dict(),
            "config": vars(args),
            "freqs": train_bundle.freqs,
            "feature_dim": train_bundle.feature_dim,
            "scaler": None if train_bundle.scaler is None else [train_bundle.scaler[0].tolist(), train_bundle.scaler[1].tolist()],
            "metrics": metrics,
        }
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            patience_counter = 0
            torch.save(payload, best_path)
            pred_df.to_csv(run_dir / "best_val_predictions_new.csv", index=False)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    save_json(
        run_dir / "config.json",
        {
            **vars(args),
            "train_records": [record.record_id for record in train_records],
            "val_records": [record.record_id for record in val_records],
            "feature_dim": train_bundle.feature_dim,
            "n_freq": len(train_bundle.freqs),
            "prediction_table": "best_val_predictions_new.csv",
        },
    )
    plot_training_curve(run_dir / "training_log.csv", run_dir / "training_curve.png")
    print(f"Saved training outputs to {run_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train SOC model and export validation detail table")
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None, help="Only use files whose series name contains this substring")
    parser.add_argument("--include-phase", action="store_true", help="Require matched phase files and build [mag,sin,cos] tokens")
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="db_to_linear", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--freq-min", type=float, default=1900000)
    parser.add_argument("--freq-max", type=float, default=2150000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--ffn", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

