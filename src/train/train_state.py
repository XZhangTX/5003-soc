import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import GroupShuffleSplit
from torch.optim.lr_scheduler import LinearLR, SequentialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import S11AlignedDataset, build_bundle, discover_s11_records
from src.data.dataset_soh_proxy import S11SOHProxyDataset, build_bundle_soh_proxy, inverse_cycle_scale
from src.models import ConvTransformerRegressor, SpectrumTransformerRegressor
from src.utils import ensure_dir, regression_metrics, save_json, set_seed, timestamp
from src.visualization import plot_attention_heatmap, plot_prediction_scatter, plot_training_curve


TASK_DEFAULTS = {
    "soc": {
        "model_arch": "transformer",
        "freq_min": 1900000.0,
        "freq_max": 2150000.0,
        "output_subdir": "train",
        "label_name": "soc_percent",
        "title": "Validation SOC Scatter",
        "xlabel": "True SOC",
        "ylabel": "Predicted SOC",
    },
    "soh": {
        "model_arch": "conv_transformer",
        "freq_min": None,
        "freq_max": None,
        "output_subdir": "train_soh_proxy",
        "label_name": "soh_proxy_0_100",
        "title": "Validation SOH Proxy Scatter",
        "xlabel": "True SOH Proxy",
        "ylabel": "Predicted SOH Proxy",
    },
}


@torch.no_grad()
def evaluate_model_state(model, loader, device, inverse_target, collect_attention: bool = False):
    model.eval()
    ys, ps = [], []
    attn_list = [] if collect_attention else None
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        pred = model(xb).cpu().numpy()
        ys.append(yb.numpy())
        ps.append(pred)
        if collect_attention and getattr(model, "last_freq_attn", None) is not None:
            attn = model.last_freq_attn
            if attn is not None:
                attn_list.append(attn.numpy())

    y_true_norm = np.concatenate(ys)
    y_pred_norm = np.concatenate(ps)
    y_true = inverse_target(y_true_norm)
    y_pred = inverse_target(y_pred_norm)
    metrics = regression_metrics(y_true, y_pred)
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



def _resolve_task_defaults(args):
    defaults = TASK_DEFAULTS[args.task]
    if args.model_arch is None:
        args.model_arch = defaults["model_arch"]
    if args.freq_min is None and defaults["freq_min"] is not None:
        args.freq_min = defaults["freq_min"]
    if args.freq_max is None and defaults["freq_max"] is not None:
        args.freq_max = defaults["freq_max"]
    return defaults



def build_model(args, train_bundle):
    common = dict(
        d_in=train_bundle.feature_dim,
        n_freq=len(train_bundle.freqs),
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.layers,
        dim_feedforward=args.ffn,
        dropout=args.dropout,
    )
    if args.model_arch == "transformer":
        return SpectrumTransformerRegressor(**common)
    if args.model_arch == "conv_transformer":
        return ConvTransformerRegressor(
            **common,
            conv_channels=args.conv_channels,
            kernel_size=args.kernel_size,
            patch_stride=args.patch_stride,
        )
    raise ValueError(f"Unsupported model_arch={args.model_arch}")



def prepare_task(args, train_records, val_records):
    if args.task == "soc":
        train_bundle = build_bundle(
            train_records,
            include_phase=args.include_phase,
            dc_mode=args.dc_mode,
            amp_mode=args.amp_mode,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
        )
        val_bundle = build_bundle(
            val_records,
            include_phase=args.include_phase,
            dc_mode=args.dc_mode,
            amp_mode=args.amp_mode,
            freq_min=args.freq_min,
            freq_max=args.freq_max,
            scaler=train_bundle.scaler,
        )
        return {
            "train_bundle": train_bundle,
            "val_bundle": val_bundle,
            "train_ds": S11AlignedDataset(train_bundle),
            "val_ds": S11AlignedDataset(val_bundle),
            "inverse_target": lambda y: np.asarray(y, dtype=np.float32) * 100.0,
        }

    train_bundle = build_bundle_soh_proxy(
        train_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
    )
    val_bundle = build_bundle_soh_proxy(
        val_records,
        include_phase=args.include_phase,
        dc_mode=args.dc_mode,
        amp_mode=args.amp_mode,
        freq_min=args.freq_min,
        freq_max=args.freq_max,
        scaler=train_bundle.scaler,
        label_scale=train_bundle.label_scale,
    )
    return {
        "train_bundle": train_bundle,
        "val_bundle": val_bundle,
        "train_ds": S11SOHProxyDataset(train_bundle),
        "val_ds": S11SOHProxyDataset(val_bundle),
        "inverse_target": lambda y: inverse_cycle_scale(y, train_bundle.label_scale),
    }



def main(args):
    defaults = _resolve_task_defaults(args)
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
    prepared = prepare_task(args, train_records, val_records)
    train_bundle = prepared["train_bundle"]
    val_bundle = prepared["val_bundle"]
    train_ds = prepared["train_ds"]
    val_ds = prepared["val_ds"]
    inverse_target = prepared["inverse_target"]

    pin_memory = args.pin_memory and torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin_memory)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.val_num_workers, pin_memory=pin_memory)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, train_bundle).to(device)

    print(
        f"Loaded {len(records)} records | train={len(train_records)} val={len(val_records)} | "
        f"n_freq={len(train_bundle.freqs)} feature_dim={train_bundle.feature_dim} | "
        f"task={args.task} model={args.model_arch} | "
        f"batch_size={args.batch_size} num_workers={args.num_workers}/{args.val_num_workers} pin_memory={pin_memory}"
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    warmup_epochs = max(1, args.warmup_epochs)
    cosine_epochs = max(1, args.epochs - warmup_epochs)
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=args.warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_epochs,
    )
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cosine_epochs,
    )
    loss_fn = nn.SmoothL1Loss(beta=args.smooth_l1_beta)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[warmup_epochs],
    )

    run_dir = ensure_dir(Path(args.output_root) / defaults["output_subdir"] / (args.tag or timestamp()))
    best_path = run_dir / "best.pt"
    log_rows = []
    best_rmse = float("inf")
    patience_counter = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_losses = []
        for xb, yb in tqdm(train_loader, desc=f"train-{args.task}-{epoch}", leave=False):
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            pred = model(xb).squeeze(-1)
            if yb.ndim > 1:
                yb = yb.squeeze(-1)

            if epoch == 1 and not train_losses:
                print(f"pred.shape={tuple(pred.shape)}, yb.shape={tuple(yb.shape)}")
                print(f"pred[:5]={pred[:5].detach().cpu().numpy()}")
                print(f"yb[:5]={yb[:5].detach().cpu().numpy()}")

            loss = loss_fn(pred, yb)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()
        metrics, y_true, y_pred, attention = evaluate_model_state(model, val_loader, device, inverse_target, collect_attention=True)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **metrics}
        log_rows.append(row)
        print(f"Epoch {epoch:03d} | lr={optimizer.param_groups[0]['lr']:.6g} | train_loss={row['train_loss']:.4f} | val_rmse={metrics['rmse']:.4f} | val_mae={metrics['mae']:.4f} | val_r2={metrics['r2']:.4f}")

        pd.DataFrame(log_rows).to_csv(run_dir / "training_log.csv", index=False)
        pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(run_dir / "val_predictions.csv", index=False)
        plot_prediction_scatter(
            y_true,
            y_pred,
            run_dir / "val_scatter.png",
            title=defaults["title"],
            xlabel=defaults["xlabel"],
            ylabel=defaults["ylabel"],
        )
        if attention is not None and attention.ndim == 1 and len(attention) == len(train_bundle.freqs):
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
        if args.task == "soh":
            payload["label_scale"] = list(train_bundle.label_scale)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            patience_counter = 0
            torch.save(payload, best_path)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                break

    config_payload = {
        **vars(args),
        "train_records": [record.record_id for record in train_records],
        "val_records": [record.record_id for record in val_records],
        "feature_dim": train_bundle.feature_dim,
        "n_freq": len(train_bundle.freqs),
        "label_name": defaults["label_name"],
    }
    if args.task == "soh":
        config_payload["label_scale"] = list(train_bundle.label_scale)
    save_json(run_dir / "config.json", config_payload)
    plot_training_curve(run_dir / "training_log.csv", run_dir / "training_curve.png")
    print(f"Saved {args.task.upper()} training outputs to {run_dir}")



def build_arg_parser(task_default=None, include_task=True):
    parser = argparse.ArgumentParser(description="Train state estimation model directly from aligned S11 CSV files")
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    if include_task:
        parser.add_argument("--task", type=str, default=task_default or "soc", choices=["soc", "soh"])
    parser.add_argument("--model-arch", type=str, default=None, choices=["transformer", "conv_transformer"])
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--include-phase", action="store_true")
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="db_to_linear", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--warmup-start-factor", type=float, default=0.1)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
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
    parser.add_argument("--conv-channels", type=int, default=32)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--patch-stride", type=int, default=4)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
