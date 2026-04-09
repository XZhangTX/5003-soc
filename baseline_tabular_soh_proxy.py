import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from src.data.discovery import discover_s11_records
from src.data.dataset_soh_proxy import build_bundle_soh_proxy, inverse_cycle_scale
from src.models import fit_baseline_regressor
from src.utils import ensure_dir, regression_metrics, save_json, set_seed, timestamp
from src.visualization import plot_prediction_scatter


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

    groups = np.arange(len(records))
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_ratio, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(groups, groups=groups))
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]

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

    x_train = train_bundle.x.reshape(len(train_bundle.x), -1)
    x_val = val_bundle.x.reshape(len(val_bundle.x), -1)
    model = fit_baseline_regressor(
        args.model,
        x_train,
        train_bundle.y,
        seed=args.seed,
        x_val=x_val,
        y_val=val_bundle.y,
        hidden_layer_sizes=tuple(int(v) for v in args.mlp_hidden.split(",") if v.strip()),
        max_iter=args.mlp_max_iter,
        n_estimators=args.xgb_rounds,
        learning_rate=args.xgb_lr,
        max_depth=args.xgb_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample,
        patience=args.patience,
    )
    y_pred_norm = model.predict(x_val)
    y_true = inverse_cycle_scale(val_bundle.y, train_bundle.label_scale)
    y_pred = inverse_cycle_scale(y_pred_norm, train_bundle.label_scale)
    metrics = regression_metrics(y_true, y_pred)

    out_dir = ensure_dir(Path(args.output_root) / "baseline_soh_proxy" / f"{args.model}-{args.tag or timestamp()}")
    pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).to_csv(out_dir / "predictions.csv", index=False)
    save_json(out_dir / "metrics.json", metrics)
    plot_prediction_scatter(
        y_true,
        y_pred,
        out_dir / "scatter.png",
        title=f"{args.model.upper()} SOH Proxy Scatter",
        xlabel="True SOH Proxy",
        ylabel="Predicted SOH Proxy",
    )
    print(f"Saved SOH proxy baseline outputs to {out_dir}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train tabular baselines on aligned S11 CSV files for SOH proxy")
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--model", type=str, default="xgb", choices=["mlp", "xgb"])
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--include-phase", action="store_true")
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="db_to_linear", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--freq-min", type=float, default=None)
    parser.add_argument("--freq-max", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--mlp-hidden", type=str, default="256,128")
    parser.add_argument("--mlp-max-iter", type=int, default=500)
    parser.add_argument("--xgb-rounds", type=int, default=800)
    parser.add_argument("--xgb-lr", type=float, default=0.05)
    parser.add_argument("--xgb-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample", type=float, default=0.8)
    parser.add_argument("--patience", type=int, default=20)
    return parser


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
