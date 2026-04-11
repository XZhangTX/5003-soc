import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, regression_metrics, save_json, timestamp


def _run_command(cmd, cwd: Path, dry_run: bool):
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _predictions_path(experiment, output_root: Path) -> Path:
    kind = experiment["kind"]
    tag = experiment["tag"]
    model = experiment["model_key"]
    if kind == "baseline_soc":
        return output_root / "baseline" / f"{model}-{tag}" / "predictions.csv"
    if kind == "baseline_soh":
        return output_root / "baseline_soh_proxy" / f"{model}-{tag}" / "predictions.csv"
    if kind == "train_soc":
        return output_root / "train" / tag / "val_predictions.csv"
    if kind == "train_soh":
        return output_root / "train_soh_proxy" / tag / "val_predictions.csv"
    raise ValueError(f"Unsupported kind={kind}")


def _build_experiments(args):
    common_root = str(args.input_root)
    output_root = str(args.output_root)
    experiments = [
        {
            "task": "soc",
            "model": "XGBoost",
            "model_key": "xgb",
            "kind": "baseline_soc",
            "tag": f"{args.tag}-soc-xgb",
            "cmd": [
                sys.executable,
                "baseline_tabular.py",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--model",
                "xgb",
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--xgb-depth",
                "3",
                "--xgb-rounds",
                "200",
                "--tag",
                f"{args.tag}-soc-xgb",
            ],
        },
        {
            "task": "soc",
            "model": "MLP",
            "model_key": "mlp",
            "kind": "baseline_soc",
            "tag": f"{args.tag}-soc-mlp",
            "cmd": [
                sys.executable,
                "baseline_tabular.py",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--model",
                "mlp",
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--mlp-hidden",
                "16,8",
                "--mlp-max-iter",
                "80",
                "--tag",
                f"{args.tag}-soc-mlp",
            ],
        },
        {
            "task": "soc",
            "model": "Transformer",
            "model_key": "transformer",
            "kind": "train_soc",
            "tag": f"{args.tag}-soc-transformer",
            "cmd": [
                sys.executable,
                "-u",
                "-m",
                "src.train.train",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--model-arch",
                "transformer",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-num-workers",
                str(args.val_num_workers),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--loss-type",
                "smooth_l1",
                "--smooth-l1-beta",
                str(args.smooth_l1_beta),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--tag",
                f"{args.tag}-soc-transformer",
            ] + (["--pin-memory"] if args.pin_memory else []),
        },
        {
            "task": "soc",
            "model": "Conv-Transformer",
            "model_key": "conv_transformer",
            "kind": "train_soc",
            "tag": f"{args.tag}-soc-convtransformer",
            "cmd": [
                sys.executable,
                "-u",
                "-m",
                "src.train.train",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--model-arch",
                "conv_transformer",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-num-workers",
                str(args.val_num_workers),
                "--d-model",
                "32",
                "--nhead",
                "2",
                "--layers",
                "1",
                "--ffn",
                "128",
                "--conv-channels",
                "12",
                "--kernel-size",
                "15",
                "--patch-stride",
                "2",
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--loss-type",
                "smooth_l1",
                "--smooth-l1-beta",
                str(args.smooth_l1_beta),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--tag",
                f"{args.tag}-soc-convtransformer",
            ] + (["--pin-memory"] if args.pin_memory else []),
        },
        {
            "task": "soh",
            "model": "XGBoost",
            "model_key": "xgb",
            "kind": "baseline_soh",
            "tag": f"{args.tag}-soh-xgb",
            "cmd": [
                sys.executable,
                "baseline_tabular_soh_proxy.py",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--model",
                "xgb",
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--xgb-depth",
                "3",
                "--xgb-rounds",
                "200",
                "--tag",
                f"{args.tag}-soh-xgb",
            ],
        },
        {
            "task": "soh",
            "model": "MLP",
            "model_key": "mlp",
            "kind": "baseline_soh",
            "tag": f"{args.tag}-soh-mlp",
            "cmd": [
                sys.executable,
                "baseline_tabular_soh_proxy.py",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--model",
                "mlp",
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--mlp-hidden",
                "16,8",
                "--mlp-max-iter",
                "80",
                "--tag",
                f"{args.tag}-soh-mlp",
            ],
        },
        {
            "task": "soh",
            "model": "Transformer",
            "model_key": "transformer",
            "kind": "train_soh",
            "tag": f"{args.tag}-soh-transformer",
            "cmd": [
                sys.executable,
                "-u",
                "-m",
                "src.train.train_soh_proxy",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--model-arch",
                "transformer",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-num-workers",
                str(args.val_num_workers),
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--loss-type",
                "smooth_l1",
                "--smooth-l1-beta",
                str(args.smooth_l1_beta),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--tag",
                f"{args.tag}-soh-transformer",
            ] + (["--pin-memory"] if args.pin_memory else []),
        },
        {
            "task": "soh",
            "model": "Conv-Transformer",
            "model_key": "conv_transformer",
            "kind": "train_soh",
            "tag": f"{args.tag}-soh-convtransformer",
            "cmd": [
                sys.executable,
                "-u",
                "-m",
                "src.train.train_soh_proxy",
                "--input-root",
                common_root,
                "--output-root",
                output_root,
                "--data-mode",
                args.data_mode,
                "--dc-mode",
                args.dc_mode,
                "--amp-mode",
                args.amp_mode,
                "--model-arch",
                "conv_transformer",
                "--batch-size",
                str(args.batch_size),
                "--num-workers",
                str(args.num_workers),
                "--val-num-workers",
                str(args.val_num_workers),
                "--d-model",
                "32",
                "--nhead",
                "2",
                "--layers",
                "1",
                "--ffn",
                "128",
                "--conv-channels",
                "8",
                "--kernel-size",
                "9",
                "--patch-stride",
                "4",
                "--lr",
                str(args.lr),
                "--weight-decay",
                str(args.weight_decay),
                "--loss-type",
                "smooth_l1",
                "--smooth-l1-beta",
                str(args.smooth_l1_beta),
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--tag",
                f"{args.tag}-soh-convtransformer",
            ] + (["--pin-memory"] if args.pin_memory else []),
        },
    ]
    if args.tasks:
        allowed = set(args.tasks)
        experiments = [exp for exp in experiments if exp["task"] in allowed]
    return experiments


def _summarize(experiments, output_root: Path, report_dir: Path):
    rows = []
    for exp in experiments:
        pred_path = _predictions_path(exp, output_root)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        pred_df = pd.read_csv(pred_path)
        metrics = regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
        rows.append(
            {
                "task": exp["task"].upper(),
                "model": exp["model"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "predictions_path": str(pred_path),
            }
        )

    summary_df = pd.DataFrame(rows).sort_values(["task", "rmse", "model"]).reset_index(drop=True)
    summary_df.to_csv(report_dir / "main_results_summary.csv", index=False)

    markdown_lines = []
    for task in ["SOC", "SOH"]:
        task_df = summary_df[summary_df["task"] == task].copy()
        if task_df.empty:
            continue
        markdown_lines.append(f"## {task}")
        markdown_lines.append("")
        markdown_lines.append("| Model | RMSE | MAE | R2 |")
        markdown_lines.append("| --- | ---: | ---: | ---: |")
        for _, row in task_df.iterrows():
            markdown_lines.append(
                f"| {row['model']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
            )
        markdown_lines.append("")

    (report_dir / "main_results_summary.md").write_text("\n".join(markdown_lines), encoding="utf-8")
    save_json(report_dir / "experiments.json", experiments)
    return summary_df


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run and summarize main SOC/SOH comparison experiments")
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--report-root", type=Path, default=Path("output") / "main_results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", choices=["soc", "soh"], default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="zscore", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--d-model", type=int, default=32)
    parser.add_argument("--nhead", type=int, default=2)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--ffn", type=int, default=128)
    parser.add_argument("--conv-channels", type=int, default=8)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--patch-stride", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(args):
    run_tag = args.tag or f"main-results-{timestamp()}"
    args.tag = run_tag
    report_dir = ensure_dir(args.report_root / run_tag)
    experiments = _build_experiments(args)

    if not args.summary_only:
        for experiment in experiments:
            _run_command(experiment["cmd"], cwd=Path.cwd(), dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run complete. Planned report dir: {report_dir}")
        return

    summary_df = _summarize(experiments, args.output_root, report_dir)
    print(summary_df.to_string(index=False))
    print(f"Saved summary to {report_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

