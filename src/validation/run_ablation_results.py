import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, regression_metrics, save_json, timestamp


TASK_OUTPUT_DIR = {
    "soc": "train",
    "soh": "train_soh_proxy",
}


ABLATION_STUDIES = {
    "soc": {
        "preprocess": [
            {"variant": "raw_db", "amp_mode": "raw_db", "model_arch": "conv_transformer"},
            {"variant": "db_to_linear", "amp_mode": "db_to_linear", "model_arch": "conv_transformer"},
            {"variant": "zscore", "amp_mode": "zscore", "model_arch": "conv_transformer"},
        ],
        "architecture": [
            {"variant": "transformer", "amp_mode": "zscore", "model_arch": "transformer"},
            {"variant": "conv_transformer", "amp_mode": "zscore", "model_arch": "conv_transformer"},
        ],
        "phase": [
            {"variant": "mag_only", "amp_mode": "zscore", "model_arch": "conv_transformer", "include_phase": False},
            {"variant": "mag_phase", "amp_mode": "zscore", "model_arch": "conv_transformer", "include_phase": True},
        ],
        "loss": [
            {"variant": "hybrid", "amp_mode": "zscore", "model_arch": "conv_transformer", "loss_type": "hybrid"},
            {"variant": "smooth_l1", "amp_mode": "zscore", "model_arch": "conv_transformer", "loss_type": "smooth_l1"},
        ],
    },
    "soh": {
        "preprocess": [
            {"variant": "raw_db", "amp_mode": "raw_db", "model_arch": "conv_transformer"},
            {"variant": "db_to_linear", "amp_mode": "db_to_linear", "model_arch": "conv_transformer"},
            {"variant": "zscore", "amp_mode": "zscore", "model_arch": "conv_transformer"},
        ],
        "architecture": [
            {"variant": "transformer", "amp_mode": "zscore", "model_arch": "transformer"},
            {"variant": "conv_transformer", "amp_mode": "zscore", "model_arch": "conv_transformer"},
        ],
        "phase": [
            {"variant": "mag_only", "amp_mode": "zscore", "model_arch": "conv_transformer", "include_phase": False},
            {"variant": "mag_phase", "amp_mode": "zscore", "model_arch": "conv_transformer", "include_phase": True},
        ],
        "loss": [
            {"variant": "hybrid", "amp_mode": "zscore", "model_arch": "conv_transformer", "loss_type": "hybrid"},
            {"variant": "smooth_l1", "amp_mode": "zscore", "model_arch": "conv_transformer", "loss_type": "smooth_l1"},
        ],
    },
}


def _run_command(cmd, cwd: Path, dry_run: bool):
    print("$", " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _prediction_path(exp, output_root: Path):
    return output_root / TASK_OUTPUT_DIR[exp["task"]] / exp["tag"] / "val_predictions.csv"


def _entry_module(task: str) -> str:
    return "src.train.train" if task == "soc" else "src.train.train_soh_proxy"


def _base_train_cmd(args, task: str):
    cmd = [
        sys.executable,
        "-u",
        "-m",
        _entry_module(task),
        "--input-root",
        str(args.input_root),
        "--output-root",
        str(args.output_root),
        "--data-mode",
        args.data_mode,
        "--dc-mode",
        args.dc_mode,
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--val-num-workers",
        str(args.val_num_workers),
        "--d-model",
        str(args.d_model),
        "--nhead",
        str(args.nhead),
        "--layers",
        str(args.layers),
        "--ffn",
        str(args.ffn),
        "--conv-channels",
        str(args.conv_channels),
        "--kernel-size",
        str(args.kernel_size),
        "--patch-stride",
        str(args.patch_stride),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--loss-type",
        args.loss_type,
        "--smooth-l1-beta",
        str(args.smooth_l1_beta),
        "--epochs",
        str(args.epochs),
        "--patience",
        str(args.patience),
        "--seed",
        str(args.seed),
    ]
    if args.pin_memory:
        cmd.append("--pin-memory")
    return cmd


def _build_experiments(args):
    studies = args.studies or list(ABLATION_STUDIES[args.task].keys())
    experiments = []
    for study in studies:
        variants = ABLATION_STUDIES[args.task][study]
        for variant_cfg in variants:
            tag = f"{args.tag}-{args.task}-{study}-{variant_cfg['variant']}"
            cmd = _base_train_cmd(args, args.task)
            cmd.extend(["--amp-mode", variant_cfg.get("amp_mode", args.amp_mode)])
            cmd.extend(["--model-arch", variant_cfg.get("model_arch", args.model_arch)])
            cmd.extend(["--loss-type", variant_cfg.get("loss_type", args.loss_type)])
            if variant_cfg.get("include_phase", False):
                cmd.append("--include-phase")
            cmd.extend(["--tag", tag])
            experiments.append(
                {
                    "task": args.task,
                    "study": study,
                    "variant": variant_cfg["variant"],
                    "model_arch": variant_cfg.get("model_arch", args.model_arch),
                    "amp_mode": variant_cfg.get("amp_mode", args.amp_mode),
                    "include_phase": bool(variant_cfg.get("include_phase", False)),
                    "loss_type": variant_cfg.get("loss_type", args.loss_type),
                    "tag": tag,
                    "cmd": cmd,
                }
            )
    return experiments


def _summarize(task: str, experiments, output_root: Path, report_dir: Path):
    rows = []
    for exp in experiments:
        pred_path = _prediction_path(exp, output_root)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        pred_df = pd.read_csv(pred_path)
        metrics = regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
        rows.append(
            {
                "task": exp["task"].upper(),
                "study": exp["study"],
                "variant": exp["variant"],
                "model_arch": exp["model_arch"],
                "amp_mode": exp["amp_mode"],
                "include_phase": exp["include_phase"],
                "loss_type": exp["loss_type"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "predictions_path": str(pred_path),
            }
        )
    df = pd.DataFrame(rows).sort_values(["study", "rmse", "variant"]).reset_index(drop=True)
    df.to_csv(report_dir / "ablation_summary.csv", index=False)

    lines = [f"# {task.upper()} Ablation", ""]
    for study in df["study"].drop_duplicates():
        study_df = df[df["study"] == study]
        lines.append(f"## {study}")
        lines.append("")
        lines.append("| Variant | Model | Preprocess | Phase | Loss | RMSE | MAE | R2 |")
        lines.append("| --- | --- | --- | --- | --- | ---: | ---: | ---: |")
        for _, row in study_df.iterrows():
            phase_label = "yes" if row["include_phase"] else "no"
            lines.append(
                f"| {row['variant']} | {row['model_arch']} | {row['amp_mode']} | {phase_label} | {row['loss_type']} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
            )
        lines.append("")
    (report_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(report_dir / "experiments.json", experiments)
    return df


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run task-specific ablation studies for SOC/SOH")
    parser.add_argument("--task", type=str, required=True, choices=["soc", "soh"])
    parser.add_argument("--studies", nargs="*", default=None)
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--report-root", type=Path, default=Path("output") / "ablation_results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default="zscore", choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--model-arch", type=str, default="conv_transformer", choices=["transformer", "conv_transformer"])
    parser.add_argument("--loss-type", type=str, default="smooth_l1", choices=["smooth_l1", "hybrid"])
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--val-num-workers", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--smooth-l1-beta", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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
    valid_studies = set(ABLATION_STUDIES[args.task].keys())
    if args.studies is not None:
        unknown = set(args.studies) - valid_studies
        if unknown:
            raise ValueError(f"Unsupported studies for {args.task}: {sorted(unknown)}")

    run_tag = args.tag or f"ablation-{args.task}-{timestamp()}"
    args.tag = run_tag
    report_dir = ensure_dir(args.report_root / run_tag)
    experiments = _build_experiments(args)

    if not args.summary_only:
        for exp in experiments:
            _run_command(exp["cmd"], cwd=Path.cwd(), dry_run=args.dry_run)

    if args.dry_run:
        print(f"Dry run complete. Planned report dir: {report_dir}")
        return

    summary_df = _summarize(args.task, experiments, args.output_root, report_dir)
    print(summary_df.to_string(index=False))
    print(f"Saved ablation summary to {report_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())


