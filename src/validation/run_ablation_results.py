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


TASK_FINAL_CONFIGS = {
    "soc": {
        "amp_mode": "zscore",
        "model_arch": "conv_transformer",
        "loss_type": "smooth_l1",
        "d_model": 32,
        "nhead": 2,
        "layers": 1,
        "ffn": 128,
        "conv_channels": 12,
        "kernel_size": 15,
        "patch_stride": 2,
        "use_pos_enc": True,
    },
    "soh": {
        "amp_mode": "zscore",
        "model_arch": "conv_transformer",
        "loss_type": "smooth_l1",
        "d_model": 32,
        "nhead": 2,
        "layers": 1,
        "ffn": 128,
        "conv_channels": 8,
        "kernel_size": 9,
        "patch_stride": 4,
        "use_pos_enc": True,
    },
}


ABLATION_STUDIES = {
    "soc": {
        "preprocess": [
            {"variant": "raw_db", "amp_mode": "raw_db"},
            {"variant": "db_to_linear", "amp_mode": "db_to_linear"},
            {"variant": "zscore", "amp_mode": "zscore"},
        ],
        "architecture": [
            {"variant": "transformer", "model_arch": "transformer"},
            {"variant": "cnn_only", "model_arch": "cnn_only"},
            {"variant": "conv_transformer", "model_arch": "conv_transformer"},
        ],
        "phase": [
            {"variant": "mag_only", "include_phase": False},
            {"variant": "mag_phase", "include_phase": True},
        ],
        "loss": [
            {"variant": "hybrid", "loss_type": "hybrid"},
            {"variant": "smooth_l1", "loss_type": "smooth_l1"},
        ],
        "patch_stride": [
            {"variant": "ps1", "patch_stride": 1},
            {"variant": "ps2", "patch_stride": 2},
            {"variant": "ps4", "patch_stride": 4},
        ],
        "position_encoding": [
            {"variant": "with_pe", "use_pos_enc": True},
            {"variant": "without_pe", "use_pos_enc": False},
        ],
        "patch_overlap": [
            {"variant": "overlap_k15_s2", "kernel_size": 15, "patch_stride": 2},
            {"variant": "nonoverlap_k2_s2", "kernel_size": 2, "patch_stride": 2},
        ],
    },
    "soh": {
        "preprocess": [
            {"variant": "raw_db", "amp_mode": "raw_db"},
            {"variant": "db_to_linear", "amp_mode": "db_to_linear"},
            {"variant": "zscore", "amp_mode": "zscore"},
        ],
        "architecture": [
            {"variant": "transformer", "model_arch": "transformer"},
            {"variant": "cnn_only", "model_arch": "cnn_only"},
            {"variant": "conv_transformer", "model_arch": "conv_transformer"},
        ],
        "phase": [
            {"variant": "mag_only", "include_phase": False},
            {"variant": "mag_phase", "include_phase": True},
        ],
        "loss": [
            {"variant": "hybrid", "loss_type": "hybrid"},
            {"variant": "smooth_l1", "loss_type": "smooth_l1"},
        ],
        "patch_stride": [
            {"variant": "ps2", "patch_stride": 2},
            {"variant": "ps4", "patch_stride": 4},
            {"variant": "ps8", "patch_stride": 8},
        ],
        "position_encoding": [
            {"variant": "with_pe", "use_pos_enc": True},
            {"variant": "without_pe", "use_pos_enc": False},
        ],
        "patch_overlap": [
            {"variant": "overlap_k9_s4", "kernel_size": 9, "patch_stride": 4},
            {"variant": "nonoverlap_k4_s4", "kernel_size": 4, "patch_stride": 4},
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


def _resolved_task_config(args, task: str):
    config = dict(TASK_FINAL_CONFIGS[task])
    for key in [
        "amp_mode",
        "model_arch",
        "loss_type",
        "d_model",
        "nhead",
        "layers",
        "ffn",
        "conv_channels",
        "kernel_size",
        "patch_stride",
        "use_pos_enc",
    ]:
        value = getattr(args, key, None)
        if value is not None:
            config[key] = value
    config.setdefault("use_pos_enc", True)
    return config


def _base_train_cmd(args, task: str):
    task_cfg = _resolved_task_config(args, task)
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
        str(task_cfg["d_model"]),
        "--nhead",
        str(task_cfg["nhead"]),
        "--layers",
        str(task_cfg["layers"]),
        "--ffn",
        str(task_cfg["ffn"]),
        "--conv-channels",
        str(task_cfg["conv_channels"]),
        "--kernel-size",
        str(task_cfg["kernel_size"]),
        "--patch-stride",
        str(task_cfg["patch_stride"]),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--loss-type",
        task_cfg["loss_type"],
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


def _selected_tasks(task_arg: str):
    return ["soc", "soh"] if task_arg == "all" else [task_arg]


def _build_experiments_for_task(args, task: str):
    studies = args.studies or list(ABLATION_STUDIES[task].keys())
    experiments = []
    task_cfg = _resolved_task_config(args, task)
    for study in studies:
        variants = ABLATION_STUDIES[task][study]
        for variant_cfg in variants:
            tag = f"{args.tag}-{task}-{study}-{variant_cfg['variant']}"
            cmd = _base_train_cmd(args, task)

            amp_mode = variant_cfg.get("amp_mode", task_cfg["amp_mode"])
            model_arch = variant_cfg.get("model_arch", task_cfg["model_arch"])
            loss_type = variant_cfg.get("loss_type", task_cfg["loss_type"])
            kernel_size = variant_cfg.get("kernel_size", task_cfg["kernel_size"])
            patch_stride = variant_cfg.get("patch_stride", task_cfg["patch_stride"])
            include_phase = bool(variant_cfg.get("include_phase", False))
            use_pos_enc = bool(variant_cfg.get("use_pos_enc", task_cfg["use_pos_enc"]))

            cmd.extend(["--amp-mode", amp_mode])
            cmd.extend(["--model-arch", model_arch])
            cmd.extend(["--loss-type", loss_type])
            cmd.extend(["--kernel-size", str(kernel_size)])
            cmd.extend(["--patch-stride", str(patch_stride)])
            if include_phase:
                cmd.append("--include-phase")
            if not use_pos_enc:
                cmd.append("--disable-pos-enc")
            cmd.extend(["--tag", tag])

            experiments.append(
                {
                    "task": task,
                    "study": study,
                    "variant": variant_cfg["variant"],
                    "model_arch": model_arch,
                    "amp_mode": amp_mode,
                    "include_phase": include_phase,
                    "loss_type": loss_type,
                    "kernel_size": kernel_size,
                    "patch_stride": patch_stride,
                    "use_pos_enc": use_pos_enc,
                    "tag": tag,
                    "cmd": cmd,
                }
            )
    return experiments


def _build_experiments(args):
    experiments = []
    for task in _selected_tasks(args.task):
        experiments.extend(_build_experiments_for_task(args, task))
    return experiments


def _summarize(experiments, output_root: Path, report_dir: Path):
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
                "kernel_size": exp["kernel_size"],
                "patch_stride": exp["patch_stride"],
                "use_pos_enc": exp["use_pos_enc"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "predictions_path": str(pred_path),
            }
        )
    df = pd.DataFrame(rows).sort_values(["task", "study", "rmse", "variant"]).reset_index(drop=True)
    df.to_csv(report_dir / "ablation_summary.csv", index=False)

    lines = ["# Ablation Summary", ""]
    for task in df["task"].drop_duplicates():
        task_df = df[df["task"] == task]
        lines.append(f"## {task}")
        lines.append("")
        for study in task_df["study"].drop_duplicates():
            study_df = task_df[task_df["study"] == study]
            lines.append(f"### {study}")
            lines.append("")
            lines.append("| Variant | Model | Preprocess | Phase | Loss | PE | Kernel | Patch Stride | RMSE | MAE | R2 |")
            lines.append("| --- | --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
            for _, row in study_df.iterrows():
                phase_label = "yes" if row["include_phase"] else "no"
                pe_label = "yes" if row["use_pos_enc"] else "no"
                lines.append(
                    f"| {row['variant']} | {row['model_arch']} | {row['amp_mode']} | {phase_label} | {row['loss_type']} | {pe_label} | {int(row['kernel_size'])} | {int(row['patch_stride'])} | {row['rmse']:.4f} | {row['mae']:.4f} | {row['r2']:.4f} |"
                )
            lines.append("")
    (report_dir / "ablation_summary.md").write_text("\n".join(lines), encoding="utf-8")
    save_json(report_dir / "experiments.json", experiments)
    return df


def _parse_optional_bool(value: str):
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "yes", "y"}:
        return True
    if lowered in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run task-specific ablation studies for SOC/SOH")
    parser.add_argument("--task", type=str, required=True, choices=["soc", "soh", "all"])
    parser.add_argument("--studies", nargs="*", default=None)
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--report-root", type=Path, default=Path("output") / "ablation_results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--amp-mode", type=str, default=None, choices=["db_to_linear", "raw_db", "zscore"])
    parser.add_argument("--model-arch", type=str, default=None, choices=["transformer", "conv_transformer", "cnn_only"])
    parser.add_argument("--loss-type", type=str, default=None, choices=["smooth_l1", "hybrid"])
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
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--ffn", type=int, default=None)
    parser.add_argument("--conv-channels", type=int, default=None)
    parser.add_argument("--kernel-size", type=int, default=None)
    parser.add_argument("--patch-stride", type=int, default=None)
    parser.add_argument("--use-pos-enc", type=_parse_optional_bool, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(args):
    for task in _selected_tasks(args.task):
        valid_studies = set(ABLATION_STUDIES[task].keys())
        if args.studies is not None:
            unknown = set(args.studies) - valid_studies
            if unknown:
                raise ValueError(f"Unsupported studies for {task}: {sorted(unknown)}")

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

    summary_df = _summarize(experiments, args.output_root, report_dir)
    print(summary_df.to_string(index=False))
    print(f"Saved ablation summary to {report_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
