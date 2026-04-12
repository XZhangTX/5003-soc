import argparse
from pathlib import Path

import pandas as pd

from src.utils import ensure_dir, regression_metrics, save_json, timestamp
from src.validation.run_main_results import _build_experiments, _predictions_path, _run_command


def _with_seed(experiments, seed: int):
    seeded = []
    for exp in experiments:
        cloned = {
            "task": exp["task"],
            "dc_mode": exp.get("dc_mode", "all"),
            "model": exp["model"],
            "model_key": exp["model_key"],
            "kind": exp["kind"],
            "tag": f"{exp['tag']}-seed{seed}",
            "cmd": list(exp["cmd"]),
        }
        if "--tag" in cloned["cmd"]:
            tag_idx = cloned["cmd"].index("--tag") + 1
            cloned["cmd"][tag_idx] = cloned["tag"]
        if "--seed" in cloned["cmd"]:
            seed_idx = cloned["cmd"].index("--seed") + 1
            cloned["cmd"][seed_idx] = str(seed)
        else:
            cloned["cmd"].extend(["--seed", str(seed)])
        seeded.append(cloned)
    return seeded


def _load_seed_rows(experiments, output_root: Path, seed: int):
    rows = []
    for exp in experiments:
        pred_path = _predictions_path(exp, output_root)
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing predictions file: {pred_path}")
        pred_df = pd.read_csv(pred_path)
        metrics = regression_metrics(pred_df["y_true"].to_numpy(), pred_df["y_pred"].to_numpy())
        rows.append(
            {
                "seed": seed,
                "task": exp["task"].upper(),
                "dc_mode": exp.get("dc_mode", "all"),
                "model": exp["model"],
                "rmse": metrics["rmse"],
                "mae": metrics["mae"],
                "r2": metrics["r2"],
                "predictions_path": str(pred_path),
            }
        )
    return rows


def _aggregate(seed_df: pd.DataFrame):
    grouped = (
        seed_df.groupby(["task", "dc_mode", "model"], as_index=False)
        .agg(
            rmse_mean=("rmse", "mean"),
            rmse_std=("rmse", "std"),
            mae_mean=("mae", "mean"),
            mae_std=("mae", "std"),
            r2_mean=("r2", "mean"),
            r2_std=("r2", "std"),
            n_seeds=("seed", "nunique"),
        )
        .sort_values(["task", "dc_mode", "rmse_mean", "model"])
        .reset_index(drop=True)
    )
    for col in ["rmse_std", "mae_std", "r2_std"]:
        grouped[col] = grouped[col].fillna(0.0)
    return grouped


def _write_markdown(summary_df: pd.DataFrame, report_dir: Path):
    lines = []
    for task in ["SOC", "SOH"]:
        task_df = summary_df[summary_df["task"] == task]
        if task_df.empty:
            continue
        lines.append(f"## {task}")
        lines.append("")
        for dc_mode in ["all", "D", "C"]:
            dc_df = task_df[task_df["dc_mode"] == dc_mode]
            if dc_df.empty:
                continue
            lines.append(f"### dc-mode = {dc_mode}")
            lines.append("")
            lines.append("| Model | RMSE (mean+/-std) | MAE (mean+/-std) | R2 (mean+/-std) | Seeds |")
            lines.append("| --- | ---: | ---: | ---: | ---: |")
            for _, row in dc_df.iterrows():
                lines.append(
                    f"| {row['model']} | {row['rmse_mean']:.4f} +/- {row['rmse_std']:.4f} | "
                    f"{row['mae_mean']:.4f} +/- {row['mae_std']:.4f} | "
                    f"{row['r2_mean']:.4f} +/- {row['r2_std']:.4f} | {int(row['n_seeds'])} |"
                )
            lines.append("")
    (report_dir / "multi_seed_summary.md").write_text("\n".join(lines), encoding="utf-8")


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Run and summarize multi-seed SOC/SOH comparison experiments")
    parser.add_argument("--input-root", type=Path, default=Path(r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new"))
    parser.add_argument("--output-root", type=Path, default=Path("output"))
    parser.add_argument("--report-root", type=Path, default=Path("output") / "multi_seed_results")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--tasks", nargs="*", choices=["soc", "soh"], default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 52, 62, 72, 82])
    parser.add_argument("--data-mode", type=str, default="raw", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--dc-mode", type=str, default="all", choices=["all", "D", "C"])
    parser.add_argument("--dc-modes", nargs="*", choices=["all", "D", "C"], default=["all", "D", "C"])
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
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(args):
    run_tag = args.tag or f"multi-seed-{timestamp()}"
    args.tag = run_tag
    report_dir = ensure_dir(args.report_root / run_tag)
    base_experiments = _build_experiments(args)

    all_rows = []
    experiment_manifest = []
    for seed in args.seeds:
        seeded_experiments = _with_seed(base_experiments, seed)
        experiment_manifest.extend(seeded_experiments)
        if not args.summary_only:
            for experiment in seeded_experiments:
                _run_command(experiment["cmd"], cwd=Path.cwd(), dry_run=args.dry_run)
        if not args.dry_run:
            all_rows.extend(_load_seed_rows(seeded_experiments, args.output_root, seed))

    if args.dry_run:
        print(f"Dry run complete. Planned report dir: {report_dir}")
        return

    seed_df = pd.DataFrame(all_rows).sort_values(["task", "dc_mode", "model", "seed"]).reset_index(drop=True)
    summary_df = _aggregate(seed_df)
    seed_df.to_csv(report_dir / "multi_seed_raw.csv", index=False)
    summary_df.to_csv(report_dir / "multi_seed_summary.csv", index=False)
    _write_markdown(summary_df, report_dir)
    save_json(report_dir / "experiments.json", experiment_manifest)
    print(summary_df.to_string(index=False))
    print(f"Saved multi-seed summary to {report_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
