import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.utils import ensure_dir


STUDY_TITLES = {
    "architecture": "Architecture Ablation",
    "preprocess": "Preprocessing Ablation",
    "phase": "Phase Ablation",
    "loss": "Loss Ablation",
    "patch_stride": "Patch Stride Ablation",
    "position_encoding": "Position Encoding Ablation",
    "tokenization": "Tokenization Ablation",
}

COLOR_MAP = {
    "SOC": "#003D7C",
    "SOH": "#EF7C00",
}

STUDY_COLORS = {
    "architecture": "#4C78A8",
    "preprocess": "#F58518",
    "phase": "#54A24B",
    "loss": "#E45756",
    "patch_stride": "#72B7B2",
    "position_encoding": "#B279A2",
    "tokenization": "#FF9DA6",
}


def _resolve_input_table(input_path: Path) -> Path:
    if input_path.is_file():
        return input_path
    candidate = input_path / "ablation_summary.csv"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not find ablation_summary.csv under {input_path}")


def _plot_single_task(ax, df: pd.DataFrame, study: str, task: str, metric: str):
    subset = df[(df["study"] == study) & (df["task"] == task)].copy()
    if subset.empty:
        ax.axis("off")
        return

    subset = subset.sort_values(metric, ascending=(metric != "r2")).reset_index(drop=True)
    x = np.arange(len(subset))
    bars = ax.bar(x, subset[metric].astype(float), color=COLOR_MAP.get(task, "#4a5568"), alpha=0.88, width=0.65)

    ax.set_xticks(x)
    ax.set_xticklabels(subset["variant"], rotation=25, ha="right", fontsize=9)
    ax.set_title(task, fontsize=12, fontweight="bold")
    ax.set_ylabel(metric.upper() if metric != "r2" else r"$R^2$")
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, subset[metric].astype(float).to_numpy()):
        if metric == "r2":
            label = f"{value:.3f}"
        else:
            label = f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)


def _plot_task_only_panels(df: pd.DataFrame, metric: str, studies: list[str], task: str, out_path: Path):
    studies = [study for study in studies if study in df["study"].unique()]
    if not studies:
        raise ValueError("No matching studies found in ablation summary")

    n_cols = 2
    n_rows = int(np.ceil(len(studies) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(13, max(4.2 * n_rows, 5.2)), squeeze=False)
    axes_flat = axes.flatten()

    for idx, study in enumerate(studies):
        ax = axes_flat[idx]
        ax.text(
            0.0,
            1.10,
            STUDY_TITLES.get(study, study.replace("_", " ").title()),
            transform=ax.transAxes,
            fontsize=13,
            fontweight="bold",
            ha="left",
        )
        _plot_single_task(ax, df, study=study, task=task, metric=metric)

    for idx in range(len(studies), len(axes_flat)):
        axes_flat[idx].axis("off")

    fig.suptitle(f"{task} Ablation Results ({metric.upper() if metric != 'r2' else '$R^2$'})", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_task_single_axis(df: pd.DataFrame, metric: str, studies: list[str], task: str, out_path: Path):
    studies = [study for study in studies if study in df["study"].unique()]
    if not studies:
        raise ValueError("No matching studies found in ablation summary")

    subset = df[(df["task"] == task) & (df["study"].isin(studies))].copy()
    if subset.empty:
        raise ValueError(f"No rows found for task={task}")

    subset["variant_label"] = subset["variant"].astype(str)
    positions = []
    labels = []
    values = []
    colors = []
    group_centers = []
    separators = []
    cursor = 0.0

    for study in studies:
        study_df = subset[subset["study"] == study].copy()
        if study_df.empty:
            continue
        study_df = study_df.sort_values(metric, ascending=(metric != "r2")).reset_index(drop=True)
        local_positions = []
        for _, row in study_df.iterrows():
            positions.append(cursor)
            local_positions.append(cursor)
            labels.append(str(row["variant_label"]))
            values.append(float(row[metric]))
            colors.append(STUDY_COLORS.get(study, COLOR_MAP.get(task, "#4a5568")))
            cursor += 1.0
        group_centers.append((np.mean(local_positions), STUDY_TITLES.get(study, study.replace("_", " ").title())))
        separators.append(cursor - 0.5)
        cursor += 0.8

    fig, ax = plt.subplots(figsize=(max(12, 0.72 * len(labels)), 5.8))
    bars = ax.bar(positions, values, color=colors, width=0.7, alpha=0.9)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel(metric.upper() if metric != "r2" else r"$R^2$")
    ax.set_title(f"{task} Ablation Results ({metric.upper() if metric != 'r2' else '$R^2$'})", fontsize=17, pad=24)
    ax.grid(True, axis="y", alpha=0.25)

    for bar, value in zip(bars, values):
        label = f"{value:.3f}" if metric == "r2" else f"{value:.2f}"
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=8)

    ymin, ymax = ax.get_ylim()
    for separator in separators[:-1]:
        ax.axvline(separator + 0.4, color="#bbbbbb", linestyle="--", linewidth=1.0, alpha=0.8)
    label_y = ymax + 0.002 * (ymax - ymin)
    for center, title in group_centers:
        ax.text(center, label_y, title, ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=STUDY_COLORS.get(study, "#999999"), edgecolor="none", label=STUDY_TITLES.get(study, study))
        for study in studies
        if study in subset["study"].unique()
    ]
    ax.legend(handles=legend_handles, loc="upper right", frameon=False, fontsize=8, handlelength=1.2, borderpad=0.2, labelspacing=0.3)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_ablation_panels(df: pd.DataFrame, metric: str, studies: list[str], out_path: Path):
    studies = [study for study in studies if study in df["study"].unique()]
    if not studies:
        raise ValueError("No matching studies found in ablation summary")

    fig, axes = plt.subplots(len(studies), 2, figsize=(13, max(4.2 * len(studies), 5.5)), squeeze=False)
    for row_idx, study in enumerate(studies):
        axes[row_idx, 0].text(
            0.0,
            1.12,
            STUDY_TITLES.get(study, study.replace("_", " ").title()),
            transform=axes[row_idx, 0].transAxes,
            fontsize=14,
            fontweight="bold",
            ha="left",
        )
        _plot_single_task(axes[row_idx, 0], df, study=study, task="SOC", metric=metric)
        _plot_single_task(axes[row_idx, 1], df, study=study, task="SOH", metric=metric)

    fig.suptitle(f"Ablation Results ({metric.upper() if metric != 'r2' else '$R^2$'})", fontsize=18, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.985])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    input_table = _resolve_input_table(Path(args.input))
    out_dir = ensure_dir(Path(args.output_root))
    df = pd.read_csv(input_table)
    studies = [item.strip() for item in args.studies.split(",") if item.strip()] if args.studies else list(df["study"].drop_duplicates())
    if args.layout == "combined":
        out_path = out_dir / f"ablation_{args.metric}.png"
        plot_ablation_panels(df, metric=args.metric, studies=studies, out_path=out_path)
        print(f"Saved ablation plot to {out_path}")
    elif args.layout == "by_task":
        soc_path = out_dir / f"ablation_soc_{args.metric}.png"
        soh_path = out_dir / f"ablation_soh_{args.metric}.png"
        _plot_task_only_panels(df, metric=args.metric, studies=studies, task="SOC", out_path=soc_path)
        _plot_task_only_panels(df, metric=args.metric, studies=studies, task="SOH", out_path=soh_path)
        print(f"Saved ablation plots to {soc_path} and {soh_path}")
    else:
        soc_path = out_dir / f"ablation_soc_{args.metric}_single_axis.png"
        soh_path = out_dir / f"ablation_soh_{args.metric}_single_axis.png"
        _plot_task_single_axis(df, metric=args.metric, studies=studies, task="SOC", out_path=soc_path)
        _plot_task_single_axis(df, metric=args.metric, studies=studies, task="SOH", out_path=soh_path)
        print(f"Saved ablation plots to {soc_path} and {soh_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot ablation results from ablation_summary.csv")
    parser.add_argument("--input", type=str, required=True, help="Path to ablation_summary.csv or its parent directory")
    parser.add_argument("--output-root", type=str, default="output/ablation_plots")
    parser.add_argument("--metric", type=str, default="rmse", choices=["rmse", "mae", "r2"])
    parser.add_argument("--studies", type=str, default=None, help="Comma-separated studies to plot")
    parser.add_argument("--layout", type=str, default="by_task", choices=["combined", "by_task", "single_axis"])
    main(parser.parse_args())
