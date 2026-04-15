import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.utils import ensure_dir


SUMMARY_DATA = {
    "SOC": {
        "XGBoost": {"rmse": (3.750, 0.868), "mae": (2.151, 0.301)},
        "MLP": {"rmse": (2.976, 0.135), "mae": (2.013, 0.016)},
        "Vanilla Transformer": {"rmse": (3.530, 0.715), "mae": None},
        "Conv-Transformer": {"rmse": (2.740, 0.279), "mae": (1.594, 0.021)},
    },
    "SOH": {
        "XGBoost": {"rmse": (2.427, 1.713), "mae": (1.278, 0.592)},
        "MLP": {"rmse": (30.721, 5.043), "mae": (24.620, 4.674)},
        "Vanilla Transformer": {"rmse": None, "mae": None},
        "Conv-Transformer": {"rmse": (3.631, 0.066), "mae": (2.627, 0.001)},
    },
}

MODEL_ORDER = ["Conv-Transformer", "Vanilla Transformer", "MLP", "XGBoost"]
MODEL_COLORS = {
    "Conv-Transformer": "#8FD3FF",
    "Vanilla Transformer": "#FFD6A5",
    "MLP": "#9DE0D3",
    "XGBoost": "#FFB3BA",
}
DEFAULT_METRICS = ["mae", "rmse"]
METRIC_LABELS = {"mae": "MAE", "rmse": "RMSE", "r2": r"$R^2$"}


def _sample_from_summary(mean: float, std: float, metric: str, n_samples: int, rng: np.random.Generator):
    if std <= 0:
        return np.full(n_samples, mean, dtype=np.float32)
    vals = rng.normal(loc=mean, scale=std, size=n_samples)
    if metric in {"mae", "rmse"}:
        vals = np.clip(vals, 0.0, None)
    return vals.astype(np.float32)


def _plot_panel(ax, task: str, task_data: dict, metrics: list[str], n_samples: int, seed: int):
    rng = np.random.default_rng(seed)
    models = [model for model in MODEL_ORDER if model in task_data]
    metric_positions = np.arange(len(metrics)) * 1.8 + 1.0
    offsets = np.linspace(-0.45, 0.45, len(models))

    legend_patches = []
    for model_idx, model in enumerate(models):
        color = MODEL_COLORS[model]
        legend_patches.append(plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor=color, alpha=0.8, label=model))
        for metric_idx, metric in enumerate(metrics):
            summary = task_data[model].get(metric)
            if summary is None:
                continue
            mean, std = summary
            samples = _sample_from_summary(mean, std, metric, n_samples=n_samples, rng=rng)
            pos = metric_positions[metric_idx] + offsets[model_idx]
            vp = ax.violinplot(
                dataset=[samples],
                positions=[pos],
                widths=0.28,
                showmeans=False,
                showmedians=False,
                showextrema=False,
            )
            for body in vp["bodies"]:
                body.set_facecolor(color)
                body.set_edgecolor(color)
                body.set_alpha(0.75)

            ax.hlines(mean, pos - 0.14, pos + 0.14, colors="red", linewidth=1.6, zorder=4)
            ax.vlines(pos, mean - std, mean + std, colors="black", linewidth=1.1, zorder=4)
            ax.text(pos, mean, f"{mean:.2f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax.set_xticks(metric_positions)
    ax.set_xticklabels([METRIC_LABELS[m] for m in metrics], fontsize=12)
    ax.set_title(f"{task} Estimation", fontsize=15)
    ax.grid(True, axis="y", alpha=0.25)

    mean_handle = plt.Line2D([0], [0], color="red", lw=1.6, label="Mean")
    std_handle = plt.Line2D([0], [0], color="black", lw=1.1, label="Mean ± Std")
    ax.legend(handles=legend_patches + [mean_handle, std_handle], loc="upper right", frameon=False, fontsize=10)


def plot_summary_violin(output_path: Path, metrics: list[str], n_samples: int, seed: int):
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2), sharey=False)
    _plot_panel(axes[0], "SOC", SUMMARY_DATA["SOC"], metrics=metrics, n_samples=n_samples, seed=seed)
    _plot_panel(axes[1], "SOH", SUMMARY_DATA["SOH"], metrics=metrics, n_samples=n_samples, seed=seed + 1)
    axes[0].set_ylabel("Metric Value", fontsize=12)
    fig.suptitle("Approximate Violin Plots from Reported Mean ± Std", fontsize=17)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main(args):
    out_dir = ensure_dir(Path(args.output_root))
    out_path = out_dir / "summary_violin_plot.png"
    metrics = [item.strip().lower() for item in args.metrics.split(",") if item.strip()] or list(DEFAULT_METRICS)
    plot_summary_violin(out_path, metrics=metrics, n_samples=args.n_samples, seed=args.seed)
    print(f"Saved summary violin plot to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw a violin plot from manually entered mean ± std summary values")
    parser.add_argument("--output-root", type=str, default="output/summary_violin")
    parser.add_argument("--metrics", type=str, default="mae,rmse")
    parser.add_argument("--n-samples", type=int, default=256, help="Synthetic sample count per violin")
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
