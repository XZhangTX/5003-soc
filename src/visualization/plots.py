from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score


def plot_training_curve(log_path: str | Path, out_path: str | Path):
    df = pd.read_csv(log_path)
    rmse_col = "val_rmse" if "val_rmse" in df.columns else "rmse"
    mae_col = "val_mae" if "val_mae" in df.columns else ("mae" if "mae" in df.columns else None)

    fig, ax1 = plt.subplots(figsize=(7, 4))
    ax1.plot(df["epoch"], df["train_loss"], label="train_loss", color="#1f77b4", linewidth=1.8)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Train Loss", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(df["epoch"], df[rmse_col], label=rmse_col, color="#ff7f0e", linewidth=1.8)
    if mae_col is not None:
        ax2.plot(df["epoch"], df[mae_col], label=mae_col, color="#2ca02c", linewidth=1.4, alpha=0.9)
    ax2.set_ylabel("Validation Metric", color="#ff7f0e")
    ax2.tick_params(axis="y", labelcolor="#ff7f0e")

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [line.get_label() for line in lines]
    ax1.legend(lines, labels, loc="upper right")

    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def plot_prediction_scatter(
    y_true,
    y_pred,
    out_path: str | Path,
    title: str = "Prediction Scatter",
    xlabel: str = "True SOC",
    ylabel: str = "Predicted SOC",
):
    y_true = np.asarray(y_true, dtype=np.float32)
    y_pred = np.asarray(y_pred, dtype=np.float32)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(np.mean(np.square(y_true - y_pred))))
    r2 = float(r2_score(y_true, y_pred))

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_true, y_pred, s=18, alpha=0.6)
    ax.plot([low, high], [low, high], "k--", linewidth=1)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    metrics_text = f"R2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
    ax.text(
        0.04,
        0.96,
        metrics_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc"},
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)



def plot_attention_heatmap(freqs, weights, out_path: str | Path, title: str = "Frequency Attention"):
    arr = np.asarray(weights).reshape(1, -1)
    freqs = np.asarray(freqs)
    plt.figure(figsize=(10, 2))
    plt.imshow(arr, aspect="auto", cmap="viridis")
    plt.colorbar(label="Attention Weight")
    step = max(1, len(freqs) // 20)
    xticks = np.arange(0, len(freqs), step)
    plt.xticks(xticks, freqs[::step], rotation=45, ha="right")
    plt.yticks([])
    plt.xlabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
