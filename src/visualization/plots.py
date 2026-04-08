from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_training_curve(log_path: str | Path, out_path: str | Path):
    df = pd.read_csv(log_path)
    rmse_col = "val_rmse" if "val_rmse" in df.columns else "rmse"
    plt.figure(figsize=(7, 4))
    plt.plot(df["epoch"], df["train_loss"], label="train_loss")
    plt.plot(df["epoch"], df[rmse_col], label=rmse_col)
    plt.xlabel("Epoch")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_prediction_scatter(y_true, y_pred, out_path: str | Path, title: str = "Prediction Scatter"):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    low = float(min(y_true.min(), y_pred.min()))
    high = float(max(y_true.max(), y_pred.max()))
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, s=10, alpha=0.5)
    plt.plot([low, high], [low, high], "k--", linewidth=1)
    plt.xlabel("True SOC")
    plt.ylabel("Predicted SOC")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


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
