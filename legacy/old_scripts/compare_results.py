import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def collect_preds(run_dir: Path):
    """Return concatenated (y_true, y_pred) from all *_preds.csv in run_dir.
    If none found, return None, None.
    """
    files = sorted(run_dir.glob("*_preds.csv"))
    if not files:
        return None, None
    ys, ps = [], []
    for f in files:
        try:
            df = pd.read_csv(f)
            if {"y_true", "y_pred"}.issubset(df.columns):
                ys.append(df["y_true"].values)
                ps.append(df["y_pred"].values)
        except Exception:
            continue
    if not ys:
        return None, None
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return y, p


def compute_metrics(y, p):
    mae = mean_absolute_error(y, p)
    mse = mean_squared_error(y, p)
    rmse = np.sqrt(mse)
    r2 = r2_score(y, p)
    return {"mae": mae, "rmse": rmse, "mse": mse, "r2": r2}


def plot_scatter_compare(results, out_path: Path, title: str = "True vs Predicted (Compare)"):
    """results: list of tuples (label, y, p). Values should be in same scale (e.g., 0-100)."""
    plt.figure(figsize=(7, 6))
    # global limits
    all_y = np.concatenate([y for _, y, _ in results])
    all_p = np.concatenate([p for _, _, p in results])
    low, high = float(min(all_y.min(), all_p.min())), float(max(all_y.max(), all_p.max()))
    for label, y, p in results:
        plt.scatter(y, p, s=10, alpha=0.5, label=label)
    plt.plot([low, high], [low, high], 'k--', linewidth=1)
    plt.xlabel('True SOC')
    plt.ylabel('Predicted SOC')
    plt.title(title)
    plt.legend(loc='best', fontsize=8)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Compare multiple evaluation runs (scatter + metrics table)")
    ap.add_argument('--run-dir', action='append', required=True,
                    help='Path to a run directory (repeat for multiple). Accepts runs/eval-* or compared_model/*-*.')
    ap.add_argument('--labels', type=str, nargs='*', default=None,
                    help='Optional labels for each run (same length as run-dir). Defaults to folder names.')
    ap.add_argument('--out', type=str, default=None, help='Output folder for comparison results (default vis_outputs/compare-<timestamp>)')
    args = ap.parse_args()

    run_dirs = [Path(p) for p in args.run_dir]
    if args.labels and len(args.labels) != len(run_dirs):
        raise ValueError('--labels length must match number of --run-dir')

    labels = args.labels or [d.name for d in run_dirs]

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out) if args.out else Path('vis_outputs') / f'compare-{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    compare_rows = []
    scatter_data = []

    for label, d in zip(labels, run_dirs):
        y, p = collect_preds(d)
        if y is None:
            print(f"Skip {d} (no *_preds.csv)")
            continue
        # Assume values are 0-100 scale as produced by eval_infer/baseline scripts
        metrics = compute_metrics(y, p)
        row = {"label": label, **metrics}
        compare_rows.append(row)
        scatter_data.append((label, y, p))

    if not scatter_data:
        print("No valid runs to compare.")
        return

    # Save metrics table
    df_cmp = pd.DataFrame(compare_rows)
    df_cmp.to_csv(out_dir / 'metrics_overall_compare.csv', index=False)

    # Plot scatter overlay
    plot_scatter_compare(scatter_data, out_dir / 'all_scatter_compare.png')

    print(f"Saved comparison to {out_dir}")


if __name__ == '__main__':
    main()

