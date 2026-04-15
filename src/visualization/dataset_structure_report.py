import argparse
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from src.data.discovery import discover_s11_records
from src.utils import ensure_dir, save_json, timestamp


def _iter_tree_lines(root: Path, max_depth: int, max_entries_per_dir: int) -> list[str]:
    lines = [str(root.resolve())]

    def walk(path: Path, prefix: str, depth: int):
        if depth >= max_depth:
            return
        entries = sorted(path.iterdir(), key=lambda item: (not item.is_dir(), item.name.lower()))
        shown = entries[:max_entries_per_dir]
        hidden = len(entries) - len(shown)
        for idx, entry in enumerate(shown):
            is_last = idx == len(shown) - 1 and hidden == 0
            connector = "└── " if is_last else "├── "
            lines.append(f"{prefix}{connector}{entry.name}")
            if entry.is_dir():
                child_prefix = prefix + ("    " if is_last else "│   ")
                walk(entry, child_prefix, depth + 1)
        if hidden > 0:
            lines.append(f"{prefix}└── ... ({hidden} more entries)")

    walk(root, "", 0)
    return lines


def _count_files_by_suffix(root: Path) -> pd.DataFrame:
    counter = Counter()
    for path in root.rglob("*"):
        if path.is_file():
            counter[path.suffix.lower() or "<no_suffix>"] += 1
    rows = [{"suffix": suffix, "count": count} for suffix, count in sorted(counter.items(), key=lambda x: (-x[1], x[0]))]
    return pd.DataFrame(rows)


def _summarize_days(root: Path, records) -> pd.DataFrame:
    file_rows = []
    day_dirs = [p for p in sorted(root.iterdir(), key=lambda p: p.name.lower()) if p.is_dir()]
    record_count_by_day = Counter(record.day for record in records)
    phase_count_by_day = Counter(record.day for record in records if record.has_phase)

    for day_dir in day_dirs:
        csv_count = sum(1 for _ in day_dir.rglob("*.csv"))
        file_rows.append(
            {
                "day": day_dir.name,
                "csv_file_count": csv_count,
                "record_count": int(record_count_by_day.get(day_dir.name, 0)),
                "phase_record_count": int(phase_count_by_day.get(day_dir.name, 0)),
            }
        )
    return pd.DataFrame(file_rows)


def _build_record_index(records, inspect_content: bool) -> pd.DataFrame:
    rows = []
    for record in records:
        row = {
            "day": record.day,
            "series_name": record.series_name,
            "record_id": record.record_id,
            "has_phase": record.has_phase,
            "mag_path": str(record.mag_path),
            "pha_path": str(record.pha_path) if record.pha_path else None,
        }
        if inspect_content:
            mag_df = pd.read_csv(record.mag_path, usecols=lambda c: c in {"Cycle", "SOC", "DC"})
            row["row_count"] = int(len(mag_df))
            row["cycle_count"] = int(mag_df["Cycle"].nunique()) if "Cycle" in mag_df.columns else None
            row["soc_min"] = float(mag_df["SOC"].min()) if "SOC" in mag_df.columns else None
            row["soc_max"] = float(mag_df["SOC"].max()) if "SOC" in mag_df.columns else None
            row["dc_modes"] = ",".join(sorted(mag_df["DC"].astype(str).str.upper().unique())) if "DC" in mag_df.columns else None
        rows.append(row)
    return pd.DataFrame(rows)


def _plot_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, ylabel: str, out_path: Path):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(df[x_col].astype(str), df[y_col].astype(float), color="#2b6cb0")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_col.replace("_", " ").title())
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_modality_coverage(records_df: pd.DataFrame, out_path: Path):
    if records_df.empty:
        return
    counts = pd.Series(
        {
            "Mag only": int((~records_df["has_phase"]).sum()),
            "Mag + Phase": int(records_df["has_phase"].sum()),
        }
    )
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(counts.index, counts.values, color=["#718096", "#2f855a"])
    ax.set_title("Record Modality Coverage")
    ax.set_ylabel("Record Count")
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _plot_day_series_heatmap(records_df: pd.DataFrame, out_path: Path):
    if records_df.empty:
        return
    prefix_rows = []
    for series in records_df["series_name"]:
        parts = str(series).split("_")
        prefix_rows.append("_".join(parts[:3]) if len(parts) >= 3 else str(series))
    tmp = records_df.copy()
    tmp["series_prefix"] = prefix_rows
    pivot = pd.pivot_table(tmp, index="day", columns="series_prefix", values="record_id", aggfunc="count", fill_value=0)
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(max(6, 0.8 * len(pivot.columns)), max(4, 0.5 * len(pivot.index))))
    im = ax.imshow(pivot.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_title("Record Count by Day and Series Prefix")
    fig.colorbar(im, ax=ax, label="Count")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main(args):
    root = Path(args.input_root)
    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")

    records = discover_s11_records(
        root=root,
        name_contains=args.name_contains,
        require_phase=args.require_phase,
        data_mode=args.data_mode,
    )

    out_dir = ensure_dir(Path(args.output_root) / "dataset_structure" / (args.tag or timestamp()))
    fig_dir = ensure_dir(out_dir / "figures")

    tree_lines = _iter_tree_lines(root, max_depth=args.max_depth, max_entries_per_dir=args.max_entries_per_dir)
    (out_dir / "tree.txt").write_text("\n".join(tree_lines), encoding="utf-8")

    file_suffix_df = _count_files_by_suffix(root)
    file_suffix_df.to_csv(out_dir / "file_suffix_counts.csv", index=False)

    day_summary_df = _summarize_days(root, records)
    day_summary_df.to_csv(out_dir / "day_summary.csv", index=False)

    record_index_df = _build_record_index(records, inspect_content=args.inspect_content)
    record_index_df.to_csv(out_dir / "record_index.csv", index=False)

    summary = {
        "input_root": str(root.resolve()),
        "record_count": int(len(records)),
        "day_count": int(record_index_df["day"].nunique()) if not record_index_df.empty else 0,
        "phase_record_count": int(record_index_df["has_phase"].sum()) if not record_index_df.empty else 0,
        "mag_only_record_count": int((~record_index_df["has_phase"]).sum()) if not record_index_df.empty else 0,
        "data_mode": args.data_mode,
        "require_phase": bool(args.require_phase),
    }
    if args.inspect_content and not record_index_df.empty:
        summary["total_row_count"] = int(record_index_df["row_count"].fillna(0).sum())
        summary["total_cycle_count"] = int(record_index_df["cycle_count"].fillna(0).sum())
    save_json(out_dir / "summary.json", summary)

    _plot_bar(day_summary_df, "day", "csv_file_count", "CSV File Count by Day", "CSV Files", fig_dir / "csv_files_by_day.png")
    _plot_bar(day_summary_df, "day", "record_count", "Discovered Records by Day", "Records", fig_dir / "records_by_day.png")
    _plot_modality_coverage(record_index_df, fig_dir / "modality_coverage.png")
    _plot_day_series_heatmap(record_index_df, fig_dir / "day_series_heatmap.png")

    print(f"Saved dataset structure report to {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Summarize and visualize dataset directory structure")
    parser.add_argument("--input-root", type=str, default=r"D:\SOC_DATA\data_complete\data_processing\S11_ALIGN_SOC_new")
    parser.add_argument("--output-root", type=str, default="output")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--data-mode", type=str, default="all", choices=["all", "raw", "socip0p1", "socip0p5", "socip1p0"])
    parser.add_argument("--name-contains", type=str, default=None)
    parser.add_argument("--require-phase", action="store_true")
    parser.add_argument("--inspect-content", action="store_true")
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--max-entries-per-dir", type=int, default=30)
    main(parser.parse_args())
