import argparse
from pathlib import Path
import pandas as pd


def collect_and_merge(src: Path, pattern: str, add_source: bool = True) -> pd.DataFrame:
    files = sorted(src.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matching {pattern} in {src}")
    dfs = []
    all_cols = set()
    for f in files:
        df = pd.read_csv(f)
        if add_source:
            df["source_file"] = f.stem
        dfs.append(df)
        all_cols.update(df.columns)
    # Align columns across files (union), preserving order from first df
    base_cols = list(dfs[0].columns)
    for c in all_cols:
        if c not in base_cols:
            base_cols.append(c)
    aligned = []
    for df in dfs:
        aligned.append(df.reindex(columns=base_cols))
    merged = pd.concat(aligned, ignore_index=True)
    return merged


def main():
    ap = argparse.ArgumentParser(description="Merge all charge/discharge combined CSVs into unified files")
    ap.add_argument("--src-charge", default="data/combined_split/combined_charge", help="Directory with charge combined_*.csv")
    ap.add_argument("--src-discharge", default="data/combined_split/combined_discharge", help="Directory with discharge combined_*.csv")
    ap.add_argument("--pattern", default="combined_*.csv", help="Glob pattern")
    ap.add_argument("--out-charge", default="data/combined_split/merged_charge.csv", help="Output CSV for charge")
    ap.add_argument("--out-discharge", default="data/combined_split/merged_discharge.csv", help="Output CSV for discharge")
    ap.add_argument("--add-source-col", action="store_true", help="Append source_file column for traceability")
    args = ap.parse_args()

    if args.src_charge:
        charge_df = collect_and_merge(Path(args.src_charge), args.pattern, add_source=args.add_source_col)
        Path(args.out_charge).parent.mkdir(parents=True, exist_ok=True)
        charge_df.to_csv(args.out_charge, index=False)
        print(f"Saved charge merged: {args.out_charge} | rows={len(charge_df)}")

    if args.src_discharge:
        discharge_df = collect_and_merge(Path(args.src_discharge), args.pattern, add_source=args.add_source_col)
        Path(args.out_discharge).parent.mkdir(parents=True, exist_ok=True)
        discharge_df.to_csv(args.out_discharge, index=False)
        print(f"Saved discharge merged: {args.out_discharge} | rows={len(discharge_df)}")


if __name__ == "__main__":
    main()
