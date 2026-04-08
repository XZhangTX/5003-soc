import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def find_monotonic_segments(soc: np.ndarray):
    diff = np.diff(soc)
    sign = np.sign(diff)
    segments = []
    start = 0
    cur_sign = None
    for i, s in enumerate(sign):
        if s == 0:
            continue
        if cur_sign is None:
            cur_sign = s
            start = i
            continue
        if s != cur_sign:
            segments.append((start, i + 1, cur_sign))  # inclusive end index of soc
            start = i + 1
            cur_sign = s
    if cur_sign is not None:
        segments.append((start, len(soc) - 1, cur_sign))
    return segments


def pick_longest(segments, sign):
    candidates = [seg for seg in segments if (seg[2] < 0 if sign < 0 else seg[2] > 0)]
    if not candidates:
        return None
    return max(candidates, key=lambda s: s[1] - s[0])


def split_file(path: Path, out_dir: Path, min_len: int = 10):
    df = pd.read_csv(path)
    if "soc" not in df.columns:
        print(f"[skip] {path.name} no soc column")
        return None
    soc = df["soc"].to_numpy()
    segments = find_monotonic_segments(soc)
    discharge = pick_longest(segments, sign=-1)
    charge = pick_longest(segments, sign=1)
    if discharge is None or charge is None:
        print(f"[skip] {path.name} cannot find both discharge & charge segments")
        return None

    def slice_seg(seg):
        a, b, _ = seg
        return df.iloc[a : b + 1].reset_index(drop=True)

    d_df = slice_seg(discharge)
    c_df = slice_seg(charge)
    if len(d_df) < min_len or len(c_df) < min_len:
        print(f"[skip] {path.name} segments too short (d={len(d_df)}, c={len(c_df)})")
        return None

    out_dir.mkdir(parents=True, exist_ok=True)
    d_out = out_dir / f"{path.stem}_discharge.csv"
    c_out = out_dir / f"{path.stem}_charge.csv"
    d_df.to_csv(d_out, index=False)
    c_df.to_csv(c_out, index=False)
    print(
        f"[ok] {path.name} -> discharge({len(d_df)}) charge({len(c_df)}) | "
        f"segments={segments}"
    )
    return d_out, c_out


def main():
    ap = argparse.ArgumentParser(description="Split combined CSV into discharge/charge segments")
    ap.add_argument("--src", default="data/combined", help="Source dir with combined_*.csv")
    ap.add_argument("--pattern", default="combined_*.csv", help="Glob pattern")
    ap.add_argument("--out", default="data/combined_split", help="Output directory")
    ap.add_argument("--min-len", type=int, default=10, help="Minimum rows for a segment")
    args = ap.parse_args()

    src = Path(args.src)
    out_dir = Path(args.out)
    paths = sorted(src.glob(args.pattern))
    if not paths:
        print(f"No files matching {args.pattern} in {src}")
        return
    for p in paths:
        split_file(p, out_dir, min_len=args.min_len)


if __name__ == "__main__":
    main()
