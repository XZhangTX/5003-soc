import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd


def parse_excel_layout(infile: Path):
    # Expected layout:
    # row0: headers, row1: Ah (unused), row2: SOC, row3+: freqs/values
    df = pd.read_excel(infile, header=None)
    soc = df.iloc[2, 1:].to_numpy()
    freqs = df.iloc[3:, 0].to_numpy()
    values = df.iloc[3:, 1:].to_numpy()  # shape [freqs, samples]
    return freqs, values, soc


def pair_files(phase_dir: Path, mag_dir: Path, pattern: str):
    # Pair files by cycle token like 'c1', 'c2' from filename
    pairs = []
    phase_files = sorted(phase_dir.glob(pattern))
    for pf in phase_files:
        m = re.search(r"c\d+", pf.stem, re.IGNORECASE)
        key = m.group(0).lower() if m else pf.stem.lower()
        # try exact counterpart names
        candidates = list(mag_dir.glob(f"*{key}*.xlsx"))
        if not candidates:
            print(f"鏈壘鍒板搴斿箙搴︽枃浠? {pf.name} (key={key})")
            continue
        # choose first deterministic (sorted) candidate
        mf = sorted(candidates)[0]
        pairs.append((pf, mf, key))
    return pairs


def align_by_freq(freqs_a: np.ndarray, vals_a: np.ndarray,
                  freqs_b: np.ndarray, vals_b: np.ndarray):
    """Align by intersection of frequencies with tolerance, preserving A's order.

    Converts both freq arrays to numeric, rounds to 6 decimals to stabilize float
    comparisons, maps B->index, and selects common freqs in A's order.
    Returns: (common_freqs_from_A, vals_a_aligned, vals_b_aligned)
    """
    # Coerce to numeric floats
    A = pd.to_numeric(pd.Series(freqs_a), errors="coerce").to_numpy(dtype=float)
    B = pd.to_numeric(pd.Series(freqs_b), errors="coerce").to_numpy(dtype=float)

    A_norm = np.round(A, 6)
    B_norm = np.round(B, 6)

    b_index = {}
    for i, v in enumerate(B_norm):
        if v not in b_index:  # keep first occurrence
            b_index[v] = i

    idx_a = []
    idx_b = []
    for i, v in enumerate(A_norm):
        if v in b_index:
            idx_a.append(i)
            idx_b.append(b_index[v])

    if not idx_a:
        raise ValueError("骞呭害涓庣浉浣嶆棤鍏卞悓棰戠巼锛屾棤娉曞悎骞?)

    common = A[idx_a]
    a_aligned = vals_a[idx_a, :]
    b_aligned = vals_b[idx_b, :]
    return common, a_aligned, b_aligned


def build_combined_df(freqs: np.ndarray,
                      amp_vals: np.ndarray,
                      phase_vals: np.ndarray,
                      soc: np.ndarray,
                      phase_unit: str = "deg") -> pd.DataFrame:
    """Build per-frequency grouped features: [amp_f, sin_f, cos_f, ... , soc].

    amp_vals/phase_vals shapes: [n_freq, n_samples]
    """
    # sample-major
    X_amp = amp_vals.T              # [n_samples, n_freq]
    radians = np.deg2rad(phase_vals) if phase_unit == "deg" else phase_vals
    sin_mat = np.sin(radians).T     # [n_samples, n_freq]
    cos_mat = np.cos(radians).T     # [n_samples, n_freq]

    # Interleave columns per frequency: amp_f, sin_f, cos_f
    cols = []
    data_cols = []
    for j, f in enumerate(freqs):
        cols.extend([f"amp_{f}", f"sin_{f}", f"cos_{f}"])
        data_cols.append(X_amp[:, j])
        data_cols.append(sin_mat[:, j])
        data_cols.append(cos_mat[:, j])

    data = np.column_stack(data_cols) if data_cols else np.empty((X_amp.shape[0], 0))
    df = pd.DataFrame(data, columns=cols)
    df["soc"] = pd.to_numeric(soc, errors="coerce")
    return df


def process_pair(phase_file: Path, mag_file: Path, out_dir: Path, prefix: str, phase_unit: str,
                 freq_min: float | None = None, freq_max: float | None = None):
    f_phase, v_phase, soc = parse_excel_layout(phase_file)
    f_amp, v_amp, _ = parse_excel_layout(mag_file)

    # Validate sample counts
    if v_phase.shape[1] != v_amp.shape[1]:
        raise ValueError(f"鏍锋湰鏁颁笉涓€鑷? phase={v_phase.shape[1]} amp={v_amp.shape[1]} ({phase_file.name} vs {mag_file.name})")

    # Align by common freqs (order from mag by default)
    freqs, amp_aligned, phase_aligned = align_by_freq(f_amp, v_amp, f_phase, v_phase)\n    df = build_combined_df(freqs, amp_aligned, phase_aligned, soc, phase_unit=phase_unit)

    out_dir.mkdir(parents=True, exist_ok=True)
    key_match = re.search(r"c\d+", phase_file.stem, re.IGNORECASE)
    key = key_match.group(0).lower() if key_match else phase_file.stem
    out_csv = out_dir / f"{prefix}{key}.csv"
    df.to_csv(out_csv, index=False)
    print(f"宸茬敓鎴? {out_csv}")


def process_all(phase_dir: Path, mag_dir: Path, out_dir: Path, pattern: str, prefix: str, phase_unit: str,
                freq_min: float | None = None, freq_max: float | None = None):
    pairs = pair_files(phase_dir, mag_dir, pattern)
    if not pairs:
        print("鏈壘鍒板彲澶勭悊鐨勯厤瀵规枃浠?)
        return
    for pf, mf, key in pairs:
        process_pair(pf, mf, out_dir, prefix, phase_unit)
    print(f"瀹屾垚锛屽叡澶勭悊 {len(pairs)} 涓惊鐜?)


def main():
    parser = argparse.ArgumentParser(description="Combine magnitude and phase(sin/cos) per cycle into one CSV")
    parser.add_argument("--mag-dir", default=r"C:\\Users\\86182\\Desktop\\SOC_DATA\\0.3c_Cycle1-Cycle6\\magnitude", help="骞呭害鐩綍")
    parser.add_argument("--phase-dir", default=r"C:\\Users\\86182\\Desktop\\SOC_DATA\\0.3c_Cycle1-Cycle6\\phase", help="鐩镐綅鐩綍")
    parser.add_argument("--out-dir", default="data/combined", help="杈撳嚭鐩綍")
    parser.add_argument("--pattern", default="*.xlsx", help="鏂囦欢鍖归厤妯″紡锛岄粯璁?.xlsx")
    parser.add_argument("--prefix", default="combined_", help="杈撳嚭鏂囦欢鍓嶇紑")
    parser.add_argument("--phase-unit", choices=["deg", "rad"], default="deg", help="鐩镐綅鍗曚綅")\n    parser.add_argument("--freq-max", type=float, default=2_150_000, help="棰戠巼涓婄晫(鍚?")

    args = parser.parse_args()
    process_all(Path(args.phase_dir), Path(args.mag_dir), Path(args.out_dir), args.pattern, args.prefix, args.phase_unit,
                )


if __name__ == "__main__":
    main()

