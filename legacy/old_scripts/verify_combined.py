import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def read_excel_layout(xlsx: Path):
    df = pd.read_excel(xlsx, header=None)
    soc = pd.to_numeric(df.iloc[2, 1:], errors='coerce').to_numpy()
    freqs = pd.to_numeric(df.iloc[3:, 0], errors='coerce').to_numpy()
    vals = df.iloc[3:, 1:].apply(pd.to_numeric, errors='coerce').to_numpy()
    return freqs, vals, soc


def freqs_from_combined_cols(cols):
    amp_cols = cols[:-1][0::3]
    freqs = []
    for c in amp_cols:
        try:
            f = float(str(c).replace('amp_', ''))
        except Exception:
            f = np.nan
        freqs.append(f)
    return np.array(freqs, dtype=float)


def main():
    ap = argparse.ArgumentParser(description='Verify combined CSV against raw magnitude/phase Excel')
    ap.add_argument('--combined', type=str, default='data/combined/combined_c1.csv')
    ap.add_argument('--magnitude', type=str, default=r"C:\Users\86182\Desktop\SOC_DATA\0.3c_Cycle1-Cycle6\magnitude\magnitude_c1.xlsx")
    ap.add_argument('--phase', type=str, default=r"C:\Users\86182\Desktop\SOC_DATA\0.3c_Cycle1-Cycle6\phase\phase_c1.xlsx")
    ap.add_argument('--round', type=int, default=6, help='round decimals when matching freqs')
    args = ap.parse_args()

    comb = pd.read_csv(args.combined)
    comb_cols = list(comb.columns)
    assert comb_cols[-1].lower() == 'soc', 'combined 最后一列必须是 soc'

    # counts in combined
    n_samples_comb = len(comb)
    n_feat_cols = len(comb_cols) - 1
    assert n_feat_cols % 3 == 0, 'combined 特征列应为 3 的倍数'
    n_freq_comb = n_feat_cols // 3
    freqs_comb = freqs_from_combined_cols(comb_cols)

    # read raw excel
    f_mag, v_mag, soc_mag = read_excel_layout(Path(args.magnitude))
    f_ph, v_ph, soc_ph = read_excel_layout(Path(args.phase))

    print('=== 样本数(列)检查 ===')
    print(f'combined 行(样本) = {n_samples_comb}; phase 列(样本) = {v_ph.shape[1]}; mag 列(样本) = {v_mag.shape[1]}')
    if not (n_samples_comb == v_ph.shape[1] == v_mag.shape[1]):
        print('样本数量不一致：请检查相位/幅度 Excel 的列数与 combined 行数是否对齐。')

    print('\n=== 频点数检查 ===')
    print(f'magnitude 频点数 = {len(f_mag)}; phase 频点数 = {len(f_ph)}; combined 频点数 = {n_freq_comb}')

    r = args.round
    A = np.round(pd.to_numeric(pd.Series(f_mag), errors='coerce').to_numpy(dtype=float), r)
    B = np.round(pd.to_numeric(pd.Series(f_ph), errors='coerce').to_numpy(dtype=float), r)
    C = np.round(freqs_comb, r)

    setA = set(A[~np.isnan(A)])
    setB = set(B[~np.isnan(B)])
    setC = set(C[~np.isnan(C)])

    inter = setA & setB
    only_mag = sorted(list(setA - setB))[:10]
    only_phase = sorted(list(setB - setA))[:10]
    only_comb_not_inter = sorted(list(setC - inter))[:10]

    print(f'交集(幅度∩相位) 频点数 = {len(inter)}')
    print(f'combined 频点数 = {len(setC)} (应≈交集)')
    if len(setC) != len(inter):
        print('警告：combined 的频点集合与(幅度∩相位)交集不一致，可能存在命名或解析问题。')

    if only_mag:
        print(f'仅在 magnitude 中存在的前10个频点: {only_mag}')
    if only_phase:
        print(f'仅在 phase 中存在的前10个频点: {only_phase}')
    if only_comb_not_inter:
        print(f'仅在 combined 中存在但不在交集的前10个频点: {only_comb_not_inter}')

    print('\n=== 列名与频点解析抽查 ===')
    sample_cols = comb_cols[:9]
    print('combined 前9列:', sample_cols)

    print('\n结论参考：')
    print('- 如果 combined 频点数 < magnitude 频点数，通常因为只保留了与 phase 的交集（对齐所致）。')
    print('- 如果 combined 频点集合与交集不一致，可能是列名 amp_<freq> 解析失败，或 excel 频率存在格式/小数差异。')
    print('- 样本数不一致则说明列数(样本)对不上，需要检查 excel 的有效列范围与空列。')


if __name__ == '__main__':
    main()
