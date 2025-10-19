#!/usr/bin/env python3
"""
research_pacf.py

Analyze PACF peaks across multiple ranges and write results to CSV.

Notes:
- Preprocessing: log-returns of Close price.
- Peak/significance: PACF lags exceeding approx 99% CI: |pacf| > 2.5758/sqrt(n)
- Outputs top-10 lags per range (by absolute PACF), with % contribution.
- CLI parsing robust to PowerShell comma splitting (borrowed from research_psd.py).
- CSV layout: two columns per range (Lag, % Contribution) with one empty column delimiter
  between adjacent ranges/groups (matches PSD layout style).
"""

import os
import sys
import argparse
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import pacf

# === Configuration ===
RANGES: List[Tuple[int, int]] = [
    (15, 40),
    (30, 60),
    (50, 90),
    (150, 350),
    (200, 500),
    (350, 700)
]

TOP_N = 10


# === Analysis utilities ===
def load_series_from_csv(file_name: str) -> Optional[pd.Series]:
    """
    Load price series (Close/close/Price/price) from historical_data/<file_name> and return as pandas Series.
    Returns None on error.
    """
    file_path = os.path.join('historical_data', file_name)
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        return None

    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
    except Exception as e:
        print(f"Error loading CSV {file_path}: {e}")
        return None

    # Flexible column detection
    if 'Close' in df.columns:
        series = df['Close']
    elif 'close' in df.columns:
        series = df['close']
    elif 'Price' in df.columns:
        series = df['Price']
    elif 'price' in df.columns:
        series = df['price']
    else:
        print(f"No price-like column found in {file_path}. Expected one of: Close, close, Price, price.")
        return None

    series = series.astype(float)
    if np.any(series <= 0):
        print("Non-positive 'Close/Price' values found — aborting (log-returns require positive prices).")
        return None

    return series


def compute_log_returns(series: pd.Series) -> np.ndarray:
    """Compute log returns from price series (close_t)."""
    prices = series.values
    logrets = np.log(prices[1:]) - np.log(prices[:-1])
    return logrets


def analyze_single_range(series: pd.Series, range_min: int, range_max: int) -> Optional[List[Tuple[int, float]]]:
    """
    Compute PACF for the preprocessed series and return list of (lag, pacf_value)
    for significant lags (|pacf| > 2.5758/sqrt(n)), sorted by absolute pacf desc, top TOP_N.
    """
    x = compute_log_returns(series)
    n = len(x)
    if n <= 0:
        print("Not enough data after preprocessing.")
        return None

    # Determine nlags reasonably: use range_max as the maximum lag to compute PACF
    nlags = max(range_max, int(0.5 * n))
    try:
        pacf_vals = pacf(x, nlags=nlags, method='ywm')
    except Exception:
        pacf_vals = pacf(x, nlags=nlags)

    selected = []
    threshold = 2.5758 / np.sqrt(n)  # ~99% conf bound for PACF
    for lag in range(range_min, min(range_max, len(pacf_vals) - 1) + 1):
        val = pacf_vals[lag]
        if np.abs(val) > threshold:
            selected.append((lag, float(val)))

    if not selected:
        return []

    selected.sort(key=lambda t: abs(t[1]), reverse=True)
    top = selected[:TOP_N]
    return top


# === CSV building utilities (PSD style layout) ===
def make_csv_table_for_ranges(all_results: List[List[Tuple[int, float]]], ranges: List[Tuple[int, int]]) -> pd.DataFrame:
    """
    Build a DataFrame where each range gets two columns: 'Lag (min-max)' and '% Contribution'.
    Insert a single empty column (separator) between ranges/groups (matching PSD style).
    """
    headers = []
    for rmin, rmax in ranges:
        headers.append(f"Lag ({rmin}-{rmax})")
        headers.append("% Contribution")
        headers.append("")  # blank separator column

    num_rows = TOP_N
    rows = []

    abs_sums = []
    for res in all_results:
        abs_sums.append(sum(abs(v) for (_, v) in res) if res else 0.0)

    for i in range(num_rows):
        row = []
        for idx, res in enumerate(all_results):
            if i < len(res):
                lag, val = res[i]
                pct = (abs(val) / abs_sums[idx] * 100.0) if abs_sums[idx] > 0 else 0.0
                row.append(str(lag))
                row.append(f"{pct:.2f}")
            else:
                row += ["", ""]
            row.append("")  # separator column
        rows.append(row)

    return pd.DataFrame(rows, columns=headers)

def append_pacf_all_file(file_base, group2_results, clear_flag=False):
    """
    Append a single-row summary to research_output/research_pacf_ALL.csv.
    Row format: file_base,lag1,lag2,...
    lag1.. are unique lag values from group2_results (last 3 ranges), duplicates removed,
    sorted descending (integers).
    If clear_flag True, truncate (overwrite) the ALL file first.
    """
    outdir = 'research_output'
    os.makedirs(outdir, exist_ok=True)
    all_file = os.path.join(outdir, 'research_pacf_ALL.csv')

    # Collect lags from the three group2_results lists
    lags = []
    for res in group2_results:
        if not res:
            continue
        for lag, _ in res:
            lags.append(int(lag))

    # Remove duplicates and sort descending
    unique_lags = sorted({int(x) for x in lags}, reverse=True)

    # Prepare row: start with file_base
    row = [file_base] + [str(x) for x in unique_lags]

    # If clear flag: truncate file (write headerless empty file)
    mode = 'a'
    if clear_flag:
        # overwrite/truncate
        with open(all_file, 'w', newline='') as fh:
            pass
        mode = 'a'

    # Append CSV row (no header)
    with open(all_file, mode, newline='') as fh:
        import csv
        writer = csv.writer(fh)
        writer.writerow(row)


# === Main ===
def main():
    # Borrow PSD's PowerShell-safe argv parsing
    fixed_argv = []
    if len(sys.argv) > 1:
        i = 0
        while i < len(sys.argv):
            token = sys.argv[i]
            if token.isdigit() and i >= 1 and sys.argv[i - 1] in ('-r', '--range'):
                if i + 1 < len(sys.argv) and sys.argv[i + 1].isdigit():
                    merged = token + ',' + sys.argv[i + 1]
                    fixed_argv.append(merged)
                    i += 2
                    continue
            fixed_argv.append(token)
            i += 1
    else:
        fixed_argv = sys.argv[:]

    parser = argparse.ArgumentParser(description='Analyze PACF for a single instrument')
    parser.add_argument('-f', '--file', type=str, required=True,
                        help='CSV file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-r', '--range', type=str,
                        help='Comma-separated range to analyze (e.g., 30,60)', default=None)
    parser.add_argument('-c', '--clear-summary', action='store_true', help='If provided, clear research_pacf_ALL.csv before appending.')
    args = parser.parse_args(fixed_argv[1:])

    series = load_series_from_csv(args.file)
    if series is None:
        return

    # --- Single range ---
    if args.range:
        try:
            range_min, range_max = map(int, args.range.split(','))
        except Exception:
            print("Invalid -r/--range format. Provide as 'min,max' (e.g., 30,60).")
            return

        print(f"Analyzing PACF range {range_min}-{range_max} for {args.file} ...")
        results = analyze_single_range(series, range_min, range_max)
        if results is None:
            print("Analysis failed.")
            return
        if not results:
            print("No significant PACF lags found in that range.")
            return

        total_abs = sum(abs(v) for (_, v) in results)
        print(f"\nPACF ({range_min}-{range_max})\tPACF Value\t% Contribution")
        for lag, val in results:
            pct = (abs(val) / total_abs * 100.0) if total_abs > 0 else 0.0
            print(f"{lag}\t{val:.4f}\t{pct:.2f}")
        return

    # --- All ranges ---
    all_results = []
    for rmin, rmax in RANGES:
        res = analyze_single_range(series, rmin, rmax)
        all_results.append(res if res is not None else [])

    output_df = make_csv_table_for_ranges(all_results, RANGES)

    os.makedirs('research_output', exist_ok=True)
    file_base = os.path.splitext(args.file)[0]
    output_file = os.path.join('research_output', f"research_pacf_{file_base}.csv")
    output_df.to_csv(output_file, index=False)
    print(f"Results saved to {output_file}")

    # Append the "ALL" file (last three ranges -> group2_results)
    group2_results = [all_results[3], all_results[4], all_results[5]]  # 150-350,200-500,350-700
    append_pacf_all_file(file_base, group2_results, clear_flag=args.clear_summary)

if __name__ == "__main__":
    main()
