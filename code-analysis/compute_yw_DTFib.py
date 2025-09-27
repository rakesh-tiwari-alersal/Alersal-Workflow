#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import pandas as pd

def load_series(filepath):
    df = pd.read_csv(filepath, parse_dates=True, index_col=0)
    for col in ("Close", "close", "Price", "price"):
        if col in df.columns:
            return df[col].astype(float)
    raise ValueError("No price column found (tried Close/close/Price/price).")

def compute_detrended(series, l1, l2):
    Y = series
    Ydt = (Y - Y.shift(l1) - Y.shift(l2)) / Y
    Ydt = Ydt.dropna()
    return Ydt

def find_peaks_and_valleys(arr):
    peaks, valleys = [], []
    n = len(arr)
    if n < 3:
        return peaks, valleys
    a = np.asarray(arr)
    for i in range(1, n - 1):
        if a[i] > a[i - 1] and a[i] > a[i + 1]:
            peaks.append(i)
        elif a[i] < a[i - 1] and a[i] < a[i + 1]:
            valleys.append(i)
    return peaks, valleys

def count_fib_hits(values, indices, level, tol):
    return sum(1 for idx in indices if abs(values[idx] - level) <= tol)

def main():
    parser = argparse.ArgumentParser(
        description="Compute detrended stats (Median, Stdev) and Fibonacci peak/valley hits."
    )
    parser.add_argument("-f", "--file", required=True, help="CSV filename inside historical_data/")
    parser.add_argument("-l", "--lags", required=True, help="Two comma-separated integer lags, e.g. 17,179")
    parser.add_argument("-t", "--tolerance", type=float, default=0.002,
                        help="Tolerance for counting a 'touch' of a Fibonacci level (default 0.002 = 0.2%)")
    args = parser.parse_args()

    # parse lags
    parts = [p.strip() for p in args.lags.split(",") if p.strip()]
    if len(parts) != 2:
        print("Error: this script requires exactly two lags.", file=sys.stderr)
        sys.exit(1)
    try:
        l1, l2 = int(parts[0]), int(parts[1])
    except ValueError:
        print("Error: lags must be integers.", file=sys.stderr)
        sys.exit(1)

    # load series
    filepath = os.path.join("historical_data", args.file) if not os.path.isabs(args.file) else args.file
    if not os.path.exists(filepath):
        print(f"Error: file not found: {filepath}", file=sys.stderr)
        sys.exit(1)

    series = load_series(filepath)

    # compute detrended series
    Ydt = compute_detrended(series, l1, l2)

    # basic stats
    med = float(np.median(Ydt)) if len(Ydt) > 0 else float("nan")
    stdev = float(np.std(Ydt, ddof=1)) if len(Ydt) > 1 else float("nan")

    print(f"Median(Ydt): {med:.6f}")
    print(f"Stdev(Ydt): {stdev:.6f}")

    # Fibonacci hits
    fib_levels = [0.1459, 0.236, 0.382, 0.5, 0.618]
    tol = args.tolerance

    if len(Ydt) < 3:
        print("Fibonacci hits: insufficient data.")
        return

    yvals = np.asarray(Ydt.values)
    peaks_idx, valleys_idx = find_peaks_and_valleys(yvals)

    total_hits = 0
    print("Fibonacci hits per level (format: Level%  peaks valleys total)")

    for lvl in fib_levels:
        # positive level
        p_hits = count_fib_hits(yvals, peaks_idx, lvl, tol)
        v_hits = count_fib_hits(yvals, valleys_idx, lvl, tol)
        tot = p_hits + v_hits
        total_hits += tot
        print(f"+{lvl*100:.2f}%  {p_hits} {v_hits} {tot}")

        # negative level
        nlvl = -lvl
        p_hits_n = count_fib_hits(yvals, peaks_idx, nlvl, tol)
        v_hits_n = count_fib_hits(yvals, valleys_idx, nlvl, tol)
        tot_n = p_hits_n + v_hits_n
        total_hits += tot_n
        print(f"-{lvl*100:.2f}%  {p_hits_n} {v_hits_n} {tot_n}")

    print(f"Total fib hits: {total_hits}")

if __name__ == "__main__":
    main()
