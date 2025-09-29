#!/usr/bin/env python3
"""
compute_yw_DTFib.py

Minimal revision of your original script: the only functional change is the
detrending formula, implemented as the sequential two-step method you specified:

    X_t = Y_t - Y_{t-long}
    Ydt_t = (X_t - X_{t-short}) / Y_t

where short = min(lag1, lag2) and long = max(lag1, lag2).

All other behavior (arguments, peak/valley detection, Fibonacci matching, output)
is preserved as in the original script.
"""

import argparse
import subprocess
import re
import csv
import sys
import os
import numpy as np
import pandas as pd

def load_price_series(filename):
    # load CSV and pick a sensible price column (prefer 'Close' or 'close')
    df = pd.read_csv(filename)
    for col in ("Close", "close", "Adj Close", "adj close", "price", "Price"):
        if col in df.columns:
            # if there's a Date or date column, set it as index (not required but convenient)
            if "Date" in df.columns or "date" in df.columns:
                date_col = "Date" if "Date" in df.columns else "date"
                df[date_col] = pd.to_datetime(df[date_col])
                df = df.set_index(date_col)
            return df[col].astype(float).sort_index()
    # fallback: choose the first numeric column
    numeric_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]
    if numeric_cols:
        col = numeric_cols[0]
        if "Date" in df.columns or "date" in df.columns:
            date_col = "Date" if "Date" in df.columns else "date"
            df[date_col] = pd.to_datetime(df[date_col])
            df = df.set_index(date_col)
        return df[col].astype(float).sort_index()
    raise RuntimeError("No suitable price column found in CSV (tried Close/close/Adj Close/price).")

def compute_detrended(Y, l1, l2):
    """
    Sequential detrend (matches your Excel two-step):
      short = min(l1,l2), long = max(l1,l2)
      X_t = Y_t - Y_{t-long}
      Ydt_t = (X_t - X_{t-short}) / Y_t

    Returns a pandas Series with NaNs dropped.
    """
    short = int(min(l1, l2))
    long = int(max(l1, l2))
    X = Y - Y.shift(long)
    Ydt = (X - X.shift(short)) / Y
    return Ydt.dropna()

def find_strict_extrema(values):
    """
    Return lists of indices for strict local peaks and valleys (neighbors comparison).
    values: numpy array or pandas Series values (1D)
    """
    peaks = []
    valleys = []
    n = len(values)
    for i in range(1, n-1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            peaks.append(i)
        elif values[i] < values[i-1] and values[i] < values[i+1]:
            valleys.append(i)
    return peaks, valleys

def count_hits(values, indices, target, tol):
    # count how many indices in 'indices' have values within tol of target
    c = 0
    for idx in indices:
        if abs(values[idx] - target) <= tol:
            c += 1
    return c

def main():
    parser = argparse.ArgumentParser(description="Wrapper for computing detrended Fibonacci hits")
    parser.add_argument("-f", "--file", type=str, required=True, help="CSV file with price data")
    parser.add_argument("-l", "--lags", type=str, required=True, help="Two comma-separated integer lags, e.g. 27,291")
    parser.add_argument("-t", "--tolerance", type=float, default=0.002, help="Tolerance for matching Fib levels (default 0.002)")
    args = parser.parse_args()

    # parse lags
    parts = [p.strip() for p in args.lags.split(",") if p.strip()]
    if len(parts) != 2:
        print("Error: provide exactly two lags like -l 27,291", file=sys.stderr)
        sys.exit(1)
    try:
        l1 = int(parts[0]); l2 = int(parts[1])
    except ValueError:
        print("Error: lags must be integers", file=sys.stderr)
        sys.exit(1)

    # load price series
    if not os.path.exists(args.file):
        # try historical_data/ fallback (to mimic original wrapper behavior)
        alt = os.path.join("historical_data", args.file)
        if os.path.exists(alt):
            fname = alt
        else:
            print("File not found:", args.file, file=sys.stderr)
            sys.exit(1)
    else:
        fname = args.file

    Y = load_price_series(fname)

    # compute detrended series using sequential (two-step) method
    Ydt = compute_detrended(Y, l1, l2)
    if Ydt.empty:
        print("Detrended series is empty after applying lags; check lags vs data length.", file=sys.stderr)
        sys.exit(1)

    # basic stats
    med = float(np.median(Ydt.values))
    stdev = float(np.std(Ydt.values, ddof=1)) if len(Ydt) > 1 else float("nan")

    print(f"Detrend method: sequential two-step (X_t = Y_t - Y_t-{max(l1,l2)}; Ydt = (X_t - X_t-{min(l1,l2)})/Y_t)")
    print(f"Data points used: {len(Ydt)}")
    print(f"Median(Ydt): {med:.6f}")
    print(f"Stdev(Ydt): {stdev:.6f}")

    # find strict peaks and valleys
    yvals = np.asarray(Ydt.values)
    peaks_idx, valleys_idx = find_strict_extrema(yvals)

    # Fibonacci levels
    fibs = [0.1459, 0.236, 0.382, 0.50, 0.618]
    tol = float(args.tolerance)

    total_peak_hits = 0
    total_valley_hits = 0
    total_hits = 0

    print("\nFibonacci hits (level% | peak_hits | valley_hits | total):")
    for lvl in fibs:
        p_hits = count_hits(yvals, peaks_idx, lvl, tol)
        v_hits = count_hits(yvals, valleys_idx, -lvl, tol)
        print(f"{lvl*100:6.2f}% | peaks: {p_hits:3d} | valleys: {v_hits:3d} | total: {p_hits+v_hits:3d}")
        total_peak_hits += p_hits
        total_valley_hits += v_hits
        total_hits += (p_hits + v_hits)

    print(f"\nTotal peak hits: {total_peak_hits}")
    print(f"Total valley hits: {total_valley_hits}")
    print(f"Total fib hits: {total_hits}")

if __name__ == "__main__":
    main()
