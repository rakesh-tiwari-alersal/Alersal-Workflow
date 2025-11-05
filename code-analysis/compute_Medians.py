#!/usr/bin/env python3
"""
compute_median_matches.py

Compute median-match counts for odd cycles 17..99 using the *direct median* method.

For each odd cycle N and for each index t >= N-1:
    - Take the window of prices: prices[t-N+1 : t+1]  (INCLUDES current price)
    - Compute the median of this window normally (odd-length median)
    - Count a match when abs(prices[t] - median_window) <= 0.0001

No timestamps/dates are processed; the time series is treated as a 1D array of prices.

Usage:
    python compute_median_matches.py -f BTC-USD.csv

Input file is expected at: ./historical_data/<filename>
Output file will be: ./results_output/median_matches_<basename>.csv
"""

import argparse
import os
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Fixed cycles: odd integers from 17 to 99 inclusive
CYCLES = list(range(17, 100, 2))
TOL = 1e-4


def load_price_array(input_path: str) -> np.ndarray:
    """
    Load CSV without any date parsing. Return a 1D numpy array of prices (float).
    Accept common price column names. If only one column exists, use it.
    """
    df = pd.read_csv(input_path)

    # If there's exactly one column, assume it's the price column.
    if df.shape[1] == 1:
        col = df.columns[0]
        prices = pd.to_numeric(df[col], errors="coerce").to_numpy()
        return prices

    # Otherwise try common price column names (case variants)
    for col in ['Close', 'close', 'PRICE', 'Price', 'price']:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy()

    # If nothing matched, try to pick the first numeric-looking column.
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(CYCLES):  # enough numeric values
            return s.to_numpy()

    raise ValueError(
        "CSV must contain a usable price column (e.g., Close/close/PRICE/Price/price) "
        "or a single column of prices."
    )


def count_matches_inclusive_median(prices: np.ndarray, cycle: int) -> Tuple[int, List[int]]:
    """
    For a given odd cycle N, count indices t where the current price equals
    the median of the inclusive window prices[t-N+1 : t+1] within tolerance TOL.
    Returns (match_count, list_of_indices).
    """
    n = len(prices)
    if n < cycle:
        return 0, []

    matches = 0
    idxs: List[int] = []

    k = cycle // 2  # middle index for odd-length median

    # Loop over valid end positions
    for t in range(cycle - 1, n):
        window = prices[t - cycle + 1 : t + 1]

        # Skip if current or any window value is NaN
        if np.isnan(prices[t]) or np.isnan(window).any():
            continue

        # Compute odd-length median efficiently: nth-element selection
        # np.partition places the k-th smallest element at position k
        wcopy = np.array(window, copy=True)
        median_val = np.partition(wcopy, k)[k]

        if abs(prices[t] - median_val) <= TOL:
            matches += 1
            idxs.append(t)

    return matches, idxs


def compute_all_cycles(prices: np.ndarray, cycles: List[int]) -> List[Tuple[int, int]]:
    """
    Compute match counts for all cycles and return list of (cycle, count),
    sorted by count desc, then cycle asc.
    """
    results = []
    for n in cycles:
        cnt, _ = count_matches_inclusive_median(prices, n)
        results.append((n, cnt))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def save_results_csv(results: List[Tuple[int, int]], out_path: str) -> None:
    """
    Save results to a comma-separated CSV with header:
      Cycle Length,Median Matches,% Drop After Prev
    """
    df = pd.DataFrame(results, columns=["Cycle Length", "Median Matches"])
    prev = df["Median Matches"].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_drop = np.where(prev > 0, (prev - df["Median Matches"]) / prev * 100.0, np.nan)
    df["% Drop After Prev"] = pd.Series(pct_drop).round(2)
    df.to_csv(out_path, index=False)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    parser = argparse.ArgumentParser(
        description="Compute median matches (inclusive median) for odd cycles 17..99."
    )
    parser.add_argument(
        '-f', '--file', required=True,
        help="Input CSV filename in ./historical_data/ (e.g. BTC-USD.csv)"
    )
    args = parser.parse_args(argv)

    input_path = Path('historical_data') / args.file
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(2)

    try:
        prices = load_price_array(str(input_path))
    except Exception as e:
        print(f"Error loading price series: {e}")
        sys.exit(3)

    results = compute_all_cycles(prices, CYCLES)

    out_dir = Path('research_output')
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"median_matches_{input_path.stem}.csv"

    save_results_csv(results, str(out_file))

    print(f"Results saved to: {out_file}")

if __name__ == "__main__":
    main()

