#!/usr/bin/env python3
"""
compute_median_matches.py

Compute median-match counts for odd cycles 17..99 using the *direct median* method.

Now extended with:
- A 4th column: Bounce-Back

Bounce-back definition (final version):
A median match at index t qualifies if:

1) Incoming direction:
       direction = sign(price[t] - price[t-1])

2) Incoming move based on 5 bars INCLUDING today:
       win5 = prices[t-4 : t+1]   # last 5 bars: t-4...t

       If direction < 0 (approaching downward):
            incoming_move = (max(win5) - price[t]) / price[t]

       If direction > 0 (approaching upward):
            incoming_move = (price[t] - min(win5)) / price[t]

       Must satisfy:
            incoming_move >= 0.02   (2% relative to today's price)

3) Bounce-back next day:
       sign(price[t+1] - price[t]) == -direction
       (any magnitude allowed)
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

# Final constants for bounce-back logic
INCOMING_WINDOW = 5
INCOMING_THRESHOLD = 0.02


def load_price_array(input_path: str) -> np.ndarray:
    df = pd.read_csv(input_path)

    if df.shape[1] == 1:
        col = df.columns[0]
        prices = pd.to_numeric(df[col], errors="coerce").to_numpy()
        return prices

    for col in ['Close', 'close', 'PRICE', 'Price', 'price']:
        if col in df.columns:
            return pd.to_numeric(df[col], errors="coerce").to_numpy()

    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce")
        if s.notna().sum() >= max(CYCLES):
            return s.to_numpy()

    raise ValueError(
        "CSV must contain a usable price column (Close/close/PRICE/Price/price) "
        "or a single-column price file."
    )


def count_matches_inclusive_median(prices: np.ndarray, cycle: int) -> Tuple[int, int, List[int]]:
    n = len(prices)
    if n < cycle:
        return 0, 0, []

    matches = 0
    bounce_backs = 0
    idxs: List[int] = []

    k = cycle // 2  # middle index for odd-length median

    for t in range(cycle - 1, n):
        window = prices[t - cycle + 1 : t + 1]

        if np.isnan(prices[t]) or np.isnan(window).any():
            continue

        wcopy = np.array(window, copy=True)
        median_val = np.partition(wcopy, k)[k]

        # Median match
        if abs(prices[t] - median_val) <= TOL:
            matches += 1
            idxs.append(t)

            # t+1 must exist for bounce-back check
            if t + 1 >= n:
                continue

            # Determine incoming direction
            incoming_direction = np.sign(prices[t] - prices[t - 1])
            if incoming_direction == 0:
                continue

            # Need last 5 bars INCLUDING price[t]
            # That means indices: t-4, t-3, t-2, t-1, t
            if t - (INCOMING_WINDOW - 1) < 0:
                continue

            win5 = prices[t - (INCOMING_WINDOW - 1) : t + 1]
            if np.isnan(win5).any():
                continue

            # Compute incoming move magnitude (relative to today's price)
            if incoming_direction < 0:  # came downward
                high5 = np.max(win5)
                incoming_move = (high5 - prices[t]) / prices[t]
            else:  # incoming_direction > 0, came upward
                low5 = np.min(win5)
                incoming_move = (prices[t] - low5) / prices[t]

            # Must meet 2% threshold
            if incoming_move < INCOMING_THRESHOLD:
                continue

            # Bounce-back = next day reversal
            bounce_direction = np.sign(prices[t + 1] - prices[t])
            if bounce_direction == -incoming_direction:
                bounce_backs += 1

    return matches, bounce_backs, idxs


def compute_all_cycles(prices: np.ndarray, cycles: List[int]) -> List[Tuple[int, int, int]]:
    results = []
    for n in cycles:
        cnt, bnc, _ = count_matches_inclusive_median(prices, n)
        results.append((n, cnt, bnc))
    results.sort(key=lambda x: (-x[1], x[0]))
    return results


def save_results_csv(results: List[Tuple[int, int, int]], out_path: str) -> None:
    df = pd.DataFrame(results, columns=["Cycle Length", "Median Matches", "Bounce-Back"])
    prev = df["Median Matches"].shift(1)
    with np.errstate(divide='ignore', invalid='ignore'):
        pct_drop = np.where(prev > 0, (prev - df["Median Matches"]) / prev * 100.0, np.nan)
    df["% Drop"] = pd.Series(pct_drop).round().astype("Int64")

    # --------------------------------------------------------
    # NEW 5th COLUMN: %Bounce = BounceBack / Matches * 100
    # --------------------------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        df["%Bounce"] = (df["Bounce-Back"] / df["Median Matches"]) * 100.0  # ← added
    df["%Bounce"] = df["%Bounce"].round().astype(int)  # ← added
    # --------------------------------------------------------

    # Reorder columns (now including the new one)
    df = df[[
        "Cycle Length",
        "Median Matches",
        "% Drop",
        "Bounce-Back",
        "%Bounce"                  # ← added
    ]]

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
