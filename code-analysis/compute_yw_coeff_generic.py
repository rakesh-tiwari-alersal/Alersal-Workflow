#!/usr/bin/env python3

import argparse
import csv
import sys
import numpy as np
import pandas as pd
from pathlib import Path


# ---------------------------------------------------------------------
# Plastic cycle table (unchanged)
# ---------------------------------------------------------------------

TABLE_CYCLES = [
    179, 183, 189, 196, 202, 206, 220, 237,
    243, 250, 260, 268, 273, 291, 308, 314,
    322, 331, 345, 355, 362, 368, 385, 403,
    408, 416, 426, 439, 457, 470, 480, 487,
    493, 510, 528, 534, 541, 551, 564, 582,
    605, 622, 636, 645, 653, 659, 676
]

# ---------------------------------------------------------------------
# NEW: nearest plastic anchors (annotation only)
# ---------------------------------------------------------------------

def nearest_plastic_cycles(lag, table_cycles):
    lower = max((c for c in table_cycles if c <= lag), default=None)
    upper = min((c for c in table_cycles if c >= lag), default=None)
    return lower, upper


# ---------------------------------------------------------------------
# Core Yule–Walker logic (unchanged)
# ---------------------------------------------------------------------

def compute_yw_coeffs(series, max_lag):
    from statsmodels.regression.linear_model import yule_walker
    rho, _ = yule_walker(series, order=max_lag, method="mle")
    return rho


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--file", required=True, help="CSV file")
    parser.add_argument("-r", "--range", nargs=2, type=int, help="Lag range")
    parser.add_argument("-d", "--dominant", type=int, help="Dominant short cycle")
    parser.add_argument(
        "-p", "--plastic", action="store_true",
        help="Restrict display to plastic cycles only"
    )

    args = parser.parse_args()

    input_path = Path("historical_data") / args.file
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}")
        sys.exit(2)

    df = pd.read_csv(input_path)
    prices = df["close"].values

    max_lag = args.range[1] if args.range else 700
    coeffs = compute_yw_coeffs(prices, max_lag)

    # Build lag → coefficient map
    results = {
        lag + 1: coeffs[lag]
        for lag in range(len(coeffs))
        if (not args.range or args.range[0] <= lag + 1 <= args.range[1])
    }

    # Display filter (UNCHANGED behavior)
    if args.plastic:
        display_results = {
            lag: coef for lag, coef in results.items()
            if lag in TABLE_CYCLES
        }
    else:
        display_results = results

    if not display_results:
        print("No lags to display.")
        return

    # Rank by absolute coefficient (UNCHANGED)
    sorted_lags = sorted(
        display_results.keys(),
        key=lambda k: abs(display_results[k]),
        reverse=True
    )

    top_3_lags = sorted_lags[:3]

    print("\nSummary: 3 Most Significant Lags (by absolute coefficient):")
    for lag in top_3_lags:
        coef = display_results[lag]

        # NEW: annotation only
        if lag < TABLE_CYCLES[0]:
            anchor = ""
        else:
            lower, upper = nearest_plastic_cycles(lag, TABLE_CYCLES)

            if lower is not None and upper is not None and lower != upper:
                anchor = f"({lower}, {upper})"
            elif lower is not None:
                anchor = f"({lower})"
            elif upper is not None:
                anchor = f"({upper})"
            else:
                anchor = ""

        print(f"Lag {lag} {anchor}: {coef:.6f}")

if __name__ == "__main__":
    main()