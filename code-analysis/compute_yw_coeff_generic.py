import argparse
import pandas as pd
import os
import numpy as np


def yule_walker_solver(series, order):
    """
    Solves the Yule-Walker equations to find the autoregressive coefficients.
    """
    n = len(series)
    if n <= order:
        return np.array([])

    autocov = np.zeros(order + 1)
    for lag in range(order + 1):
        autocov[lag] = np.sum(series[lag:] * series[:n - lag]) / n

    if autocov[0] == 0:
        return np.zeros(order)

    R = np.zeros((order, order))
    r = autocov[1:]

    for i in range(order):
        for j in range(order):
            R[i, j] = autocov[abs(i - j)]

    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        print("Warning: Could not solve Yule-Walker equations. Matrix is singular.")
        return np.zeros(order)

    return phi


def nearest_plastic_cycles(lag, table):
    lower = max((c for c in table if c <= lag), default=None)
    upper = min((c for c in table if c >= lag), default=None)
    return lower, upper


def main():
    TABLE_CYCLES = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    parser = argparse.ArgumentParser(
        description="Generate Yule-Walker coefficients (plastic-aligned annotation for Top-3 only)."
    )

    parser.add_argument("-f", "--file", required=True)
    parser.add_argument("-p", "--plastic_cycles", action="store_true")

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("-b", "--base", type=int)
    group.add_argument("-r", nargs=2, metavar=("BEGIN", "END"), type=int)

    parser.add_argument("-d", "--differencing_lag", type=int, default=1)

    args = parser.parse_args()

    file_path = os.path.join("historical_data", args.file)

    try:
        df = pd.read_csv(file_path)
        series = df.iloc[:, 6].dropna()
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    differenced = series.diff(periods=args.differencing_lag).dropna()

    # Select lags
    if args.base is not None:
        base = args.base
        lags_to_compute = [
            c for c in TABLE_CYCLES
            if base - 31 <= c <= base + 31
        ]
        if not lags_to_compute:
            print(f"No TABLE cycles found within ±31 of base {base}.")
            return
    else:
        begin, end = args.r
        if begin > end:
            print("Error: BEGIN must be <= END.")
            return
        lags_to_compute = list(range(begin, end + 1))

    ar_order = max(lags_to_compute)
    coeffs = yule_walker_solver(differenced, ar_order)

    if len(coeffs) == 0:
        print("Could not compute coefficients.")
        return

    results = {
        lag: coeffs[lag - 1]
        for lag in lags_to_compute
        if lag - 1 < len(coeffs)
    }

    # Apply plastic display filter ONLY if -p is supplied
    if args.plastic_cycles:
        results = {k: v for k, v in results.items() if k in TABLE_CYCLES}

    # ---- OUTPUT ----

    print("\nYule-Walker Coefficients:\n")
    for lag, coef in sorted(results.items()):
        print(f"Lag {lag}: {coef:.6f}")

    # ---- TOP 3 SIGNIFICANT ----

    if results:
        top3 = sorted(results, key=lambda k: abs(results[k]), reverse=True)[:3]

        print("\nSummary: 3 Most Significant Lags:")
        for lag in top3:
            coef = results[lag]

            # Plastic annotation ONLY when -p is NOT supplied
            if not args.plastic_cycles:
                lower, upper = nearest_plastic_cycles(lag, TABLE_CYCLES)
                if lower is not None and upper is not None:
                    print(f"Lag {lag} ({lower},{upper}) : {coef:.6f}")
                elif lower is not None:
                    print(f"Lag {lag} ({lower}) : {coef:.6f}")
                else:
                    print(f"Lag {lag}: {coef:.6f}")
            else:
                print(f"Lag {lag}: {coef:.6f}")


if __name__ == "__main__":
    main()
