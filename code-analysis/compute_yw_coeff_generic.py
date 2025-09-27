import argparse
import pandas as pd
import os
import numpy as np


def yule_walker_solver(series, order):
    """
    Solves the Yule-Walker equations to find the autoregressive coefficients.
    This is a custom implementation as statsmodels is not available.
    """
    n = len(series)
    if n <= order:
        return np.array([])

    # Calculate autocovariance
    autocov = np.zeros(order + 1)
    for lag in range(order + 1):
        autocov[lag] = np.sum(series[lag:] * series[:n-lag]) / n

    if autocov[0] == 0:
        return np.zeros(order)

    # Build R matrix and r vector
    R = np.zeros((order, order))
    r = autocov[1:]

    for i in range(order):
        for j in range(order):
            R[i, j] = autocov[abs(i - j)]

    # Solve for phi (coefficients)
    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        print("Warning: Could not solve Yule-Walker equations. Matrix is singular.")
        return np.zeros(order)

    return phi


def main():
    """
    Main function to parse arguments and generate Yule-Walker coefficients.

    Behavior summary (important):
    - Exactly one of the following *must* be provided: either -r BEGIN END  OR -b/--base BASE.
      These are mutually exclusive.
    - The -b / --base argument *selects which TABLE cycles are used for calculations* (cycles within BASE ± 54).
      It does NOT affect display filtering.
    - The -p / --plastic_cycles flag is strictly a *display filter* — it controls which of the computed
      coefficients are printed/shown. It does NOT change which lags are used for the calculation.
    - The -d / --differencing_lag works as before (default 1) and is unchanged by these modes.
    """

    TABLE_CYCLES = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    parser = argparse.ArgumentParser(description="Generate Yule-Walker coefficients for a specified historical data file.")

    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=True,
        help='The input CSV file (e.g., BTC-USD.csv) to analyze from the historical_data/ folder.'
    )

    parser.add_argument(
        "-p", "--plastic_cycles", 
        action='store_true',
        help="Display filter: if set, only display coefficients for lags that are present in the Plastic Cycles table."
    )

    # Mutually-exclusive selector: either base OR range
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-b', '--base', type=int,
                       help='Base cycle (integer). When provided, the script ignores any range and uses TABLE_CYCLES within BASE±54 for calculations.')
    group.add_argument('-r', nargs=2, metavar=('BEGIN', 'END'), type=int,
                       help='Specify begin and end of the coefficient range (e.g., -r 190 250).')

    parser.add_argument('-d', '--differencing_lag', type=int, default=1, 
                        help='Lag for differencing (e.g., 1 for Yt - Yt-1). Default is 1.')

    args = parser.parse_args()

    # Construct the full file path and load the historical data
    file_path = os.path.join('historical_data', args.file)

    try:
        # Read the data from the 7th column (index 6)
        df = pd.read_csv(file_path)
        time_series = df.iloc[:, 6]
        time_series = time_series.dropna()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except IndexError:
        print(f"Error: Column 7 does not exist in the file.")
        return

    # Apply differencing
    differenced_series = time_series.diff(periods=args.differencing_lag).dropna()

    # Decide which lags to compute based on mode
    if args.base is not None:
        base = args.base
        lower = base - 54
        upper = base + 54
        # Select cycles from TABLE_CYCLES that fall in the window [base-54, base+54]
        lags_to_compute = [c for c in TABLE_CYCLES if lower <= c <= upper]
        if not lags_to_compute:
            print(f"No TABLE cycles found within ±54 of base {base}. Exiting.")
            return
        # IMPORTANT: -p does NOT change lags_to_compute; it's only a display filter
    else:
        # Range mode (-r BEGIN END)
        begin_range, end_range = args.r
        if begin_range > end_range:
            print("Error: BEGIN must be <= END.")
            return
        # Compute coefficients for every integer lag in the inclusive range
        lags_to_compute = list(range(begin_range, end_range + 1))

    # Determine the AR order to solve up to
    ar_order = max(lags_to_compute)

    # Calculate Yule-Walker coefficients for the full specified order
    ar_coeffs = yule_walker_solver(differenced_series, ar_order)

    if len(ar_coeffs) == 0:
        print("Could not compute coefficients. Check data and parameters.")
        return

    # Collect results for requested lags
    full_results = {}
    for lag in lags_to_compute:
        if lag - 1 < len(ar_coeffs):
            full_results[lag] = ar_coeffs[lag - 1]

    # Apply display filter if requested (-p). This does NOT affect which lags were computed.
    if args.plastic_cycles:
        # Show only those computed lags that are also in TABLE_CYCLES
        display_results = {lag: coef for lag, coef in full_results.items() if lag in TABLE_CYCLES}
    else:
        display_results = full_results

    # Output
    if args.base is not None:
        print(f"Yule-Walker Coefficients (calculated for TABLE cycles within ±54 of base {args.base}):")
    else:
        print(f"Yule-Walker Coefficients for Lags {begin_range} to {end_range}:")

    if args.plastic_cycles:
        print("Note: Display filtered to Plastic Cycles (requested via -p).")

    if display_results:
        for lag, coef in sorted(display_results.items()):
            print(f"Lag {lag}: {coef:.6f}")
    else:
        print("\nNo coefficients found in the specified selection (after display filter).")

    # Summary: 3 most significant lags from the displayed results
    if display_results:
        sorted_lags = sorted(display_results.keys(), key=lambda k: abs(display_results[k]), reverse=True)
        top_3_lags = sorted_lags[:3]
        print("\nSummary: 3 Most Significant Lags (by absolute coefficient):")
        for lag in top_3_lags:
            print(f"Lag {lag}: {display_results[lag]:.6f}")


if __name__ == "__main__":
    main()
