import argparse
import subprocess
import re
import csv
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Wrapper for compute_yw_R2.py across lag combinations"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="CSV filename inside historical_data/"
    )
    parser.add_argument(
        "-b", "--base",
        type=int,
        required=True,
        help="Base cycle (integer). Only cycles in range BASE±54 will be used."
    )
    args = parser.parse_args()

    # Short-term lags (fixed)
    short_lags = [17, 20, 23, 27, 31, 36, 41, 47, 54]

    # Long-term lags (full list from Table 2)
    all_long_lags = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    # Filter long-term lags to within BASE±54
    lower_bound = args.base - 54
    upper_bound = args.base + 54
    long_lags = [lag for lag in all_long_lags if lower_bound <= lag <= upper_bound]

    if not long_lags:
        print(f"No long-term cycles within ±54 of {args.base}. Exiting.")
        sys.exit(1)

    print(f"Using base cycle {args.base}. Long-term cycles in range: {long_lags}")

    results = {}

    for st_lag in short_lags:
        results[st_lag] = {}
        for lt_lag in long_lags:
            lags = f"{st_lag},{lt_lag}"
            try:
                # Call compute_yw_R2.py
                completed = subprocess.run(
                    [sys.executable, "compute_yw_R2.py", "-f", args.file, "-l", lags],
                    capture_output=True,
                    text=True,
                    check=True
                )
                # Extract R² using regex
                match = re.search(r"Linear OLS \(Polynomial Degree 1\):\s*(-?\d+\.\d+)", completed.stdout)
                if match:
                    results[st_lag][lt_lag] = float(match.group(1))
                else:
                    results[st_lag][lt_lag] = None
            except subprocess.CalledProcessError as e:
                print(f"Error with lags {lags}: {e.stderr.strip()}")
                results[st_lag][lt_lag] = None

    # Prepare output directory
    os.makedirs("R2_results", exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    out_file = os.path.join("R2_results", f"R2_{base_filename}.csv")

    # Write matrix to CSV
    with open(out_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Short Cycle/Long Cycle"] + long_lags
        writer.writerow(header)
        for st_lag in short_lags:
            row = [st_lag] + [results[st_lag].get(lt, None) for lt in long_lags]
            writer.writerow(row)

    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
