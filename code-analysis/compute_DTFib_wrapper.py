#!/usr/bin/env python3
import argparse
import subprocess
import re
import csv
import sys
import os

def main():
    parser = argparse.ArgumentParser(
        description="Wrapper for compute_DTFib.py across short×long lag combinations"
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
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.005,
        help="Tolerance passed to compute_DTFib.py for Fibonacci matching (default 0.005)"
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
                cmd = [
                    sys.executable,
                    "compute_DTFib.py",
                    "-f", args.file,
                    "-l", lags,
                    "-t", str(args.tolerance)
                ]
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                stdout = completed.stdout

                # Parse Total fib hits: N
                match = re.search(r"Total fib hits:\s*(\d+)", stdout)
                if match:
                    results[st_lag][lt_lag] = int(match.group(1))
                else:
                    # If Total not found, set None (or optionally parse per-level lines later)
                    results[st_lag][lt_lag] = None

            except subprocess.CalledProcessError as e:
                print(f"Error with lags {lags}: {e.stderr.strip()}", file=sys.stderr)
                results[st_lag][lt_lag] = None

    # Prepare output directory
    os.makedirs("DTFib_results", exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    out_file = os.path.join("DTFib_results", f"DTFib_{base_filename}.csv")

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
