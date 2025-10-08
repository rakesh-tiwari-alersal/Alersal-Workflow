#!/usr/bin/env python3
import argparse
import subprocess
import re
import csv
import sys
import os

def parse_output(stdout):
    """
    Parse stdout of compute_DTFib.py to extract summary metrics:
    Confirmed peaks, Confirmed valleys, Peak hits, Valley hits, Total hits
    """
    lines = stdout.splitlines()
    # Defensive parsing: look for lines and raise if missing
    try:
        confirmed_peaks = int([l for l in lines if "Confirmed peaks" in l][0].split()[-1])
        confirmed_valleys = int([l for l in lines if "Confirmed valleys" in l][0].split()[-1])
        total_hits = int([l for l in lines if "Total fib hits" in l][0].split()[-1])
        peak_hits = int([l for l in lines if "Total peak hits" in l][0].split()[-1])
        valley_hits = int([l for l in lines if "Total valley hits" in l][0].split()[-1])
    except Exception as e:
        raise RuntimeError(f"Failed to parse compute_DTFib output: {e}\nSTDOUT:\n{stdout[:2000]}") from e

    return confirmed_peaks, confirmed_valleys, peak_hits, valley_hits, total_hits

def parse_lag_list(s, max_allowed, name):
    """
    Parse a comma-separated list of integers, strip whitespace, validate count.
    """
    items = [x.strip() for x in s.split(",") if x.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one lag value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    try:
        ints = [int(x) for x in items]
    except ValueError:
        raise ValueError(f"{name} must be integers, comma-separated.")
    return ints

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
        "-l", "--long_lags",
        type=str,
        required=True,
        help="Comma-separated list of long lags (up to 4), e.g. 237,273,291,309"
    )
    parser.add_argument(
        "-s", "--short_lags",
        type=str,
        required=True,
        help="Comma-separated list of short lags (up to 3), e.g. 23,27,31"
    )
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.075,
        help="Tolerance passed to compute_DTFib.py for Fibonacci matching (default 0.005)"
    )
    args = parser.parse_args()

    # Construct full path inside historical_data/
    data_file = os.path.join("historical_data", args.file)

    # Validate input file exists
    if not os.path.isfile(data_file):
        print(f"Error: data file not found: {data_file}", file=sys.stderr)
        sys.exit(2)

    # Parse lag lists (allow up to 4 long lags)
    try:
        long_lags = parse_lag_list(args.long_lags, max_allowed=4, name="long_lags")
        short_lags = parse_lag_list(args.short_lags, max_allowed=3, name="short_lags")
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    results = []

    for st_lag in short_lags:
        for lt_lag in long_lags:
            lags = f"{st_lag},{lt_lag}"
            try:
                cmd = [
                    sys.executable,
                    "compute_DTFib.py",
                    "-f", data_file,
                    "-l", lags,
                    "-t", str(args.tolerance)
                ]
                completed = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )

                confirmed_peaks, confirmed_valleys, peak_hits, valley_hits, total_hits = parse_output(completed.stdout)

                confirmed_total = confirmed_peaks + confirmed_valleys
                hit_pct = (total_hits / confirmed_total * 100.0) if confirmed_total > 0 else 0.0

                results.append({
                    "Lag (long,short)": f"{lt_lag},{st_lag}",
                    "Confirmed Peaks": confirmed_peaks,
                    "Confirmed Valleys": confirmed_valleys,
                    "Confirmed Total": confirmed_total,
                    "Peak Hits": peak_hits,
                    "Valley Hits": valley_hits,
                    "Total Hits": total_hits,
                    "Hit %": f"{hit_pct:.2f}"
                })

            except subprocess.CalledProcessError as e:
                # Print stderr from compute_DTFib for debugging and continue
                print(f"Error running compute_DTFib for lags {lags}:\n{e.stderr}", file=sys.stderr)
                # Append a row with None values so matrix remains complete
                results.append({
                    "Lag (long,short)": f"{lt_lag},{st_lag}",
                    "Confirmed Peaks": None,
                    "Confirmed Valleys": None,
                    "Confirmed Total": None,
                    "Peak Hits": None,
                    "Valley Hits": None,
                    "Total Hits": None,
                    "Hit %": None
                })
            except Exception as e:
                print(f"Unexpected error for lags {lags}: {e}", file=sys.stderr)
                results.append({
                    "Lag (long,short)": f"{lt_lag},{st_lag}",
                    "Confirmed Peaks": None,
                    "Confirmed Valleys": None,
                    "Confirmed Total": None,
                    "Peak Hits": None,
                    "Valley Hits": None,
                    "Total Hits": None,
                    "Hit %": None
                })

    # Prepare output directory
    os.makedirs("DTFib_results", exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    # NOTE: changed suffix to _9 as requested
    out_file = os.path.join("DTFib_results", f"DTFib_{base_filename}_9.csv")

    # Sort by Hit % (numeric) descending, then write results to CSV
    def _hit_key(r):
        try:
            return float(r["Hit %"]) if r.get("Hit %") is not None else -1.0
        except Exception:
            return -1.0

    results.sort(key=_hit_key, reverse=True)

    # Write results to CSV
    with open(out_file, "w", newline="") as csvfile:
        fieldnames = ["Lag (long,short)", "Confirmed Peaks", "Confirmed Valleys",
                      "Confirmed Total", "Peak Hits", "Valley Hits", "Total Hits", "Hit %"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"Results written to {out_file}")

if __name__ == "__main__":
    main()
