#!/usr/bin/env python3
import argparse
import subprocess
import csv
import sys
import os


def parse_output(stdout):
    """
    Parse stdout of compute_DTFib.py to extract summary metrics:
    Confirmed peaks, Confirmed valleys, Peak hits, Valley hits, Total hits
    """
    lines = stdout.splitlines()
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
        description="Wrapper for compute_DTFib.py across ALL short×long lag combinations"
    )
    parser.add_argument(
        "-f", "--file",
        type=str,
        required=True,
        help="CSV filename inside historical_data/"
    )
    parser.add_argument(
        "-t", "--tolerance",
        type=float,
        default=0.0075,
        help="Tolerance passed to compute_DTFib.py for Fibonacci matching (default 0.0075)"
    )
    args = parser.parse_args()

    # Construct full path inside historical_data/
    data_file = os.path.join("historical_data", args.file)

    # Validate input file exists
    if not os.path.isfile(data_file):
        print(f"Error: data file not found: {data_file}", file=sys.stderr)
        sys.exit(2)

    # Always use all short-term cycles (full universe)
    short_lags = [17, 20, 23, 25, 27, 31, 36, 41, 47]

    # Full long-term lags table (complete set)
    all_long_lags = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    long_lags = list(all_long_lags)

    print(f"Using all long-term cycles: {long_lags}")
    print(f"Using short cycles (full universe): {short_lags}")

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

                if valley_hits is not None and valley_hits > 0:
                    try:
                        pvs_val = peak_hits / valley_hits
                        pvs_str = f"{pvs_val:.2f}"
                    except Exception:
                        pvs_str = None
                else:
                    pvs_str = None

                results.append({
                    "Lag (long,short)": f"{lt_lag},{st_lag}",
                    "Confirmed Peaks": confirmed_peaks,
                    "Confirmed Valleys": confirmed_valleys,
                    "Confirmed Total": confirmed_total,
                    "Peak Hits": peak_hits,
                    "Valley Hits": valley_hits,
                    "Total Hits": total_hits,
                    "Hit %": f"{hit_pct:.2f}",
                    "% PVS": pvs_str
                })

            except subprocess.CalledProcessError as e:
                print(f"Error running compute_DTFib for lags {lags}:\n{e.stderr}", file=sys.stderr)
                results.append({
                    "Lag (long,short)": f"{lt_lag},{st_lag}",
                    "Confirmed Peaks": None,
                    "Confirmed Valleys": None,
                    "Confirmed Total": None,
                    "Peak Hits": None,
                    "Valley Hits": None,
                    "Total Hits": None,
                    "Hit %": None,
                    "% PVS": None
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
                    "Hit %": None,
                    "% PVS": None
                })

    # Prepare output directory
    os.makedirs("DTFib_results", exist_ok=True)
    base_filename = os.path.splitext(os.path.basename(args.file))[0]
    out_file = os.path.join("DTFib_results", f"DTFib_{base_filename}_N.csv")

    # Sort by Total Hits (numeric) descending
    def _total_hits_key(r):
        try:
            return int(r["Total Hits"]) if r.get("Total Hits") is not None else -1
        except Exception:
            return -1

    results.sort(key=_total_hits_key, reverse=True)

    # Write results to CSV
    with open(out_file, "w", newline="") as csvfile:
        fieldnames = [
            "Lag (long,short)", "Confirmed Peaks", "Confirmed Valleys",
            "Confirmed Total", "Peak Hits", "Valley Hits", "Total Hits", "Hit %", "% PVS"
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

    print(f"[DONE] Wrote {len(results)} rows to {out_file}")


if __name__ == "__main__":
    main()
