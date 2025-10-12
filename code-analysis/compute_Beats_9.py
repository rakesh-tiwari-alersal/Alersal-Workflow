#!/usr/bin/env python3
"""
compute_Beats_9.py

Standalone script to compute beat cycles for all (short, long) lag combinations.
Writes results to Beat_results/Beat_<file>_9.csv.

Example:
  ./compute_Beats_9.py -f mydata.csv -l 237,273,291 -s 23,27 -r 291
  or multiple refs:
  ./compute_Beats_9.py -f mydata.csv -l 237,273,291 -s 23,27 -r 224.0,288.0,367.0
"""

import argparse
import csv
import math
import os
import sys
from typing import List, Optional, Tuple


# ---------- Utility Functions ----------

def parse_lag_list(s: str, max_allowed: int, name: str) -> List[int]:
    items = [it.strip() for it in s.split(",") if it.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    try:
        return [int(x) for x in items]
    except ValueError:
        raise ValueError(f"{name} must contain integers, comma-separated.")


def parse_reference_list(s: str, max_allowed: int, name: str) -> List[float]:
    """
    Parse up to max_allowed comma-separated reference values.
    Accepts floats; values returned rounded to 1 decimal.
    """
    items = [it.strip() for it in s.split(",") if it.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    vals = []
    for it in items:
        try:
            v = float(it)
        except ValueError:
            raise ValueError(f"{name} must contain numbers (optionally with one decimal).")
        # store with one decimal precision
        vals.append(round(v, 1))
    return vals


def compute_beat_cycle(short: int, long: int) -> Optional[float]:
    """Compute beat cycle (period) for given short and long lags."""
    if short <= 0 or long <= 0:
        return None
    try:
        bf = abs((1.0 / float(long)) - (1.0 / float(short)))
        return None if bf == 0 else 1.0 / bf
    except Exception:
        return None


def compute_nbeats_single(long: int, beat_cycle: Optional[float], ref: float) -> Optional[float]:
    """Compute single N-beats distance for a given reference (ref can be float)."""
    if not beat_cycle or beat_cycle == 0:
        return None
    try:
        return abs(long - ref) / beat_cycle
    except Exception:
        return None


# ---------- Main ----------

def main():
    parser = argparse.ArgumentParser(description="Compute beat cycles for lag combinations.")
    parser.add_argument("-f", "--file", required=True, help="CSV filename inside historical_data/")
    parser.add_argument("-l", "--long_lags", required=True, help="Comma-separated long lags (up to 4), e.g. 237,273,291,309")
    parser.add_argument("-s", "--short_lags", required=True, help="Comma-separated short lags (up to 3), e.g. 23,27,31")
    parser.add_argument("-r", "--reference", type=str, required=True, help="Comma-separated reference cycle(s) (up to 4) from PSD analysis; may be floats and will be rounded to 1 decimal")
    args = parser.parse_args()

    data_path = os.path.join("historical_data", args.file)
    if not os.path.isfile(data_path):
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    try:
        long_lags = parse_lag_list(args.long_lags, 4, "long_lags")
        short_lags = parse_lag_list(args.short_lags, 3, "short_lags")
        # references may be floats (one-decimal precision); parse accordingly
        reference_list = parse_reference_list(args.reference, 4, "reference")
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    results = []
    for s in short_lags:
        for l in long_lags:
            beat_cycle = compute_beat_cycle(s, l)
            # store numeric rounded values (1 decimal) instead of formatted strings
            row = {
                "Lag (long,short)": f"{l},{s}",
                "Beat Cycle": round(beat_cycle, 1) if beat_cycle is not None else ""
            }
            for ref in reference_list:
                n_val = compute_nbeats_single(l, beat_cycle, ref)
                # Column name uses one-decimal formatting for the reference
                col_name = f"N {ref:.1f}"
                row[col_name] = round(n_val, 1) if n_val is not None else ""
            results.append(row)

    # Sort by N <primary reference>
    primary_ref = reference_list[0]
    ref_key = f"N {primary_ref:.1f}"
    def sort_key(r):
        try:
            v = r.get(ref_key, "")
            return float(v) if v != "" else 0.0
        except Exception:
            return float("inf")

    results.sort(key=sort_key)

    os.makedirs("Beat_results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = os.path.join("Beat_results", f"Beat_{base}_9.csv")

    fieldnames = ["Lag (long,short)", "Beat Cycle"]
    for ref in reference_list:
        fieldnames.append(f"N {ref:.1f}")

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            writer.writerow({k: row.get(k, "") for k in fieldnames})

    print(f"Beat results written to: {out_path}")


if __name__ == "__main__":
    main()


