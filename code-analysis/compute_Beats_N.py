#!/usr/bin/env python3
"""
compute_Beats_N.py

Usage example:
  python compute_Beats_N.py -f GSPC.csv -r 501.6,372.1,607.2 -s 41,45 -b 470

Behavior:
 - -r/--reference accepts floats with up to 2 decimal places (validated).
 - All internal calculations are done to 2-decimal precision.
 - When writing CSV, displayed values are rounded to 1 decimal using ROUND_HALF_UP.
 - Integer Count is computed after rounding each 2-decimal N to 1 decimal with ROUND_HALF_UP.
 - Long lags are selected automatically as all values within ±54 of the base (-b) cycle.
 - Results are now sorted by Integer Count (descending).
"""

import argparse
import csv
import os
import sys
from typing import List, Optional
from decimal import Decimal, ROUND_HALF_UP


# ----------------- Helpers -----------------

def _to_decimal_str(x) -> Decimal:
    """Create a Decimal from x using str(x) to avoid binary float artifacts."""
    return Decimal(str(x))


def round_half_up_decimal(x: float, ndigits: int) -> Decimal:
    """Round numeric x to ndigits decimal places using ROUND_HALF_UP and return Decimal."""
    if ndigits < 0:
        raise ValueError("ndigits must be >= 0")
    d = _to_decimal_str(x)
    quant = Decimal('1').scaleb(-ndigits)  # 10**-ndigits
    return d.quantize(quant, rounding=ROUND_HALF_UP)


def decimal_to_float_one_decimal_half_up(x: float) -> float:
    """Return a float rounded to 1 decimal using ROUND_HALF_UP (for display)."""
    return float(round_half_up_decimal(x, 1))


def decimal_str_one_decimal_label(x: float) -> str:
    """Format a reference value into a label with 1 decimal place using ROUND_HALF_UP."""
    d = round_half_up_decimal(x, 1)
    return f"{d:.1f}"


# ---------- Argument parsing & validation ----------

def parse_lag_list(s: str, max_allowed: int, name: str) -> List[int]:
    items = [it.strip() for it in s.split(",") if it.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    vals = []
    for it in items:
        try:
            v = int(it)
        except Exception:
            raise ValueError(f"{name} must contain integers, comma-separated (invalid: '{it}').")
        vals.append(v)
    return vals


def parse_reference_list(s: str, max_allowed: int, name: str) -> List[float]:
    """Parse up to max_allowed comma-separated reference values."""
    items = [it.strip() for it in s.split(",") if it.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    vals: List[float] = []
    for it in items:
        try:
            if '.' in it:
                frac = it.split('.', 1)[1]
                if len(frac) > 2:
                    raise ValueError(f"{name} values may have at most 2 decimal places (got '{it}').")
            v = float(it)
        except ValueError:
            raise ValueError(f"{name} must contain numbers (optionally with up to 2 decimals).")
        d2 = float(round_half_up_decimal(v, 2))
        vals.append(d2)
    return vals


# ---------- Core calculations ----------

def compute_beat_cycle(short: int, long: int) -> Optional[float]:
    """Compute beat cycle (period) for given short and long lags."""
    if short <= 0 or long <= 0:
        return None
    try:
        bf = abs((1.0 / float(long)) - (1.0 / float(short)))
        if bf == 0:
            return None
        cycle = 1.0 / bf
        return float(round_half_up_decimal(cycle, 2))
    except Exception:
        return None


def compute_nbeats_single(long: int, beat_cycle: Optional[float], ref: float) -> Optional[float]:
    """Compute single N-beats distance for a given reference (ref can be float)."""
    if not beat_cycle or beat_cycle == 0:
        return None
    try:
        n = abs(float(long) - float(ref)) / float(beat_cycle)
        return float(round_half_up_decimal(n, 2))
    except Exception:
        return None


# ----------------- Main -----------------

def main():
    parser = argparse.ArgumentParser(description="Compute beat cycles for lag combinations.")
    parser.add_argument("-f", "--file", required=True, help="CSV filename inside historical_data/")
    parser.add_argument("-b", "--base", type=int, required=True,
                        help="Base cycle (integer). Only cycles in range BASE±54 will be used.")
    parser.add_argument("-s", "--short_lags", required=True,
                        help="Comma-separated short lags (up to 10), e.g. 23,27,31")
    parser.add_argument("-r", "--reference", type=str, required=True,
                        help="Comma-separated reference cycle(s) (may be floats with up to 2 decimals)")
    args = parser.parse_args()

    data_path = os.path.join("historical_data", args.file)
    if not os.path.isfile(data_path):
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    # Parse inputs
    try:
        short_lags = parse_lag_list(args.short_lags, 10, "short_lags")
        reference_list = parse_reference_list(args.reference, 100, "reference")
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    # Define all long-lag cycles (same Table 2 values as DTFib wrapper)
    all_long_lags = [
        179, 183, 189, 196, 202, 206, 220, 237, 243, 250, 260, 268, 273, 291,
        308, 314, 322, 331, 345, 355, 362, 368, 385, 403, 408, 416, 426, 439,
        457, 470, 480, 487, 493, 510, 528, 534, 541, 551, 564, 582, 605, 622,
        636, 645, 653, 659, 676
    ]

    # Filter ±54 around base
    lower_bound = args.base - 54
    upper_bound = args.base + 54
    long_lags = [lag for lag in all_long_lags if lower_bound <= lag <= upper_bound]

    if not long_lags:
        print(f"No long-term cycles within ±54 of {args.base}. Exiting.")
        sys.exit(1)

    print(f"Using base cycle {args.base}. Long-term cycles in range: {long_lags}")

    results = []

    # Prepare header labels for N columns
    n_header_labels = [decimal_str_one_decimal_label(r) for r in reference_list]

    for s in short_lags:
        for l in long_lags:
            beat_cycle_2d = compute_beat_cycle(s, l)
            beat_cycle_display = "" if beat_cycle_2d is None else decimal_to_float_one_decimal_half_up(beat_cycle_2d)

            row = {
                "Lag (long,short)": f"{l},{s}",
                "Beat Cycle": beat_cycle_display
            }

            # Compute N values
            n_values_2d = [compute_nbeats_single(l, beat_cycle_2d, ref) for ref in reference_list]

            # Count integer hits
            integer_count = 0
            for n2 in n_values_2d:
                if n2 is None:
                    continue
                n_1dec = round_half_up_decimal(n2, 1)
                if n_1dec == n_1dec.to_integral():
                    integer_count += 1
            row["Integer Count"] = integer_count

            # Add columns for each reference
            for header_label, n2 in zip(n_header_labels, n_values_2d):
                col_name = f"N {header_label}"
                row[col_name] = "" if n2 is None else float(round_half_up_decimal(n2, 1))

            results.append(row)

    # === Changed section: sort results by Integer Count (descending) ===
    results.sort(key=lambda r: r.get("Integer Count", 0), reverse=True)

    os.makedirs("Beat_results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = os.path.join("Beat_results", f"Beat_{base}_N.csv")

    fieldnames = ["Lag (long,short)", "Beat Cycle", "Integer Count"] + [f"N {label}" for label in n_header_labels]

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            out_row = {k: row.get(k, "") for k in fieldnames}
            writer.writerow(out_row)

    print(f"Beat results written to: {out_path}")


if __name__ == "__main__":
    main()
