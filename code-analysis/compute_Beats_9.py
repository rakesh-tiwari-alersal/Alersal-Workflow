#!/usr/bin/env python3
"""
compute_Beats_9.py

Usage example:
  python compute_Beats_9.py -f GSPC.csv -r 501.6,372.1,607.2 -s 41,45 -l 470,480,709

Behavior:
 - -r/--reference accepts floats with up to 2 decimal places (validated).
 - All internal calculations are done to 2-decimal precision.
 - When writing CSV, displayed values are rounded to 1 decimal using ROUND_HALF_UP.
 - Integer Count is computed after rounding each 2-decimal N to 1 decimal with ROUND_HALF_UP.
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
    """
    Round numeric x to ndigits decimal places using ROUND_HALF_UP and return Decimal.
    ndigits >= 0.
    """
    if ndigits < 0:
        raise ValueError("ndigits must be >= 0")
    d = _to_decimal_str(x)
    quant = Decimal('1').scaleb(-ndigits)  # 10**-ndigits
    return d.quantize(quant, rounding=ROUND_HALF_UP)


def decimal_to_float_one_decimal_half_up(x: float) -> float:
    """Return a float rounded to 1 decimal using ROUND_HALF_UP (for display)."""
    return float(round_half_up_decimal(x, 1))


def decimal_str_one_decimal_label(x: float) -> str:
    """
    Format a reference value into a label with 1 decimal place using ROUND_HALF_UP.
    This ensures headers like 'N 501.6' are consistent with standard rounding.
    """
    d = round_half_up_decimal(x, 1)
    # Keep one decimal in the string (e.g. '501.6', '250.0')
    # Use normalized quantize to have exactly one decimal place in string
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
    """
    Parse up to max_allowed comma-separated reference values.
    Accepts floats with up to 2 decimal places; values returned rounded to 2 decimals.
    """
    items = [it.strip() for it in s.split(",") if it.strip()]
    if not items:
        raise ValueError(f"{name} must contain at least one value.")
    if len(items) > max_allowed:
        raise ValueError(f"{name} accepts up to {max_allowed} values (got {len(items)}).")
    vals: List[float] = []
    for it in items:
        # Validate numeric and up to 2 decimal places
        try:
            # reject if more than 2 decimal digits
            if '.' in it:
                frac = it.split('.', 1)[1]
                if len(frac) > 2:
                    raise ValueError(f"{name} values may have at most 2 decimal places (got '{it}').")
            v = float(it)
        except ValueError:
            raise ValueError(f"{name} must contain numbers (optionally with up to 2 decimals).")
        # store as two-decimal rounded float (internal representation)
        d2 = float(round_half_up_decimal(v, 2))
        vals.append(d2)
    return vals


# ---------- Core calculations ----------

def compute_beat_cycle(short: int, long: int) -> Optional[float]:
    """Compute beat cycle (period) for given short and long lags.
    Result returned as float rounded to 2 decimals (internal representation).
    """
    if short <= 0 or long <= 0:
        return None
    try:
        bf = abs((1.0 / float(long)) - (1.0 / float(short)))
        if bf == 0:
            return None
        cycle = 1.0 / bf
        # return two-decimal precision (standard rounding half-up)
        return float(round_half_up_decimal(cycle, 2))
    except Exception:
        return None


def compute_nbeats_single(long: int, beat_cycle: Optional[float], ref: float) -> Optional[float]:
    """Compute single N-beats distance for a given reference (ref can be float).
    Returns value rounded to 2 decimals (internal representation).
    """
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
    parser.add_argument("-l", "--long_lags", required=True,
                        help="Comma-separated long lags (up to 10), e.g. 237,273,291,309")
    parser.add_argument("-s", "--short_lags", required=True,
                        help="Comma-separated short lags (up to 10), e.g. 23,27,31")
    parser.add_argument("-r", "--reference", type=str, required=True,
                        help="Comma-separated reference cycle(s) (may be floats with up to 2 decimals)")
    args = parser.parse_args()

    data_path = os.path.join("historical_data", args.file)
    if not os.path.isfile(data_path):
        print(f"Error: data file not found: {data_path}", file=sys.stderr)
        sys.exit(2)

    try:
        long_lags = parse_lag_list(args.long_lags, 10, "long_lags")
        short_lags = parse_lag_list(args.short_lags, 10, "short_lags")
        reference_list = parse_reference_list(args.reference, 100, "reference")
    except ValueError as e:
        print(f"Argument error: {e}", file=sys.stderr)
        sys.exit(2)

    results = []

    # Prepare header labels for N columns using ROUND_HALF_UP to one decimal for header naming
    n_header_labels = [decimal_str_one_decimal_label(r) for r in reference_list]

    for s in short_lags:
        for l in long_lags:
            # Internal 2-decimal beat cycle
            beat_cycle_2d = compute_beat_cycle(s, l)

            # display beat cycle 1-decimal (ROUND_HALF_UP)
            beat_cycle_display = "" if beat_cycle_2d is None else decimal_to_float_one_decimal_half_up(beat_cycle_2d)

            row = {
                "Lag (long,short)": f"{l},{s}",
                "Beat Cycle": beat_cycle_display
            }

            # compute all N values (internal 2-decimal)
            n_values_2d = []
            for ref in reference_list:
                n_val = compute_nbeats_single(l, beat_cycle_2d, ref)
                n_values_2d.append(n_val)

            # Count integer matches:
            # - round each internal 2-decimal N to 1 decimal using ROUND_HALF_UP
            # - if resulting 1-decimal equals an integer (i.e., .0), count it
            integer_count = 0
            for n2 in n_values_2d:
                if n2 is None:
                    continue
                n_1dec = round_half_up_decimal(n2, 1)  # Decimal
                if n_1dec == n_1dec.to_integral():
                    integer_count += 1

            row["Integer Count"] = integer_count

            # Add individual N columns (displayed as 1-decimal using ROUND_HALF_UP)
            for header_label, n2 in zip(n_header_labels, n_values_2d):
                col_name = f"N {header_label}"
                if n2 is None:
                    row[col_name] = ""
                else:
                    row[col_name] = float(round_half_up_decimal(n2, 1))

            results.append(row)

    # Sort by the first reference column (using its header label)
    primary_label = n_header_labels[0]
    primary_col = f"N {primary_label}"

    def sort_key(r):
        try:
            v = r.get(primary_col, "")
            return float(v) if v != "" else float("inf")
        except Exception:
            return float("inf")

    results.sort(key=sort_key)

    os.makedirs("Beat_results", exist_ok=True)
    base = os.path.splitext(os.path.basename(args.file))[0]
    out_path = os.path.join("Beat_results", f"Beat_{base}_9.csv")

    # Build fieldnames with Integer Count right after Beat Cycle
    fieldnames = ["Lag (long,short)", "Beat Cycle", "Integer Count"]
    for label in n_header_labels:
        fieldnames.append(f"N {label}")

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in results:
            # Ensure the writer receives basic Python types (str/float/int)
            out_row = {}
            for k in fieldnames:
                v = row.get(k, "")
                out_row[k] = v
            writer.writerow(out_row)

    print(f"Beat results written to: {out_path}")


if __name__ == "__main__":
    main()
