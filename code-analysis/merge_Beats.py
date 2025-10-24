#!/usr/bin/env python3
"""
merge_Beats.py

Merge Beat_results/Beat_<SYM>_N.csv across multiple run directories.

Behavior (key points):
- Integer Count is summed exactly as before (no change to calculations or sorting).
- For non-Integer Count fields (PSD / N columns, etc.) the script collects unique values
  from all input files for the same 'Lag (long,short)'. Values are preserved in
  first-seen order and joined with " | " inside the CSV cell.
- Beat Cycle is **treated specially**: for a given lag pair it is deterministic/unique,
  so we keep the **first non-empty Beat Cycle** observed and do NOT join multiple values.
- The final header is the union of all input headers, but the script ensures
  **'Beat Cycle' appears immediately before 'Integer Count'** in the output.
- Output is standard comma-separated CSV. Empty cells are written as ,, (empty).
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

LAG_KEY = "Lag (long,short)"
INT_COL = "Integer Count"
BEAT_CYCLE_COL = "Beat Cycle"
JOIN_SEP = " | "


def read_csv(path: str) -> List[Dict[str, str]]:
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]


def parse_int_safe(x: Optional[str]) -> int:
    if x is None:
        return 0
    s = str(x).strip()
    if s == "":
        return 0
    try:
        return int(float(s))
    except Exception:
        return 0


def write_csv(path: str, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            out = {}
            for k in fieldnames:
                v = r.get(k, "")
                out[k] = "" if v is None else v
            writer.writerow(out)


def ensure_beat_and_int_order(header_order: List[str]) -> List[str]:
    """
    Ensure header_order contains BEAT_CYCLE_COL immediately before INT_COL.
    If either is missing, insert them (Beat Cycle before Integer Count).
    Preserve other columns and relative ordering as much as possible.
    """
    headers = [h for h in header_order]
    if BEAT_CYCLE_COL in headers:
        headers.remove(BEAT_CYCLE_COL)
    if INT_COL in headers:
        headers.remove(INT_COL)

    insert_idx = 0
    if LAG_KEY in headers:
        insert_idx = headers.index(LAG_KEY) + 1
    else:
        insert_idx = 0

    headers[insert_idx:insert_idx] = [BEAT_CYCLE_COL, INT_COL]
    return headers


def merge_for_symbol(sym: str, input_dirs: List[str], out_dir: str) -> None:
    filename = f"Beat_{sym}_N.csv"
    input_paths = [os.path.join(d, "Beat_results", filename) for d in input_dirs]

    agg: Dict[str, Dict[str, object]] = {}
    agg_values: Dict[str, Dict[str, List[str]]] = {}
    header_order: Optional[List[str]] = None

    for p in input_paths:
        rows = read_csv(p)
        if not rows:
            continue

        file_headers = list(rows[0].keys())
        if header_order is None:
            header_order = file_headers[:]
            if BEAT_CYCLE_COL not in header_order:
                header_order.append(BEAT_CYCLE_COL)
            if INT_COL not in header_order:
                header_order.append(INT_COL)
        else:
            for col in file_headers:
                if col not in header_order:
                    header_order.append(col)

        for r in rows:
            key = r.get(LAG_KEY)
            if key is None or str(key).strip() == "":
                continue

            int_cnt = parse_int_safe(r.get(INT_COL))

            if key not in agg:
                copy = dict(r)
                copy[INT_COL] = int_cnt
                if BEAT_CYCLE_COL not in copy:
                    copy[BEAT_CYCLE_COL] = copy.get(BEAT_CYCLE_COL, "")
                agg[key] = copy

                vals: Dict[str, List[str]] = {}
                for col, v in r.items():
                    if col in (LAG_KEY, INT_COL):
                        continue
                    s = "" if v is None else str(v).strip()
                    # For Beat Cycle: keep first non-empty value only
                    if col == BEAT_CYCLE_COL:
                        vals[col] = [s] if s != "" else []
                    else:
                        vals[col] = [s] if s != "" else []
                if BEAT_CYCLE_COL not in vals:
                    vals[BEAT_CYCLE_COL] = []
                agg_values[key] = vals

            else:
                agg[key][INT_COL] = parse_int_safe(agg[key].get(INT_COL)) + int_cnt

                for col, v in r.items():
                    if col in (LAG_KEY, INT_COL):
                        continue
                    s = "" if v is None else str(v).strip()
                    if s == "":
                        continue

                    # Special-case Beat Cycle: keep first seen value only
                    if col == BEAT_CYCLE_COL:
                        if col not in agg_values[key] or not agg_values[key][col]:
                            agg_values[key].setdefault(col, []).append(s)
                        # otherwise already have a Beat Cycle, skip
                        continue

                    # Normal collection for other columns: unique-first-seen
                    if col not in agg_values[key]:
                        agg_values[key][col] = [s]
                    else:
                        if s not in agg_values[key][col]:
                            agg_values[key][col].append(s)

    if not agg:
        print(f"[WARN] No Beat files found for {sym} in provided dirs; skipping.", file=sys.stderr)
        return

    if header_order is None:
        header_order = [LAG_KEY, BEAT_CYCLE_COL, INT_COL]

    header_order = ensure_beat_and_int_order(header_order)

    rows_out: List[Dict[str, object]] = []
    for key, stored in agg.items():
        row = dict(stored)
        row[INT_COL] = int(parse_int_safe(row.get(INT_COL)))
        if LAG_KEY not in row or row.get(LAG_KEY) == "":
            row[LAG_KEY] = key

        vals_map = agg_values.get(key, {})
        for col, seen_list in vals_map.items():
            if col in (LAG_KEY, INT_COL):
                continue
            if col == BEAT_CYCLE_COL:
                # keep first seen Beat Cycle (no join)
                row[col] = seen_list[0] if seen_list else row.get(col, "")
            else:
                if seen_list:
                    row[col] = JOIN_SEP.join(seen_list)
                else:
                    row[col] = row.get(col, "")

        if BEAT_CYCLE_COL not in row:
            row[BEAT_CYCLE_COL] = ""

        for col in header_order:
            if col not in row:
                row[col] = ""

        rows_out.append(row)

    rows_out.sort(key=lambda r: parse_int_safe(r.get(INT_COL)), reverse=True)

    out_path = os.path.join(out_dir, "Beat_results", filename)
    write_csv(out_path, header_order, rows_out)
    print(f"[OK] Wrote merged Beat file for {sym} -> {out_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Merge Beat CSVs from multiple run directories.")
    parser.add_argument("-d", "--dirs", required=True,
                        help="Comma-separated list of run directories (2 or more).")
    parser.add_argument("-s", "--symbols", required=True,
                        help="Comma-separated symbols, e.g. BTC-USD,GSPC,XLE")
    parser.add_argument("-o", "--out", required=True,
                        help="Output directory for merged results.")
    args = parser.parse_args(argv)

    input_dirs = [p.strip() for p in args.dirs.split(",") if p.strip()]
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    out_dir = args.out

    if not input_dirs:
        print("[ERROR] No input directories provided.", file=sys.stderr)
        sys.exit(2)
    if not symbols:
        print("[ERROR] No symbols provided.", file=sys.stderr)
        sys.exit(2)

    for sym in symbols:
        merge_for_symbol(sym, input_dirs, out_dir)


if __name__ == "__main__":
    main()
