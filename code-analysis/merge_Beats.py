#!/usr/bin/env python3
"""
merge_beats_runs.py

Merge Beat_results/Beat_<SYM>_9.csv across multiple run directories.

Usage:
  python merge_beats_runs.py -d run1,run2[,run3] -s BTC-USD,GSPC -o merged_out
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

LAG_KEY = "Lag (long,short)"
INT_COL = "Integer Count"


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


def merge_for_symbol(sym: str, input_dirs: List[str], out_dir: str) -> None:
    filename = f"Beat_{sym}_9.csv"
    input_paths = [os.path.join(d, "Beat_results", filename) for d in input_dirs]

    agg: Dict[str, Dict[str, object]] = {}
    header_order: Optional[List[str]] = None

    for p in input_paths:
        rows = read_csv(p)
        if not rows:
            continue
        if header_order is None:
            header_order = list(rows[0].keys())
            # Ensure Integer Count exists in header_order
            if INT_COL not in header_order:
                # append it near the front after Beat Cycle if possible
                try:
                    idx = header_order.index("Beat Cycle")
                    header_order.insert(idx + 1, INT_COL)
                except ValueError:
                    header_order.append(INT_COL)
        for r in rows:
            key = r.get(LAG_KEY)
            if key is None or str(key).strip() == "":
                continue
            int_cnt = parse_int_safe(r.get(INT_COL))
            if key not in agg:
                # store row copy and convert Integer Count to int
                copy = dict(r)
                copy[INT_COL] = int_cnt
                agg[key] = copy
            else:
                # sum only integer count, keep other fields from first-seen
                agg[key][INT_COL] = parse_int_safe(agg[key].get(INT_COL)) + int_cnt

    if not agg:
        print(f"[WARN] No Beat files found for {sym} in provided dirs; skipping.", file=sys.stderr)
        return

    if header_order is None:
        # fallback
        header_order = [LAG_KEY, "Beat Cycle", INT_COL]

    rows_out: List[Dict[str, object]] = []
    for key, stored in agg.items():
        row = dict(stored)
        row[INT_COL] = int(parse_int_safe(row.get(INT_COL)))
        # ensure the LAG_KEY is present
        if LAG_KEY not in row or row.get(LAG_KEY) == "":
            row[LAG_KEY] = key
        rows_out.append(row)

    # sort descending by Integer Count
    rows_out.sort(key=lambda r: parse_int_safe(r.get(INT_COL)), reverse=True)

    out_path = os.path.join(out_dir, "Beat_results", filename)
    write_csv(out_path, header_order, rows_out)
    print(f"[OK] Wrote merged Beat file for {sym} -> {out_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Merge Beat CSVs from multiple run directories.")
    parser.add_argument("-d", "--dirs", required=True,
                        help="Comma-separated list of run directories (2 or 3).")
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
