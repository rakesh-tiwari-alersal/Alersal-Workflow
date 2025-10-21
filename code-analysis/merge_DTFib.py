#!/usr/bin/env python3
"""
merge_dtfib_runs.py

Merge DTFib_results/DTFib_<SYM>_N.csv across multiple run directories.

Usage:
  python merge_dtfib_runs.py -d run1,run2[,run3] -s BTC-USD,GSPC -o merged_out
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import Dict, List, Optional

LAG_KEY = "Lag (long,short)"

# Common alternative header names to be flexible
CONF_PEAK_KEYS = ["Confirmed Peaks", "ConfirmedPeaks", "Confirmed_Peaks", "Peaks"]
CONF_VAL_KEYS = ["Confirmed Valleys", "ConfirmedValleys", "Confirmed_Valleys", "Valleys"]
PEAK_HIT_KEYS = ["Peak Hits", "PeakHits", "Peak_Hits", "PeaksHits"]
VALLEY_HIT_KEYS = ["Valley Hits", "ValleyHits", "Valley_Hits", "ValleysHits"]


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
    filename = f"DTFib_{sym}_N.csv"
    input_paths = [os.path.join(d, "DTFib_results", filename) for d in input_dirs]

    # Aggregation map: lag -> aggregated counters
    agg: Dict[str, Dict[str, int]] = {}
    sample_row: Optional[Dict[str, str]] = None

    for p in input_paths:
        rows = read_csv(p)
        if not rows:
            continue
        if sample_row is None:
            sample_row = rows[0]
        for r in rows:
            key = r.get(LAG_KEY)
            if key is None or str(key).strip() == "":
                continue
            # find columns using flexible key lists
            cp = 0
            cv = 0
            ph = 0
            vh = 0
            for k in CONF_PEAK_KEYS:
                if k in r and str(r[k]).strip() != "":
                    cp = parse_int_safe(r.get(k))
                    break
            for k in CONF_VAL_KEYS:
                if k in r and str(r[k]).strip() != "":
                    cv = parse_int_safe(r.get(k))
                    break
            for k in PEAK_HIT_KEYS:
                if k in r and str(r[k]).strip() != "":
                    ph = parse_int_safe(r.get(k))
                    break
            for k in VALLEY_HIT_KEYS:
                if k in r and str(r[k]).strip() != "":
                    vh = parse_int_safe(r.get(k))
                    break

            if key not in agg:
                agg[key] = {"Confirmed Peaks": cp, "Confirmed Valleys": cv,
                            "Peak Hits": ph, "Valley Hits": vh}
            else:
                a = agg[key]
                a["Confirmed Peaks"] = a.get("Confirmed Peaks", 0) + cp
                a["Confirmed Valleys"] = a.get("Confirmed Valleys", 0) + cv
                a["Peak Hits"] = a.get("Peak Hits", 0) + ph
                a["Valley Hits"] = a.get("Valley Hits", 0) + vh

    if not agg:
        print(f"[WARN] No DTFib files found for {sym} in provided dirs; skipping.", file=sys.stderr)
        return

    # Determine output fieldnames: prefer to preserve original sample header order if available,
    # otherwise use a standard ordering.
    if sample_row is not None:
        out_fields = list(sample_row.keys())
        # ensure computed fields exist
        for req in ["Confirmed Total", "Total Hits", "Hit %", "% PVS"]:
            if req not in out_fields:
                out_fields.append(req)
    else:
        out_fields = [
            LAG_KEY, "Confirmed Peaks", "Confirmed Valleys", "Confirmed Total",
            "Peak Hits", "Valley Hits", "Total Hits", "Hit %", "% PVS"
        ]

    out_rows: List[Dict[str, object]] = []
    for key, counts in agg.items():
        cp = int(counts.get("Confirmed Peaks", 0))
        cv = int(counts.get("Confirmed Valleys", 0))
        ph = int(counts.get("Peak Hits", 0))
        vh = int(counts.get("Valley Hits", 0))

        confirmed_total = cp + cv
        total_hits = ph + vh
        hit_pct = 0.0
        if confirmed_total > 0:
            hit_pct = (total_hits / confirmed_total) * 100.0

        pvs = ""
        if vh > 0:
            try:
                pvs = f"{(ph / vh):.2f}"
            except Exception:
                pvs = ""

        row: Dict[str, object] = {}
        for fn in out_fields:
            if fn == LAG_KEY:
                row[fn] = key
            elif fn in CONF_PEAK_KEYS:
                row[fn] = cp
            elif fn in CONF_VAL_KEYS:
                row[fn] = cv
            elif fn == "Confirmed Total":
                row[fn] = confirmed_total
            elif fn in PEAK_HIT_KEYS:
                row[fn] = ph
            elif fn in VALLEY_HIT_KEYS:
                row[fn] = vh
            elif fn == "Total Hits":
                row[fn] = total_hits
            elif fn == "Hit %":
                row[fn] = f"{hit_pct:.2f}"
            elif fn == "% PVS":
                row[fn] = pvs
            else:
                # unknown/sample columns -> blank because we don't have aggregated value for those
                row[fn] = ""
        out_rows.append(row)

    # sort by Total Hits descending
    out_rows.sort(key=lambda r: parse_int_safe(r.get("Total Hits")), reverse=True)

    out_path = os.path.join(out_dir, "DTFib_results", filename)
    write_csv(out_path, out_fields, out_rows)
    print(f"[OK] Wrote merged DTFib file for {sym} -> {out_path}")


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Merge DTFib CSVs from multiple run directories.")
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