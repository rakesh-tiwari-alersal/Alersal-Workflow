#!/usr/bin/env python3
"""
DTFib_Beats_Top3.py

Usage:
  python DTFib_Beats_Top3.py -s GSPC,BTC-USD,CLF,XLE

This script looks for:
  Beat_results/Beat_<SYMBOL>_9.csv
  DTFib_results/DTFib_<SYMBOL>_9.csv

It merges results on "Lag (long,short)", computes a score (60% Hit%, 40% Integer Count),
and writes the top 3 lag pairs per symbol to DTFib_Beats_Top3.csv in the current directory.
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List, Dict, Any, Optional

OUT_CSV = "DTFib_Beats_Results.csv"

def read_csv_to_dict(path: str) -> List[Dict[str, str]]:
    """Read CSV to list of dict rows. Returns [] on missing file."""
    if not os.path.isfile(path):
        return []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return [dict(row) for row in reader]

def parse_float(x: Optional[str]) -> Optional[float]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    # drop trailing percent sign if present
    if s.endswith("%"):
        s = s[:-1].strip()
    try:
        return float(s)
    except Exception:
        return None

def parse_int(x: Optional[str]) -> Optional[int]:
    if x is None:
        return None
    s = str(x).strip()
    if s == "":
        return None
    try:
        return int(float(s))
    except Exception:
        return None

def merge_rows(beat_rows: List[Dict[str, str]], dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Merge on 'Lag (long,short)'. Returns list of merged rows with numeric fields normalized.
    """
    beat_map = {r.get("Lag (long,short)"): r for r in beat_rows if r.get("Lag (long,short)") is not None}
    dt_map = {r.get("Lag (long,short)"): r for r in dt_rows if r.get("Lag (long,short)") is not None}

    keys = sorted(set(beat_map.keys()) | set(dt_map.keys()))
    merged = []
    for k in keys:
        b = beat_map.get(k, {})
        d = dt_map.get(k, {})
        beat_cycle = parse_float(b.get("Beat Cycle"))
        int_count = parse_int(b.get("Integer Count"))
        hit_pct = parse_float(d.get("Hit %"))
        total_hits = parse_int(d.get("Total Hits"))

        merged.append({
            "Lag (long,short)": k,
            "Beat Cycle": beat_cycle if beat_cycle is not None else "",
            "Integer Count": int_count if int_count is not None else 0,
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
        })
    return merged

def score_and_select_top3(merged: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Compute normalized score and return top 3 rows sorted by score desc.
    Score = 0.6 * (hit% / max_hit%) + 0.4 * (int_count / max_int_count)
    If max is zero, that component contributes 0.
    """
    if not merged:
        return []

    max_hit = max((row["DTFib Hit %"] for row in merged), default=0.0)
    max_int = max((row["Integer Count"] for row in merged), default=0)

    scored = []
    for row in merged:
        hit_norm = (row["DTFib Hit %"] / max_hit) if (max_hit and max_hit > 0) else 0.0
        int_norm = (row["Integer Count"] / max_int) if (max_int and max_int > 0) else 0.0
        score = 0.6 * hit_norm + 0.4 * int_norm
        r = dict(row)
        r["Score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["Score"], reverse=True)
    # take top 3 (or fewer if not available)
    top3 = scored[:3]
    # add Rank within selection
    for i, r in enumerate(top3, start=1):
        r["Rank"] = i
    return top3

def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Select top 3 lag pairs per symbol by combining Beats and DTFib results.")
    parser.add_argument("-s", "--symbols", type=str, required=True,
                        help="Comma-separated symbols (no spaces). Example: -s GSPC,BTC-USD,CLF")
    args = parser.parse_args(argv)

    # Accept single comma-separated string and split into list
    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("[ERROR] No symbols provided after parsing -s. Use -s GSPC,BTC-USD,...", file=sys.stderr)
        sys.exit(2)

    out_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        beat_path = os.path.join("Beat_results", f"Beat_{sym}_9.csv")
        dt_path = os.path.join("DTFib_results", f"DTFib_{sym}_9.csv")

        beat_rows = read_csv_to_dict(beat_path)
        dt_rows = read_csv_to_dict(dt_path)

        if not beat_rows and not dt_rows:
            print(f"[WARN] No Beat or DTFib files found for '{sym}' (expected {beat_path} and {dt_path}). Skipping.", file=sys.stderr)
            continue

        merged = merge_rows(beat_rows, dt_rows)
        if not merged:
            print(f"[WARN] No merged rows for '{sym}' (empty merge).", file=sys.stderr)
            continue

        top3 = score_and_select_top3(merged)
        if not top3:
            print(f"[INFO] No candidate pairs for '{sym}'.", file=sys.stderr)
            continue

        for r in top3:
            out_rows.append({
                "Symbol": sym,
                "Rank": r.get("Rank"),
                "Lag (long,short)": r.get("Lag (long,short)"),
                "Beat Cycle": r.get("Beat Cycle"),
                "Integer Count": int(r.get("Integer Count", 0)),
                "DTFib Hit %": round(float(r.get("DTFib Hit %", 0.0)), 4),
                "Total Hits": int(r.get("Total Hits", 0)),
                "Score": round(float(r.get("Score", 0.0)), 6)
            })

        print(f"[OK] {sym}: selected {len(top3)} top pairs.")

    # Write CSV
    if not out_rows:
        print("[ERROR] No results to write; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Symbol", "Rank", "Lag (long,short)", "Beat Cycle", "Integer Count",
                  "DTFib Hit %", "Total Hits", "Score"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in out_rows:
            writer.writerow(row)

    print(f"[DONE] Wrote {len(out_rows)} rows to {OUT_CSV}")

if __name__ == "__main__":
    main()

