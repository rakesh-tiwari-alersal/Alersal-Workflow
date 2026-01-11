#!/usr/bin/env python3
"""
DTFib-only selection script (no R², per-symbol output)

- Accepts symbols via -s/--symbols (comma-separated).
- Reads merged DTFib results produced by compute_DTFib_N_wrapper.py.
- Produces ONE output CSV PER SYMBOL.
- Output files are written into the current working directory.
- Scoring uses HIT vs PHASE weights.
"""

from __future__ import annotations

import csv
import os
import sys
from typing import List, Dict, Any, Optional


# ==========================================================
# Configuration
# ==========================================================

TOP_N = 20

HIT_WEIGHT: float = 0.75
PHASE_WEIGHT: float = 1.0 - HIT_WEIGHT

DTFIB_ROOT = os.path.join("DTFibRuns", "DTFib_results")


# ==========================================================
# Utility functions
# ==========================================================

def read_csv_to_dict(path: str) -> List[Dict[str, str]]:
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


def _find_phase_keys(row: Dict[str, str]) -> Dict[str, Optional[int]]:
    possible_peak_keys = ["Peak Hits", "PeakHits", "Peak_Hits", "Peaks"]
    possible_valley_keys = ["Valley Hits", "ValleyHits", "Valley_Hits", "Valleys"]

    peak = None
    valley = None

    for k in possible_peak_keys:
        if k in row and row.get(k):
            peak = parse_int(row.get(k))
            break

    for k in possible_valley_keys:
        if k in row and row.get(k):
            valley = parse_int(row.get(k))
            break

    return {"peak": peak, "valley": valley}


# ==========================================================
# Core logic
# ==========================================================

def build_rows_from_dtfib(dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []

    for d in dt_rows:
        lag_key = d.get("Lag (long,short)")
        if not lag_key:
            continue

        pv = _find_phase_keys(d)

        results.append({
            "Lag (long,short)": lag_key,
            "DTFib Hit %": parse_float(d.get("Hit %")) or 0.0,
            "Total Hits": parse_int(d.get("Total Hits")) or 0,
            "Peak Hits": pv.get("peak"),
            "Valley Hits": pv.get("valley"),
            "% PVS": d.get("% PVS"),
        })

    return results


def filter_by_min_average(rows: List[Dict[str, Any]], min_average: int) -> List[Dict[str, Any]]:
    out = []
    for r in rows:
        lag = r.get("Lag (long,short)")
        if not lag or "," not in lag:
            continue
        try:
            long_part = int(lag.split(",")[0].strip())
            if long_part >= min_average:
                out.append(r)
        except Exception:
            continue
    return out


def filter_by_short_lags(rows: List[Dict[str, Any]], allowed_shorts: List[int]) -> List[Dict[str, Any]]:
    allowed = set(allowed_shorts)
    out = []

    for r in rows:
        lag = r.get("Lag (long,short)")
        if not lag or "," not in lag:
            continue
        try:
            _, short_lag = [int(x.strip()) for x in lag.split(",")]
            if short_lag in allowed:
                out.append(r)
        except Exception:
            continue

    return out


def compute_phase_balance(peak: Optional[int], valley: Optional[int]) -> float:
    if peak is None or valley is None:
        return 0.5
    if peak == 0 and valley == 0:
        return 0.5
    low = min(peak, valley)
    high = max(peak, valley)
    if high == 0:
        return 0.5
    return float(low) / float(high)


def score_and_select_topN(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not rows:
        return []

    max_hits = max((r["Total Hits"] for r in rows), default=0)
    scored: List[Dict[str, Any]] = []

    for r in rows:
        hit_norm = (r["Total Hits"] / max_hits) if max_hits > 0 else 0.0
        phase_balance = compute_phase_balance(r["Peak Hits"], r["Valley Hits"])
        score = HIT_WEIGHT * hit_norm + PHASE_WEIGHT * phase_balance

        rr = dict(r)
        rr["Score"] = score
        scored.append(rr)

    scored.sort(key=lambda x: x["Score"], reverse=True)

    topN = scored[:TOP_N]
    for i, r in enumerate(topN, start=1):
        r["Rank"] = i

    return topN


# ==========================================================
# Main
# ==========================================================

def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(
        description="Select top lag pairs per symbol using merged DTFib results only"
    )

    parser.add_argument(
        "-s", "--symbols",
        required=True,
        help="Comma-separated symbols (e.g. Uber,Roku,GSPC)"
    )

    parser.add_argument(
        "-ma", "--min-average",
        nargs="?",
        const=200,
        type=int,
        default=200,
        help="Minimum long cycle (default 200)"
    )

    parser.add_argument(
        "-v", "--volatile",
        action="store_true",
        help="Enable volatile mode: include lower short-lags (17, 20)"
    )

    args = parser.parse_args(argv)

    short_lags = (
        [17, 20, 23, 27, 31, 36, 41, 47]
        if args.volatile
        else [23, 27, 31, 36, 41, 47]
    )

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("[ERROR] No symbols provided.", file=sys.stderr)
        sys.exit(2)

    for sym in symbols:
        dt_path = os.path.join(DTFIB_ROOT, f"DTFib_{sym}_N.csv")
        dt_rows = read_csv_to_dict(dt_path)

        if not dt_rows:
            print(f"[WARN] No DTFib file found for '{sym}'. Skipping.")
            continue

        rows = build_rows_from_dtfib(dt_rows)
        rows = filter_by_short_lags(rows, short_lags)
        rows = filter_by_min_average(rows, args.min_average)

        if not rows:
            print(f"[WARN] No valid rows for '{sym}'. Skipping.")
            continue

        topN = score_and_select_topN(rows)

        out_filename = f"DTFib_Results_{sym}.csv"
        out_path = os.path.join("DTFibRuns", out_filename)

        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "Rank", "Lag (long,short)",
                    "DTFib Hit %", "Total Hits", "% PVS"
                ]
            )
            writer.writeheader()
            for r in topN:
                writer.writerow({
                    "Rank": r["Rank"],
                    "Lag (long,short)": r["Lag (long,short)"],
                    "DTFib Hit %": round(r["DTFib Hit %"], 4),
                    "Total Hits": r["Total Hits"],
                    "% PVS": r["% PVS"],
                })

        print(f"[DONE] Wrote {len(topN)} rows to {out_path}")


if __name__ == "__main__":
    main()
