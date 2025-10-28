#!/usr/bin/env python3
"""
DTFib-only selection script

This version:
- Uses only DTFib results (DTFib_results/DTFib_<sym>_N.csv).
- Removes Integer Count and Beat Cycle (those fields are not used or expected).
- Keeps phase-balance scoring (searches for Peak/Valley keys in DTFib CSV).
- Keeps -ma / --min-average filtering on the long lag (default 200).
- Selects the top 5 candidates by score (normalized Total Hits + phase balance).
"""
from __future__ import annotations
import csv
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

OUT_CSV = "DTFib_Results_Top5.csv"

# === Scoring weights ===
GROWTH_HIT_W: float = 0.80
GROWTH_PHASE_W: float = 0.20

CYCLIC_HIT_W: float = 0.75
CYCLIC_PHASE_W: float = 0.25
# ========================


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
    """
    Look for common variants of peak/valley hit column names and return parsed ints
    as {'peak': int or None, 'valley': int or None}
    """
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


def build_rows_from_dtfib(dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Construct rows entirely from DTFib CSV rows.
    Expects 'Lag (long,short)' and DTFib fields such as 'Total Hits', 'Hit %', and possible peak/valley columns.
    """
    results: List[Dict[str, Any]] = []
    for d in dt_rows:
        k = d.get("Lag (long,short)")
        if not k:
            continue
        hit_pct = parse_float(d.get("Hit %"))
        total_hits = parse_int(d.get("Total Hits"))

        p_d = _find_phase_keys(d)
        peak_hits = p_d.get("peak")
        valley_hits = p_d.get("valley")

        # Preserve the '% PVS' column from source (no transformation)
        pvs_value = d.get("% PVS")

        results.append({
            "Lag (long,short)": k,
            # No Beat Cycle or Integer Count fields expected or used
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
            "Peak Hits": peak_hits if peak_hits is not None else None,
            "Valley Hits": valley_hits if valley_hits is not None else None,
            "% PVS": pvs_value
        })
    return results


def filter_by_min_average(rows: List[Dict[str, Any]], min_average: int) -> List[Dict[str, Any]]:
    """
    Filters rows by minimum 'long' cycle length in 'Lag (long,short)'.
    Keeps only rows where long >= min_average.
    """
    filtered = []
    for row in rows:
        lag_str = row.get("Lag (long,short)")
        if not lag_str or "," not in lag_str:
            continue
        try:
            long_part = int(lag_str.split(",")[0].strip())
            if long_part >= min_average:
                filtered.append(row)
        except Exception:
            # if parsing fails, skip the row
            continue
    return filtered


def compute_phase_balance(peak: Optional[int], valley: Optional[int]) -> float:
    """Compute [0,1] phase balance: 1.0 = perfect balance."""
    if peak is None or valley is None:
        return 0.5
    if peak == 0 and valley == 0:
        return 0.5
    try:
        low = min(peak, valley)
        high = max(peak, valley)
        if high == 0:
            return 0.5
        return float(low) / float(high)
    except Exception:
        return 0.5


def score_and_select_topN(rows: List[Dict[str, Any]], mode: str = "growth") -> Tuple[List[Dict[str, Any]], None]:
    """
    Compute normalized score using hits and phase balance.
    Returns (topN_list, None).

    - Uses all rows provided (no Integer Count filtering).
    - Returns top 5 rows by score (or fewer if not enough rows).
    """
    if not rows:
        return [], None

    if mode == "growth":
        hit_w, phase_w = GROWTH_HIT_W, GROWTH_PHASE_W
    else:
        hit_w, phase_w = CYCLIC_HIT_W, CYCLIC_PHASE_W

    max_hits = max((int(row.get("Total Hits", 0)) for row in rows), default=0)
    scored: List[Dict[str, Any]] = []

    for row in rows:
        hit_cnt = int(row.get("Total Hits", 0))
        peak = row.get("Peak Hits")
        valley = row.get("Valley Hits")
        hit_norm = (hit_cnt / max_hits) if max_hits > 0 else 0.0
        phase_balance = compute_phase_balance(peak, valley)
        score = hit_w * hit_norm + phase_w * phase_balance

        r = dict(row)
        r["Score"] = score
        scored.append(r)

    # sort by score descending
    scored.sort(key=lambda x: x["Score"], reverse=True)

    # pick top 5 (if fewer available, return what's there)
    topN = scored[:5]
    for i, r in enumerate(topN, start=1):
        r["Rank"] = i

    return topN, None


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Select top lag pairs per symbol using DTFib results only.")
    parser.add_argument("-s", "--symbols", type=str, required=False,
                        help="Comma-separated symbols for growth mode. Example: -s GSPC,BTC-USD")
    parser.add_argument("-c", "--cyclic", type=str, required=False,
                        help="Comma-separated symbols to score with cyclic algorithm. Example: -c CLF,USO")
    parser.add_argument("-ma", "--min-average", nargs="?", const=200, type=int, default=200,
                        help="Minimum long-term cycle (default 200 if omitted or no value supplied)")

    args = parser.parse_args(argv)

    min_average = args.min_average
    s_list = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
    c_list = [s.strip() for s in args.cyclic.split(",")] if args.cyclic else []

    symbols: List[str] = []
    symbol_modes: Dict[str, str] = {}
    for s in s_list:
        if s:
            symbols.append(s)
            symbol_modes[s] = "growth"
    for s in c_list:
        if s:
            if s not in symbols:
                symbols.append(s)
            symbol_modes[s] = "cyclic"

    if not symbols:
        print("[ERROR] No symbols provided.", file=sys.stderr)
        sys.exit(2)

    out_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        mode = symbol_modes.get(sym, "growth")
        dt_path = os.path.join("DTFib_results", f"DTFib_{sym}_N.csv")

        dt_rows = read_csv_to_dict(dt_path)

        if not dt_rows:
            print(f"[WARN] No DTFib file found for '{sym}'. Skipping.")
            continue

        # Build rows from DTFib only
        rows = build_rows_from_dtfib(dt_rows)

        # Apply minimum-average filter on long lag
        rows = filter_by_min_average(rows, min_average)

        if not rows:
            print(f"[WARN] No rows for '{sym}' after applying -ma {min_average} filter. Skipping.")
            continue

        topN, _ = score_and_select_topN(rows, mode=mode)

        for r in topN:
            out_rows.append({
                "Symbol": sym,
                "Rank": r.get("Rank"),
                "Lag (long,short)": r.get("Lag (long,short)"),
                "DTFib Hit %": round(float(r.get("DTFib Hit %", 0.0)), 4),
                "Total Hits": int(r.get("Total Hits", 0)),
                "% PVS": r.get("% PVS"),
                "Score": round(float(r.get("Score", 0.0)), 6)
            })

        print(f"[OK] {sym}: selected {len(topN)} top pairs (mode={mode}), min long lag {min_average} days.")

    if not out_rows:
        print("[ERROR] No results to write; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Symbol", "Rank", "Lag (long,short)", "DTFib Hit %", "Total Hits", "% PVS", "Score"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
