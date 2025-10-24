#!/usr/bin/env python3
"""
DTFib_Beats_Top3 - adjusted output & reporting

- Integer Count is used qualitatively: keep only the upper half by Integer Count (ties included).
- Both growth and cyclic modes use two weights: HIT and PHASE.
- Removed Beat Cycle from the CSV output.
- For each symbol, prints the number of selected top pairs and the ignored Integer Count cutoff:
  [OK] GSPC: selected 3 top pairs (mode=growth), ignored Integer Count X and below.

Change added in this version:
- The row with the highest Total Hits (as seen in the merged DTFib rows) is always
  forced into the Top-3 and is placed at Rank 1 (hard rule). The other two ranks
  are selected by the existing rules unchanged.
"""
from __future__ import annotations
import csv
import os
import sys
from typing import List, Dict, Any, Optional, Tuple

OUT_CSV = "DTFib_Beats_Results.csv"

# === Scoring weights ===
GROWTH_HIT_W: float = 0.80
GROWTH_PHASE_W: float = 0.20

CYCLIC_HIT_W: float = 0.80
CYCLIC_PHASE_W: float = 0.20
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


def merge_rows(beat_rows: List[Dict[str, str]], dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Merge on 'Lag (long,short)'."""
    beat_map = {r.get("Lag (long,short)"): r for r in beat_rows if r.get("Lag (long,short)") is not None}
    dt_map = {r.get("Lag (long,short)"): r for r in dt_rows if r.get("Lag (long,short)") is not None}

    keys = sorted(set(beat_map.keys()) | set(dt_map.keys()))
    merged: List[Dict[str, Any]] = []
    for k in keys:
        b = beat_map.get(k, {})
        d = dt_map.get(k, {})
        beat_cycle = parse_float(b.get("Beat Cycle"))
        int_count = parse_int(b.get("Integer Count"))
        hit_pct = parse_float(d.get("Hit %"))
        total_hits = parse_int(d.get("Total Hits"))

        p_b = _find_phase_keys(b)
        p_d = _find_phase_keys(d)
        peak_hits = p_b.get("peak") if p_b.get("peak") is not None else p_d.get("peak")
        valley_hits = p_b.get("valley") if p_b.get("valley") is not None else p_d.get("valley")

        merged.append({
            "Lag (long,short)": k,
            "Beat Cycle": beat_cycle if beat_cycle is not None else "",  # still stored internally but not output
            "Integer Count": int_count if int_count is not None else 0,
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
            "Peak Hits": peak_hits if peak_hits is not None else None,
            "Valley Hits": valley_hits if valley_hits is not None else None,
        })
    return merged


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


def filter_by_integer_count_qualitative(merged: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """
    Keep only the upper half of rows by Integer Count (qualitative use).
    Include all rows that have the cutoff Integer Count value (i.e., ties at the cutoff are retained).
    Returns tuple (filtered_list, cutoff_value). If no filtering is applied (small n), cutoff_value is None.
    """
    n = len(merged)
    if n <= 2:
        return merged[:], None

    # sort descending by Integer Count
    sorted_merged = sorted(merged, key=lambda x: int(x.get("Integer Count", 0)), reverse=True)

    # threshold index to pick cutoff (include ties at the cutoff)
    threshold_idx = ((n + 1) // 2) - 1  # e.g., n=5 => threshold_idx=2 (0-based), keeps ceil(n/2) items or more if ties
    cutoff_value = int(sorted_merged[threshold_idx].get("Integer Count", 0))

    # include all rows with Integer Count > cutoff_value
    filtered = [r for r in sorted_merged if int(r.get("Integer Count", 0)) > cutoff_value]
    return filtered, cutoff_value


def score_and_select_top3(merged: List[Dict[str, Any]], mode: str = "growth") -> Tuple[List[Dict[str, Any]], Optional[int]]:
    """Compute normalized score using hits and phase balance. Returns (top3_list, cutoff_value).

    New behavior: The single merged row with the maximum Total Hits is always included
    and forced into Rank 1 (hard rule). The other two ranks are selected by the usual scoring.
    """
    if not merged:
        return [], None

    # find the merged row that has the highest Total Hits (tie-breaker: first encountered)
    top_hit_row_key = None
    max_hits_val = -1
    for r in merged:
        th = int(r.get("Total Hits", 0))
        if th > max_hits_val:
            max_hits_val = th
            top_hit_row_key = r.get("Lag (long,short)")

    # apply the qualitative Integer Count filter as before
    filtered, cutoff_value = filter_by_integer_count_qualitative(merged)
    if not filtered:
        # If filtering produced empty, ensure top_hit_row is included (fallback)
        # find the top row in merged and include it
        top_row = next((r for r in merged if r.get("Lag (long,short)") == top_hit_row_key), None)
        if top_row is not None:
            filtered = [top_row]
        else:
            return [], cutoff_value

    # If the top-hit row was excluded by the filter, add it back (preserve unique by Lag key)
    if top_hit_row_key is not None and not any(r.get("Lag (long,short)") == top_hit_row_key for r in filtered):
        top_row = next((r for r in merged if r.get("Lag (long,short)") == top_hit_row_key), None)
        if top_row is not None:
            filtered.append(top_row)

    if mode == "growth":
        hit_w, phase_w = GROWTH_HIT_W, GROWTH_PHASE_W
    else:
        hit_w, phase_w = CYCLIC_HIT_W, CYCLIC_PHASE_W

    max_hits = max((row["Total Hits"] for row in filtered), default=0)
    scored: List[Dict[str, Any]] = []

    for row in filtered:
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

    # Ensure top-hit row is Rank 1: if present in scored, move it to front
    if top_hit_row_key is not None:
        for i, r in enumerate(scored):
            if r.get("Lag (long,short)") == top_hit_row_key:
                top_r = scored.pop(i)
                scored.insert(0, top_r)
                break

    # pick top 3 (if fewer available, return what's there)
    top3 = scored[:3]
    for i, r in enumerate(top3, start=1):
        r["Rank"] = i
    return top3, cutoff_value


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Select top 3 lag pairs per symbol by combining Beats and DTFib results.")
    parser.add_argument("-s", "--symbols", type=str, required=False,
                        help="Comma-separated symbols for growth mode. Example: -s GSPC,BTC-USD")
    parser.add_argument("-c", "--cyclic", type=str, required=False,
                        help="Comma-separated symbols to score with cyclic algorithm. Example: -c CLF,USO")

    args = parser.parse_args(argv)

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
        beat_path = os.path.join("Beat_results", f"Beat_{sym}_N.csv")
        dt_path = os.path.join("DTFib_results", f"DTFib_{sym}_N.csv")

        beat_rows = read_csv_to_dict(beat_path)
        dt_rows = read_csv_to_dict(dt_path)

        if not beat_rows and not dt_rows:
            print(f"[WARN] No Beat or DTFib files found for '{sym}'. Skipping.")
            continue

        merged = merge_rows(beat_rows, dt_rows)
        if not merged:
            print(f"[WARN] No merged rows for '{sym}'. Skipping.")
            continue

        top3, cutoff_value = score_and_select_top3(merged, mode=mode)
        for r in top3:
            out_rows.append({
                "Symbol": sym,
                "Rank": r.get("Rank"),
                "Lag (long,short)": r.get("Lag (long,short)"),
                # "Beat Cycle" intentionally omitted from output as requested
                "Integer Count": int(r.get("Integer Count", 0)),
                "DTFib Hit %": round(float(r.get("DTFib Hit %", 0.0)), 4),
                "Total Hits": int(r.get("Total Hits", 0)),
                "Score": round(float(r.get("Score", 0.0)), 6)
            })

        # Print OK message including ignored Integer Count cutoff info
        if cutoff_value is None:
            cutoff_msg = "none"
        else:
            cutoff_msg = str(cutoff_value)
        print(f"[OK] {sym}: selected {len(top3)} top pairs (mode={mode}), ignored Integer Count {cutoff_msg} and below.")

    if not out_rows:
        print("[ERROR] No results to write; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Symbol", "Rank", "Lag (long,short)",
                  "Integer Count", "DTFib Hit %", "Total Hits", "Score"]  # Beat Cycle removed
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
