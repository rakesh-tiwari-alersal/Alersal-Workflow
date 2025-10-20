#!/usr/bin/env python3
"""
DTFib_Beats_Top3 (absolute-hit normalized version)

Changes from original compute_DTFibBeats.py:
- Scoring uses **Total Hits** (absolute hit count), normalized by the maximum Total Hits across candidates,
  instead of using DTFib Hit %.
- Removed all command-line weight override options. Scoring weights are controlled only by the
  module-level constants near the top of the file (easy to edit in-code).
- Cyclic-mode uses **Phase Balance** computed directly from Peak Hits and Valley Hits
  (a 0..1 balance where 1.0 means perfectly balanced and 0 means fully imbalanced).
- No mapped hit-weight (no HIT_WEIGHT_BOTTOM/TOP). Hit contribution is purely proportional
  to normalized Total Hits × GROWTH_HIT_W.

Usage examples (same as before):
  python compute_DTFibBeats.py -s GSPC,BTC-USD
  python compute_DTFibBeats.py -c CLF

Output: DTFib_Beats_Results.csv
"""
from __future__ import annotations
import csv
import os
import sys
from typing import List, Dict, Any, Optional

OUT_CSV = "DTFib_Beats_Results.csv"

# === Scoring weight configuration (edit these values for quick tweaks) ===
# Growth-mode (used for growth symbols)
GROWTH_HIT_W: float = 0.80
GROWTH_INT_W: float = 0.20

# Cyclic-mode (used for cyclic symbols like CLF)
CYCLIC_HIT_W: float = 0.80
CYCLIC_INT_W: float = 0.10
CYCLIC_PHASE_W: float = 0.10    
# ========================================================================


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
        if k in row and row.get(k) is not None and str(row.get(k)).strip() != "":
            peak = parse_int(row.get(k))
            break
    for k in possible_valley_keys:
        if k in row and row.get(k) is not None and str(row.get(k)).strip() != "":
            valley = parse_int(row.get(k))
            break
    return {"peak": peak, "valley": valley}


def merge_rows(beat_rows: List[Dict[str, str]], dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """
    Merge on 'Lag (long,short)'. Returns list of merged rows with numeric fields normalized.
    Also extracts peak/valley hit counts (if present) to compute PhaseBalance later.
    """
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

        # try to find peak/valley hits in either source row (beat or dt)
        p_b = _find_phase_keys(b)
        p_d = _find_phase_keys(d)
        peak_hits = p_b.get("peak") if p_b.get("peak") is not None else p_d.get("peak")
        valley_hits = p_b.get("valley") if p_b.get("valley") is not None else p_d.get("valley")

        merged.append({
            "Lag (long,short)": k,
            "Beat Cycle": beat_cycle if beat_cycle is not None else "",
            "Integer Count": int_count if int_count is not None else 0,
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
            "Peak Hits": peak_hits if peak_hits is not None else None,
            "Valley Hits": valley_hits if valley_hits is not None else None,
        })
    return merged


def compute_phase_balance(peak: Optional[int], valley: Optional[int]) -> float:
    """
    Compute PhaseBalance in [0,1] from peak and valley hit counts.
    Returns 1.0 when peak == valley (perfect balance), and approaches 0 when one side
    dominates. If missing, returns neutral 0.5.
    """
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


def score_and_select_top3(merged: List[Dict[str, Any]], mode: str = "growth") -> List[Dict[str, Any]]:
    """
    Compute normalized score using Total Hits and return top 3 rows sorted by score desc.
    Uses module-level weights for growth/cyclic modes.
    """
    if not merged:
        return []

    g_hit = GROWTH_HIT_W
    g_int = GROWTH_INT_W
    c_hit = CYCLIC_HIT_W
    c_int = CYCLIC_INT_W
    c_phase = CYCLIC_PHASE_W

    max_hits = max((row["Total Hits"] for row in merged), default=0)
    max_int = max((row["Integer Count"] for row in merged), default=0)

    scored: List[Dict[str, Any]] = []
    for row in merged:
        hit_cnt = int(row.get("Total Hits", 0))
        int_cnt = int(row.get("Integer Count", 0))

        # normalize using absolute hits (linear normalization)
        hit_norm = (hit_cnt / max_hits) if (max_hits and max_hits > 0) else 0.0
        int_norm = (int_cnt / max_int) if (max_int and max_int > 0) else 0.0

        if mode == "growth":
            score = g_hit * hit_norm + g_int * int_norm
        elif mode == "cyclic":
            peak = row.get("Peak Hits")
            valley = row.get("Valley Hits")
            phase_balance = compute_phase_balance(peak, valley)
            score = c_hit * hit_norm + c_int * int_norm + c_phase * phase_balance
        else:
            score = g_hit * hit_norm + g_int * int_norm

        r = dict(row)
        r["Score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["Score"], reverse=True)
    top3 = scored[:3]
    for i, r in enumerate(top3, start=1):
        r["Rank"] = i
    return top3


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Select top 3 lag pairs per symbol by combining Beats and DTFib results.")
    parser.add_argument("-s", "--symbols", type=str, required=False,
                        help="Comma-separated symbols to score with growth algorithm. Example: -s GSPC,BTC-USD")
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

        top3 = score_and_select_top3(merged, mode=mode)
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
        print(f"[OK] {sym}: selected {len(top3)} top pairs (mode={mode}).")

    if not out_rows:
        print("[ERROR] No results to write; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Symbol", "Rank", "Lag (long,short)", "Beat Cycle",
                  "Integer Count", "DTFib Hit %", "Total Hits", "Score"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
