#!/usr/bin/env python3
"""
DTFib_Beats_Top3 (tweakable)

Usage examples:
  python compute_DTFibBeats_tweakable.py -s GSPC,BTC-USD
  python compute_DTFibBeats_tweakable.py -c CLF
  python compute_DTFibBeats_tweakable.py -c CLF -C --cyclic-int-weight 0.35 --cyclic-phase-weight 0.15

What changed:
- All scoring-weights live near the top as clearly labeled variables (easy for a layman to edit).
- You can optionally override weights from the command line with --growth-* and --cyclic-* flags.
- Behavior otherwise mirrors the original script: it merges Beat_results/Beat_<SYM>_9.csv and
  DTFib_results/DTFib_<SYM>_9.csv, computes scores, and writes DTFib_Beats_Results.csv.

Scoring formulas:
- Growth mode (default): Score = GROWTH_HIT_W * HitNorm + GROWTH_INT_W * IntNorm
- Cyclic mode (-c): Score = CYCLIC_HIT_W * HitNorm + CYCLIC_INT_W * IntNorm + CYCLIC_PHASE_W * PhaseBalance

PhaseBalance is calculated from Peak/Valley hit counts (if present in your CSVs). If not
present, PhaseBalance defaults to 0.5 (neutral).
"""
from __future__ import annotations
import argparse
import csv
import os
import sys
from typing import List, Dict, Any, Optional

OUT_CSV = "DTFib_Beats_Results.csv"

# === Scoring weight configuration (edit these values for quick tweaks) ===
# Growth-mode (used for growth symbols; old default behaviour)
GROWTH_HIT_W: float = 0.75
GROWTH_INT_W: float = 0.25

# Cyclic-mode (used for cyclic symbols like CLF). Recommended starting point: 0.5/0.35/0.15
CYCLIC_HIT_W: float = 0.70
CYCLIC_INT_W: float = 0.15
CYCLIC_PHASE_W: float = 0.15
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
    If either is missing or both missing, return neutral 0.5.
    Use min/max ratio; guard against zero counts.
    """
    if peak is None or valley is None:
        return 0.5  # neutral when no phase data available
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


def score_and_select_top3(merged: List[Dict[str, Any]], mode: str = "growth",
                          growth_hit_w: Optional[float] = None, growth_int_w: Optional[float] = None,
                          cyclic_hit_w: Optional[float] = None, cyclic_int_w: Optional[float] = None,
                          cyclic_phase_w: Optional[float] = None) -> List[Dict[str, Any]]:
    """
    Compute normalized score and return top 3 rows sorted by score desc.
    Parameters allow overriding the global constants (useful for CLI overrides).
    """
    if not merged:
        return []

    # use passed weights or fall back to module-level constants
    g_hit = growth_hit_w if growth_hit_w is not None else GROWTH_HIT_W
    g_int = growth_int_w if growth_int_w is not None else GROWTH_INT_W
    c_hit = cyclic_hit_w if cyclic_hit_w is not None else CYCLIC_HIT_W
    c_int = cyclic_int_w if cyclic_int_w is not None else CYCLIC_INT_W
    c_phase = cyclic_phase_w if cyclic_phase_w is not None else CYCLIC_PHASE_W

    max_hit = max((row["DTFib Hit %"] for row in merged), default=0.0)
    max_int = max((row["Integer Count"] for row in merged), default=0)

    scored: List[Dict[str, Any]] = []
    for row in merged:
        hit_pct = float(row.get("DTFib Hit %", 0.0))
        int_cnt = int(row.get("Integer Count", 0))
        hit_norm = (hit_pct / max_hit) if (max_hit and max_hit > 0) else 0.0
        int_norm = (int_cnt / max_int) if (max_int and max_int > 0) else 0.0

        if mode == "growth":
            score = g_hit * hit_norm + g_int * int_norm
        elif mode == "cyclic":
            peak = row.get("Peak Hits")
            valley = row.get("Valley Hits")
            phase_balance = compute_phase_balance(peak, valley)
            score = c_hit * hit_norm + c_int * int_norm + c_phase * phase_balance
        else:
            # fallback to growth if unknown mode
            score = g_hit * hit_norm + g_int * int_norm

        r = dict(row)
        r["Score"] = score
        scored.append(r)

    scored.sort(key=lambda x: x["Score"], reverse=True)
    top3 = scored[:3]
    for i, r in enumerate(top3, start=1):
        r["Rank"] = i
    return top3


def parse_weight_arg(val: Optional[str], default: float) -> Optional[float]:
    if val is None:
        return None
    try:
        v = float(val)
        return v
    except Exception:
        return default


def main(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="Select top 3 lag pairs per symbol by combining Beats and DTFib results.")
    parser.add_argument("-s", "--symbols", type=str, required=False,
                        help="Comma-separated symbols to score with growth algorithm (old 0.6/0.4). Example: -s GSPC,BTC-USD")
    parser.add_argument("-c", "--cyclic", type=str, required=False,
                        help="Comma-separated symbols to score with cyclic algorithm (0.5/0.25/0.25). Example: -c CLF,USO")

    # optional overrides for weights (easy CLI tweaking)
    parser.add_argument("--growth-hit-weight", type=str, required=False, help="Override growth hit weight (float).")
    parser.add_argument("--growth-int-weight", type=str, required=False, help="Override growth int weight (float).")
    parser.add_argument("--cyclic-hit-weight", type=str, required=False, help="Override cyclic hit weight (float).")
    parser.add_argument("--cyclic-int-weight", type=str, required=False, help="Override cyclic int weight (float).")
    parser.add_argument("--cyclic-phase-weight", type=str, required=False, help="Override cyclic phase weight (float).")

    args = parser.parse_args(argv)

    s_list = [s.strip() for s in args.symbols.split(",")] if args.symbols else []
    c_list = [s.strip() for s in args.cyclic.split(",")] if args.cyclic else []

    # collect all symbols; preserve explicit mode selection per symbol
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
            # cyclic takes precedence if provided in both
            symbol_modes[s] = "cyclic"

    if not symbols:
        print("[ERROR] No symbols provided. Use -s for growth symbols and/or -c for cyclic symbols.", file=sys.stderr)
        sys.exit(2)

    # parse optional overrides
    grow_hit_override = parse_weight_arg(args.growth_hit_weight, GROWTH_HIT_W)
    grow_int_override = parse_weight_arg(args.growth_int_weight, GROWTH_INT_W)
    cyclic_hit_override = parse_weight_arg(args.cyclic_hit_weight, CYCLIC_HIT_W)
    cyclic_int_override = parse_weight_arg(args.cyclic_int_weight, CYCLIC_INT_W)
    cyclic_phase_override = parse_weight_arg(args.cyclic_phase_weight, CYCLIC_PHASE_W)

    out_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        mode = symbol_modes.get(sym, "growth")
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

        top3 = score_and_select_top3(
            merged,
            mode=mode,
            growth_hit_w=grow_hit_override,
            growth_int_w=grow_int_override,
            cyclic_hit_w=cyclic_hit_override,
            cyclic_int_w=cyclic_int_override,
            cyclic_phase_w=cyclic_phase_override,
        )
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

        print(f"[OK] {sym}: selected {len(top3)} top pairs (mode={mode}).")

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
