#!/usr/bin/env python3
"""
DTFib-only selection script

This version:
- Uses only DTFib results (DTFib_results/DTFib_<sym>_N.csv).
- Keeps phase-balance scoring (searches for Peak/Valley keys in DTFib CSV).
- Keeps -ma / --min-average filtering on the long lag (default 200).
- Selects the top 5 candidates by score (normalized Total Hits + phase balance).
- Adds "Beat Cycle" column immediately after "Lag (long,short)".
- Optional -p / --psd-peak <value> to compute "PSD Alignment" per row using nearest beat harmonic:
      x = round((PSD - long) / beat)
      PSD Alignment = abs(PSD - (long + x * beat))
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
    """Find peak/valley hit columns."""
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


def compute_beat_cycle(lag_str: str) -> Optional[float]:
    """Compute beat cycle given 'Lag (long,short)'."""
    if not lag_str or "," not in lag_str:
        return None
    try:
        long_lag, short_lag = [float(x.strip()) for x in lag_str.split(",")]
        if long_lag <= 0 or short_lag <= 0 or long_lag == short_lag:
            return None
        return 1.0 / abs((1.0 / long_lag) - (1.0 / short_lag))
    except Exception:
        return None


def build_rows_from_dtfib(dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Build rows entirely from DTFib CSV."""
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
        pvs_value = d.get("% PVS")
        beat_cycle_value = compute_beat_cycle(k)

        # Store both display and numeric beat cycle
        beat_cycle_display = round(beat_cycle_value, 1) if beat_cycle_value is not None else ""

        results.append({
            "Lag (long,short)": k,
            "Beat Cycle": beat_cycle_display,
            "_BeatCycleNum": beat_cycle_value,  # numeric version for calculations
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
            "Peak Hits": peak_hits if peak_hits is not None else None,
            "Valley Hits": valley_hits if valley_hits is not None else None,
            "% PVS": pvs_value
        })
    return results


def filter_by_min_average(rows: List[Dict[str, Any]], min_average: int) -> List[Dict[str, Any]]:
    """Filter rows by minimum long cycle."""
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
            continue
    return filtered


def compute_phase_balance(peak: Optional[int], valley: Optional[int]) -> float:
    """Compute [0,1] phase balance."""
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
    """Score rows and select top 5."""
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

    scored.sort(key=lambda x: x["Score"], reverse=True)

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
    parser.add_argument("-p", "--psd-peak", type=float, required=False,
                        help="Optional PSD peak value (in days). Adds 'PSD Alignment' column per row using nearest beat harmonic.")
    args = parser.parse_args(argv)

    min_average = args.min_average
    psd_peak = args.psd_peak
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

        rows = build_rows_from_dtfib(dt_rows)
        rows = filter_by_min_average(rows, min_average)
        if not rows:
            print(f"[WARN] No rows for '{sym}' after applying -ma {min_average} filter. Skipping.")
            continue

        topN, _ = score_and_select_topN(rows, mode=mode)
        for r in topN:
            psd_align_val: Optional[float] = None
            if psd_peak is not None:
                try:
                    long_part = float(r.get("Lag (long,short)").split(",")[0].strip())
                    beat_num = r.get("_BeatCycleNum")
                    if beat_num and beat_num > 0:
                        x = round((psd_peak - long_part) / beat_num)
                        psd_align_val = abs(psd_peak - (long_part + x * beat_num))
                except Exception:
                    psd_align_val = None

            row_out: Dict[str, Any] = {
                "Symbol": sym,
                "Rank": r.get("Rank"),
                "Lag (long,short)": r.get("Lag (long,short)"),
                "Beat Cycle": r.get("Beat Cycle"),
                "DTFib Hit %": round(float(r.get("DTFib Hit %", 0.0)), 4),
                "Total Hits": int(r.get("Total Hits", 0)),
                "% PVS": r.get("% PVS"),
                "Score": round(float(r.get("Score", 0.0)), 2)
            }
            if psd_peak is not None:
                row_out["PSD Alignment"] = round(psd_align_val, 1) if psd_align_val is not None else ""
            out_rows.append(row_out)

        print(f"[OK] {sym}: selected {len(topN)} top pairs (mode={mode}), min long lag {min_average} days.")

    if not out_rows:
        print("[ERROR] No results to write; exiting.", file=sys.stderr)
        sys.exit(1)

    fieldnames = ["Symbol", "Rank", "Lag (long,short)", "Beat Cycle",
                  "DTFib Hit %", "Total Hits", "% PVS", "Score"]
    if psd_peak is not None:
        fieldnames.append("PSD Alignment")

    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"[DONE] Wrote {len(out_rows)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()
