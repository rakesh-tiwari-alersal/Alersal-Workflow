#!/usr/bin/env python3
"""
DTFib-only selection script (updated)

Changes in this variant:
- If multiple symbols are given, a single combined CSV is produced with the symbols
  concatenated with underscores (e.g. DTFib_Results_CLF_XLE_DX-Y.NYB_GCF.csv).
- Output file is written into the current working directory (not DTFib_Results/).
- TOP_N is configurable at the top of the file.
- Invokes compute_yw_R2.py for each top candidate and appends OOS R^2 (2-decimal).
"""
from __future__ import annotations
import csv
import os
import sys
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple

# === Config ===
TOP_N = 10  # change this to 10/20 as you like
YW_R2_SCRIPT = "compute_yw_R2.py"  # expected to be in same directory and callable with sys.executable
HISTORICAL_SUFFIX = ".csv"  # historical file expected at historical_data/<SYMBOL>.csv
# ==============

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
    """Score rows and select top N (TOP_N)."""
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

    topN = scored[:TOP_N]
    for i, r in enumerate(topN, start=1):
        r["Rank"] = i

    return topN, None


def invoke_compute_yw_r2(symbol: str, lag_str: str) -> Optional[float]:
    """
    Call compute_yw_R2.py with '-f <symbol>.csv -l <lag_str>'.
    Returns parsed OOS R^2 (float) on success, or None on failure.
    """
    # Build file argument (script expects filename relative to historical_data/)
    file_arg = f"{symbol}{HISTORICAL_SUFFIX}"
    cmd = [sys.executable, YW_R2_SCRIPT, "-f", file_arg, "-l", lag_str]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = completed.stdout or ""
        # Regex same as compute_yw_R2_Grid uses
        match = re.search(r"Linear OLS \(Polynomial Degree 1\):\s*(-?\d+\.\d+)", stdout)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        else:
            # no match found
            return None
    except subprocess.CalledProcessError as e:
        # log error but continue
        print(f"[R2 ERROR] symbol={symbol} lags={lag_str} -> subprocess error: {e.stderr.strip()}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"[R2 ERROR] Could not find {YW_R2_SCRIPT} or python executable.", file=sys.stderr)
        return None


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

    # Combined output accumulator (we will write a single CSV containing all symbols)
    combined_out_rows: List[Dict[str, Any]] = []

    # Keep track of whether PSD column is used (global arg)
    psd_included = psd_peak is not None

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

            # Invoke compute_yw_R2.py for this candidate and capture OOS R^2
            lag_value_str = r.get("Lag (long,short)")
            oos_r2_val = None
            if lag_value_str:
                oos_r2_val = invoke_compute_yw_r2(sym, lag_value_str)

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
            if psd_included:
                row_out["PSD Alignment"] = round(psd_align_val, 1) if psd_align_val is not None else ""

            # Add OOS R^2 column (rounded to 2 decimals) at the end
            row_out["OOS R^2"] = f"{oos_r2_val:.2f}" if (oos_r2_val is not None) else ""

            combined_out_rows.append(row_out)

    # After processing all symbols, write a single combined CSV to the current working directory.
    if not combined_out_rows:
        print("[WARN] No output rows generated (check input files/paths).")
        return

    # Build a safe filename based on supplied symbols
    joined_symbols = "_".join([s.replace("/", "-").replace("\\", "-") for s in symbols])
    out_filename = f"DTFib_Results_{joined_symbols}.csv"

    # Fieldnames: preserve order, include PSD column only if used
    fieldnames = ["Symbol", "Rank", "Lag (long,short)", "Beat Cycle",
                  "DTFib Hit %", "Total Hits", "% PVS", "Score"]
    if psd_included:
        fieldnames.append("PSD Alignment")
    fieldnames.append("OOS R^2")

    # Write to current directory (not into a DTFib_Results folder)
    out_path = os.path.join(os.getcwd(), out_filename)
    try:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(combined_out_rows)
        print(f"[DONE] Wrote {len(combined_out_rows)} rows to {out_path}")
    except Exception as e:
        print(f"[ERROR] Could not write output file {out_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
