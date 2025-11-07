#!/usr/bin/env python3
"""
DTFib-only selection script (no growth/cyclic modes)

- Accepts symbols via -s/--symbols (comma-separated).
- Produces a single combined CSV for all provided symbols.
- Output file is written into the current working directory (not DTFib_Results/).
- TOP_N is configurable at the top of the file.
- Invokes compute_yw_R2.py for each top candidate and appends OOS R^2 (2-decimal).
- Scoring uses HIT vs PHASE weights: HIT_WEIGHT and (1 - HIT_WEIGHT), set at top.
"""
from __future__ import annotations
import csv
import os
import sys
import subprocess
import re
from typing import List, Dict, Any, Optional, Tuple

# === Config ===
TOP_N = 10
YW_R2_SCRIPT = "compute_yw_R2.py"
HISTORICAL_SUFFIX = ".csv"

HIT_WEIGHT: float = 0.75
PHASE_WEIGHT: float = 1.0 - HIT_WEIGHT
# ==============


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


def build_rows_from_dtfib(dt_rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
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

        results.append({
            "Lag (long,short)": k,
            "DTFib Hit %": hit_pct if hit_pct is not None else 0.0,
            "Total Hits": total_hits if total_hits is not None else 0,
            "Peak Hits": peak_hits if peak_hits is not None else None,
            "Valley Hits": valley_hits if valley_hits is not None else None,
            "% PVS": pvs_value
        })
    return results


def filter_by_min_average(rows: List[Dict[str, Any]], min_average: int) -> List[Dict[str, Any]]:
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


def filter_by_short_lags(rows: List[Dict[str, Any]], allowed_shorts: List[int]) -> List[Dict[str, Any]]:
    out = []
    allowed = set(int(x) for x in allowed_shorts)
    for row in rows:
        lag_str = row.get("Lag (long,short)")
        if not lag_str or "," not in lag_str:
            continue
        try:
            parts = [int(x.strip()) for x in lag_str.split(",")]
            if len(parts) != 2:
                continue
            _, short_lag = parts[0], parts[1]
            if short_lag in allowed:
                out.append(row)
        except Exception:
            continue
    return out


def compute_phase_balance(peak: Optional[int], valley: Optional[int]) -> float:
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


def score_and_select_topN(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], None]:
    if not rows:
        return [], None

    hit_w, phase_w = HIT_WEIGHT, PHASE_WEIGHT
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
    file_arg = f"{symbol}{HISTORICAL_SUFFIX}"
    cmd = [sys.executable, YW_R2_SCRIPT, "-f", file_arg, "-l", lag_str]
    try:
        completed = subprocess.run(cmd, capture_output=True, text=True, check=True)
        stdout = completed.stdout or ""
        match = re.search(r"Linear OLS \(Polynomial Degree 1\):\s*(-?\d+\.\d+)", stdout)
        if match:
            try:
                return float(match.group(1))
            except Exception:
                return None
        return None
    except subprocess.CalledProcessError as e:
        print(f"[R2 ERROR] symbol={symbol} lags={lag_str} -> subprocess error: {e.stderr.strip()}", file=sys.stderr)
        return None
    except FileNotFoundError:
        print(f"[R2 ERROR] Could not find {YW_R2_SCRIPT} or python executable.", file=sys.stderr)
        return None


def main(argv: Optional[List[str]] = None):
    import argparse

    parser = argparse.ArgumentParser(description="Select top lag pairs per symbol using DTFib results only")
    parser.add_argument("-s", "--symbols", type=str, required=True,
                        help="Comma-separated symbols. Example: -s GSPC,BTC-USD")
    parser.add_argument("-ma", "--min-average", nargs="?", const=200, type=int, default=200,
                        help="Minimum long cycle (default 200)")

    # ✅ Simplified volatility switch
    parser.add_argument("-v", "--volatile", action="store_true",
                        help="Enable volatile mode: include lower short-lags (17, 20)")

    args = parser.parse_args(argv)

    # ✅ Default short lags, expand if -vl passed
    if args.volatile:
        short_lags = [17, 20, 23, 25, 27, 31, 36, 41, 47]
    else:
        short_lags = [23, 25, 27, 31, 36, 41, 47]

    min_average = args.min_average

    symbols = [s.strip() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        print("[ERROR] No symbols provided after parsing -s.", file=sys.stderr)
        sys.exit(2)

    combined_out_rows: List[Dict[str, Any]] = []

    for sym in symbols:
        dt_path = os.path.join("DTFib_results", f"DTFib_{sym}_N.csv")
        dt_rows = read_csv_to_dict(dt_path)
        if not dt_rows:
            print(f"[WARN] No DTFib file found for '{sym}'. Skipping.")
            continue

        rows = build_rows_from_dtfib(dt_rows)
        rows = filter_by_short_lags(rows, short_lags)
        rows = filter_by_min_average(rows, min_average)
        if not rows:
            print(f"[WARN] No rows for '{sym}' after -ma {min_average}. Skipping.")
            continue

        topN, _ = score_and_select_topN(rows)

        for r in topN:
            lag_value_str = r.get("Lag (long,short)")
            oos_r2_val = None
            if lag_value_str:
                oos_r2_val = invoke_compute_yw_r2(sym, lag_value_str)

            row_out: Dict[str, Any] = {
                "Symbol": sym,
                "Rank": r.get("Rank"),
                "Lag (long,short)": r.get("Lag (long,short)"),
                "DTFib Hit %": round(float(r.get("DTFib Hit %", 0.0)), 4),
                "Total Hits": int(r.get("Total Hits", 0)),
                "% PVS": r.get("% PVS"),
                "Score": round(float(r.get("Score", 0.0)), 2),
                "OOS R^2": f"{oos_r2_val:.2f}" if (oos_r2_val is not None) else ""
            }

            combined_out_rows.append(row_out)

    if not combined_out_rows:
        print("[WARN] No output rows generated.")
        return

    joined_symbols = "_".join([s.replace("/", "-").replace("\\", "-") for s in symbols])
    out_filename = f"DTFib_Results_{joined_symbols}.csv"

    fieldnames = [
        "Symbol", "Rank", "Lag (long,short)",
        "DTFib Hit %", "Total Hits", "% PVS", "Score", "OOS R^2"
    ]

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
