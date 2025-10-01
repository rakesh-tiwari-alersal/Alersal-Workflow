#!/usr/bin/env python3
"""
compute_DTFib.py

Detrended Fibonacci peak/valley detection.

- Two-step detrend (long lag then short lag)
- Confirmation threshold (drop_thresh; default 0.055)
- Continuation threshold cont_min (default 0.0275). This single parameter is used both
  to decide whether to ignore a tiny bump (continuation) and whether to create a nested candidate.
- Candidate expiry: a prior candidate is removed if EITHER it has exceeded max_days (peak-days)
  OR a later higher peak appears (OR condition).
- Separate look-ahead windows for peaks and valleys (defaults 10 and 30 days).
- Compatible with wrapper calling: accepts -f (filename), -l (lags), -t (tolerance) as before.
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def detect_confirmed_extrema(values,
                             drop_thresh=0.055,
                             cont_min=0.0275,
                             max_days_peak=10,
                             max_days_valley=30):
    """
    Detect confirmed peaks and valleys using:
      - drop_thresh: confirmation threshold (e.g. 0.055 for 5.5% move)
      - cont_min: continuation / nested threshold (if up_move < cont_min => ignore; if >= cont_min => candidate)
      - max_days_peak / max_days_valley: look-ahead windows for confirmation (None = unlimited)

    Removal (expiry) rule: a prior candidate is removed when EITHER
      - age > max_days_* (i.e. it has timed out), OR
      - a new higher (for peaks) / lower (for valleys) extremum appears.

    Returns:
      confirmed_peaks, confirmed_valleys -> lists of dicts {'idx': index, 'value': value}
    """
    values = np.asarray(values)
    n = len(values)
    if n < 3:
        return [], []

    # Build strict extrema
    extrema = []
    for i in range(1, n - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            extrema.append((i, 'peak', values[i]))
        elif values[i] < values[i - 1] and values[i] < values[i + 1]:
            extrema.append((i, 'valley', values[i]))

    confirmed_peaks = []
    confirmed_valleys = []
    peak_candidates = []    # [{'idx':..., 'value':...}, ...]
    valley_candidates = []

    for (i, typ, v) in extrema:
        if typ == 'peak':
            # ---- EXPIRE old peak candidates if (age>max_days_peak) OR (new peak is higher) ----
            if peak_candidates:
                kept = []
                for c in peak_candidates:
                    age = i - c['idx']
                    # Remove c if age exceeded OR new peak v is higher than candidate value
                    if ((max_days_peak is not None and age > max_days_peak) or (v > c['value'])):
                        # expire/drop c
                        continue
                    else:
                        kept.append(c)
                peak_candidates = kept

            # ---- Decide whether to append this new peak candidate or ignore as tiny bump ----
            if peak_candidates:
                prev = peak_candidates[-1]
                start = prev['idx'] + 1
                end = i
                if start < end:
                    min_between = np.min(values[start:end])
                    up_move = v - min_between
                    # up_move < cont_min -> tiny bump -> ignore (do not add)
                    # up_move >= cont_min -> add as new candidate (nested / meaningful bump)
                    if up_move < cont_min:
                        pass
                    else:
                        peak_candidates.append({'idx': i, 'value': v})
                else:
                    peak_candidates.append({'idx': i, 'value': v})
            else:
                peak_candidates.append({'idx': i, 'value': v})

            # Check valley candidates to see if they are confirmed now (valley -> rise)
            new_valley_candidates = []
            for c in valley_candidates:
                rise = v - c['value']
                age = i - c['idx']
                if rise >= drop_thresh and (max_days_valley is None or age <= max_days_valley):
                    confirmed_valleys.append(c)
                else:
                    new_valley_candidates.append(c)
            valley_candidates = new_valley_candidates

        else:  # valley
            # ---- EXPIRE old valley candidates if (age>max_days_valley) OR (new valley is lower) ----
            if valley_candidates:
                kept = []
                for c in valley_candidates:
                    age = i - c['idx']
                    if ((max_days_valley is not None and age > max_days_valley) or (v < c['value'])):
                        # expire/drop c
                        continue
                    else:
                        kept.append(c)
                valley_candidates = kept

            # ---- Decide whether to append the new valley candidate or ignore tiny dip ----
            if valley_candidates:
                prev = valley_candidates[-1]
                start = prev['idx'] + 1
                end = i
                if start < end:
                    max_between = np.max(values[start:end])
                    down_move = max_between - v
                    if down_move < cont_min:
                        pass
                    else:
                        valley_candidates.append({'idx': i, 'value': v})
                else:
                    valley_candidates.append({'idx': i, 'value': v})
            else:
                valley_candidates.append({'idx': i, 'value': v})

            # Check peak candidates to see if they are confirmed now (peak -> drop)
            new_peak_candidates = []
            for c in peak_candidates:
                drop = c['value'] - v
                age = i - c['idx']
                if drop >= drop_thresh and (max_days_peak is None or age <= max_days_peak):
                    confirmed_peaks.append(c)
                else:
                    new_peak_candidates.append(c)
            peak_candidates = new_peak_candidates

    return confirmed_peaks, confirmed_valleys


def count_hits_on_confirmed(confirmed, target, tol):
    """Count how many confirmed extrema have values within tol of target."""
    cnt = 0
    for d in confirmed:
        if abs(d['value'] - target) <= tol:
            cnt += 1
    return cnt


def find_price_column(df):
    for c in ["Close", "close", "Adj Close", "AdjClose", "Price", "price"]:
        if c in df.columns:
            return c
    return df.columns[-1]


def main():
    parser = argparse.ArgumentParser(description="Detrended Fibonacci peak/valley detection (compute_DTFib.py)")
    parser.add_argument("-f", "--file", required=True, help="CSV filename inside historical_data/")
    parser.add_argument("-l", "--lags", required=True, help="short_lag,long_lag (e.g. 27,243)")
    parser.add_argument("-t", "--tolerance", type=float, default=0.005,
                        help="Tolerance for Fibonacci level matching (default 0.005)")
    args = parser.parse_args()

    # Resolve file path relative to historical_data/
    hist_dir = "historical_data"
    file_path = args.file
    if not os.path.isabs(file_path) and not os.path.exists(file_path):
        candidate = os.path.join(hist_dir, file_path)
        if os.path.exists(candidate):
            file_path = candidate
        else:
            file_path = os.path.join(hist_dir, file_path)

    if not os.path.exists(file_path):
        print(f"Error: input file not found at {file_path}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(file_path)
    price_col = find_price_column(df)
    Y = pd.to_numeric(df[price_col], errors='coerce').copy()

    # parse lags
    try:
        short_lag, long_lag = map(int, args.lags.split(","))
    except Exception:
        print("Error parsing lags (-l). Provide as short,long (e.g. 27,243).", file=sys.stderr)
        sys.exit(1)

    # two-step detrend
    X = Y - Y.shift(long_lag)
    Ydt = (X - X.shift(short_lag)) / Y
    Ydt = Ydt.dropna()
    if Ydt.size == 0:
        print("No data after detrending; check lags and input file.", file=sys.stderr)
        sys.exit(1)

    yvals = np.asarray(Ydt.values)

    # detection parameters (these are fixed defaults; wrapper remains compatible)
    drop_thresh = 0.055    # 5.5% confirm
    cont_min = 0.0275      # 2.75% continuation/nested threshold
    max_days_peak = 10
    max_days_valley = 30
    tol = float(args.tolerance)

    confirmed_peaks, confirmed_valleys = detect_confirmed_extrema(
        yvals,
        drop_thresh=drop_thresh,
        cont_min=cont_min,
        max_days_peak=max_days_peak,
        max_days_valley=max_days_valley
    )

    # Fibonacci levels
    fibs = [0.0902, 0.1459, 0.236, 0.309, 0.382, 0.441, 0.50, 0.559, 0.618]

    total_peak_hits = 0
    total_valley_hits = 0
    total_hits = 0

    print("\nFibonacci hits (level% | peak_hits | valley_hits | total):")
    for lvl in fibs:
        p_hits = count_hits_on_confirmed(confirmed_peaks, lvl, tol)
        v_hits = count_hits_on_confirmed(confirmed_valleys, -lvl, tol)
        print(f"{lvl*100:6.2f}% | peaks: {p_hits:3d} | valleys: {v_hits:3d} | total: {p_hits+v_hits:3d}")
        total_peak_hits += p_hits
        total_valley_hits += v_hits
        total_hits += (p_hits + v_hits)

    print(f"\nTotal peak hits: {total_peak_hits}")
    print(f"Total valley hits: {total_valley_hits}")
    print(f"Total fib hits: {total_hits}")

    # Summary
    print("\nSummary:")
    print(f"Confirmed peaks:   {len(confirmed_peaks)}")
    print(f"Confirmed valleys: {len(confirmed_valleys)}")


if __name__ == "__main__":
    main()
