#!/usr/bin/env python3
"""compute_DTFib.py

Detrended Fibonacci peak/valley detection.

- Two-step detrend (long lag then short lag)
- Confirmation threshold (drop_thresh; default 0.05)
- Continuation threshold cont_min (default 0.025). This parameter is used to
  ignore tiny bumps during a move (not to block legitimate replacements).
- Candidate expiry: a prior candidate is removed if EITHER it has exceeded max_days
  OR a later higher (for peaks) / lower (for valleys) extremum replaces it.
- Separate look-ahead windows for peaks and valleys (defaults 10 and 30 days).
- Compatible with wrapper calling: accepts -f (filename), -l (lags), -t (tolerance).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd


def find_price_column(df):
    """Pick a sensible price column name from the dataframe."""
    candidates = ['Close', 'close', 'Adj Close', 'adj_close', 'Adj_Close', 'Price', 'price', 'close_price']
    for c in candidates:
        if c in df.columns:
            return c
    # fallback to last numeric column
    for c in reversed(df.columns):
        if pd.api.types.is_numeric_dtype(df[c]):
            return c
    return df.columns[-1]


def detect_confirmed_extrema(values,
                             drop_thresh=0.05,
                             cont_min=0.025,
                             max_days_peak=10,
                             max_days_valley=30):
    """Detect confirmed peaks and valleys using candidate lifecycle rules.

    Revision: replacement behavior changed so that **a new higher peak always replaces
    the prior peak candidate**, and **a new lower valley always replaces the prior valley candidate**,
    regardless of cont_min. The cont_min only gates appending of nested candidates (tiny bumps/dips
    are ignored and not appended).
    """
    confirmed_peaks = []
    confirmed_valleys = []
    peak_candidates = []
    valley_candidates = []
    strict_extrema = []
    n = len(values)
    if n < 3:
        return [], []

    # 1) find strict extrema (simple 3-point test)
    for i in range(1, n-1):
        if values[i] > values[i-1] and values[i] > values[i+1]:
            strict_extrema.append((i, 'peak', values[i]))
        elif values[i] < values[i-1] and values[i] < values[i+1]:
            strict_extrema.append((i, 'valley', values[i]))

    # 2) walk strict extrema and manage candidates
    for idx, typ, v in strict_extrema:
        if typ == 'peak':
            # Confirm valley candidates when a peak arrives
            for c in valley_candidates[:]:
                drop = v - c['value']
                if drop >= drop_thresh and (max_days_valley is None or (idx - c['idx'] <= max_days_valley)):
                    confirmed_valleys.append(c)
                    valley_candidates.remove(c)

            # Peak candidate lifecycle (REVISED):
            # - If no prior peak candidate -> append
            # - Else: always replace last if this peak is higher than last['value']
            #         otherwise append only if up_move >= cont_min (prevent tiny bumps)
            if not peak_candidates:
                peak_candidates.append({'idx': idx, 'value': v})
            else:
                last = peak_candidates[-1]
                if last['idx'] + 1 < idx:
                    min_between = np.min(values[last['idx']+1:idx])
                else:
                    min_between = values[idx]
                up_move = v - min_between
                # Replacement happens regardless of cont_min
                if v > last['value']:
                    peak_candidates[-1] = {'idx': idx, 'value': v}
                else:
                    # Only append if the up_move is meaningful
                    if up_move < cont_min:
                        # tiny bump -> ignore
                        pass
                    else:
                        peak_candidates.append({'idx': idx, 'value': v})

            # Expire old peak candidates by age
            if peak_candidates:
                kept = []
                for c in peak_candidates:
                    if (max_days_peak is not None) and ((idx - c['idx']) > max_days_peak):
                        continue
                    kept.append(c)
                peak_candidates = kept

        elif typ == 'valley':
            # Confirm peak candidates when a valley arrives
            for c in peak_candidates[:]:
                drop = c['value'] - v
                if drop >= drop_thresh and (max_days_peak is None or (idx - c['idx'] <= max_days_peak)):
                    confirmed_peaks.append(c)
                    peak_candidates.remove(c)

            # Valley candidate lifecycle (REVISED):
            # - If no prior valley candidate -> append
            # - Else: always replace last if this valley is deeper than last['value']
            #         otherwise append only if down_move >= cont_min (prevent tiny dips)
            if not valley_candidates:
                valley_candidates.append({'idx': idx, 'value': v})
            else:
                last = valley_candidates[-1]
                if last['idx'] + 1 < idx:
                    max_between = np.max(values[last['idx']+1:idx])
                else:
                    max_between = values[idx]
                down_move = max_between - v
                # Replacement happens regardless of cont_min
                if v < last['value']:
                    valley_candidates[-1] = {'idx': idx, 'value': v}
                else:
                    # Only append if the down_move is meaningful
                    if down_move < cont_min:
                        # tiny dip -> ignore
                        pass
                    else:
                        valley_candidates.append({'idx': idx, 'value': v})

            # Expire old valley candidates by age
            if valley_candidates:
                kept = []
                for c in valley_candidates:
                    if (max_days_valley is not None) and ((idx - c['idx']) > max_days_valley):
                        continue
                    kept.append(c)
                valley_candidates = kept

    return confirmed_peaks, confirmed_valleys


def count_hits_on_confirmed(confirmed, target, tol):
    """Count how many confirmed extrema have values within tol of target."""
    cnt = 0
    for d in confirmed:
        if abs(d['value'] - target) <= tol:
            cnt += 1
    return cnt


def main():
    parser = argparse.ArgumentParser(description='Compute detrended fib hits on timeseries.')
    parser.add_argument('-f', '--file', required=True, help='CSV file with time series')
    parser.add_argument('-l', '--lags', required=True, help='short_lag,long_lag (e.g. 27,243)')
    parser.add_argument('-t', '--tolerance', type=float, default=0.0075, help='tolerance for fib matching')
    args = parser.parse_args()

    fname = args.file
    if not os.path.exists(fname):
        # try historical_data/ fallback
        alt = os.path.join('historical_data', fname)
        if os.path.exists(alt):
            fname = alt
        else:
            print(f"File not found: {args.file}")
            sys.exit(1)

    lparts = args.lags.split(',')
    if len(lparts) != 2:
        print('Provide two comma-separated lags: short,long')
        sys.exit(1)
    short_lag = int(lparts[0])
    long_lag = int(lparts[1])

    df = pd.read_csv(fname)
    price_col = find_price_column(df)
    Y = pd.to_numeric(df[price_col], errors='coerce')

    # Detrend
    X = Y - Y.shift(long_lag)
    Ydt = (X - X.shift(short_lag)) / Y
    Ydt = Ydt.replace([np.inf, -np.inf], np.nan).dropna()

    values = Ydt.values

    confirmed_peaks, confirmed_valleys = detect_confirmed_extrema(values,
                                                                  drop_thresh=0.05,
                                                                  cont_min=0.025,
                                                                  max_days_peak=10,
                                                                  max_days_valley=30)

    fibs = [0.0902, 0.1459, 0.19095, 0.236, 0.309, 0.382, 0.441, 0.50, 0.559, 0.618]
    tol = args.tolerance

    total_peak_hits = 0
    total_valley_hits = 0
    total_hits = 0
    for lvl in fibs:
        p_hits = count_hits_on_confirmed(confirmed_peaks, lvl, tol)
        v_hits = count_hits_on_confirmed(confirmed_valleys, -lvl, tol)
        total_peak_hits += p_hits
        total_valley_hits += v_hits
        total_hits += (p_hits + v_hits)
        print(f"Fib {lvl:.5f}: peaks={p_hits}, valleys={v_hits}")

    print(f"\nTotal peak hits: {total_peak_hits}")
    print(f"Total valley hits: {total_valley_hits}")
    print(f"Total fib hits: {total_hits}")

    # Summary
    print("\nSummary:")
    print(f"Confirmed peaks:   {len(confirmed_peaks)}")
    print(f"Confirmed valleys: {len(confirmed_valleys)}")


if __name__ == "__main__":
    main()
