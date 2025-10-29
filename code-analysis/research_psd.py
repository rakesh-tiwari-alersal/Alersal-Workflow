#!/usr/bin/env python3
"""
research_psd.py

Produces per-symbol PSD peak tables across multiple period ranges and appends
a small summary row into research_output/research_psd_ALL.csv.

Behavior:
- Six initial ranges return top 10 peaks each.
- The final range (RANGES[-1], now 150-700) returns top 20 peaks.
- Headers and output blocks derive from RANGES to avoid hard-coded strings.
- Centroid/ALL-file logic left unchanged (computed from 200..700 in existing code).
"""

from __future__ import annotations
import os
import sys
import argparse
import csv
import pandas as pd
import numpy as np
from scipy.signal import periodogram, find_peaks

# === CONFIG ===
# Analysis ranges (first 6 remain as before; final range expanded to 150-700)
RANGES = [
    (15, 40),
    (30, 60),
    (50, 90),
    (150, 350),
    (200, 500),
    (350, 700),
    (150, 700)   # final expanded block: top-20 peaks for this block only
]
# =============

def analyze_single_range(series, range_min, range_max):
    """Analyze a single period range and return top peaks DataFrame."""
    closes = series.values
    if len(closes) < 10:
        return None

    # Standard differencing
    closes_diff = closes[1:] - closes[:-1]

    # Compute PSD using periodogram
    frequencies, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')

    # Remove DC (zero frequency)
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]

    if len(frequencies) == 0:
        return None

    # Convert to periods
    periods = 1.0 / frequencies

    # Filter to requested range
    range_mask = (periods >= range_min) & (periods <= range_max)
    range_periods = periods[range_mask]
    range_psd = psd[range_mask]

    if len(range_psd) == 0:
        return None

    # Peak detection — use relative height threshold against local max
    try:
        height_thresh = 0.001 * np.max(range_psd)
    except Exception:
        height_thresh = None

    if height_thresh is None or np.isnan(height_thresh):
        return None

    peak_indices, _ = find_peaks(range_psd, height=height_thresh, distance=1)
    if len(peak_indices) == 0:
        return None

    peak_periods = range_periods[peak_indices].astype(float)
    peak_powers = range_psd[peak_indices].astype(float)

    # Build DataFrame and sort by power descending
    peaks_df = pd.DataFrame({'Period': peak_periods, 'Power': peak_powers})
    peaks_df = peaks_df.sort_values('Power', ascending=False)

    # Choose 20 peaks only for the final 150-700 block; otherwise 10 peaks
    top_n = 20 if (range_min == 150 and range_max == 700) else 10
    top_peaks = peaks_df.head(top_n).copy()

    # Compute percent contributions (relative to the selected top-N powers)
    total_power = top_peaks['Power'].sum()
    if total_power > 0:
        top_peaks['% Power'] = (top_peaks['Power'] / total_power) * 100.0
    else:
        top_peaks['% Power'] = 0.0

    # Round and format for CSV-friendly output
    top_peaks['Period'] = top_peaks['Period'].round(2)
    top_peaks['% Power'] = top_peaks['% Power'].round(2)

    # Only keep the output columns we want
    top_peaks = top_peaks[['Period', '% Power']]

    return top_peaks


def compute_range_weighted_all_peaks(series, range_min, range_max):
    """
    Compute the highest-power peak and power-weighted average across *all* detected
    peaks inside [range_min, range_max]. Returns formatted strings (2 decimals).
    """
    closes = series.values
    if len(closes) < 10:
        return "", ""

    closes_diff = closes[1:] - closes[:-1]
    frequencies, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]
    if len(frequencies) == 0:
        return "", ""

    periods = 1.0 / frequencies
    range_mask = (periods >= range_min) & (periods <= range_max)
    range_periods = periods[range_mask]
    range_psd = psd[range_mask]
    if len(range_psd) == 0:
        return "", ""

    peak_indices, _ = find_peaks(range_psd, height=0.001 * np.max(range_psd), distance=1)
    if len(peak_indices) == 0:
        return "", ""

    peak_periods = range_periods[peak_indices].astype(float)
    peak_powers = range_psd[peak_indices].astype(float)

    valid_mask = (~np.isnan(peak_periods)) & (~np.isnan(peak_powers)) & (peak_powers > 0)
    if not np.any(valid_mask):
        return "", ""

    peak_periods = peak_periods[valid_mask]
    peak_powers = peak_powers[valid_mask]

    # Highest-power period
    max_idx = int(np.argmax(peak_powers))
    highest_period = float(peak_periods[max_idx])

    # Power-weighted average
    sum_weights = float(np.sum(peak_powers))
    if sum_weights > 0:
        weighted_avg = float((peak_periods * peak_powers).sum() / sum_weights)
    else:
        weighted_avg = float('nan')

    highest_str = f"{highest_period:.2f}"
    weighted_str = f"{weighted_avg:.2f}" if not np.isnan(weighted_avg) else ""

    return highest_str, weighted_str


def compute_top_n_peaks_in_range(series, range_min, range_max, n=3):
    """
    Select up to n top periods with strictly increasing period constraint:
    - first = absolute highest-power peak
    - second = highest-power peak with period > first_period
    - third = highest-power peak with period > second_period
    """
    closes = series.values
    if len(closes) < 10:
        return [""] * n

    closes_diff = closes[1:] - closes[:-1]
    frequencies, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]
    if len(frequencies) == 0:
        return [""] * n

    periods = 1.0 / frequencies
    range_mask = (periods >= range_min) & (periods <= range_max)
    range_periods = periods[range_mask]
    range_psd = psd[range_mask]
    if len(range_psd) == 0:
        return [""] * n

    peak_indices, _ = find_peaks(range_psd, height=0.001 * np.max(range_psd), distance=1)
    if len(peak_indices) == 0:
        return [""] * n

    peak_periods = range_periods[peak_indices].astype(float)
    peak_powers = range_psd[peak_indices].astype(float)

    valid_mask = (~np.isnan(peak_periods)) & (~np.isnan(peak_powers)) & (peak_powers > 0)
    if not np.any(valid_mask):
        return [""] * n

    peaks = np.vstack([peak_periods[valid_mask], peak_powers[valid_mask]]).T

    def pick_highest_after(prev_period, candidates):
        mask = candidates[:, 0] > prev_period
        if not np.any(mask):
            return None
        sub = candidates[mask]
        idx = int(np.argmax(sub[:, 1]))
        return sub[idx]

    out = [""] * n
    if peaks.shape[0] == 0:
        return out

    # pick absolute top
    idx0 = int(np.argmax(peaks[:, 1]))
    first_peak = peaks[idx0]
    out[0] = f"{float(first_peak[0]):.2f}"
    remaining = np.delete(peaks, idx0, axis=0) if peaks.shape[0] > 1 else np.empty((0, 2))

    # pick second (period > first)
    pick1 = pick_highest_after(first_peak[0], remaining)
    if pick1 is not None:
        out[1] = f"{float(pick1[0]):.2f}"
        # remove
        mask_match = (remaining[:, 0] == pick1[0]) & (remaining[:, 1] == pick1[1])
        if np.any(mask_match):
            remaining = remaining[~mask_match]

    # pick third (period > second)
    if out[1]:
        pick2 = pick_highest_after(float(out[1]), remaining)
        if pick2 is not None:
            out[2] = f"{float(pick2[0]):.2f}"

    return out


def append_psd_all_file_from_values(file_base,
                                   most_powerful_cycle_str,
                                   second_powerful_cycle_str,
                                   third_powerful_cycle_str,
                                   power_weighted_cycle_str,
                                   clear_flag=False):
    """Append a single row to research_output/research_psd_ALL.csv."""
    outdir = 'research_output'
    os.makedirs(outdir, exist_ok=True)
    all_file = os.path.join(outdir, 'research_psd_ALL.csv')

    row = [
        file_base,
        most_powerful_cycle_str,
        second_powerful_cycle_str,
        third_powerful_cycle_str,
        power_weighted_cycle_str
    ]

    mode = 'a'
    if clear_flag:
        with open(all_file, 'w', newline='') as fh:
            pass
        mode = 'a'

    with open(all_file, mode, newline='') as fh:
        writer = csv.writer(fh)
        writer.writerow(row)


def main():
    # Fix PowerShell arg parsing oddities
    fixed_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('-r') and ',' in arg and '=' not in arg:
            fixed_argv.append('-r')
            fixed_argv.append(arg[2:])
        else:
            fixed_argv.append(arg)
        i += 1

    parser = argparse.ArgumentParser(description='Analyze PSD cycles for a single instrument')

    # Mutually exclusive volatility switch (still present; not used elsewhere)
    vol_group = parser.add_mutually_exclusive_group(required=True)
    vol_group.add_argument("-vh", "--vol-high", action="store_true",
                           help="High volatility preset: use 200-day cutoff for power-weighted cycle calculations")
    vol_group.add_argument("-vl", "--vol-low", action="store_true",
                           help="Low volatility preset: use 300-day cutoff for power-weighted cycle calculations")

    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-r', '--range', type=str, help='Comma-separated range to analyze (e.g., 30,60)', default=None)
    parser.add_argument('-c', '--clear-summary', action='store_true', help='If provided, clear research_psd_ALL.csv before appending.')
    args = parser.parse_args(fixed_argv[1:])

    # Validate file path
    file_path = os.path.join('historical_data', args.file)
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Load CSV and detect Close column
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        if 'Close' in df.columns:
            series = df['Close']
        elif 'close' in df.columns:
            series = df['close']
        elif 'Price' in df.columns:
            series = df['Price']
        elif 'price' in df.columns:
            series = df['price']
        else:
            print("Error: CSV file must contain a 'Close', 'close', 'Price', or 'price' column.")
            return
    except Exception as e:
        print(f"Error loading data from file: {e}")
        return

    # If a single range requested, short-circuit
    if args.range:
        try:
            range_min, range_max = map(int, args.range.split(','))
            print(f"Analyzing range {range_min}-{range_max} for {args.file}")
            results = analyze_single_range(series, range_min, range_max)
            if results is not None:
                print(f"\nCycle ({range_min}-{range_max})\t% Power")
                for _, row in results.iterrows():
                    print(f"{row['Period']}\t{row['% Power']}%")
            else:
                print("No cycles found")
            return
        except ValueError:
            print("Error: Range format should be like '30,60'")
            return

    # Analyze all defined ranges
    all_results = {}
    for (rmin, rmax) in RANGES:
        key = f"{rmin}-{rmax}"
        res = analyze_single_range(series, rmin, rmax)
        if res is not None:
            all_results[key] = res

    if not all_results:
        print("No significant cycles found in any range")
        return

    # Build grouped blocks:
    group1_ranges = [f"{RANGES[0][0]}-{RANGES[0][1]}", f"{RANGES[1][0]}-{RANGES[1][1]}", f"{RANGES[2][0]}-{RANGES[2][1]}"]
    group2_ranges = [f"{RANGES[3][0]}-{RANGES[3][1]}", f"{RANGES[4][0]}-{RANGES[4][1]}", f"{RANGES[5][0]}-{RANGES[5][1]}"]
    # group3: derive from final RANGES entry so header always matches
    last_min, last_max = RANGES[-1]
    group3_ranges = [f"{last_min}-{last_max}"]

    group1_data = [all_results.get(r, pd.DataFrame(columns=['Period', '% Power'])) for r in group1_ranges]
    group2_data = [all_results.get(r, pd.DataFrame(columns=['Period', '% Power'])) for r in group2_ranges]
    group3_data = [all_results.get(r, pd.DataFrame(columns=['Period', '% Power'])) for r in group3_ranges]

    # Determine number of rows to output
    max_rows = max(
        max((len(df) for df in group1_data), default=0),
        max((len(df) for df in group2_data), default=0),
        max((len(df) for df in group3_data), default=0)
    )

    # Build header row
    header_row = []
    for i, range_name in enumerate(group1_ranges):
        header_row.extend([f"Cycle ({range_name})", "% Power"])
        if i < len(group1_ranges) - 1:
            header_row.append("")  # empty column between ranges
    header_row.append("")  # empty column between group1 and group2
    for i, range_name in enumerate(group2_ranges):
        header_row.extend([f"Cycle ({range_name})", "% Power"])
        if i < len(group2_ranges) - 1:
            header_row.append("")
    header_row.append("")  # empty between group2 and group3
    for i, range_name in enumerate(group3_ranges):
        header_row.extend([f"Cycle ({range_name})", "% Power"])
        if i < len(group3_ranges) - 1:
            header_row.append("")

    output_rows = [header_row]

    # Fill rows
    for i in range(max_rows):
        row = []
        # group1
        for j, df in enumerate(group1_data):
            if i < len(df):
                p = df.iloc[i]['Period']
                pct = df.iloc[i]['% Power']
                row.extend([f"{p:.2f}" if not pd.isna(p) else "", f"{pct:.2f}" if not pd.isna(pct) else ""])
            else:
                row.extend(["", ""])
            if j < len(group1_data) - 1:
                row.append("")
        row.append("")  # gap between group1 and group2

        # group2
        for j, df in enumerate(group2_data):
            if i < len(df):
                p = df.iloc[i]['Period']
                pct = df.iloc[i]['% Power']
                row.extend([f"{p:.2f}" if not pd.isna(p) else "", f"{pct:.2f}" if not pd.isna(pct) else ""])
            else:
                row.extend(["", ""])
            if j < len(group2_data) - 1:
                row.append("")
        row.append("")  # gap between group2 and group3

        # group3 (final block)
        for j, df in enumerate(group3_data):
            if i < len(df):
                p = df.iloc[i]['Period']
                pct = df.iloc[i]['% Power']
                row.extend([f"{p:.2f}" if not pd.isna(p) else "", f"{pct:.2f}" if not pd.isna(pct) else ""])
            else:
                row.extend(["", ""])
            if j < len(group3_data) - 1:
                row.append("")
        output_rows.append(row)

    # Write out CSV
    os.makedirs('research_output', exist_ok=True)
    file_name = os.path.splitext(args.file)[0]
    out_file = os.path.join('research_output', f"research_psd_{file_name}.csv")

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(out_file, index=False, header=False)
    print(f"Saved per-range PSD table to {out_file}")

    # Decide cutoff for centroid/ALL calculation based on vol flags (left as-is)
    if args.vol_high:
        cutoff = 200
    elif args.vol_low:
        cutoff = 300
    else:
        cutoff = 200

    # Compute top 3 most-powerful cycles from 200-700 for the _ALL file
    top3 = compute_top_n_peaks_in_range(series, 200, 700, n=3)
    most_powerful_cycle_str = top3[0]
    second_powerful_cycle_str = top3[1]
    third_powerful_cycle_str = top3[2]

    # Compute power-weighted cycle using selected cutoff..700
    _, power_weighted_cycle_str = compute_range_weighted_all_peaks(series, cutoff, 700)

    # Append to ALL file (unchanged)
    append_psd_all_file_from_values(
        file_name,
        most_powerful_cycle_str,
        second_powerful_cycle_str,
        third_powerful_cycle_str,
        power_weighted_cycle_str,
        clear_flag=args.clear_summary
    )

if __name__ == "__main__":
    main()
