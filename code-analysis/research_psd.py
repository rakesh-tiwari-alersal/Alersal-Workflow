#!/usr/bin/env python3
import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy.signal import periodogram, find_peaks

# Define analysis ranges (used for per-symbol CSV outputs; unchanged)
RANGES = [
    (15, 40),
    (30, 60),
    (50, 90),
    (150, 350),
    (200, 500),
    (350, 700)
]

def analyze_single_range(series, range_min, range_max):
    """Analyze a single range and return results"""
    closes = series.values
    if np.any(closes <= 0):
        print("Error: Non-positive closing prices detected")
        return None

    # Standard differencing 
    closes_diff = closes[1:] - closes[:-1]
    
    # Compute PSD
    frequencies, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')
    
    # Filter out zero frequency
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]
    
    if len(frequencies) == 0:
        print("Error: No valid frequencies found")
        return None
    
    # Convert frequencies to periods
    periods = 1 / frequencies
    
    # Filter to current range
    range_mask = (periods >= range_min) & (periods <= range_max)
    range_periods = periods[range_mask]
    range_psd = psd[range_mask]
    
    if len(range_psd) == 0:
        print(f"No cycles found in range {range_min}-{range_max}")
        return None
        
    # Find peaks in this range
    peak_indices, _ = find_peaks(range_psd, height=0.001 * np.max(range_psd), distance=1)
    
    if len(peak_indices) == 0:
        print(f"No significant peaks found in range {range_min}-{range_max}")
        return None
        
    # Extract peak data
    peak_periods = range_periods[peak_indices]
    peak_powers = range_psd[peak_indices]
    
    # Sort by power (descending)
    peaks_df = pd.DataFrame({'Period': peak_periods, 'Power': peak_powers})
    peaks_df = peaks_df.sort_values('Power', ascending=False)
    
    # Take top 10 (this behavior remains exactly as before for per-range outputs)
    top_peaks = peaks_df.head(10).copy()
    
    # Calculate percentage contribution
    total_power = top_peaks['Power'].sum()
    if total_power > 0:
        top_peaks['% Power'] = (top_peaks['Power'] / total_power) * 100
    else:
        top_peaks['% Power'] = 0
        
    # Round values and drop Power column
    # Keep Period as two-decimal precision
    top_peaks['Period'] = top_peaks['Period'].round(2)
    top_peaks['% Power'] = top_peaks['% Power'].round(2)
    top_peaks = top_peaks[['Period', '% Power']]  # Drop Power column
    
    return top_peaks

# Helper: compute power-weighted summary from full PSD for a given period range
def compute_range_weighted_all_peaks(series, range_min, range_max):
    """
    Compute:
      - highest_period: period (not frequency) with maximum power among all PSD peaks
        whose period lies in [range_min, range_max].
      - weighted_avg: power-weighted average of all detected peak periods in that range.
    Returns (highest_str, weighted_str) where each is a formatted string (2 decimals)
    or ("", "") if no valid peaks are found.
    This function detects ALL peaks (no top-N truncation) inside the specified period range.
    """
    closes = series.values
    if np.any(closes <= 0):
        return "", ""

    # Standard differencing
    closes_diff = closes[1:] - closes[:-1]

    # Compute PSD on full differenced series
    frequencies, psd = periodogram(closes_diff, fs=1, scaling='density', window='hann')

    # Remove zero frequency
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]

    if len(frequencies) == 0:
        return "", ""

    # Convert to periods
    periods = 1.0 / frequencies

    # Filter desired period range
    range_mask = (periods >= range_min) & (periods <= range_max)
    range_periods = periods[range_mask]
    range_psd = psd[range_mask]

    if len(range_psd) == 0:
        return "", ""

    # Find peaks within this filtered PSD (use same threshold idea as analyze_single_range)
    peak_indices, _ = find_peaks(range_psd, height=0.001 * np.max(range_psd), distance=1)

    if len(peak_indices) == 0:
        return "", ""

    # Extract all detected peaks (do NOT truncate to top N)
    peak_periods = range_periods[peak_indices].astype(float)
    peak_powers = range_psd[peak_indices].astype(float)

    # If any NaNs or non-positive weights, filter them out
    valid_mask = (~np.isnan(peak_periods)) & (~np.isnan(peak_powers)) & (peak_powers > 0)
    if not np.any(valid_mask):
        return "", ""

    peak_periods = peak_periods[valid_mask]
    peak_powers = peak_powers[valid_mask]

    # Highest-power period (period corresponding to max peak power)
    max_idx = int(np.argmax(peak_powers))
    highest_period = float(peak_periods[max_idx])

    # Power-weighted average (weights = peak_powers)
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
    Return a list of up to n top periods (strings formatted to 2 decimals) selected by power
    but ensuring strictly increasing period sequence:
      - first = absolute top-power peak
      - second = highest-power peak among peaks with period > first_period
      - third = highest-power peak among peaks with period > second_period
    If fewer than n qualifying peaks exist, remaining entries are empty strings.
    """
    closes = series.values
    if np.any(closes <= 0):
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

    peak_periods = peak_periods[valid_mask]
    peak_powers = peak_powers[valid_mask]

    # Build array of peaks (period, power)
    peaks = np.vstack([peak_periods, peak_powers]).T  # shape (m,2)

    # Helper to pick highest-power peak among peaks with period > prev_period
    def pick_highest_after(prev_period, candidates):
        # candidates: array of shape (k,2): column0=period, col1=power
        mask = candidates[:,0] > prev_period
        if not np.any(mask):
            return None
        sub = candidates[mask]
        # pick row with max power
        idx = int(np.argmax(sub[:,1]))
        return sub[idx]  # returns [period, power]

    out = [""] * n

    # 1) pick absolute top-power peak
    if len(peaks) == 0:
        return out
    # find index of max power among all peaks
    idx0 = int(np.argmax(peaks[:,1]))
    first_peak = peaks[idx0]  # [period, power]
    out[0] = f"{float(first_peak[0]):.2f}"

    # Remove the selected peak from candidates to avoid reselection
    remaining = np.delete(peaks, idx0, axis=0) if len(peaks) > 1 else np.empty((0,2))

    # 2) pick highest-power among remaining peaks with period > first_period
    pick1 = pick_highest_after(first_peak[0], remaining)
    if pick1 is not None:
        out[1] = f"{float(pick1[0]):.2f}"
        # remove that picked peak from remaining
        # find matching row in remaining (match period and power)
        mask_match = (remaining[:,0] == pick1[0]) & (remaining[:,1] == pick1[1])
        if np.any(mask_match):
            remaining = remaining[~mask_match]
    else:
        # no qualifying second peak; return with empty slots
        return out

    # 3) pick highest-power among remaining peaks with period > second_period
    pick2 = pick_highest_after(float(pick1[0]), remaining)
    if pick2 is not None:
        out[2] = f"{float(pick2[0]):.2f}"
    # else leave as empty

    return out

def append_psd_all_file_from_values(file_base,
                                   most_powerful_cycle_str,
                                   second_powerful_cycle_str,
                                   third_powerful_cycle_str,
                                   power_weighted_cycle_str,
                                   clear_flag=False):
    """
    Append a single-row summary to research_output/research_psd_ALL.csv using
    precomputed string values.

    Row format (columns):
      0: file_base (symbol)
      1: most_powerful_cycle  (top peak in 200-700)
      2: second_most_powerful_cycle (2nd peak in 200-700, period > first)
      3: third_most_powerful_cycle  (3rd peak in 200-700, period > second)
      4: power_weighted_cycle (power-weighted average period computed with chosen cutoff..700)
    """
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

    # Handle clear flag (truncate file) then append
    mode = 'a'
    if clear_flag:
        with open(all_file, 'w', newline='') as fh:
            pass
        mode = 'a'

    with open(all_file, mode, newline='') as fh:
        import csv
        writer = csv.writer(fh)
        writer.writerow(row)

def main():
    # Handle PowerShell's argument parsing issue with commas
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

    # Required mutually exclusive volatility switch
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

    # Load data
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

    if args.range:
        try:
            range_min, range_max = map(int, args.range.split(','))
            print(f"Analyzing range {range_min}-{range_max} for {args.file}")

            results = analyze_single_range(series, range_min, range_max)

            if results is not None:
                print(f"\nCycle ({range_min}-{range_max})\t% Power")
                for _, row in results.iterrows():
                    print(f"{row['Period']}\t{row['% Power']}%")
            return
        except ValueError:
            print("Error: Range format should be like '30,60'")
            return

    # Analyze all ranges (unchanged behavior)
    all_results = {}
    for range_min, range_max in RANGES:
        results = analyze_single_range(series, range_min, range_max)
        if results is not None:
            all_results[f"{range_min}-{range_max}"] = results

    if not all_results:
        print("No significant cycles found in any range")
        return

    # Prepare output for CSV (unchanged original behavior)
    output_rows = []

    # First group: 15-40, 30-60, 50-90
    group1_ranges = ["15-40", "30-60", "50-90"]
    group1_data = []
    for range_name in group1_ranges:
        if range_name in all_results:
            group1_data.append(all_results[range_name])
        else:
            group1_data.append(pd.DataFrame(columns=['Period', '% Power']))

    # Second group: 150-350, 200-500, 350-700
    group2_ranges = ["150-350", "200-500", "350-700"]
    group2_data = []
    for range_name in group2_ranges:
        if range_name in all_results:
            group2_data.append(all_results[range_name])
        else:
            group2_data.append(pd.DataFrame(columns=['Period', '% Power']))

    # Find maximum rows needed
    max_rows = max(
        max(len(df) for df in group1_data),
        max(len(df) for df in group2_data)
    )

    # Build header row with exactly one empty column between ranges
    header_row = []
    for i, range_name in enumerate(group1_ranges):
        header_row.extend([f"Cycle ({range_name})", "% Power"])
        if i < len(group1_ranges) - 1:
            header_row.append("")  # Add empty column between ranges

    # Add one empty column between groups
    header_row.append("")

    for i, range_name in enumerate(group2_ranges):
        header_row.extend([f"Cycle ({range_name})", "% Power"])
        if i < len(group2_ranges) - 1:
            header_row.append("")  # Add empty column between ranges

    output_rows.append(header_row)

    # Add data rows with exactly one empty column between ranges
    for i in range(max_rows):
        row = []

        # Add group1 data
        for j, df in enumerate(group1_data):
            if i < len(df):
                # Format numbers to 2 decimal places for CSV output
                p = df.iloc[i]['Period']
                pct = df.iloc[i]['% Power']
                row.extend([
                    f"{p:.2f}" if not pd.isna(p) else "",
                    f"{pct:.2f}" if not pd.isna(pct) else ""
                ])
            else:
                row.extend(["", ""])
            if j < len(group1_data) - 1:
                row.append("")  # Add empty column between ranges

        # Add one empty column between groups
        row.append("")

        # Add group2 data
        for j, df in enumerate(group2_data):
            if i < len(df):
                # Format numbers to 2 decimal places for CSV output
                p = df.iloc[i]['Period']
                pct = df.iloc[i]['% Power']
                row.extend([
                    f"{p:.2f}" if not pd.isna(p) else "",
                    f"{pct:.2f}" if not pd.isna(pct) else ""
                ])
            else:
                row.extend(["", ""])
            if j < len(group2_data) - 1:
                row.append("")  # Add empty column between ranges

        output_rows.append(row)

    # Create output directory if it doesn't exist
    os.makedirs('research_output', exist_ok=True)

    # Create DataFrame and save to CSV (exact same layout as original)
    output_df = pd.DataFrame(output_rows)
    file_name = os.path.splitext(args.file)[0]
    output_file = f"research_output/research_psd_{file_name}.csv"
    output_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")

    # Determine cutoff for power-weighted calculation based on volatility switch
    if args.vol_high:
        cutoff = 200
    elif args.vol_low:
        cutoff = 300
    else:
        cutoff = 200  # fallback (shouldn't occur because group required=True)

    # Compute the three most powerful cycles from 200-700 (exclude <200),
    # but enforce strictly increasing periods for the 2nd and 3rd picks.
    top3 = compute_top_n_peaks_in_range(series, 200, 700, n=3)
    most_powerful_cycle_str = top3[0]
    second_powerful_cycle_str = top3[1]
    third_powerful_cycle_str = top3[2]

    # Compute power-weighted cycle using selected cutoff..700
    _, power_weighted_cycle_str = compute_range_weighted_all_peaks(series, cutoff, 700)

    # Append to ALL file:
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
