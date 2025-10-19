import pandas as pd
import numpy as np
import os
import sys
import argparse
from scipy.signal import periodogram, find_peaks

# Define analysis ranges
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
    
    # Take top 10
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

def append_psd_all_file(file_base, group2_data, clear_flag=False):
    """
    Append a single-row summary to research_output/research_psd_ALL.csv.
    Row format: file_base,period1,period2,...
    period1.. are unique Period values from group2_data (last 3 ranges), duplicates removed,
    sorted descending, formatted with two decimals.
    If clear_flag True, truncate (overwrite) the ALL file first.
    """
    outdir = 'research_output'
    os.makedirs(outdir, exist_ok=True)
    all_file = os.path.join(outdir, 'research_psd_ALL.csv')

    # Collect Periods from the three group2_data DataFrames
    periods = []
    for df in group2_data:
        if df is None or df.empty:
            continue
        # df['Period'] is rounded to 2 decimals already
        vals = df['Period'].dropna().tolist()
        periods.extend(vals)

    # Remove duplicates, convert to floats, sort descending
    unique_periods = sorted({float(p) for p in periods}, reverse=True)

    # Format with two decimals
    formatted_periods = [f"{p:.2f}" for p in unique_periods]

    # Prepare row: start with file_base
    row = [file_base] + formatted_periods

    # If clear flag: truncate file (write headerless empty file)
    mode = 'a'
    if clear_flag:
        # overwrite/truncate
        with open(all_file, 'w', newline='') as fh:
            pass
        mode = 'a'

    # Append CSV row (no header)
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
    
    # Analyze all ranges
    all_results = {}
    
    for range_min, range_max in RANGES:
        results = analyze_single_range(series, range_min, range_max)
        if results is not None:
            all_results[f"{range_min}-{range_max}"] = results
    
    if not all_results:
        print("No significant cycles found in any range")
        return
    
    # Prepare output for CSV
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
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    file_name = os.path.splitext(args.file)[0]
    output_file = f"research_output/research_psd_{file_name}.csv"
    output_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")

    # Append the "ALL" file (last three ranges -> group2_data)
    append_psd_all_file(file_name, group2_data, clear_flag=args.clear_summary)

if __name__ == "__main__":
    main()
