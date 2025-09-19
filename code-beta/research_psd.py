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
    (300, 700)
]

def analyze_single_range(series, range_min, range_max):
    """Analyze a single range and return results"""
    # Calculate log returns
    closes = series.values
    if np.any(closes <= 0):
        print("Error: Non-positive closing prices detected")
        return None
    
    closes_diff = closes[1:] - closes[:-1]
#   closes_diff = np.log(closes[1:]) - np.log(closes[:-1])
        
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
        top_peaks['% Contribution'] = (top_peaks['Power'] / total_power) * 100
    else:
        top_peaks['% Contribution'] = 0
        
    # Round values
    top_peaks['Period'] = top_peaks['Period'].round().astype(int)
    top_peaks['Power'] = top_peaks['Power'].round(0).astype(int)
    top_peaks['% Contribution'] = top_peaks['% Contribution'].round(2)
    
    return top_peaks

def main():
    # Handle PowerShell's argument parsing issue with commas
    # If we have an argument that starts with -r and contains a comma, we need to fix it
    fixed_argv = []
    i = 0
    while i < len(sys.argv):
        arg = sys.argv[i]
        if arg.startswith('-r') and ',' in arg and '=' not in arg:
            # This is the problematic case: -r30,60
            # Split it into -r and 30,60
            fixed_argv.append('-r')
            fixed_argv.append(arg[2:])  # Remove the -r prefix
        else:
            fixed_argv.append(arg)
        i += 1
    
    parser = argparse.ArgumentParser(description='Analyze PSD cycles for a single instrument')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-r', '--range', type=str, help='Comma-separated range to analyze (e.g., 30,60)', default=None)
    
    # Use our fixed argument list
    args = parser.parse_args(fixed_argv[1:])  # Skip the script name

    # Validate file path
    file_path = os.path.join('historical_data', args.file)
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Load data
    try:
        df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
        # Check for both upper and lower case column names
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
        # Parse the range
        try:
            range_min, range_max = map(int, args.range.split(','))
            print(f"Analyzing range {range_min}-{range_max} for {args.file}")
            
            # Analyze the specific range
            results = analyze_single_range(series, range_min, range_max)
            
            if results is not None:
                print(f"\nCycle ({range_min}-{range_max})\tPower\t% Contribution")
                for _, row in results.iterrows():
                    print(f"{row['Period']}\t{row['Power']}\t{row['% Contribution']}%")
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
            group1_data.append(pd.DataFrame(columns=['Period', 'Power', '% Contribution']))
    
    # Second group: 150-350, 200-500, 300-700
    group2_ranges = ["150-350", "200-500", "300-700"]
    group2_data = []
    
    for range_name in group2_ranges:
        if range_name in all_results:
            group2_data.append(all_results[range_name])
        else:
            group2_data.append(pd.DataFrame(columns=['Period', 'Power', '% Contribution']))
    
    # Find maximum rows needed
    max_rows = max(
        max(len(df) for df in group1_data),
        max(len(df) for df in group2_data)
    )
    
    # Add headers with empty columns between ranges
    header_row = []
    for range_name in group1_ranges:
        header_row.extend([f"Cycle ({range_name})", "Power", "% Contribution", ""])
    
    # Remove the last empty column
    header_row = header_row[:-1]
    
    # Add separator between groups
    header_row.extend(["", "", ""])
    
    for range_name in group2_ranges:
        header_row.extend([f"Cycle ({range_name})", "Power", "% Contribution", ""])
    
    # Remove the last empty column
    header_row = header_row[:-1]
    
    output_rows.append(header_row)
    
    # Add data rows with empty columns between ranges
    for i in range(max_rows):
        row = []
        
        # Add group1 data with empty columns
        for df in group1_data:
            if i < len(df):
                row.extend([df.iloc[i]['Period'], df.iloc[i]['Power'], df.iloc[i]['% Contribution'], ""])
            else:
                row.extend(["", "", "", ""])
        
        # Remove the last empty column
        row = row[:-1]
        
        # Add separator between groups
        row.extend(["", "", ""])
        
        # Add group2 data with empty columns
        for df in group2_data:
            if i < len(df):
                row.extend([df.iloc[i]['Period'], df.iloc[i]['Power'], df.iloc[i]['% Contribution'], ""])
            else:
                row.extend(["", "", "", ""])
        
        # Remove the last empty column
        row = row[:-1]
        
        output_rows.append(row)
    
    # Create output directory if it doesn't exist
    os.makedirs('research_output', exist_ok=True)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    file_name = os.path.splitext(args.file)[0]
    output_file = f"research_output/research_psd_{file_name}.csv"
    output_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()