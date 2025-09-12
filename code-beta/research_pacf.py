import pandas as pd
import numpy as np
import os
import argparse
from statsmodels.tsa.stattools import pacf

# Define analysis ranges
RANGES = [
    (15, 50),
    (30, 60),
    (60, 90),
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
        
    log_returns = np.log(closes[1:]) - np.log(closes[:-1])
    
    # Compute PACF with maximum lag set to the maximum range
    max_lag = max(range_max for _, range_max in RANGES)
    pacf_vals = pacf(log_returns, nlags=max_lag, method='ols')
    
    # Calculate 95% confidence threshold
    n = len(log_returns)
    ci_threshold = 1.96 / np.sqrt(n)
    
    # Prepare results for the specific range
    results = []
    for lag in range(range_min, range_max + 1):
        if lag < len(pacf_vals):
            pacf_val = pacf_vals[lag]
            if abs(pacf_val) > ci_threshold:
                results.append({
                    'Lag': lag,
                    'PACF': pacf_val,
                    'Absolute PACF': abs(pacf_val)
                })
    
    if not results:
        print(f"No significant PACF values found in range {range_min}-{range_max}")
        return None
        
    # Convert to DataFrame and sort by absolute value (descending)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('Absolute PACF', ascending=False)
    
    # Take top 10
    top_results = results_df.head(10).copy()
    
    # Calculate percentage contribution based on absolute values
    total_abs_pacf = top_results['Absolute PACF'].sum()
    if total_abs_pacf > 0:
        top_results['% Contribution'] = (top_results['Absolute PACF'] / total_abs_pacf) * 100
    else:
        top_results['% Contribution'] = 0
        
    # Round values
    top_results['PACF'] = top_results['PACF'].round(4)
    top_results['Absolute PACF'] = top_results['Absolute PACF'].round(4)
    top_results['% Contribution'] = top_results['% Contribution'].round(2)
    
    return top_results[['Lag', 'PACF', '% Contribution']]

def main():
    parser = argparse.ArgumentParser(description='Analyze PACF for a single instrument')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-r', '--range', type=str, help='Comma-separated range to analyze (e.g., 30,60)', default=None)
    
    args = parser.parse_args()

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
            print(f"Analyzing PACF range {range_min}-{range_max} for {args.file}")
            
            # Analyze the specific range
            results = analyze_single_range(series, range_min, range_max)
            
            if results is not None:
                print(f"\nPACF ({range_min}-{range_max})\tPACF Value\t% Contribution")
                for _, row in results.iterrows():
                    print(f"{row['Lag']}\t{row['PACF']}\t{row['% Contribution']}%")
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
        print("No significant PACF values found in any range")
        return
    
    # Prepare output for CSV
    output_rows = []
    
    # First group: 15-50, 30-60, 60-90
    group1_ranges = ["15-50", "30-60", "60-90"]
    group1_data = []
    
    for range_name in group1_ranges:
        if range_name in all_results:
            group1_data.append(all_results[range_name])
        else:
            group1_data.append(pd.DataFrame(columns=['Lag', 'PACF', '% Contribution']))
    
    # Second group: 150-350, 200-500, 300-700
    group2_ranges = ["150-350", "200-500", "300-700"]
    group2_data = []
    
    for range_name in group2_ranges:
        if range_name in all_results:
            group2_data.append(all_results[range_name])
        else:
            group2_data.append(pd.DataFrame(columns=['Lag', 'PACF', '% Contribution']))
    
    # Find maximum rows needed
    max_rows = max(
        max(len(df) for df in group1_data),
        max(len(df) for df in group2_data)
    )
    
    # Add headers
    header_row = []
    for range_name in group1_ranges:
        header_row.extend([f"PACF ({range_name})", "PACF Value", "% Contribution"])
    
    # Add two empty columns
    header_row.extend(["", "", ""])
    
    for range_name in group2_ranges:
        header_row.extend([f"PACF ({range_name})", "PACF Value", "% Contribution"])
    
    output_rows.append(header_row)
    
    # Add data rows
    for i in range(max_rows):
        row = []
        
        # Add group1 data
        for df in group1_data:
            if i < len(df):
                row.extend([df.iloc[i]['Lag'], df.iloc[i]['PACF'], df.iloc[i]['% Contribution']])
            else:
                row.extend(["", "", ""])
        
        # Add two empty columns
        row.extend(["", "", ""])
        
        # Add group2 data
        for df in group2_data:
            if i < len(df):
                row.extend([df.iloc[i]['Lag'], df.iloc[i]['PACF'], df.iloc[i]['% Contribution']])
            else:
                row.extend(["", "", ""])
        
        output_rows.append(row)
    
    # Create output directory if it doesn't exist
    os.makedirs('research_output', exist_ok=True)
    
    # Create DataFrame and save to CSV
    output_df = pd.DataFrame(output_rows)
    file_name = os.path.splitext(args.file)[0]
    output_file = f"research_output/research_pacf_{file_name}.csv"
    output_df.to_csv(output_file, index=False, header=False)
    print(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()