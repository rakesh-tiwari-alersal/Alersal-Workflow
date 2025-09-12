import argparse
import pandas as pd
import os
import numpy as np

def yule_walker_solver(series, order):
    """
    Solves the Yule-Walker equations to find the autoregressive coefficients.
    This is a custom implementation as statsmodels is not available.
    """
    n = len(series)
    if n <= order:
        return np.array([])
    
    # Calculate autocovariance
    autocov = np.zeros(order + 1)
    for lag in range(order + 1):
        autocov[lag] = np.sum(series[lag:] * series[:n-lag]) / n
    
    if autocov[0] == 0:
        return np.zeros(order)
    
    # Build R matrix and r vector
    R = np.zeros((order, order))
    r = autocov[1:]
    
    for i in range(order):
        for j in range(order):
            R[i, j] = autocov[abs(i - j)]
            
    # Solve for phi (coefficients)
    try:
        phi = np.linalg.solve(R, r)
    except np.linalg.LinAlgError:
        print("Warning: Could not solve Yule-Walker equations. Matrix is singular.")
        return np.zeros(order)

    return phi

def main():
    """
    Main function to parse arguments and generate Yule-Walker coefficients.
    """
    TABLE_CYCLES = [
        179, 183, 189, 196, 202, 206, 220, 237,
        243, 250, 260, 268, 273, 291, 308, 314,
        322, 331, 345, 355, 362, 368, 385, 403,
        408, 416, 426, 439, 457, 470, 480, 487,
        493, 510, 528, 534, 541, 551, 564, 582,
        605, 622, 636, 645, 653, 659, 676
    ]

    parser = argparse.ArgumentParser(description="Generate Yule-Walker coefficients for a specified historical data file.")
    
    parser.add_argument(
        '-f', '--file', 
        type=str, 
        required=True,
        help='The input CSV file (e.g., BTC-USD.csv) to analyze from the historical_data/ folder.'
    )
    
    parser.add_argument(
        "-p", "--plastic_cycles", 
        action='store_true',
        help="If set, only display coefficients for lags that are also in the Plastic Cycles table."
    )
    
    parser.add_argument('begin_range', type=int, help="Start of the coefficient range (e.g., 190).")
    parser.add_argument('end_range', type=int, help="End of the coefficient range (e.g., 250).")
    parser.add_argument('-d', '--differencing_lag', type=int, default=1, 
                        help="Lag for differencing (e.g., 1 for Yt - Yt-1). Default is 1.")

    args = parser.parse_args()

    # Construct the full file path and load the historical data
    file_path = os.path.join('historical_data', args.file)
    
    try:
        # Read the data from the 7th column
        df = pd.read_csv(file_path)
        time_series = df.iloc[:, 6]
        time_series = time_series.dropna()
    except FileNotFoundError:
        print(f"Error: The file {file_path} was not found.")
        return
    except IndexError:
        print(f"Error: Column 7 does not exist in the file.")
        return

    # Apply differencing
    differenced_series = time_series.diff(periods=args.differencing_lag).dropna()
    
    # Determine the order for the Yule-Walker solver
    # The order must be at least the end_range to compute coefficients up to that point.
    ar_order = args.end_range

    # Calculate Yule-Walker coefficients for the full specified range
    ar_coeffs = yule_walker_solver(differenced_series, ar_order)
    
    if len(ar_coeffs) == 0:
        print("Could not compute coefficients. Check data and parameters.")
        return
    
    # Get coefficients within the specified range
    full_results = {}
    for lag in range(args.begin_range, args.end_range + 1):
        if lag - 1 < len(ar_coeffs):
            full_results[lag] = ar_coeffs[lag - 1]
    
    # Filter the results if the plastic_cycles flag is set
    display_results = {}
    if args.plastic_cycles:
        lags_to_display = [lag for lag in TABLE_CYCLES if args.begin_range <= lag <= args.end_range]
        for lag in lags_to_display:
            if lag in full_results:
                display_results[lag] = full_results[lag]
    else:
        display_results = full_results

    # Output all coefficients in the range
    print(f"Yule-Walker Coefficients for Lags {args.begin_range} to {args.end_range}:")
    if args.plastic_cycles:
        print("Note: Only displaying coefficients for Plastic Cycles within the range.")
        
    for lag, coef in sorted(display_results.items()):
        print(f"Lag {lag}: {coef:.4f}")
        
    # Find and print the 3 most significant lags from the DISPLAYED results
    if display_results:
        sorted_lags = sorted(display_results.keys(), key=lambda k: abs(display_results[k]), reverse=True)
        top_3_lags = sorted_lags[:3]
        
        print("\nSummary: 3 Most Significant Lags:")
        for lag in top_3_lags:
            print(f"Lag {lag}: {display_results[lag]:.4f}")
    else:
        print("\nNo coefficients found in the specified range.")

if __name__ == "__main__":
    main()