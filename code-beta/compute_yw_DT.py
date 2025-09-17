# compute_yw_DT.py
import argparse
import pandas as pd
import numpy as np
import os

def calculate_detrended_stats(series, lags):
    """
    Computes the detrended series and returns median and stdev of Ydt.
    
    Args:
        series (pd.Series): The time series data (raw prices).
        lags (list): A list of exactly 2 integer lags.
        
    Returns:
        tuple: (median, stdev) of the detrended series.
    """
    if len(lags) != 2:
        raise ValueError("Exactly 2 lags must be provided.")

    l1, l2 = lags
    max_lag = max(l1, l2)

    # Construct detrended series
    Y = series
    Ydt = (Y - Y.shift(l1) - Y.shift(l2)) / Y
    Ydt = Ydt.iloc[max_lag:]  # drop initial NaNs

    return np.median(Ydt.dropna()), np.std(Ydt.dropna(), ddof=1)

def main():
    """
    Main function to parse arguments, load data, and compute DT metrics.
    """
    parser = argparse.ArgumentParser(description='Compute detrended median and stdev for a given time series.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-l', '--lags', type=str, required=True, help='Comma-separated list of 2 integer lags (e.g., 24,291).')
    
    args = parser.parse_args()

    # Validate file path
    file_path = os.path.join('historical_data', args.file)
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Parse and validate lags
    try:
        lags = [int(lag.strip()) for lag in args.lags.split(',')]
        if len(lags) != 2:
            raise ValueError
    except (ValueError, IndexError):
        print("Error: Lags must be exactly 2 comma-separated integers.")
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

    # Calculate detrended metrics
    median_dt, stdev_dt = calculate_detrended_stats(series, lags)

    # Print output in required structure
    print(f"\n--- DT metrics for file '{args.file}' with lags {lags} ---")
    print(f"Median(Ydt): {median_dt:.6f}")
    print(f"Stdev(Ydt): {stdev_dt:.6f}")

if __name__ == "__main__":
    main()
