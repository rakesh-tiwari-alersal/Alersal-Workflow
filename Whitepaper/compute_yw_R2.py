import argparse
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

def calculate_oos_r_squared_poly(series, lags, degree):
    """
    Computes the out-of-sample R-squared for a given set of lags and polynomial degree
    using the scikit-learn library. This version matches the processing of Figure4_gold_model.py
    (raw prices, no intercept).
    
    Args:
        series (pd.Series): The time series data (raw prices).
        lags (list): A list of integer lags for the AR model.
        degree (int): The polynomial degree to use for the model features.
        
    Returns:
        float: The out-of-sample R-squared value.
    """
    max_lag = max(lags)
    
    # Create the base design matrices from lags
    X_base = pd.concat([series.shift(lag) for lag in lags], axis=1).iloc[max_lag:]
    y = series.iloc[max_lag:]

    # Build the full design matrix with polynomial features
    X = pd.DataFrame()
    for d in range(1, degree + 1):
        X = pd.concat([X, X_base**d], axis=1)

    # Align y with X
    y = y.loc[X.index]

    # Split data into training (80%) and testing (20%) sets
    split_index = int(len(y) * 0.8)
    X_train = X.iloc[:split_index]
    y_train = y.iloc[:split_index]
    X_test = X.iloc[split_index:]
    y_test = y.iloc[split_index:]
    
    # Handle NaNs
    X_train = X_train.dropna()
    y_train = y_train.loc[X_train.index]
    X_test = X_test.dropna()
    y_test = y_test.loc[X_test.index]

    if X_train.empty or X_test.empty:
        return np.nan

    # Fit the linear model to the training data (no intercept to match Figure4 script)
    model = LinearRegression(fit_intercept=False)
    model.fit(X_train, y_train)

    # Predict on the out-of-sample data
    y_predicted = model.predict(X_test)

    # Calculate out-of-sample R-squared
    r_squared = r2_score(y_test, y_predicted)
    return r_squared

def main():
    """
    Main function to parse arguments, load data, and calculate out-of-sample R-squared.
    """
    parser = argparse.ArgumentParser(description='Compute out-of-sample R-squared for a given time series.')
    parser.add_argument('-f', '--file', type=str, required=True, help='Input file name (e.g., GC=F.csv) located in historical_data/ folder.')
    parser.add_argument('-l', '--lags', type=str, required=True, help='Comma-separated list of 2 or 3 integer lags (e.g., 24,291,385).')
    
    args = parser.parse_args()

    # Validate file path
    file_path = os.path.join('historical_data', args.file)
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' does not exist.")
        return

    # Parse and validate lags
    try:
        lags = [int(lag.strip()) for lag in args.lags.split(',')]
        if not (2 <= len(lags) <= 3):
            raise ValueError
    except (ValueError, IndexError):
        print("Error: Lags must be 2 or 3 comma-separated integers.")
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

    # Use raw series 
    series_raw = series

    # Calculate and print R-squared for each model type
    print(f"\n--- R-squared for file '{args.file}' with lags {lags} ---")
    
    # Linear OLS (Degree 1)
    r_squared_1 = calculate_oos_r_squared_poly(series_raw, lags, 1)
    if not np.isnan(r_squared_1):
        print(f"Linear OLS (Polynomial Degree 1): {r_squared_1:.4f}")
    
    # Polynomial Degree 3
    r_squared_3 = calculate_oos_r_squared_poly(series_raw, lags, 3)
    if not np.isnan(r_squared_3):
        print(f"Polynomial Degree 3: {r_squared_3:.4f}")
        
    # Polynomial Degree 4
    r_squared_4 = calculate_oos_r_squared_poly(series_raw, lags, 4)
    if not np.isnan(r_squared_4):
        print(f"Polynomial Degree 4: {r_squared_4:.4f}")

if __name__ == "__main__":
    main()
