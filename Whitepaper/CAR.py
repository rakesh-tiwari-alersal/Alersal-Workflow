import pandas as pd
import numpy as np
from scipy.stats import binomtest
import os
import argparse

# Universal cycle table parameters
TABLE_CYCLES = [
    179, 183, 189, 196, 202, 206, 220, 237,
    243, 250, 260, 268, 273, 291, 308, 314,
    322, 331, 345, 355, 362, 368, 385, 403,
    408, 416, 426, 439, 457, 470, 480, 487,
    493, 510, 528, 534, 541, 551, 564, 582,
    605, 622, 636, 645, 653, 659, 676
]
MIN_PERIOD = 175
MAX_PERIOD = 680
METHOD_RESULT_PATHS = {
    'psd': "psd_results/match_psd_results_{asset_class}.csv",
    'dft': "dft_results/match_dft_results_{asset_class}.csv",
    'pacf': "pacf_results/match_pacf_results_{asset_class}.csv",
    'wavelet': "wavelet_results/match_wavelet_results_{asset_class}.csv"  # Added wavelet
}

def calculate_car(results_df, method='psd', tolerance=2):
    """
    Calculate Coverage Acceptance Ratio (CAR) with statistical significance
    Handles multiple analysis methods (PSD, DFT, PACF, WAVELET)

    Args:
        results_df: DataFrame with analysis results
        method: Analysis method name
        tolerance: Tolerance for cycle matching
    """
    # Convert delta columns to numeric, handling missing values
    for col in ['Cycle1_Delta', 'Cycle2_Delta']:
        results_df[col] = pd.to_numeric(results_df[col], errors='coerce')
    
    # Remove rows where both cycles are NaN (no cycle detected)
    # Use .copy() to avoid SettingWithCopyWarning
    results_df = results_df.dropna(subset=['Cycle1_Delta', 'Cycle2_Delta'], how='all').copy()
    
    # Identify covered instruments (at least one delta <= tolerance)
    # Use .loc to avoid SettingWithCopyWarning
    results_df.loc[:, 'Covered'] = results_df.apply(
        lambda row: (row['Cycle1_Delta'] <= tolerance) or 
                   (pd.notna(row['Cycle2_Delta']) and 
                   (row['Cycle2_Delta'] <= tolerance)), 
        axis=1
    )
    
    # Calculate CAR based on instruments with at least one cycle detected
    covered = results_df['Covered'].sum()
    processed_count = len(results_df)
    car = covered / processed_count if processed_count > 0 else 0.0
    
    # Calculate exact covered days with merging
    intervals = []
    for c in TABLE_CYCLES:
        low = max(MIN_PERIOD, c - tolerance)
        high = min(MAX_PERIOD, c + tolerance)
        intervals.append((low, high))
    
    # Merge overlapping intervals
    intervals.sort()
    merged = []
    if intervals:
        start, end = intervals[0]
        for s, e in intervals[1:]:
            if s <= end + 1:  # Overlapping or adjacent
                end = max(end, e)
            else:
                merged.append((start, end))
                start, end = s, e
        merged.append((start, end))
    
    # Calculate actual covered days
    total_covered_days = sum(e - s + 1 for s, e in merged)
    coverage_range = MAX_PERIOD - MIN_PERIOD + 1
    
    # Conservative probability model
    p_single_match = total_covered_days / coverage_range
    p_at_least_one = min(1.0, 1 - (1 - p_single_match)**2)
    
    # Binomial test uses processed_count (instruments with cycles detected) as n
    binom_result = binomtest(covered, processed_count, p_at_least_one, alternative='greater')
    
    # Calculate excess coverage
    expected_random = processed_count * p_at_least_one
    excess_coverage = car - p_at_least_one
    
    # Calculate z-score
    std_dev = np.sqrt(processed_count * p_at_least_one * (1 - p_at_least_one))
    z_score = (covered - expected_random) / std_dev if std_dev > 0 else 0
    
    return {
        'method': method,
        'car': car,
        'covered': covered,
        'processed': processed_count,
        'p_value': binom_result.pvalue,
        'z_score': z_score,
        'excess_coverage': excess_coverage,
        'expected_random': expected_random,
        'p_random_per_cycle': p_single_match,
        'p_at_least_one_match': p_at_least_one
    }

# Load analysis results for specified method
def load_results(method='psd', asset_classes=['eq', 'ix', 'cr', 'co', 'fx']):
    """Load analysis results from CSV files for specified method"""
    all_results = []
    for asset_class in asset_classes:
        try:
            file_path = METHOD_RESULT_PATHS[method].format(asset_class=asset_class)
            if os.path.exists(file_path):
                asset_df = pd.read_csv(file_path)
                all_results.append(asset_df)
        except KeyError:
            continue
    
    if not all_results:
        raise FileNotFoundError(f"No result files found for method: {method}")
    
    return pd.concat(all_results, ignore_index=True)

# Main execution
if __name__ == "__main__":
    # Configure command-line arguments
    parser = argparse.ArgumentParser(description='Calculate CAR for cycle analysis methods')
    parser.add_argument('method', nargs='?', default='psd', choices=['psd', 'dft', 'pacf', 'wavelet'],
                        help='Analysis method (psd, dft, pacf, wavelet). Default: psd')
    parser.add_argument('-t', '--tolerance', type=int, default=2, choices=[1,2,3],
                        help='Tolerance value for cycle matching (1, 2, or 3). Default: 2')
    args = parser.parse_args()

    # Load and process specified method
    try:
        results_df = load_results(args.method)
        car_result = calculate_car(results_df, args.method, args.tolerance)
        
        # Print results
        print(f"{args.method.upper()} Results (tolerance={args.tolerance}):")
        print(f"Instruments with cycles detected: {car_result['processed']}")
        print(f"CAR: {car_result['car']:.2%} ({car_result['covered']} instruments)")
        print(f"Expected random coverage: {car_result['expected_random']:.1f} instruments")
        print(f"Excess coverage: {car_result['excess_coverage']:.2%} points")
        print(f"Statistical significance: z={car_result['z_score']:.2f}, p={car_result['p_value']:.2e}")
        
    except Exception as e:
        print(f"Error processing {args.method}: {str(e)}")