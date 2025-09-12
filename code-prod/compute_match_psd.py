import pandas as pd
import numpy as np
import os
import csv
import time
import sys
from scipy.signal import welch, find_peaks, periodogram

# Validate command-line arguments
if len(sys.argv) < 2:
    print("Error: Asset class argument required (eq/ix/cr/co/fx)")
    print("Usage: python compute_match_psd.py <asset_class>")
    sys.exit(1)

asset_class = sys.argv[1].lower()
valid_classes = ['eq', 'ix', 'cr', 'co', 'fx']
if asset_class not in valid_classes:
    print(f"Error: Invalid asset class '{asset_class}'. Valid options: {', '.join(valid_classes)}")
    sys.exit(1)

# Unified cycle reference table
TABLE_CYCLES = [
    179, 183, 189, 196, 202, 206, 220, 237,
    243, 250, 260, 268, 273, 291, 308, 314,
    322, 331, 345, 355, 362, 368, 385, 403,
    408, 416, 426, 439, 457, 470, 480, 487,
    493, 510, 528, 534, 541, 551, 564, 582,
    605, 622, 636, 645, 653, 659, 676
]

# Configuration
DATA_DIR = "historical_data"
OUTPUT_DIR = "psd_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set range for universal cycle detection
MIN_PERIOD = 175
MAX_PERIOD = 680

# Output files
final_output_path = os.path.join(OUTPUT_DIR, f"match_psd_results_{asset_class}.csv")
log_path = os.path.join(OUTPUT_DIR, "psd_processing_log.csv")

# Initialize log file only if it doesn't exist
if not os.path.exists(log_path):
    with open(log_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Timestamp', 'AssetClass', 'Category', 'Instrument', 'Ticker', 'Status', 'Message'])

# Initialize final output CSV
with open(final_output_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'Category', 'Instrument', 'Ticker',
        'Cycle1', 'Cycle1_Match', 'Cycle1_Delta',
        'Cycle2', 'Cycle2_Match', 'Cycle2_Delta'
    ])

# Load instrument metadata
instruments = []
metadata_file = f'instrument_data_{asset_class}.csv'
try:
    with open(metadata_file, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            instruments.append(row)
    print(f"Loaded {len(instruments)} instruments from {metadata_file}")
except FileNotFoundError:
    print(f"Error: Metadata file not found - {metadata_file}")
    sys.exit(1)

print(f"Starting PSD computation and cycle matching for {asset_class.upper()} instruments...")
print(f"Period range: {MIN_PERIOD}-{MAX_PERIOD} days")
start_time = time.time()
processed_count = 0

def log_entry(timestamp, asset_class, category, instrument, ticker, status, message):
    """Log processing status to central log file"""
    with open(log_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, asset_class, category, instrument, ticker, status, message])

def find_closest_cycle(period, cycle_table):
    """Find closest reference cycle and calculate delta"""
    if period is None or period == '':
        return None, None
    
    try:
        period_int = int(period)
        closest = min(cycle_table, key=lambda x: abs(x - period_int))
        delta = abs(period_int - closest)
        return closest, delta
    except (ValueError, TypeError):
        return None, None

def find_dominant_cycles(frequencies, psd, min_period=MIN_PERIOD, max_period=MAX_PERIOD):
    # Filter out zero frequency
    non_zero_mask = frequencies > 0
    frequencies = frequencies[non_zero_mask]
    psd = psd[non_zero_mask]
    
    if len(frequencies) == 0:
        return []
    
    # Convert frequencies to periods
    periods = 1 / frequencies
    
    # Apply asset-specific period range filter
    period_mask = (periods >= min_period) & (periods <= max_period)
    filtered_periods = periods[period_mask]
    filtered_psd = psd[period_mask]
    
    if len(filtered_psd) == 0:
        return []
    
    # Detect spectral peaks
    peak_indices, _ = find_peaks(filtered_psd, height=0.001 * np.max(filtered_psd), distance=1)
    
    if len(peak_indices) == 0:
        return []
    
    # Extract peak data
    peak_periods = filtered_periods[peak_indices]
    peak_powers = filtered_psd[peak_indices]
    
    # Sort peaks by power (descending)
    peaks_df = pd.DataFrame({'period': peak_periods, 'power': peak_powers})
    peaks_df = peaks_df.sort_values('power', ascending=False)
  
    # Get top 10 candidate cycles
    top_10_cycles = peaks_df.head(10)
    
    if top_10_cycles.empty:
        return []

    # ============== CYCLE SELECTION CORE LOGIC ==============
    MIN_CYCLE_DISTANCE = 71  # Minimum separation between cycles (days)
    all_peaks = []
    
    # Prepare peak objects for processing
    for _, row in top_10_cycles.iterrows():
        all_peaks.append({
            'period': row['period'],
            'power': row['power']
        })
    
    # Case 1: Fewer than 3 peaks - use all available
    if len(all_peaks) <= 2:
        candidate_periods = [p['period'] for p in all_peaks]
    
    # Case 2: More than 2 peaks - apply separation logic
    else:
        candidate_periods = []
        
        # Always include strongest peak
        candidate_periods.append(all_peaks[0]['period'])
        
        # Find next strongest peak meeting separation requirement
        found_valid = False
        for i in range(1, len(all_peaks)):
            valid = True
            # Check against all selected cycles
            for existing in candidate_periods:
                if abs(all_peaks[i]['period'] - existing) < MIN_CYCLE_DISTANCE:
                    valid = False
                    break
            if valid:
                candidate_periods.append(all_peaks[i]['period'])
                found_valid = True
                break
        
        # Fallback: use second strongest if no valid peak found
        if not found_valid:
            candidate_periods.append(all_peaks[1]['period'])
    
    # Return integer periods (max 2 cycles)
    return [int(round(p)) for p in candidate_periods[:2]]


# Main processing loop
for inst in instruments:
    instrument_start = time.time()
    ticker = inst['Ticker']
    category = inst['Category']
    instrument_name = inst['Instrument']
    safe_ticker = ticker.replace('^', '').replace('=', '').replace('/', '_')
    file_path = os.path.join(DATA_DIR, f"{safe_ticker}.csv")
    
    # Initialize result container
    result = {
        'Category': category,
        'Instrument': instrument_name,
        'Ticker': ticker,
        'Cycle1': '',
        'Cycle1_Match': '',
        'Cycle1_Delta': '',
        'Cycle2': '',
        'Cycle2_Match': '',
        'Cycle2_Delta': ''
    }
    
    status = "SUCCESS"
    message = ""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    
    print(f"[{processed_count+1}/{len(instruments)}] Processing {ticker}...")
    
    try:
        # Validate and load price data
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found: {os.path.abspath(file_path)}")
        
        data = pd.read_csv(file_path, parse_dates=['Date'])
        print(f"  Rows: {len(data)} | Period: {data['Date'].min().date()} to {data['Date'].max().date()}")
        
        data = data.sort_values('Date')
        
        # Validate data sufficiency
        if len(data) < 1000:
            raise ValueError(f"Insufficient data ({len(data)} rows < 1000)")
        
        # Calculate log returns
        closes = data['close'].values
        if np.any(closes <= 0):
            raise ValueError("Non-positive closing prices detected")
        
        log_returns = np.log(closes[1:]) - np.log(closes[:-1])
        
        # Compute power spectral density
        frequencies, psd = periodogram(log_returns, fs=1, scaling='density', window='hann')
        
        # Detect dominant market cycles
        peak_periods = find_dominant_cycles(frequencies, psd)
        print(f"  Found {len(peak_periods)} cycles: {peak_periods}")
        
        # Match to reference cycles
        if peak_periods:
            for i, period in enumerate(peak_periods):
                if i < 2:  # Store max 2 cycles
                    result[f'Cycle{i+1}'] = period
                    match, delta = find_closest_cycle(period, TABLE_CYCLES)
                    result[f'Cycle{i+1}_Match'] = match
                    result[f'Cycle{i+1}_Delta'] = delta
        
    except Exception as e:
        status = "ERROR"
        message = str(e)
        print(f"  {status}: {message}")
    
    # Log and save results
    log_entry(timestamp, asset_class, category, instrument_name, ticker, status, message)
    
    with open(final_output_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            result['Category'],
            result['Instrument'],
            result['Ticker'],
            result['Cycle1'],
            result['Cycle1_Match'],
            result['Cycle1_Delta'],
            result['Cycle2'],
            result['Cycle2_Match'],
            result['Cycle2_Delta']
        ])
    
    processed_count += 1
    instrument_time = time.time() - instrument_start
    elapsed = time.time() - start_time
    avg_time = elapsed / processed_count
    remaining = avg_time * (len(instruments) - processed_count)
    
    print(f"  Status: {status} | Time: {instrument_time:.2f}s | "
          f"ETA: {remaining//60:.0f}m {remaining%60:.0f}s")

print(f"\nCompleted {asset_class.upper()} instruments!")
print(f"Final results: {final_output_path}")
print(f"Logs: {log_path}")
print(f"Total time: {(time.time()-start_time)//60:.0f}m {(time.time()-start_time)%60:.0f}s")