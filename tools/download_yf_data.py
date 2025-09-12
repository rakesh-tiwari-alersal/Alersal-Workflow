import yfinance as yf
import pandas as pd
import csv
import os
import sys
from datetime import datetime, date, timedelta

# Handle command-line arguments
start_year = 1990
end_year = 2024

if len(sys.argv) == 3:
    try:
        start_year = int(sys.argv[1])
        end_year = int(sys.argv[2])
    except ValueError:
        print("Error: Both arguments must be integers")
        sys.exit(1)
elif len(sys.argv) > 1:
    print("Error: Requires either 0 or 2 arguments (start_year end_year)")
    sys.exit(1)

# Calculate date ranges
start_date = f"{start_year}-01-01"
end_date = f"{end_year}-12-31"

# Adjust end date to yesterday if in future
end_datetime = datetime.strptime(end_date, "%Y-%m-%d").date()
if end_datetime >= date.today():
    end_date = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

print(f"Downloading data from {start_date} to {end_date}")

# Create directories
os.makedirs("historical_data", exist_ok=True)
os.makedirs("download_logs", exist_ok=True)

# Read instrument data
instruments = []
with open('instrument_data.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        instruments.append(row)

def download_instrument_data(ticker, instrument_name, start_date, end_date):
    """Download historical data for a single instrument"""
    safe_ticker = ticker.replace('^', '').replace('=', '').replace('/', '_')
    filepath = f"historical_data/{safe_ticker}.csv"
    
    # Skip if file exists
    if os.path.exists(filepath):
        print(f"Warning: File exists - skipping {ticker} ({instrument_name})")
        return "Skipped", "File already exists"
    
    try:
        print(f"Downloading {ticker} ({instrument_name})...")
        
        # Download data with 3 retries
        data = None
        for attempt in range(3):
            try:
                # Set auto_adjust=False to get raw prices
                data = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=False  # Get raw prices
                )
                if data.empty:
                    continue
                break
            except Exception as e:
                if attempt == 2:
                    raise e
                print(f"  Retry {attempt+1}/3 for {ticker}")
        
        if data is None or data.empty:
            return "Failed", "No data available"
        
        # FIXED: More robust column handling
        # Handle columns explicitly to prevent Adj Close contamination
        if 'Close' in data.columns:
            # Standard columns when auto_adjust=False
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
        elif 'Adj Close' in data.columns:
            # Explicitly avoid Adj Close contamination
            # Use raw Close if available, otherwise fail
            if 'Close' not in data.columns:
                return "Failed", "Close column missing"
            data = data[['Open', 'High', 'Low', 'Close', 'Volume']]
            data.columns = ['open', 'high', 'low', 'close', 'volume']
        else:
            # Fallback with explicit column validation
            if len(data.columns) < 5:
                return "Failed", f"Not enough columns: {data.columns.tolist()}"
            data = data.iloc[:, :5]
            data.columns = ['open', 'high', 'low', 'close', 'volume'][:len(data.columns)]
        
        # Process data
        data = data.reset_index()
        data['Date'] = pd.to_datetime(data['Date']).dt.strftime('%Y-%m-%d')
        data['ticker'] = ticker
        data['instrument'] = instrument_name
        data = data[['Date', 'ticker', 'instrument', 'open', 'high', 'low', 'close', 'volume']]
        
        # Save to CSV
        data.to_csv(filepath, index=False)
        return "Success", "Downloaded"
    
    except Exception as e:
        return "Failed", str(e)

# Download all instruments
log_entries = []
for inst in instruments:
    status, message = download_instrument_data(
        inst['Ticker'],
        inst['Instrument'],
        start_date,
        end_date
    )
    log_entries.append({
        'Category': inst['Category'],
        'Instrument': inst['Instrument'],
        'Ticker': inst['Ticker'],
        'Status': status,
        'Message': message
    })

# Save download log
log_filename = f"download_logs/download_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
log_df = pd.DataFrame(log_entries)
log_df.to_csv(log_filename, index=False)

# Print summary
status_counts = log_df['Status'].value_counts()
print("\nDownload summary:")
for status, count in status_counts.items():
    print(f"{status}: {count}/{len(log_entries)}")
print(f"Log saved to {log_filename}")