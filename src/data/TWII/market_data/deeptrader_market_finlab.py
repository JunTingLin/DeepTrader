"""
DeepTrader Taiwan Market Data Processing using FinLab API
Process 0050 ETF data with adjusted (restored) prices
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables from parent directory
env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')
load_dotenv(env_path)

import finlab
from finlab import data

# Initialize FinLab with API key
FINLAB_API_KEY = os.getenv('FINLAB_API_KEY')
if not FINLAB_API_KEY:
    raise ValueError("FINLAB_API_KEY not found in .env file")

finlab.login(FINLAB_API_KEY)

# Define date range constants
START_DATE = '2013-01-01'
END_DATE = '2025-12-31'

print(f"=== DeepTrader Market Data Processing (FinLab) ===")
print(f"Processing 0050 ETF (Taiwan 50 Index)")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Features: 0050 OHLC (4 features)")
print(f"Using FinLab API with adjusted prices")
print("=" * 60)

# Get 0050 ETF data from FinLab
# FinLab directly provides adjusted OHLC
print("\nFetching 0050 ETF data from FinLab...")
try:
    # Stock code for 0050 is "0050"
    # Get adjusted OHLC directly from FinLab
    # Note: data.get() returns DataFrame with all stocks, then slice by stock_id and date range
    open_price = data.get('etl:adj_open')['0050'][START_DATE:END_DATE]
    high_price = data.get('etl:adj_high')['0050'][START_DATE:END_DATE]
    low_price = data.get('etl:adj_low')['0050'][START_DATE:END_DATE]
    close_price = data.get('etl:adj_close')['0050'][START_DATE:END_DATE]

    # Get trading dates from FinLab data (same as deeptrader_data_tw_finlab.py)
    unique_dates = close_price.index.to_pydatetime()
    unique_dates = np.array(unique_dates)
    print(f"Trading dates from FinLab: {len(unique_dates)} days ({unique_dates[0]} to {unique_dates[-1]})")

    # Create DataFrame
    market_df = pd.DataFrame({
        'Date': close_price.index,
        'Open': open_price.values,
        'High': high_price.values,
        'Low': low_price.values,
        'Close': close_price.values
    })

    # Rename columns with prefix
    market_df = market_df.rename(columns={
        'Open': '0050_Open',
        'High': '0050_High',
        'Low': '0050_Low',
        'Close': '0050_Close'
    })

    # Fill missing values (replace zeros with NaN, then forward/backward fill)
    market_df = market_df.replace(0, np.nan)
    market_df = market_df.ffill().bfill()

    # Check for remaining NaN values
    if market_df.isnull().any().any():
        print(f"Warning: Some NaN values remain in 0050 data")
        print(market_df.isnull().sum())
    else:
        print(f"Downloaded and cleaned 0050 data: {len(market_df)} days")

    num_days = len(market_df)

except Exception as e:
    print(f"Error fetching 0050 data: {e}")
    raise RuntimeError(f"Failed to fetch 0050 data from FinLab: {e}")

# Use 0050 OHLC features (4 features)
feature_columns = ['0050_Open', '0050_High', '0050_Low', '0050_Close']
selected_data = market_df[feature_columns]
num_MSU_features = len(feature_columns)

print(f"\n=== MARKET DATA FEATURES ===")
print(f"Total features: {num_MSU_features}")
print(f"Feature names (in order):")
for i, name in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {name}")
print("=" * 30)

# Convert to numpy array and reshape
reshaped_data = selected_data.to_numpy()
if len(reshaped_data.shape) == 2:
    reshaped_data = reshaped_data.reshape(num_days, num_MSU_features)

# Get script directory for output paths
script_dir = os.path.dirname(os.path.abspath(__file__))

# Save market data as npy
output_file = os.path.join(script_dir, 'market_data.npy')
np.save(output_file, reshaped_data)

# Save market data as CSV (for plot system compatibility)
csv_df = pd.DataFrame({
    'Date': close_price.index.strftime('%Y-%m-%d'),
    'Open': open_price.values,
    'High': high_price.values,
    'Low': low_price.values,
    'Close': close_price.values
})
csv_output_file = os.path.join(script_dir, '0050_finlab_adj.csv')
csv_df.to_csv(csv_output_file, index=False)

print(f"\n=== MARKET DATA PROCESSING COMPLETE ===")
print(f"NPY Shape: {reshaped_data.shape}")
print(f"Saved to: {output_file}")
print(f"Saved to: {csv_output_file} (for plot system)")
print(f"Using adjusted (restored) prices from FinLab API")
print("=" * 40)
