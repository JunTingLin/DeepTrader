"""
DeepTrader Taiwan Market Data Processing using FinLab API
Process 0050 ETF data with adjusted (restored) prices
Supports 'basic' mode (0050 OHLC only) and 'full' mode (0050 + bonds + TWDUSD)
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
START_DATE = '2008-01-01'
END_DATE = '2025-12-31'

# Feature mode: 'basic' for 0050 OHLC only, 'full' for all market features
FEATURE_MODE = 'basic'


# Function to process bond data files
def process_bond_file(file_path):
    """Process bond CSV file and return DataFrame with Date, Open, High, Low, Close"""
    try:
        # Try different approaches to read the file
        try:
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                df = pd.read_csv(file_path, encoding='utf-8', quotechar='"')
            except:
                df = pd.read_csv(file_path, encoding='big5')

        # Rename columns from Chinese to English
        column_mapping = {
            '日期': 'Date',
            '收市': 'Close',
            '開市': 'Open',
            '高': 'High',
            '低': 'Low',
            '升跌（%）': 'Change_Pct'
        }

        rename_dict = {col: new_col for col, new_col in column_mapping.items() if col in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)

        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df = df.dropna(subset=['Date'])

        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        columns_to_select = ['Date']
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                columns_to_select.append(col)

        df = df[columns_to_select]
        df = df.sort_values(by='Date')

        return df

    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])


def concat_bond_files(file1, file2):
    """Concatenate two bond CSV files"""
    df1 = process_bond_file(file1)
    df2 = process_bond_file(file2)
    return pd.concat([df1, df2], ignore_index=True).sort_values(by='Date')


def rename_columns_with_prefix(df, prefix):
    """Rename columns with prefix (except Date)"""
    new_cols = {}
    for c in df.columns:
        if c != 'Date':
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)


print(f"=== DeepTrader Market Data Processing (FinLab) ===")
print(f"Processing 0050 ETF (Taiwan 50 Index)")
print(f"Date range: {START_DATE} to {END_DATE}")
print(f"Feature mode: {FEATURE_MODE}")
print(f"Using FinLab API with adjusted prices for 0050")
print("=" * 60)

# Get script directory for file paths
script_dir = os.path.dirname(os.path.abspath(__file__))

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

except Exception as e:
    print(f"Error fetching 0050 data: {e}")
    raise RuntimeError(f"Failed to fetch 0050 data from FinLab: {e}")

# Process additional market data for full mode
if FEATURE_MODE == 'full':
    print("\n=== Loading additional market data for full mode ===")

    # Read TWD/USD exchange rate data
    twdusd_path = os.path.join(script_dir, 'TWDUSD=X.csv')
    twdusd_df = pd.read_csv(twdusd_path, parse_dates=['Date'])
    if 'Volume' in twdusd_df.columns:
        twdusd_df = twdusd_df.drop(columns=['Volume'])
    twdusd_df = rename_columns_with_prefix(twdusd_df, 'TWDUSD')
    print(f"Loaded TWDUSD exchange rate: {len(twdusd_df)} rows")

    # Initialize list of dataframes to merge
    dfs_to_merge = [twdusd_df]

    # Process bond data
    try:
        # 5-year bonds
        tw5y_df = concat_bond_files(
            os.path.join(script_dir, 'tw5yearbonds_1.csv'),
            os.path.join(script_dir, 'tw5yearbonds_2.csv')
        )
        if not tw5y_df.empty:
            tw5y_df = rename_columns_with_prefix(tw5y_df, 'TW5Y')
            dfs_to_merge.append(tw5y_df)
            print(f"Loaded TW5Y bonds: {len(tw5y_df)} rows")

        # 10-year bonds
        tw10y_df = concat_bond_files(
            os.path.join(script_dir, 'tw10yearbonds_1.csv'),
            os.path.join(script_dir, 'tw10yearbonds_2.csv')
        )
        if not tw10y_df.empty:
            tw10y_df = rename_columns_with_prefix(tw10y_df, 'TW10Y')
            dfs_to_merge.append(tw10y_df)
            print(f"Loaded TW10Y bonds: {len(tw10y_df)} rows")

        # 20-year bonds
        tw20y_df = concat_bond_files(
            os.path.join(script_dir, 'tw20yearbonds_1.csv'),
            os.path.join(script_dir, 'tw20yearbonds_2.csv')
        )
        if not tw20y_df.empty:
            tw20y_df = rename_columns_with_prefix(tw20y_df, 'TW20Y')
            dfs_to_merge.append(tw20y_df)
            print(f"Loaded TW20Y bonds: {len(tw20y_df)} rows")

        # 30-year bonds
        tw30y_path = os.path.join(script_dir, 'tw30yearbonds.csv')
        if os.path.exists(tw30y_path):
            tw30y_df = process_bond_file(tw30y_path)
            if not tw30y_df.empty:
                tw30y_df = rename_columns_with_prefix(tw30y_df, 'TW30Y')
                dfs_to_merge.append(tw30y_df)
                print(f"Loaded TW30Y bonds: {len(tw30y_df)} rows")

    except Exception as e:
        print(f"Error processing bond files: {e}")

    # Merge all additional data with 0050 data
    for df in dfs_to_merge:
        market_df = pd.merge(market_df, df, on='Date', how='left')

    # Reindex to ensure alignment with 0050 trading dates
    unique_dates_pd = pd.to_datetime(unique_dates)
    market_df.set_index('Date', inplace=True)
    market_df = market_df.reindex(unique_dates_pd)
    market_df = market_df.reset_index()
    market_df = market_df.rename(columns={'index': 'Date'})

    # Fill missing values
    market_df = market_df.replace(0, np.nan)
    market_df = market_df.ffill().bfill()

    if market_df.isnull().any().any():
        print(f"Warning: Some NaN values remain after filling")
        print(market_df.isnull().sum())
    else:
        print(f"All data filled successfully")

# Determine which features to use based on mode
if FEATURE_MODE == 'basic':
    feature_columns = ['0050_Open', '0050_High', '0050_Low', '0050_Close']
else:
    # Full mode: all columns except Date
    feature_columns = [col for col in market_df.columns if col != 'Date']

selected_data = market_df[feature_columns]
num_MSU_features = len(feature_columns)
num_days = len(market_df)

print(f"\n=== MARKET DATA FEATURES ===")
print(f"Feature mode: {FEATURE_MODE}")
print(f"Total features: {num_MSU_features}")
print(f"Feature names (in order):")
for i, name in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {name}")
print("=" * 30)

# Convert to numpy array and reshape
reshaped_data = selected_data.to_numpy()
if len(reshaped_data.shape) == 2:
    reshaped_data = reshaped_data.reshape(num_days, num_MSU_features)

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
print(f"0050 uses adjusted (restored) prices from FinLab API")
if FEATURE_MODE == 'full':
    print(f"Bond/FX data uses raw prices from local CSV files")
print("=" * 40)
