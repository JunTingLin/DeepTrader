import pandas as pd
import numpy as np
import yfinance as yf

# Define date range constants (matching deeptrader_data_us_mp_fill.py)
START_DATE = '2013-01-01'
END_DATE = '2026-02-28'
DOWNLOAD_START_DATE = '2010-01-01'  # Extra data for buffer
DOWNLOAD_END_DATE = '2026-03-01'   # yfinance end_date is exclusive

# Feature mode: 'basic' for DIA OHLC (4 features), 'full' for all market features
FEATURE_MODE = 'basic'  # Options: 'basic', 'full'

# Download DIA ETF data from yfinance (used for trading dates and basic mode)
print(f"Downloading DIA ETF data from yfinance...")
dia_data = yf.download('DIA', start=DOWNLOAD_START_DATE, end=DOWNLOAD_END_DATE, auto_adjust=False, progress=False)

if dia_data.empty:
    raise ValueError("Cannot download DIA ETF data from yfinance")

dia_data.reset_index(inplace=True)
dia_data.columns = dia_data.columns.droplevel(level=1)

# Calculate adjusted OHLC using adjustment factor from Adj Close / Close
# This accounts for dividends and stock splits
adj_factor = dia_data['Adj Close'] / dia_data['Close']
dia_data['Open'] = dia_data['Open'] * adj_factor
dia_data['High'] = dia_data['High'] * adj_factor
dia_data['Low'] = dia_data['Low'] * adj_factor
dia_data['Close'] = dia_data['Adj Close']  # Use Adj Close as Close

# Drop Adj Close column as it's no longer needed
dia_data = dia_data.drop(columns=['Adj Close'])

# Filter to the specified date range
dia_data['Date'] = pd.to_datetime(dia_data['Date'])
dia_data = dia_data[(dia_data['Date'] >= START_DATE) & (dia_data['Date'] <= END_DATE)]
dia_data = dia_data.sort_values('Date')
dia_data = dia_data.reset_index(drop=True)

# Get unique trading dates
unique_dates = dia_data['Date'].dt.to_pydatetime()
unique_dates = np.array(unique_dates)
print(f"Using {len(unique_dates)} actual trading days from DIA ETF")
print(f"Date range: {unique_dates[0]} to {unique_dates[-1]}")

# Save trading dates for consistency with stocks data
np.save('trading_dates.npy', unique_dates)
print(f"Saved trading dates to trading_dates.npy")

# Also save DIA.csv for reference
dia_data.to_csv('DIA.csv', index=False)
print(f"Saved DIA.csv for reference")

num_days = len(dia_data)

# Determine which features to use based on mode
if FEATURE_MODE == 'basic':
    # Basic mode: OHLC (4 features) - adjusted DIA prices
    feature_columns = ['Open', 'High', 'Low', 'Close']
    selected_data = dia_data[feature_columns].copy()
elif FEATURE_MODE == 'full':
    # Full mode: all market data (from local CSV files)
    # Load additional market data files
    file_paths = ['BAMLCC0A4BBBTRIV.csv', 'BAMLCC0A0CMTRIV.csv', 'BAMLCC0A1AAATRIV.csv',
                  'BAMLHYH0A3CMTRIV.csv', 'DGS10.csv', 'DGS30.csv']
    dfs = [pd.read_csv(file) for file in file_paths]

    merged_df = pd.concat(dfs)
    merged_df.sort_values(by='observation_date', inplace=True)
    merged_df = merged_df.groupby('observation_date').mean().reset_index()
    merged_df = merged_df[(merged_df['observation_date'] >= START_DATE) & (merged_df['observation_date'] <= END_DATE)]
    merged_df = merged_df.reset_index(drop=True)
    merged_df = merged_df.rename(columns={'observation_date': 'Date'})
    merged_df['Date'] = pd.to_datetime(merged_df['Date'])

    # Additional CSV files for full mode
    csv_files = ['xauusd_d.csv', '^VIX.csv', '^GSPC.csv']

    def rename_columns_with_prefix(df, prefix):
        new_cols = {}
        for c in df.columns:
            if c != 'Date':
                new_cols[c] = f"{prefix}_{c}"
        return df.rename(columns=new_cols)

    # Add DIA data with prefix
    dia_prefixed = rename_columns_with_prefix(dia_data.copy(), 'DIA')
    merged_df = pd.merge(merged_df, dia_prefixed, on='Date', how='left')

    # Add other market data
    for file in csv_files:
        prefix = file.replace('.csv', '').replace('^', '').replace('_d', '')
        tmp_df = pd.read_csv(file, parse_dates=['Date'])
        tmp_df = rename_columns_with_prefix(tmp_df, prefix)
        merged_df = pd.merge(merged_df, tmp_df, on='Date', how='left')

    # Drop VIX_Volume if exists (usually all zeros)
    if 'VIX_Volume' in merged_df.columns:
        merged_df = merged_df.drop(columns=['VIX_Volume'])

    # Reindex to match unique_dates
    unique_dates_pd = pd.to_datetime(unique_dates)
    merged_df.set_index('Date', inplace=True)
    merged_df = merged_df.reindex(unique_dates_pd)
    merged_df = merged_df.reset_index()
    merged_df = merged_df.rename(columns={'index': 'Date'})

    # Replace zeros with NaN
    merged_df = merged_df.replace(0, np.nan)
    merged_df = merged_df.ffill().bfill()

    feature_columns = [col for col in merged_df.columns if col != 'Date']
    selected_data = merged_df.drop(columns='Date')
else:
    raise ValueError(f"Unsupported feature mode: {FEATURE_MODE}")

num_MSU_features = len(feature_columns)

print(f"\n=== MARKET DATA FEATURES ===")
print(f"Feature Mode: {FEATURE_MODE}")
print(f"Total features: {num_MSU_features}")
print(f"Feature names (in order):")
for i, name in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {name}")
print("=" * 30)

# Fill any NaN values
selected_data = selected_data.ffill().bfill()
assert not selected_data.isnull().any().any(), "There are still NaN in selected_data"

reshaped_data_new = selected_data.to_numpy().reshape(num_days, num_MSU_features)

output_file = 'market_data.npy'
np.save(output_file, reshaped_data_new)

print(f"\n=== MARKET DATA PROCESSING COMPLETE ===")
print(f"Shape: {reshaped_data_new.shape}")
print(f"Saved to: {output_file}")
print("=" * 40)
