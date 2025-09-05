import pandas as pd
import numpy as np

# Define date range constants
START_DATE = '2015-01-01'
END_DATE = '2025-08-31'

# Feature mode: 'basic' for 4 DJIA features only, 'full' for all market features
FEATURE_MODE = 'full'  # Change to 'basic' to use only DJI OHLC

# Create business day date range
unique_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
unique_dates = unique_dates.to_pydatetime()
unique_dates = np.array(unique_dates)

file_paths = ['BAMLCC0A4BBBTRIV.csv', 'BAMLCC0A0CMTRIV.csv', 'BAMLCC0A1AAATRIV.csv', 'BAMLHYH0A3CMTRIV.csv', 'DGS10.csv', 'DGS30.csv']
dfs = [pd.read_csv(file) for file in file_paths]

merged_df = pd.concat(dfs)
merged_df.sort_values(by='observation_date', inplace=True)
merged_df = merged_df.groupby('observation_date').mean().reset_index()
merged_df = merged_df[(merged_df['observation_date'] >= START_DATE) & (merged_df['observation_date'] <= END_DATE)]
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.rename(columns={'observation_date': 'Date'})
merged_df['Date'] = pd.to_datetime(merged_df['Date'])
csv_files = ['^DJI.csv', 'xauusd_d.csv', '^VIX.csv', '^GSPC.csv']
# csv_dfs = [pd.read_csv(file, parse_dates=['Date']) for file in csv_files]

def rename_columns_with_prefix(df, prefix):
    new_cols = {}
    for c in df.columns:
        if c != 'Date':
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)

csv_dfs = []
for file in csv_files:
    # ex: '^DJI.csv' -> 'DJI' as prefix
    prefix = file.replace('.csv', '').replace('^','').replace('_d','')
    
    tmp_df = pd.read_csv(file, parse_dates=['Date'])
    tmp_df = rename_columns_with_prefix(tmp_df, prefix)
    csv_dfs.append(tmp_df)

for csv_df in csv_dfs:
    merged_df = pd.merge(merged_df, csv_df, on='Date', how='left')

merged_df = merged_df.drop(columns=['VIX_Volume']) # VIX_Volume is all 0
unique_dates_pd = pd.to_datetime(unique_dates)
merged_df.set_index('Date', inplace=True)
merged_df = merged_df.reindex(unique_dates_pd)

merged_df = merged_df.reset_index()
merged_df = merged_df.rename(columns={'index': 'Date'})

# Fill in missing values
# Replace all zeros with NaN (as zero values are unrealistic for market data)
merged_df = merged_df.replace(0, np.nan)
merged_df_filled = merged_df.ffill().bfill()
assert not merged_df_filled.isnull().any().any(), "There are still NaN in merged_df_filled"

num_days = len(merged_df_filled['Date'].unique())

# Determine which features to use based on mode
if FEATURE_MODE == 'basic':
    # Basic mode: only DJI OHLC
    feature_columns = ['DJI_Open', 'DJI_High', 'DJI_Low', 'DJI_Close']
    selected_data = merged_df_filled[feature_columns]
else:
    # Full mode: all market data
    feature_columns = [col for col in merged_df_filled.columns if col != 'Date']
    selected_data = merged_df_filled.drop(columns='Date')

num_MSU_features = len(feature_columns)

print(f"\n=== MARKET DATA FEATURES ===")
print(f"Feature Mode: {FEATURE_MODE}")
print(f"Total features: {num_MSU_features}")
print(f"Feature names (in order):")
for i, name in enumerate(feature_columns, 1):
    print(f"  {i:2d}. {name}")
print("=" * 30)

reshaped_data_new = selected_data.to_numpy().reshape(num_days, num_MSU_features)

output_file = 'market_data.npy'
np.save(output_file, reshaped_data_new)

print(f"\n=== MARKET DATA PROCESSING COMPLETE ===")
print(f"Shape: {reshaped_data_new.shape}")
print(f"Saved to: {output_file}")
print("=" * 40)