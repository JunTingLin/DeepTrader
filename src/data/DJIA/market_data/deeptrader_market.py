import pandas as pd
import numpy as np

# Create business day date range
unique_dates = pd.bdate_range(start='2000-01-01', end='2023-12-31')
unique_dates = unique_dates.to_pydatetime()
unique_dates = np.array(unique_dates)

file_paths = ['BAMLCC0A4BBBTRIV.xls', 'BAMLCC0A0CMTRIV.xls', 'BAMLCC0A1AAATRIV.xls', 'BAMLHYH0A3CMTRIV.xls', 'DGS10.xls', 'DGS30.xls']
dfs = [pd.read_excel(file) for file in file_paths]

merged_df = pd.concat(dfs)
merged_df.sort_values(by='observation_date', inplace=True)
merged_df = merged_df.groupby('observation_date').mean().reset_index()
merged_df = merged_df[(merged_df['observation_date'] >= '2000-01-03') & (merged_df['observation_date'] <= '2024-03-01')]
merged_df = merged_df.reset_index(drop=True)
merged_df = merged_df.rename(columns={'observation_date': 'Date'})
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
merged_df_filled = merged_df.fillna(method='ffill').fillna(method='bfill')
assert not merged_df_filled.isnull().any().any(), "There are still NaN in merged_df_filled"

num_days = len(merged_df_filled['Date'].unique())
# change here
# used_cols = ['DJI_Open', 'DJI_High', 'DJI_Low', 'DJI_Close']
# selected_data = merged_df_filled[used_cols]
# reshaped_data_new = selected_data.to_numpy()
num_MSU_features = merged_df_filled.shape[1] - 1
reshaped_data_new = merged_df_filled.drop(columns='Date').to_numpy().reshape(num_days, num_MSU_features)

output_file = 'market_data.npy'
np.save(output_file, reshaped_data_new)