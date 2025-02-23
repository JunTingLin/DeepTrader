import pandas as pd
import numpy as np

# 建立 business day 日期區間
unique_dates = pd.bdate_range(start='2000-01-01', end='2023-12-31')
unique_dates_pd = pd.to_datetime(unique_dates)

# 定義欄位重新命名的函式 (排除 Date 欄)
def rename_columns_with_prefix(df, prefix):
    new_cols = {}
    for c in df.columns:
        if c != 'Date':
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)

# 讀取 ^DJI.csv，並將 Date 欄解析為日期
file = '^DJI.csv'
prefix = file.replace('.csv', '').replace('^','').replace('_d','')
df = pd.read_csv(file, parse_dates=['Date'])
df = rename_columns_with_prefix(df, prefix)

# 篩選指定日期區間 (2000-01-03 至 2024-03-01)
df = df[(df['Date'] >= '2000-01-03') & (df['Date'] <= '2024-03-01')]

# 依據 business day 日期重建索引
df.set_index('Date', inplace=True)
df = df.reindex(unique_dates_pd)
df = df.reset_index().rename(columns={'index': 'Date'})

# 補值 (先 forward fill 再 back fill)
df_filled = df.fillna(method='ffill').fillna(method='bfill')
assert not df_filled.isnull().any().any(), "There are still NaN in df_filled"

# 轉換成 numpy array (不包含 Date 欄)
num_days = len(unique_dates_pd)
num_MSU_features = df_filled.shape[1] - 1
reshaped_data_new = df_filled.drop(columns='Date').to_numpy().reshape(num_days, num_MSU_features)

output_file = 'market_data.npy'
np.save(output_file, reshaped_data_new)