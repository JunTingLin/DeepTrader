import pandas as pd
import numpy as np

# Create business day date range
unique_dates = pd.bdate_range(start='2015-01-01', end='2025-03-31')
unique_dates = unique_dates.to_pydatetime()
unique_dates = np.array(unique_dates)

# Function to process bond data files
def process_bond_file(file_path):
    try:
        # Try different approaches to read the file
        try:
            # First attempt: standard CSV with UTF-8 encoding
            df = pd.read_csv(file_path, encoding='utf-8')
        except:
            try:
                # Second attempt: with quote character
                df = pd.read_csv(file_path, encoding='utf-8', quotechar='"')
            except:
                # Third attempt: with different encoding
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
        
        # Only rename columns that exist
        rename_dict = {col: new_col for col, new_col in column_mapping.items() if col in df.columns}
        if rename_dict:
            df = df.rename(columns=rename_dict)
        
        # Convert date to datetime
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        
        # Drop rows with invalid dates
        df = df.dropna(subset=['Date'])
        
        # Ensure numeric columns are numeric
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Select only required columns if they exist
        columns_to_select = ['Date']
        for col in ['Open', 'High', 'Low', 'Close']:
            if col in df.columns:
                columns_to_select.append(col)
        
        df = df[columns_to_select]
        
        # Sort by date
        df = df.sort_values(by='Date')
        
        return df
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        # Return empty DataFrame with expected columns
        return pd.DataFrame(columns=['Date', 'Open', 'High', 'Low', 'Close'])

# Function to concatenate split bond files
def concat_bond_files(file1, file2):
    df1 = process_bond_file(file1)
    df2 = process_bond_file(file2)
    return pd.concat([df1, df2], ignore_index=True).sort_values(by='Date')

# Function to rename columns with prefix
def rename_columns_with_prefix(df, prefix):
    new_cols = {}
    for c in df.columns:
        if c != 'Date':
            new_cols[c] = f"{prefix}_{c}"
    return df.rename(columns=new_cols)

# Read TWII index data
twii_df = pd.read_csv('^TWII.csv', parse_dates=['Date'])
twii_df['Date'] = twii_df['Date'].dt.tz_localize(None)
# Drop Volume column as it contains mostly zeros
if 'Volume' in twii_df.columns:
    twii_df = twii_df.drop(columns=['Volume'])
twii_df = rename_columns_with_prefix(twii_df, 'TWII')

# Read TWD/USD exchange rate data
twdusd_df = pd.read_csv('TWDUSD=X.csv', parse_dates=['Date'])
# Drop Volume column as it contains mostly zeros
if 'Volume' in twdusd_df.columns:
    twdusd_df = twdusd_df.drop(columns=['Volume'])
twdusd_df = rename_columns_with_prefix(twdusd_df, 'TWDUSD')

# Initialize list of dataframes
dfs = [twii_df, twdusd_df]

# Process bond data with error handling
try:
    # 5-year bonds
    tw5y_df = concat_bond_files('tw5yearbonds_1.csv', 'tw5yearbonds_2.csv')
    if not tw5y_df.empty:
        tw5y_df = rename_columns_with_prefix(tw5y_df, 'TW5Y')
        dfs.append(tw5y_df)

    # 10-year bonds
    tw10y_df = concat_bond_files('tw10yearbonds_1.csv', 'tw10yearbonds_2.csv')
    if not tw10y_df.empty:
        tw10y_df = rename_columns_with_prefix(tw10y_df, 'TW10Y')
        dfs.append(tw10y_df)

    # 20-year bonds
    tw20y_df = concat_bond_files('tw20yearbonds_1.csv', 'tw20yearbonds_2.csv')
    if not tw20y_df.empty:
        tw20y_df = rename_columns_with_prefix(tw20y_df, 'TW20Y')
        dfs.append(tw20y_df)

    # 30-year bonds
    tw30y_df = process_bond_file('tw30yearbonds.csv')
    if not tw30y_df.empty:
        tw30y_df = rename_columns_with_prefix(tw30y_df, 'TW30Y')
        dfs.append(tw30y_df)

except Exception as e:
    print(f"Error processing bond files: {e}")

# Start with first dataframe
merged_df = dfs[0]

# Merge with remaining dataframes
for df in dfs[1:]:
    merged_df = pd.merge(merged_df, df, on='Date', how='outer')

# Sort by date
merged_df.sort_values(by='Date', inplace=True)

# Set date range for the data
merged_df = merged_df[(merged_df['Date'] >= '2015-01-05') & (merged_df['Date'] <= '2025-03-31')]
merged_df = merged_df.reset_index(drop=True)

# Reindex to ensure all business days are included
unique_dates_pd = pd.to_datetime(unique_dates)
merged_df.set_index('Date', inplace=True)
merged_df = merged_df.reindex(unique_dates_pd)

# Reset index and rename
merged_df = merged_df.reset_index()
merged_df = merged_df.rename(columns={'index': 'Date'})

# Fill missing values
merged_df_filled = merged_df.fillna(method='ffill').fillna(method='bfill')
assert not merged_df_filled.isnull().any().any(), "There are still NaN in merged_df_filled"

num_days = len(merged_df_filled['Date'].unique())
# change here
used_cols = ['TWII_Open', 'TWII_High', 'TWII_Low', 'TWII_Close']
selected_data = merged_df_filled[used_cols]
reshaped_data = selected_data.to_numpy()
# num_MSU_features = merged_df_filled.shape[1] - 1
# reshaped_data = merged_df_filled.drop(columns='Date').to_numpy().reshape(num_days, num_MSU_features)

output_file = 'market_data.npy'
np.save(output_file, reshaped_data)