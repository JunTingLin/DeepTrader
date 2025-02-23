import logging
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import multiprocessing as mp

logging.basicConfig(level=logging.INFO)

def process_one_stock(args):
    """
    args: (i, stock_id, df_us, unique_dates)
    對單一股票做資料整理，僅保留 [Open, High, Close, Low, Volume] 5個欄位，
    並依照 unique_dates 填入，缺漏資料則前向填補。
    如果最終資料中有 0 值，則打印警告訊息。
    """
    i, stock_id, df_us, unique_dates = args
    
    # 選取該股票的資料並僅保留必要欄位
    stock_data = df_us[df_us['Ticker'] == stock_id].copy()
    stock_data = stock_data[['Date', 'Open', 'High', 'Close', 'Low', 'Volume']]
    
    num_days = len(unique_dates)
    num_features = 5
    per_stock_array = np.zeros((num_days, num_features))
    
    # 依據 unique_dates 填入該日期資料（順序：Open, High, Close, Low, Volume）
    for j, date in enumerate(unique_dates):
        day_data = stock_data[stock_data['Date'] == date]
        if not day_data.empty:
            row = day_data.iloc[0]
            per_stock_array[j, :] = [row['Open'], row['High'], row['Close'], row['Low'], row['Volume']]
    
    # 前向填補
    for j in range(1, per_stock_array.shape[0]):
        if per_stock_array[j, 0] == 0:
            per_stock_array[j, :] = per_stock_array[j-1, :]
    
    return (i, per_stock_array)

if __name__ == '__main__':
    # 取得業務日曆（business days）
    unique_dates = pd.bdate_range(start='2000-01-03', end='2023-12-31')
    unique_dates = unique_dates.to_pydatetime()
    unique_dates = np.array(unique_dates)

    djia_tickers = ['MMM','AXP','AMGN','AAPL','BA','CAT','CVX','CSCO','KO','HPQ',
                    'GS','HD','HON','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT',
                    'NKE','PFE','PG','CRM','TRV','UNH','VZ','V','WBA','DIS']
    df_us = pd.DataFrame()
    
    # 下載每個股票資料
    for ticker in djia_tickers:
        print("Downloading:", ticker)
        try:
            sample_data = yf.download(ticker, start='2000-01-01', end='2023-12-31', progress=False)
            if sample_data.empty or sample_data.index[0] != pd.Timestamp('2000-01-03'):
                print("Skipping:", ticker, "due to no data on 2000-01-03.")
                continue

            stock_data = yf.download(ticker, start='2000-01-03', end='2024-03-01', auto_adjust=False, progress=False)
            if stock_data.empty:
                print("Skipping:", ticker, "due to empty DataFrame.")
                continue

            # 檢查下載資料中必須欄位是否有 0 值
            required_cols = ['Open','High','Close','Low','Volume']
            zero_mask = (stock_data[required_cols] == 0)
            if zero_mask.any().any():
                rows_with_zeros = stock_data[zero_mask.any(axis=1)]
                print(f"Warning: {ticker} has zero values in raw data on dates: {list(rows_with_zeros.index)}")

            # 將 index 轉為 Date 欄位，若有 MultiIndex 需降層
            stock_data.reset_index(inplace=True)
            if isinstance(stock_data.columns, pd.MultiIndex):
                stock_data.columns = stock_data.columns.droplevel(level=1)
            stock_data['Ticker'] = ticker
            df_us = pd.concat([df_us, stock_data], ignore_index=True)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    # 依 Ticker 與 Date 排序，並對部分欄位作 MinMaxScaler 正規化
    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us = df_us.sort_values(by=['Ticker','Date'])
    cols_to_normalize = ['Open','High','Low','Close','Adj Close','Volume']
    scaler = MinMaxScaler()
    df_us[cols_to_normalize] = scaler.fit_transform(df_us[cols_to_normalize])
    
    unique_stock_ids = df_us['Ticker'].unique()
    num_stocks = len(unique_stock_ids)
    num_days   = len(unique_dates)
    num_features = 5  # 只保留 5 個特徵
    
    # 建立多核心平行處理的任務
    tasks = []
    for i, stock_id in enumerate(unique_stock_ids):
        tasks.append((i, stock_id, df_us, unique_dates))
    
    reshaped_data = np.zeros((num_stocks, num_days, num_features))
    
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(process_one_stock, tasks)
    pool.close()
    pool.join()
    
    for (i, per_stock_arr) in results:
        reshaped_data[i, :, :] = per_stock_arr
    
    # 儲存 shape 為 (股票數, 日期數, 5) 的資料
    output_file = 'stocks_data.npy'
    np.save(output_file, reshaped_data)

    # 使用 Close 價計算每日報酬率 (採用第3欄，索引 2)
    returns = np.zeros((num_stocks, num_days))
    for i in range(1, num_days):
        returns[:, i] = (reshaped_data[:, i, 2] - reshaped_data[:, i - 1, 2]) / reshaped_data[:, i - 1, 2]
    np.save('ror.npy', returns)

    correlation_matrix = np.corrcoef(returns[:, :1000])
    np.save('industry_classification.npy', correlation_matrix)
