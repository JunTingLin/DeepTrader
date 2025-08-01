import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
import talib
import multiprocessing as mp

def calculate_returns(df):
    df['Returns'] = df['Close'].pct_change()
    return df

def calculate_alpha001(df):
    rank_close = df['Close'].rank(pct=True)
    rank_volume = df['Volume'].rank(pct=True)
    alpha001 = rank_close.rolling(window=5).apply(lambda x: np.corrcoef(x, rank_volume.loc[x.index])[0, 1], raw=False).rank(pct=True)
    return alpha001

def calculate_alpha002(df):
    log_volume = np.log(df['Volume'])
    delta_log_volume = log_volume.diff(2)
    price_change = (df['Close'] - df['Open']) / df['Open']
    alpha002 = -1 * df['Close'].rolling(window=6).apply(lambda x: np.corrcoef(delta_log_volume.loc[x.index], price_change.loc[x.index])[0, 1], raw=False).rank(pct=True)
    return alpha002

def calculate_alpha003(df):
    alpha003 = -1 * df['Open'].rolling(window=10).apply(lambda x: np.corrcoef(x.rank(), df['Volume'].loc[x.index].rank())[0, 1], raw=False).rank(pct=True)
    return alpha003

def calculate_alpha004(df):
    alpha004 = -1 * df['Low'].rank(pct=True).rolling(window=9).apply(lambda x: x.rank().iloc[-1], raw=False)
    return alpha004

def calculate_alpha006(df):
    alpha006 = -1 * df['Open'].rolling(window=10).apply(lambda x: np.corrcoef(x, df['Volume'].loc[x.index])[0, 1], raw=False)
    return alpha006

def calculate_alpha012(df):
    delta_volume = df['Volume'].diff(1)
    delta_close = df['Close'].diff(1)
    alpha012 = np.sign(delta_volume) * (-1 * delta_close)
    return alpha012

def calculate_alpha019(df):
    delayed_close = df['Close'].shift(7)
    delta_close = df['Close'].diff(7)
    rank_sum_returns = df['Returns'].rolling(window=250).sum().rank(pct=True)
    alpha019 = (-1 * np.sign((df['Close'] - delayed_close) + delta_close)) * (1 + rank_sum_returns)
    return alpha019

def calculate_alpha033(df):
    alpha033 = (-1 * (1 - (df['Open'] / df['Close'])).rank(pct=True))
    return alpha033

def calculate_alpha038(df):
    alpha038 = (-1 * df['Close'].rolling(window=10).apply(lambda x: x.rank().iloc[-1], raw=False).rank(pct=True)) * (df['Close'] / df['Open']).rank(pct=True)
    return alpha038

def calculate_alpha040(df):
    alpha040 = (-1 * df['High'].rolling(window=10).apply(lambda x: x.std()).rank(pct=True)) * df['High'].rolling(window=10).apply(lambda x: np.corrcoef(x, df['Volume'].loc[x.index])[0, 1], raw=False)
    return alpha040

def calculate_alpha044(df):
    alpha044 = -1 * df['High'].rolling(window=5).apply(lambda x: np.corrcoef(x, df['Volume'].loc[x.index].rank())[0, 1], raw=False)
    return alpha044

def calculate_alpha045(df):
    delayed_close_5 = df['Close'].shift(5)
    alpha045 = (-1 * (df['Close'].rolling(window=20).mean().shift(5).rank(pct=True) * df['Close'].rolling(window=2).apply(lambda x: np.corrcoef(x, df['Volume'].loc[x.index])[0, 1], raw=False)).rank(pct=True) * df['Close'].rolling(window=5).sum().rolling(window=20).apply(lambda x: np.corrcoef(x, df['Close'].loc[x.index])[0, 1], raw=False).rank(pct=True))
    return alpha045

def calculate_alpha046(df):
    delta_close_10 = df['Close'].diff(10)
    delta_close_20 = df['Close'].diff(20)
    term = ((delta_close_20 - delta_close_10) / 10) - ((delta_close_10 - df['Close']) / 10)
    alpha046 = np.where(0.25 < term, -1, np.where(term < 0, 1, -1 * (df['Close'] - df['Close'].shift(1))))
    return alpha046

def calculate_alpha051(df):
    delta_close_10 = df['Close'].diff(10)
    delta_close_20 = df['Close'].diff(20)
    alpha051 = np.where(((delta_close_20 - delta_close_10) / 10) - ((delta_close_10 - df['Close']) / 10) < -0.05, 1, -1 * (df['Close'] - df['Close'].shift(1)))
    return alpha051

def calculate_alpha052(df):
    ts_min_low_5 = df['Low'].rolling(window=5).min()
    delayed_ts_min_low_5 = ts_min_low_5.shift(5)
    rank_sum_returns_240_20 = ((df['Returns'].rolling(window=240).sum() - df['Returns'].rolling(window=20).sum()) / 220).rank(pct=True)
    alpha052 = ((-1 * ts_min_low_5 + delayed_ts_min_low_5) * rank_sum_returns_240_20) * df['Volume'].rolling(window=5).apply(lambda x: x.rank().iloc[-1], raw=False)
    return alpha052

def calculate_alpha053(df):
    alpha053 = -1 * (((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['Close'] - df['Low'])).diff(9)
    return alpha053

def calculate_alpha054(df):
    alpha054 = (-1 * ((df['Low'] - df['Close']) * df['Open']**5)) / ((df['Low'] - df['High']) * df['Close']**5)
    return alpha054

def calculate_alpha056(df):
    rank_sum_returns_10 = (df['Returns'].rolling(window=10).sum() / df['Returns'].rolling(window=2).sum().rolling(window=3).sum()).rank(pct=True)
    alpha056 = -rank_sum_returns_10 * (df['Returns'] * df['Volume'])
    return alpha056

def calculate_alpha060(df):
    alpha060 = -1 * (2 * ((df['High'] - df['Close']).rolling(window=10).apply(lambda x: (x - x.min()) / (x.max() - x.min()), raw=False).rank(pct=True) - (df['Close'].rolling(window=10).apply(np.argmax, raw=False).rank(pct=True))))
    return alpha060

def calculate_alpha068(df):
    rank_corr_high_adv15_8 = df['High'].rolling(window=8).apply(lambda x: np.corrcoef(x.rank(), df['Volume'].rolling(window=15).mean().loc[x.index])[0, 1], raw=False).rank(pct=True)
    delta_weighted_close = (df['Close'] * 0.518371 + df['Low'] * (1 - 0.518371)).diff(1.06157)
    alpha068 = (rank_corr_high_adv15_8 < delta_weighted_close) * -1
    return alpha068

def calculate_alpha085(df):
    rank_corr_high_close_adv30_9 = ((df['High'] * 0.876703 + df['Close'] * (1 - 0.876703)).rolling(window=9).apply(lambda x: np.corrcoef(x, df['Volume'].rolling(window=30).mean().loc[x.index])[0, 1], raw=False).rank(pct=True))
    rank_corr_median_high_low_ts_rank_volume = df['High'].rolling(window=10).apply(lambda x: np.corrcoef((df['High'] + df['Low']) / 2, df['Volume'])[0, 1], raw=False).rank(pct=True)
    alpha085 = rank_corr_high_close_adv30_9 ** rank_corr_median_high_low_ts_rank_volume
    return alpha085

def calculate_alpha092(df):
    adv30 = df['Volume'].rolling(window=30).mean()
    ts_rank1 = df[['High', 'Low', 'Close', 'Open']].mean(axis=1).rolling(window=14).apply(lambda x: (x < (df['Low'] + df['Open'])).mean(), raw=False).rank(pct=True)
    ts_rank2 = df[['Low', 'Volume']].rank().rolling(window=7).apply(lambda x: np.corrcoef(x, adv30.loc[x.index])[0, 1], raw=False).rank(pct=True)
    alpha092 = np.minimum(ts_rank1, ts_rank2)
    return alpha092

def calculate_alpha088(df):
    df['Rank_Open'] = df['Open'].rank(pct=True)
    df['Rank_Low'] = df['Low'].rank(pct=True)
    df['Rank_High'] = df['High'].rank(pct=True)
    df['Rank_Close'] = df['Close'].rank(pct=True)
    decay_linear_value = ((df['Rank_Open'] + df['Rank_Low']) - 
                          (df['Rank_High'] + df['Rank_Close'])).rolling(window=8).apply(lambda x: np.mean(x), raw=False)
    rank_decay_linear = decay_linear_value.rank(pct=True)
    ts_rank_value = (df['Close'].rolling(window=8).apply(lambda x: pd.Series(x).rank(pct=True).mean(), raw=False) -
                     df['Volume'].rolling(window=20).apply(lambda x: pd.Series(x).rank(pct=True).mean(), raw=False))
    ts_rank_value = ts_rank_value.rolling(window=6).apply(lambda x: np.mean(x), raw=False).rank(pct=True)
    alpha088 = np.minimum(rank_decay_linear, ts_rank_value)
    return alpha088

def calculate_alpha095(df):
    df['Ts_Min_Open'] = df['Open'].rolling(window=12).min()
    rank_open_min = (df['Open'] - df['Ts_Min_Open']).rank(pct=True)
    avg_high_low = (df['High'] + df['Low']) / 2
    sum_high_low = avg_high_low.rolling(window=19).sum()
    sum_adv40 = df['Volume'].rolling(window=19).sum()
    correlation_value = sum_high_low.rolling(window=12).apply(lambda x: np.corrcoef(x, sum_adv40.loc[x.index])[0, 1], raw=False)
    rank_correlation = correlation_value.rank(pct=True)**5
    ts_rank_value = rank_correlation.rolling(window=11).apply(lambda x: pd.Series(x).rank(pct=True).mean(), raw=False)
    alpha095 = rank_open_min - ts_rank_value
    return alpha095

def calculate_alpha101(df):
    alpha101 = (df['Close'] - df['Open']) / ((df['High'] - df['Low']) + 0.001)
    return alpha101

def process_one_stock(args):
    """
    args: (i, stock_id, df_us, unique_dates, alphas)
    1) 選擇該股票資料
    2) 做 Talib + alpha 計算
    3) 組出 shape=(num_days, num_ASU_features) 的陣列
    4) 回傳 (i, array)
    """
    i, stock_id, df_us, unique_dates, alphas = args
    
    # 1) 取出該股票資料
    stock_data = df_us[df_us['Ticker'] == stock_id].copy()
    
    # 2) Talib計算
    stock_data = calculate_returns(stock_data)
    stock_data['MA20'] = talib.SMA(stock_data['Close'], timeperiod=20)
    stock_data['MA60'] = talib.SMA(stock_data['Close'], timeperiod=60)
    stock_data['RSI']  = talib.RSI(stock_data['Close'], timeperiod=14)
    macd, signal, hist = talib.MACD(stock_data['Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    stock_data['MACD']        = macd
    stock_data['MACD_Signal'] = signal
    stock_data['MACD_Hist']   = hist
    k, d = talib.STOCH(stock_data['High'], stock_data['Low'], stock_data['Close'])
    stock_data['K'] = k
    stock_data['D'] = d
    upper_band, middle_band, lower_band = talib.BBANDS(stock_data['Close'], timeperiod=20, nbdevup=2, nbdevdn=2)
    stock_data['BBands_Upper']  = upper_band
    stock_data['BBands_Middle'] = middle_band
    stock_data['BBands_Lower']  = lower_band

    # 3) 計算 alpha
    for alpha in alphas:
        calc_function = globals()[f'calculate_{alpha.lower()}']
        stock_data[alpha] = calc_function(stock_data)
    
    # 4) 組出 shape=(num_days, num_ASU_features) 的暫存
    num_days = len(unique_dates)
    num_ASU_features = 5    # 34
    per_stock_array = np.zeros((num_days, num_ASU_features))
    
    # 依每個 unique_date 填值
    drop_cols = ['Date','Ticker','Adj Close','Returns','MACD','MACD_Hist']
    # used_cols = stock_data.columns.drop(drop_cols)
    used_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    
    for j, date in enumerate(unique_dates):
        day_data = stock_data[stock_data['Date'] == date]
        if not day_data.empty:
            per_stock_array[j, :] = day_data[used_cols].values
    
    # 前向填補
    for j in range(1, per_stock_array.shape[0]):
        if per_stock_array[j, 0] == 0:
            per_stock_array[j, :] = per_stock_array[j-1, :]
    # df = pd.DataFrame(per_stock_array)
    # df.replace(0, np.nan, inplace=True)
    # df.ffill(inplace=True)
    # df.bfill(inplace=True)
    # per_stock_array[:] = df.values

    return (i, per_stock_array)



if __name__ == '__main__':
    djia_tickers = ['MMM','AXP','AMGN','AAPL','BA','CAT','CVX','CSCO','KO','HPQ',
                    'GS','HD','HON','IBM','INTC','JNJ','JPM','MCD','MRK','MSFT',
                    'NKE','PFE','PG','CRM','TRV','UNH','VZ','V','WBA','DIS']
    df_us = pd.DataFrame()
    for ticker in djia_tickers:
        print("Downloading:", ticker)
        try:
            sample_data = yf.download(ticker, start='2000-01-03', end='2000-01-04', progress=False)
            if sample_data.empty or sample_data.index[0] != pd.Timestamp('2000-01-03'):
                print("Skipping:", ticker, "due to no data on 2000-01-03.")
                continue

            stock_data = yf.download(ticker, start='2000-01-03', end='2023-12-31', auto_adjust=False, progress=False)
            if stock_data.empty:
                print("Skipping:", ticker, "due to empty DataFrame.")
                continue

            # Reset index so 'Date' becomes a normal column
            stock_data.reset_index(inplace=True)
            stock_data.columns = stock_data.columns.droplevel(level=1) # (Open, MMM) -> Open
            stock_data['Ticker'] = ticker
            df_us = pd.concat([df_us, stock_data], ignore_index=True)

        except Exception as e:
            print(f"Error downloading {ticker}: {e}")
            continue
    
    # 排序 + MinMaxScaler
    df_us['Date'] = pd.to_datetime(df_us['Date'])
    df_us = df_us.sort_values(by=['Ticker','Date'])

    # cols_to_normalize = ['Open','High','Low','Close','Adj Close','Volume']
    # scaler = MinMaxScaler()
    # df_us[cols_to_normalize] = scaler.fit_transform(df_us[cols_to_normalize])

    unique_dates = pd.bdate_range(start='2000-01-03', end='2023-12-31') # business day
    unique_dates = unique_dates.to_pydatetime()
    unique_dates = np.array(unique_dates)
    
    # alpha list
    alphas = ['Alpha001','Alpha002','Alpha003','Alpha004','Alpha006','Alpha012','Alpha019',
              'Alpha033','Alpha038','Alpha040','Alpha044','Alpha045','Alpha046','Alpha051',
              'Alpha052','Alpha053','Alpha054','Alpha056','Alpha068','Alpha085']
    
    unique_stock_ids = df_us['Ticker'].unique()
    # unique_dates = df_us['Date'].unique()
    num_stocks = len(unique_stock_ids)
    num_days   = len(unique_dates)
    num_ASU_features = 5    # 34
    
    # 使用多核心平行化
    tasks = []
    for i, stock_id in enumerate(unique_stock_ids):
        tasks.append((i, stock_id, df_us, unique_dates, alphas))
    
    reshaped_data = np.zeros((num_stocks, num_days, num_ASU_features))
    
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.map(process_one_stock, tasks)
    pool.close()
    pool.join()
    
    # 將 results 填入 reshaped_data
    for (i, per_stock_arr) in results:
        reshaped_data[i, :, :] = per_stock_arr
    
    output_file = r'./stocks_data.npy'
    np.save(output_file, reshaped_data)

    returns = np.zeros((num_stocks, num_days))
    for i in range(1, num_days):
        returns[:, i] = (reshaped_data[:, i, 0] - reshaped_data[:, i - 1, 0]) / reshaped_data[:, i - 1, 0]
    np.save(r'./ror.npy', returns)

    correlation_matrix = np.corrcoef(returns[:, :1000])
    np.save(r'./industry_classification.npy', correlation_matrix)