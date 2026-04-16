"""
DeepTrader Taiwan Stock Data Processing using FinLab API
Uses adjusted (restored) prices to account for dividends and stock splits
Supports 'basic', 'basic_sentiment', and 'full' feature modes
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import talib
import finlab
from finlab import data

# Load environment variables
load_dotenv()

# Initialize FinLab with API key
FINLAB_API_KEY = os.getenv('FINLAB_API_KEY')
if not FINLAB_API_KEY:
    raise ValueError("FINLAB_API_KEY not found in .env file")

finlab.login(FINLAB_API_KEY)


# ============================================================
# Technical Indicator and Alpha Factor Functions
# (Same as deeptrader_data_common.py for consistency)
# ============================================================

def calculate_returns(df):
    df['Returns'] = df['close'].pct_change()
    return df


def calculate_alpha001(df):
    rank_close = df['close'].rank(pct=True)
    rank_volume = df['volume'].rank(pct=True)
    alpha001 = rank_close.rolling(window=5).apply(
        lambda x: np.corrcoef(x, rank_volume.loc[x.index])[0, 1], raw=False
    ).rank(pct=True)
    return alpha001


def calculate_alpha002(df):
    volume_safe = pd.Series(df['volume']).replace(0, 1)
    open_safe = pd.Series(df['open']).replace(0, 1)
    log_volume = np.log(volume_safe)
    delta_log_volume = log_volume.diff(2)
    price_change = (df['close'] - df['open']) / open_safe
    alpha002 = -1 * df['close'].rolling(window=6).apply(
        lambda x: np.corrcoef(delta_log_volume.loc[x.index], price_change.loc[x.index])[0, 1], raw=False
    ).rank(pct=True)
    return alpha002


def calculate_alpha003(df):
    alpha003 = -1 * df['open'].rolling(window=10).apply(
        lambda x: np.corrcoef(x.rank(), df['volume'].loc[x.index].rank())[0, 1], raw=False
    ).rank(pct=True)
    return alpha003


def calculate_alpha004(df):
    alpha004 = -1 * df['low'].rank(pct=True).rolling(window=9).apply(
        lambda x: x.rank().iloc[-1], raw=False
    )
    return alpha004


def calculate_alpha006(df):
    alpha006 = -1 * df['open'].rolling(window=10).apply(
        lambda x: np.corrcoef(x, df['volume'].loc[x.index])[0, 1], raw=False
    )
    return alpha006


def calculate_alpha012(df):
    delta_volume = df['volume'].diff(1)
    delta_close = df['close'].diff(1)
    alpha012 = np.sign(delta_volume) * (-1 * delta_close)
    return alpha012


def calculate_alpha019(df):
    delayed_close = df['close'].shift(7)
    delta_close = df['close'].diff(7)
    rank_sum_returns = df['Returns'].rolling(window=250).sum().rank(pct=True)
    alpha019 = (-1 * np.sign((df['close'] - delayed_close) + delta_close)) * (1 + rank_sum_returns)
    return alpha019


def calculate_alpha033(df):
    close_safe = pd.Series(df['close']).replace(0, 1)
    alpha033 = (-1 * (1 - (df['open'] / close_safe)).rank(pct=True))
    return alpha033


def calculate_alpha038(df):
    open_safe = pd.Series(df['open']).replace(0, 1)
    alpha038 = (-1 * df['close'].rolling(window=10).apply(
        lambda x: x.rank().iloc[-1], raw=False
    ).rank(pct=True)) * (df['close'] / open_safe).rank(pct=True)
    return alpha038


def calculate_alpha040(df):
    alpha040 = (-1 * df['high'].rolling(window=10).apply(
        lambda x: x.std()
    ).rank(pct=True)) * df['high'].rolling(window=10).apply(
        lambda x: np.corrcoef(x, df['volume'].loc[x.index])[0, 1], raw=False
    )
    return alpha040


def calculate_alpha044(df):
    alpha044 = -1 * df['high'].rolling(window=5).apply(
        lambda x: np.corrcoef(x, df['volume'].loc[x.index].rank())[0, 1], raw=False
    )
    return alpha044


def calculate_alpha045(df):
    alpha045 = (-1 * (df['close'].rolling(window=20).mean().shift(5).rank(pct=True) *
                      df['close'].rolling(window=2).apply(
                          lambda x: np.corrcoef(x, df['volume'].loc[x.index])[0, 1], raw=False
                      )).rank(pct=True) *
                df['close'].rolling(window=5).sum().rolling(window=20).apply(
                    lambda x: np.corrcoef(x, df['close'].loc[x.index])[0, 1], raw=False
                ).rank(pct=True))
    return alpha045


def calculate_alpha046(df):
    delta_close_10 = df['close'].diff(10)
    delta_close_20 = df['close'].diff(20)
    term = ((delta_close_20 - delta_close_10) / 10) - ((delta_close_10 - df['close']) / 10)
    alpha046 = np.where(0.25 < term, -1, np.where(term < 0, 1, -1 * (df['close'] - df['close'].shift(1))))
    return alpha046


def calculate_alpha051(df):
    delta_close_10 = df['close'].diff(10)
    delta_close_20 = df['close'].diff(20)
    alpha051 = np.where(
        ((delta_close_20 - delta_close_10) / 10) - ((delta_close_10 - df['close']) / 10) < -0.05,
        1, -1 * (df['close'] - df['close'].shift(1))
    )
    return alpha051


def calculate_alpha052(df):
    ts_min_low_5 = df['low'].rolling(window=5).min()
    delayed_ts_min_low_5 = ts_min_low_5.shift(5)
    rank_sum_returns_240_20 = ((df['Returns'].rolling(window=240).sum() -
                                df['Returns'].rolling(window=20).sum()) / 220).rank(pct=True)
    alpha052 = ((-1 * ts_min_low_5 + delayed_ts_min_low_5) * rank_sum_returns_240_20) * \
               df['volume'].rolling(window=5).apply(lambda x: x.rank().iloc[-1], raw=False)
    return alpha052


def calculate_alpha053(df):
    denom = pd.Series(df['close'] - df['low']).replace(0, 1)
    alpha053 = -1 * (((df['close'] - df['low']) - (df['high'] - df['close'])) / denom).diff(9)
    return alpha053


def calculate_alpha054(df):
    denom = pd.Series(df['low'] - df['high']).replace(0, 1)
    alpha054 = (-1 * ((df['low'] - df['close']) * df['open']**5)) / (denom * df['close']**5)
    return alpha054


def calculate_alpha056(df):
    denom = pd.Series(df['Returns'].rolling(window=2).sum().rolling(window=3).sum()).replace(0, 1)
    rank_sum_returns_10 = (df['Returns'].rolling(window=10).sum() / denom).rank(pct=True)
    alpha056 = -rank_sum_returns_10 * (df['Returns'] * df['volume'])
    return alpha056


def calculate_alpha060(df):
    alpha060 = -1 * (2 * ((df['high'] - df['close']).rolling(window=10).apply(
        lambda x: (x - x.min()) / (x.max() - x.min()) if (x.max() - x.min()) != 0 else 0, raw=False
    ).rank(pct=True) - (df['close'].rolling(window=10).apply(np.argmax, raw=False).rank(pct=True))))
    return alpha060


def calculate_alpha068(df):
    rank_corr_high_adv15_8 = df['high'].rolling(window=8).apply(
        lambda x: np.corrcoef(x.rank(), df['volume'].rolling(window=15).mean().loc[x.index])[0, 1], raw=False
    ).rank(pct=True)
    delta_weighted_close = (df['close'] * 0.518371 + df['low'] * (1 - 0.518371)).diff(1)
    alpha068 = (rank_corr_high_adv15_8 < delta_weighted_close) * -1
    return alpha068


def calculate_alpha085(df):
    rank_corr = ((df['high'] * 0.876703 + df['close'] * (1 - 0.876703)).rolling(window=9).apply(
        lambda x: np.corrcoef(x, df['volume'].rolling(window=30).mean().loc[x.index])[0, 1], raw=False
    ).rank(pct=True))
    rank_corr2 = df['high'].rolling(window=10).apply(
        lambda _: np.corrcoef((df['high'] + df['low']) / 2, df['volume'])[0, 1], raw=False
    ).rank(pct=True)
    alpha085 = rank_corr ** rank_corr2
    return alpha085


def fill_technical_indicators(df, indicators):
    """Fill NaN/Inf values in technical indicators"""
    for col in indicators:
        if col in df.columns:
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].bfill().ffill().fillna(0)
    return df


def clean_alpha_factor(alpha_series):
    """Clean alpha factor by replacing Inf/NaN"""
    # Handle both pandas Series and numpy array
    if isinstance(alpha_series, np.ndarray):
        result = pd.Series(alpha_series)
    else:
        result = alpha_series.copy()

    result = result.replace([np.inf, -np.inf], np.nan)
    result = result.bfill().ffill().fillna(0)
    return result

# 2016-01-01~2025-12-31 intersect TWII stocks list
# Removed 3711 (日月光投控) due to missing data before 2018-04-30
# TWII_STOCKS = [
#     "1216",  # 統一
#     "1301",  # 台塑
#     "1303",  # 南亞
#     "2002",  # 中鋼
#     "2303",  # 聯電
#     "2308",  # 台達電
#     "2317",  # 鴻海
#     "2330",  # 台積電
#     "2357",  # 華碩
#     "2382",  # 廣達
#     "2395",  # 研華
#     "2412",  # 中華電
#     "2454",  # 聯發科
#     "2880",  # 華南金
#     "2881",  # 富邦金
#     "2882",  # 國泰金
#     "2884",  # 玉山金
#     "2885",  # 元大金
#     "2886",  # 兆豐金
#     "2887",  # 台新新光金
#     "2891",  # 中信金
#     "2892",  # 第一金
#     "2912",  # 統一超
#     "3008",  # 大立光
#     "3045",  # 台灣大
#     "4904",  # 遠傳
#     "5880",  # 合庫金
#     "6505",  # 台塑化
# ]

TWII_STOCKS = [
    "1216",  # 統一
    "1301",  # 台塑
    "1303",  # 南亞
    "2002",  # 中鋼
    "2059",  # 川湖
    "2207",  # 和泰車
    "2301",  # 光寶科
    "2303",  # 聯電
    "2308",  # 台達電
    "2317",  # 鴻海
    "2327",  # 國巨
    "2330",  # 台積電
    "2345",  # 智邦
    "2357",  # 華碩
    "2360",  # 致茂
    "2379",  # 瑞昱
    "2382",  # 廣達
    "2383",  # 台光電
    "2395",  # 研華
    "2408",  # 南亞科
    "2412",  # 中華電
    "2454",  # 聯發科
    "2603",  # 長榮
    "2615",  # 萬海
    "2880",  # 華南金
    "2881",  # 富邦金
    "2882",  # 國泰金
    "2883",  # 凱基金
    "2884",  # 玉山金
    "2885",  # 元大金
    "2886",  # 兆豐金
    "2887",  # 台新新光金
    "2890",  # 永豐金
    "2891",  # 中信金
    "2892",  # 第一金
    "2912",  # 統一超
    "3008",  # 大立光
    "3017",  # 奇鋐
    "3034",  # 聯詠
    "3045",  # 台灣大
    "3231",  # 緯創
    "3653",  # 健策
    "3661",  # 世芯-KY
    "3665",  # 貿聯-KY
    # "3711",  # 日月光投控
    "4904",  # 遠傳
    "5880",  # 合庫金
    "6505",  # 台塑化
    # "6669",  # 緯穎
    # "6919",  # 康霈
]


def load_precomputed_sentiment(sentiment_file, num_stocks, num_days):
    """
    Load precomputed sentiment scores

    Args:
        sentiment_file: Path to sentiment scores .npy file
        num_stocks: Number of stocks
        num_days: Number of trading days

    Returns:
        sentiment_scores: Array of shape (num_stocks, num_days, 1)
    """
    if sentiment_file and os.path.exists(sentiment_file):
        print(f"\n=== Loading precomputed sentiment scores from {sentiment_file} ===")
        sentiment_data = np.load(sentiment_file)

        # Check shape
        if sentiment_data.shape == (num_stocks, num_days):
            # Reshape to (num_stocks, num_days, 1)
            sentiment_scores = sentiment_data.reshape(num_stocks, num_days, 1)
            print(f"Sentiment scores loaded: shape = {sentiment_scores.shape}")
            return sentiment_scores
        else:
            print(f"Warning: Sentiment file shape {sentiment_data.shape} doesn't match expected ({num_stocks}, {num_days})")
            return np.zeros((num_stocks, num_days, 1))
    else:
        print(f"Warning: Sentiment file {sentiment_file} not found, using zeros")
        return np.zeros((num_stocks, num_days, 1))


if __name__ == '__main__':
    # Configuration
    START_DATE = '2008-01-01'
    END_DATE = '2025-12-31'

    # For full mode, we need more historical data for indicator calculations
    DOWNLOAD_START_DATE = '2007-04-23'

    # Feature mode:
    # - 'basic': OHLCV (5 features)
    # - 'basic_sentiment': OHLCV + Sentiment (6 features)
    # - 'full': OHLCV + Technical + Alpha (34 features)
    # - 'full_sentiment': OHLCV + Technical + Alpha + Sentiment (35 features)
    FEATURE_MODE = 'basic'

    # Sentiment file (required for basic_sentiment mode)
    PRECOMPUTED_SENTIMENT_FILE = './sentiment_scores.npy'

    # Alpha factors list
    ALPHAS = ['Alpha001', 'Alpha002', 'Alpha003', 'Alpha004', 'Alpha006', 'Alpha012', 'Alpha019',
              'Alpha033', 'Alpha038', 'Alpha040', 'Alpha044', 'Alpha045', 'Alpha046', 'Alpha051',
              'Alpha052', 'Alpha053', 'Alpha054', 'Alpha056', 'Alpha068', 'Alpha085']

    print(f"=== DeepTrader Taiwan Stock Data Processing (FinLab) ===")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Feature mode: {FEATURE_MODE}")
    print(f"Number of stocks: {len(TWII_STOCKS)}")
    print("Using FinLab API with adjusted prices")
    print("=" * 60)

    # Get trading dates from FinLab data (use 0050 as reference)
    print("\n=== Getting trading dates from FinLab ===")
    # Use 0050 ETF as reference because it rarely suspends trading
    # This ensures consistency with market_data
    df_close = data.get('etl:adj_close')
    stock_dates = df_close['0050'][START_DATE:END_DATE]
    unique_dates = stock_dates.index.to_pydatetime()
    unique_dates = np.array(unique_dates)

    print(f"Trading dates: {len(unique_dates)} days ({unique_dates[0]} to {unique_dates[-1]})")

    # Get dimensions
    num_stocks = len(TWII_STOCKS)
    num_days = len(unique_dates)

    # Determine number of features based on mode
    tech_names = ['MA20', 'MA60', 'RSI', 'MACD_Signal', 'K', 'D',
                  'BBands_Upper', 'BBands_Middle', 'BBands_Lower']

    if FEATURE_MODE == 'basic':
        num_features = 5  # OHLCV
        feature_names = ['open', 'high', 'low', 'close', 'volume']
    elif FEATURE_MODE == 'basic_sentiment':
        num_features = 6  # OHLCV + Sentiment
        feature_names = ['open', 'high', 'low', 'close', 'volume', 'Sentiment']
    elif FEATURE_MODE == 'full':
        num_features = 5 + len(tech_names) + len(ALPHAS)  # 5 + 9 + 20 = 34
        feature_names = ['open', 'high', 'low', 'close', 'volume'] + tech_names + ALPHAS
    elif FEATURE_MODE == 'full_sentiment':
        num_features = 5 + len(tech_names) + len(ALPHAS) + 1  # 5 + 9 + 20 + 1 = 35
        feature_names = ['open', 'high', 'low', 'close', 'volume'] + tech_names + ALPHAS + ['Sentiment']
    else:
        raise ValueError(f"Unknown FEATURE_MODE: {FEATURE_MODE}")

    print(f"\n=== FEATURE CONFIGURATION ===")
    print(f"Mode: {FEATURE_MODE}")
    print(f"Total features: {num_features}")
    print(f"Feature names (in order):")
    for idx, name in enumerate(feature_names, 1):
        print(f"  {idx:2d}. {name}")
    print("=" * 30)

    # Initialize output array
    print(f"\nProcessing {num_stocks} stocks...")
    reshaped_data = np.zeros((num_stocks, num_days, num_features))

    # Process each stock sequentially
    for i, stock_id in enumerate(TWII_STOCKS):
        print(f"Processing stock {i+1}/{num_stocks}: {stock_id}")

        try:
            # Always use DOWNLOAD_START_DATE to ensure consistent data across all modes
            # This ensures OHLCV values are identical regardless of feature mode
            download_start = DOWNLOAD_START_DATE

            # Get adjusted (restored) prices from FinLab
            open_price = data.get('etl:adj_open')[stock_id][download_start:END_DATE]
            high_price = data.get('etl:adj_high')[stock_id][download_start:END_DATE]
            low_price = data.get('etl:adj_low')[stock_id][download_start:END_DATE]
            close_price = data.get('etl:adj_close')[stock_id][download_start:END_DATE]
            volume = data.get('price:成交股數')[stock_id][download_start:END_DATE]

            # Create DataFrame with full historical data
            stock_data = pd.DataFrame({
                'Date': close_price.index,
                'open': open_price.values,
                'high': high_price.values,
                'low': low_price.values,
                'close': close_price.values,
                'volume': volume.values
            })
            stock_data = stock_data.reset_index(drop=True)

            # Clean OHLCV data
            for col in ['open', 'high', 'low', 'close', 'volume']:
                stock_data[col] = stock_data[col].replace(0, np.nan)
                stock_data[col] = stock_data[col].ffill().bfill().fillna(0)
                stock_data[col] = stock_data[col].replace([np.inf, -np.inf], 0)

            # Calculate technical indicators and alphas for full/full_sentiment mode
            if FEATURE_MODE in ['full', 'full_sentiment']:
                # Calculate returns
                stock_data = calculate_returns(stock_data)

                # Calculate technical indicators using talib
                stock_data['MA20'] = talib.SMA(stock_data['close'], timeperiod=20)
                stock_data['MA60'] = talib.SMA(stock_data['close'], timeperiod=60)
                stock_data['RSI'] = talib.RSI(stock_data['close'], timeperiod=14)
                macd, signal, hist = talib.MACD(stock_data['close'], fastperiod=12, slowperiod=26, signalperiod=9)
                stock_data['MACD_Signal'] = signal
                k, d = talib.STOCH(stock_data['high'], stock_data['low'], stock_data['close'])
                stock_data['K'] = k
                stock_data['D'] = d
                upper, middle, lower = talib.BBANDS(stock_data['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
                stock_data['BBands_Upper'] = upper
                stock_data['BBands_Middle'] = middle
                stock_data['BBands_Lower'] = lower

                # Fill technical indicators
                tech_cols = ['MA20', 'MA60', 'RSI', 'MACD_Signal', 'K', 'D',
                             'BBands_Upper', 'BBands_Middle', 'BBands_Lower']
                stock_data = fill_technical_indicators(stock_data, tech_cols)

                # Calculate alpha factors
                for alpha in ALPHAS:
                    calc_func = globals()[f'calculate_{alpha.lower()}']
                    stock_data[alpha] = clean_alpha_factor(calc_func(stock_data))

        except Exception as e:
            print(f"  Error processing {stock_id}: {e}")
            continue

        # Fill data for each date (align with unique_dates)
        if FEATURE_MODE in ['full', 'full_sentiment']:
            # For full modes, use OHLCV + technical + alpha (sentiment added later)
            used_cols = ['open', 'high', 'low', 'close', 'volume'] + \
                        ['MA20', 'MA60', 'RSI', 'MACD_Signal', 'K', 'D',
                         'BBands_Upper', 'BBands_Middle', 'BBands_Lower'] + ALPHAS
        else:
            used_cols = ['open', 'high', 'low', 'close', 'volume']

        for j, date in enumerate(unique_dates):
            day_data = stock_data[stock_data['Date'] == pd.Timestamp(date)]
            if not day_data.empty:
                reshaped_data[i, j, :len(used_cols)] = day_data[used_cols].values[0]

    print("\n=== Stock data processing complete ===")
    print(f"Data shape: {reshaped_data.shape}")

    # Check for NaN values before filling
    nan_count_before = np.isnan(reshaped_data).sum()
    if nan_count_before > 0:
        print(f"Found {nan_count_before} NaN values before filling")
        for stock_idx in range(num_stocks):
            stock_nan = np.isnan(reshaped_data[stock_idx]).sum()
            if stock_nan > 0:
                print(f"  Stock {stock_idx} ({TWII_STOCKS[stock_idx]}): {stock_nan} NaN values")

        # Fill NaN values using forward fill + backward fill
        print("\nFilling NaN values with forward fill + backward fill...")
        df_temp = pd.DataFrame(reshaped_data.reshape(-1, num_features))
        df_temp = df_temp.ffill().bfill()
        reshaped_data = df_temp.values.reshape(num_stocks, num_days, num_features)

        # Check NaN after filling
        nan_count_after = np.isnan(reshaped_data).sum()
        if nan_count_after == 0:
            print(f"✓ All NaN values filled successfully (0 NaN remaining)")
        else:
            print(f"WARNING: {nan_count_after} NaN values remain after filling")
    else:
        print(f"✓ No NaN values found")

    # Add sentiment scores for sentiment modes
    if FEATURE_MODE == 'basic_sentiment':
        print("\n=== Adding sentiment scores ===")
        sentiment_scores = load_precomputed_sentiment(
            PRECOMPUTED_SENTIMENT_FILE,
            num_stocks,
            num_days
        )
        # Add sentiment as 6th feature (index 5)
        reshaped_data[:, :, 5:6] = sentiment_scores
        print(f"Final data shape: {reshaped_data.shape} (OHLCV + Sentiment)")
    elif FEATURE_MODE == 'full_sentiment':
        print("\n=== Adding sentiment scores ===")
        sentiment_scores = load_precomputed_sentiment(
            PRECOMPUTED_SENTIMENT_FILE,
            num_stocks,
            num_days
        )
        # Add sentiment as 35th feature (index 34)
        reshaped_data[:, :, 34:35] = sentiment_scores
        print(f"Final data shape: {reshaped_data.shape} (OHLCV + Technical + Alpha + Sentiment)")
    elif FEATURE_MODE == 'full':
        print(f"Final data shape: {reshaped_data.shape} (OHLCV + Technical + Alpha)")
    else:
        print(f"Final data shape: {reshaped_data.shape} (OHLCV only)")

    # Save stocks data (in current directory, same as original)
    output_file = 'stocks_data.npy'
    np.save(output_file, reshaped_data)
    print(f"\nSaved stocks data to: {output_file}")

    # Calculate returns using Open price (open-to-open)
    print("\nCalculating returns (open-to-open)...")
    returns = np.zeros((num_stocks, num_days))
    open_prices = reshaped_data[:, :, 0]  # Open price is at index 0
    for i in range(1, num_days):
        with np.errstate(divide='ignore', invalid='ignore'):
            returns[:, i] = (open_prices[:, i] - open_prices[:, i-1]) / open_prices[:, i-1]
    returns = np.nan_to_num(returns, nan=0, posinf=0, neginf=0)

    output_ror_file = 'ror.npy'
    np.save(output_ror_file, returns)
    print(f"Saved returns to: {output_ror_file}")

    # Calculate correlation matrix (industry classification proxy)
    print("\nCalculating correlation matrix...")
    # Use first 1000 days or all available days
    corr_days = min(1000, num_days)
    correlation_matrix = np.corrcoef(returns[:, :corr_days])

    output_corr_file = 'industry_classification.npy'
    np.save(output_corr_file, correlation_matrix)
    print(f"Saved correlation matrix to: {output_corr_file}")

    # Save trading dates
    output_dates_file = 'trading_dates.npy'
    np.save(output_dates_file, unique_dates)
    print(f"Saved trading dates to: {output_dates_file}")

    print("\n" + "=" * 60)
    print("=== ALL PROCESSING COMPLETE ===")
    print(f"Number of stocks: {num_stocks}")
    print(f"Number of trading days: {num_days}")
    print(f"Number of features: {num_features}")
    print(f"Feature mode: {FEATURE_MODE}")
    print(f"Using adjusted (restored) prices from FinLab API")
    if FEATURE_MODE in ['full', 'full_sentiment']:
        print(f"Technical indicators: MA20, MA60, RSI, MACD_Signal, K, D, BBands")
        print(f"Alpha factors: {len(ALPHAS)} factors")
    if FEATURE_MODE in ['basic_sentiment', 'full_sentiment']:
        print(f"Sentiment: loaded from {PRECOMPUTED_SENTIMENT_FILE}")
    print("=" * 60)
