"""
DeepTrader Taiwan Stock Data Processing using FinLab API
Uses adjusted (restored) prices to account for dividends and stock splits
"""

import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import finlab
from finlab import data

# Load environment variables
load_dotenv()

# Initialize FinLab with API key
FINLAB_API_KEY = os.getenv('FINLAB_API_KEY')
if not FINLAB_API_KEY:
    raise ValueError("FINLAB_API_KEY not found in .env file")

finlab.login(FINLAB_API_KEY)

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
    START_DATE = '2016-01-01'
    END_DATE = '2025-12-31'

    # Feature mode: 'basic' for OHLCV, 'basic_sentiment' for OHLCV + Sentiment
    # FEATURE_MODE = 'basic_sentiment'
    FEATURE_MODE = 'basic'

    # Sentiment file (required for basic_sentiment mode)
    PRECOMPUTED_SENTIMENT_FILE = './sentiment_scores.npy'

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

    # Initialize output array for OHLCV
    print(f"\nProcessing {num_stocks} stocks...")
    reshaped_data_ohlcv = np.zeros((num_stocks, num_days, 5))

    # Process each stock sequentially
    for i, stock_id in enumerate(TWII_STOCKS):
        print(f"Processing stock {i+1}/{num_stocks}: {stock_id}")

        try:
            # Get adjusted (restored) prices from FinLab
            open_price = data.get('etl:adj_open')[stock_id][START_DATE:END_DATE]
            high_price = data.get('etl:adj_high')[stock_id][START_DATE:END_DATE]
            low_price = data.get('etl:adj_low')[stock_id][START_DATE:END_DATE]
            close_price = data.get('etl:adj_close')[stock_id][START_DATE:END_DATE]
            volume = data.get('price:成交股數')[stock_id][START_DATE:END_DATE]

            # Create DataFrame
            stock_data = pd.DataFrame({
                'Date': close_price.index,
                'open': open_price.values,
                'high': high_price.values,
                'low': low_price.values,
                'close': close_price.values,
                'volume': volume.values
            })

            # Reset index
            stock_data = stock_data.reset_index(drop=True)

        except Exception as e:
            print(f"  Error downloading {stock_id}: {e}")
            # Fill with zeros if download fails
            continue

        # Fill data for each date
        used_cols = ['open', 'high', 'low', 'close', 'volume']
        for j, date in enumerate(unique_dates):
            day_data = stock_data[stock_data['Date'] == pd.Timestamp(date)]
            if not day_data.empty:
                reshaped_data_ohlcv[i, j, :] = day_data[used_cols].values[0]

    print("\n=== Stock data processing complete ===")
    print(f"OHLCV data shape: {reshaped_data_ohlcv.shape}")

    # Check for NaN values before filling
    nan_count_before = np.isnan(reshaped_data_ohlcv).sum()
    if nan_count_before > 0:
        print(f"Found {nan_count_before} NaN values before filling")
        for stock_idx in range(num_stocks):
            stock_nan = np.isnan(reshaped_data_ohlcv[stock_idx]).sum()
            if stock_nan > 0:
                print(f"  Stock {stock_idx} ({TWII_STOCKS[stock_idx]}): {stock_nan} NaN values")

        # Fill NaN values using forward fill + backward fill
        print("\nFilling NaN values with forward fill + backward fill...")
        df_temp = pd.DataFrame(reshaped_data_ohlcv.reshape(-1, 5))
        df_temp = df_temp.ffill().bfill()
        reshaped_data_ohlcv = df_temp.values.reshape(num_stocks, num_days, 5)

        # Check NaN after filling
        nan_count_after = np.isnan(reshaped_data_ohlcv).sum()
        if nan_count_after == 0:
            print(f"✓ All NaN values filled successfully (0 NaN remaining)")
        else:
            print(f"WARNING: {nan_count_after} NaN values remain after filling")
    else:
        print(f"✓ No NaN values found")

    # Combine with sentiment scores if needed
    if FEATURE_MODE == 'basic_sentiment':
        print("\n=== Adding sentiment scores ===")
        sentiment_scores = load_precomputed_sentiment(
            PRECOMPUTED_SENTIMENT_FILE,
            num_stocks,
            num_days
        )

        # Combine OHLCV + Sentiment
        reshaped_data = np.concatenate([reshaped_data_ohlcv, sentiment_scores], axis=2)
        num_features = 6
        print(f"Final data shape: {reshaped_data.shape} (OHLCV + Sentiment)")
    else:
        reshaped_data = reshaped_data_ohlcv
        num_features = 5
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
    print(f"Using adjusted (restored) prices from FinLab API")
    print("=" * 60)
