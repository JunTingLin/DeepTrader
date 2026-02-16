import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeptrader_data_common import process_stocks_data

# 2016-01-01~2025-12-31 intersect TWII stocks list
TWII_STOCKS = [
    "1216.TW",  # 統一
    "1301.TW",  # 台塑
    "1303.TW",  # 南亞
    "2002.TW",  # 中鋼
    "2303.TW",  # 聯電
    "2308.TW",  # 台達電
    "2317.TW",  # 鴻海
    "2330.TW",  # 台積電
    "2357.TW",  # 華碩
    "2382.TW",  # 廣達
    "2395.TW",  # 研華
    "2412.TW",  # 中華電
    "2454.TW",  # 聯發科
    "2880.TW",  # 華南金
    "2881.TW",  # 富邦金
    "2882.TW",  # 國泰金
    "2884.TW",  # 玉山金
    "2885.TW",  # 元大金
    "2886.TW",  # 兆豐金
    "2887.TW",  # 台新新光金
    "2891.TW",  # 中信金
    "2892.TW",  # 第一金
    "2912.TW",  # 統一超
    "3008.TW",  # 大立光
    "3045.TW",  # 台灣大
    "3711.TW",  # 日月光投控
    "4904.TW",  # 遠傳
    "5880.TW",  # 合庫金
    "6505.TW",  # 台塑化
]

# 學姊原始清單
# TWII_STOCKS = [
#     "2330.TW", "2454.TW", "2317.TW", "2382.TW", "2308.TW", "2303.TW", "2891.TW", "3711.TW", "2881.TW", "2412.TW",
#     "2886.TW", "2882.TW", "2884.TW", "1216.TW", "2885.TW", "3231.TW", "3034.TW", "2357.TW", "2002.TW", "2892.TW",
#     "1303.TW", "5880.TW", "2379.TW", "1301.TW", "2890.TW", "3008.TW", "3037.TW", "2345.TW", "5871.TW", "3661.TW",
#     "2880.TW", "2327.TW", "2883.TW", "2301.TW", "1101.TW", "2887.TW", "2207.TW", "4938.TW", "1326.TW", "3045.TW",
#     "2395.TW", "5876.TW", "2603.TW", "1590.TW", "2912.TW", "4904.TW", "2801.TW", "6505.TW", "2408.TW"
# ]

# 2016-01-01~2025-12-31 Difference set TWII stocks list
# TWII_STOCKS = [
#     "1101.TW", "1326.TW", "1590.TW", "2207.TW", "2301.TW", "2327.TW", "2345.TW", "2379.TW", "2408.TW", "2603.TW",
#     "2801.TW", "2883.TW", "2890.TW", "3034.TW", "3037.TW", "3231.TW", "3661.TW", "4938.TW", "5871.TW", "5876.TW"
# ]

if __name__ == '__main__':
    # Choose feature mode:
    # - 'basic': OHLCV (5 features)
    # - 'basic_sentiment': OHLCV + Sentiment (6 features)
    # - 'full': all features (34 features)
    FEATURE_MODE = 'basic_sentiment'

    # Trading dates from local CSV (ensures consistency with generate_sentiment.py)
    TRADING_DATES_CSV = './market_data/^TWII.csv'

    # Sentiment file (required for basic_sentiment mode)
    # Run generate_sentiment.py first to create this file
    PRECOMPUTED_SENTIMENT_FILE = './sentiment_scores.npy'

    # Process TWII stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=TWII_STOCKS,
        start_date='2016-01-01',       # Final data range start
        end_date='2025-12-31',         # Final data range end
        feature_mode=FEATURE_MODE,     # Automatically determines feature count
        output_prefix='./',
        download_start_date='2010-01-01',  # Download more data for technical indicators
        download_end_date='2026-01-01',    # Same as final end date
        trading_dates_csv=TRADING_DATES_CSV,  # Use local CSV for trading dates
        precomputed_sentiment_file=PRECOMPUTED_SENTIMENT_FILE if FEATURE_MODE == 'basic_sentiment' else None
    )