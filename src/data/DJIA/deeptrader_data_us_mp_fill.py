import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeptrader_data_common import process_stocks_data

# Hardcoded DJIA stocks list (學姐的版本)
# Note: Removed CRM (listed 2004-06-23) and V (listed 2008-03-19) to match feature34-Inter (28 stocks)
# DJIA_STOCKS = [
#     "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GS",
#     "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
#     "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA"
# ]

# 2025-12-31 updated DJIA stocks list (intersect)
DJIA_STOCKS = [
    "AXP", "BA", "CAT", "CSCO", "CVX", "DIS",
    "GS", "HD", "IBM", "JNJ", "JPM", "KO",
    "MCD", "MMM", "MRK", "MSFT", "NKE", "PG",
    "TRV", "UNH", "V", "VZ", "WMT"
]

if __name__ == '__main__':
    # Choose feature mode:
    # - 'basic': OHLCV (5 features)
    # - 'basic_sentiment': OHLCV + Sentiment (6 features)
    # - 'full': all features (34 features)
    FEATURE_MODE = 'full'

    # Trading dates from local CSV (ensures consistency with generate_sentiment.py)
    TRADING_DATES_CSV = './market_data/^DJI.csv'

    # Sentiment file (required for basic_sentiment mode)
    # Run generate_sentiment.py first to create this file
    PRECOMPUTED_SENTIMENT_FILE = './sentiment_scores.npy'

    # Process DJIA stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=DJIA_STOCKS,
        start_date='2016-01-01',       # Final data range start
        end_date='2025-12-31',         # Final data range end
        feature_mode=FEATURE_MODE,     # Automatically determines feature count
        output_prefix='./',
        download_start_date='2010-01-01',  # Download more data for technical indicators
        download_end_date='2026-01-01',    # yfinance end_date is exclusive
        trading_dates_csv=TRADING_DATES_CSV,  # Use local CSV for trading dates
        precomputed_sentiment_file=PRECOMPUTED_SENTIMENT_FILE if FEATURE_MODE == 'basic_sentiment' else None
    )