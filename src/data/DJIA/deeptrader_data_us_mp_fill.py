import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeptrader_data_common import process_stocks_data

# Hardcoded DJIA stocks list (學姐的版本)
# Note: Removed CRM (listed 2004-06-23) and V (listed 2008-03-19) to match feature34-Inter (28 stocks)
DJIA_STOCKS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA"
]

if __name__ == '__main__':
    # Choose feature mode: 'basic' for OHLCV (5 features) or 'full' for all features (34 features)
    FEATURE_MODE = 'full'  # Change to 'basic' if you only want OHLCV features
    
    # Process DJIA stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=DJIA_STOCKS,
        start_date='2000-01-01',      # Final data range start
        end_date='2025-03-31',         # Final data range end
        feature_mode=FEATURE_MODE,     # Automatically determines feature count
        output_prefix='./',
        download_start_date='2000-01-01',  # Download more data for technical indicators
        download_end_date='2025-03-31'     # Same as final end date
    )