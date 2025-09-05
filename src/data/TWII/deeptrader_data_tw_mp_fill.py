import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeptrader_data_common import process_stocks_data

# Hardcoded TWII stocks list
TWII_STOCKS = [
    "1101.TW", "1216.TW", "1301.TW", "1303.TW", "2002.TW", "2207.TW", "2301.TW", "2303.TW", "2308.TW", "2317.TW",
    "2327.TW", "2330.TW", "2345.TW", "2357.TW", "2379.TW", "2382.TW", "2383.TW", "2395.TW", "2412.TW", "2454.TW",
    "2603.TW", "2609.TW", "2615.TW", "2880.TW", "2881.TW", "2882.TW", "2883.TW", "2884.TW", "2885.TW", "2886.TW",
    "2887.TW", "2890.TW", "2891.TW", "2892.TW", "2912.TW", "3008.TW", "3017.TW", "3034.TW", "3045.TW", "3231.TW",
    "3661.TW", "3711.TW", "4904.TW", "4938.TW", "5871.TW", "5876.TW", "5880.TW", "6446.TW", "6505.TW"
]

if __name__ == '__main__':
    # Choose feature mode: 'basic' for OHLCV (5 features) or 'full' for all features (34 features)
    FEATURE_MODE = 'full'  # Change to 'basic' if you only want OHLCV features
    
    # Process TWII stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=TWII_STOCKS,
        start_date='2015-01-01',      # Final data range start
        end_date='2025-08-31',         # Final data range end
        feature_mode=FEATURE_MODE,     # Automatically determines feature count
        output_prefix='./',
        download_start_date='2000-01-01',  # Download more data for technical indicators
        download_end_date='2025-08-31'     # Same as final end date
    )