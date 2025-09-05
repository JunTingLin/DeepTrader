import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from deeptrader_data_common import process_stocks_data

# Hardcoded DJIA stocks list
DJIA_STOCKS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

if __name__ == '__main__':
    NUM_ASU_FEATURES = 34  # change here to 5 or 34
    
    # Process DJIA stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=DJIA_STOCKS,
        start_date='2015-01-01',      # Final data range start
        end_date='2025-08-31',         # Final data range end
        num_asu_features=NUM_ASU_FEATURES,
        output_prefix='./',
        download_start_date='2000-01-01',  # Download more data for technical indicators
        download_end_date='2025-08-31'     # Same as final end date
    )