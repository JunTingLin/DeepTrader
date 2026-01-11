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

if __name__ == '__main__':
    # Choose feature mode: 'basic' for OHLCV (5 features) or 'full' for all features (34 features)
    FEATURE_MODE = 'full'  # Change to 'basic' if you only want OHLCV features
    
    # Process TWII stocks
    unique_stock_ids, reshaped_data = process_stocks_data(
        stock_list=TWII_STOCKS,
        start_date='2016-01-01',      # Final data range start
        end_date='2025-12-31',         # Final data range end
        feature_mode=FEATURE_MODE,     # Automatically determines feature count
        output_prefix='./',
        download_start_date='2000-01-01',  # Download more data for technical indicators
        download_end_date='2026-01-01'     # Same as final end date
    )