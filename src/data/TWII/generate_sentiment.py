"""
Generate sentiment data for TWII stocks.

This script processes news summaries from summaries_v2/ and generates:
1. sentiment_scores.npy: (num_stocks, num_days) with values -1, 0, 1
2. cls_embeddings.npy: (num_stocks, num_days, 768)

Usage:
    python generate_sentiment.py [--start-date DATE] [--end-date DATE]

Example:
    python generate_sentiment.py --start-date 2016-01-01 --end-date 2025-12-31
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from process_sentiment_common import generate_sentiment_data


# Default paths
DEFAULT_SUMMARIES_DIR = './summaries_v2'
DEFAULT_TWII_CSV = './market_data/^TWII.csv'

# Default model for Taiwan market (Chinese)
DEFAULT_MODEL = 'yiyanghkust/finbert-tone-chinese'

# TWII stock IDs (must match folder names in summaries_v2)
# Order must match the order in stocks_data.npy
TWII_STOCK_IDS = [
    "1216",  # 統一
    "1301",  # 台塑
    "1303",  # 南亞
    "2002",  # 中鋼
    "2303",  # 聯電
    "2308",  # 台達電
    "2317",  # 鴻海
    "2330",  # 台積電
    "2357",  # 華碩
    "2382",  # 廣達
    "2395",  # 研華
    "2412",  # 中華電
    "2454",  # 聯發科
    "2880",  # 華南金
    "2881",  # 富邦金
    "2882",  # 國泰金
    "2884",  # 玉山金
    "2885",  # 元大金
    "2886",  # 兆豐金
    "2887",  # 台新金
    "2891",  # 中信金
    "2892",  # 第一金
    "2912",  # 統一超
    "3008",  # 大立光
    "3045",  # 台灣大
    "3711",  # 日月光投控
    "4904",  # 遠傳
    "5880",  # 合庫金
    "6505",  # 台塑化
]


def get_trading_dates_from_csv(csv_path: str, start_date: str, end_date: str) -> np.ndarray:
    """
    Get real trading dates from local CSV file (same as deeptrader_market.py).

    Args:
        csv_path: Path to ^TWII.csv file
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Array of trading dates
    """
    print(f"Loading trading dates from: {csv_path}")

    ref_df = pd.read_csv(csv_path, parse_dates=['Date'])
    ref_df['Date'] = ref_df['Date'].dt.tz_localize(None)
    ref_df = ref_df[(ref_df['Date'] >= start_date) & (ref_df['Date'] <= end_date)]
    ref_df = ref_df.sort_values('Date')

    trading_dates = ref_df['Date'].dt.to_pydatetime()
    trading_dates = np.array(trading_dates)

    return trading_dates


def main():
    parser = argparse.ArgumentParser(description='Generate sentiment data for TWII stocks')
    parser.add_argument('--summaries-dir', default=DEFAULT_SUMMARIES_DIR,
                        help=f'Path to summaries directory (default: {DEFAULT_SUMMARIES_DIR})')
    parser.add_argument('--twii-csv', default=DEFAULT_TWII_CSV,
                        help=f'Path to ^TWII.csv for trading dates (default: {DEFAULT_TWII_CSV})')
    parser.add_argument('--start-date', default='2016-01-01',
                        help='Start date for trading days (default: 2016-01-01)')
    parser.add_argument('--end-date', default='2025-12-31',
                        help='End date for trading days (default: 2025-12-31)')
    parser.add_argument('--model', default=DEFAULT_MODEL,
                        help=f'FinBERT model name (default: {DEFAULT_MODEL})')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for processing (default: 16)')

    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    script_dir = os.path.dirname(os.path.abspath(__file__))

    summaries_dir = args.summaries_dir
    if not os.path.isabs(summaries_dir):
        summaries_dir = os.path.join(script_dir, summaries_dir)

    twii_csv = args.twii_csv
    if not os.path.isabs(twii_csv):
        twii_csv = os.path.join(script_dir, twii_csv)

    # Output to current directory (same level as this script)
    output_dir = script_dir

    # Verify summaries directory exists
    if not os.path.exists(summaries_dir):
        print(f"Error: Summaries directory not found: {summaries_dir}")
        sys.exit(1)

    # Verify TWII CSV exists
    if not os.path.exists(twii_csv):
        print(f"Error: TWII CSV file not found: {twii_csv}")
        sys.exit(1)

    # Get trading dates from local CSV file (same source as deeptrader_market.py)
    trading_dates = get_trading_dates_from_csv(
        csv_path=twii_csv,
        start_date=args.start_date,
        end_date=args.end_date
    )
    print(f"  Found {len(trading_dates)} trading days")
    print(f"  Range: {trading_dates[0]} to {trading_dates[-1]}")

    # Verify stock directories exist
    print(f"\nVerifying stock directories in: {summaries_dir}")
    missing_stocks = []
    for stock_id in TWII_STOCK_IDS:
        stock_dir = os.path.join(summaries_dir, stock_id)
        if not os.path.exists(stock_dir):
            missing_stocks.append(stock_id)

    if missing_stocks:
        print(f"Warning: Missing directories for stocks: {missing_stocks}")

    # Generate sentiment data
    print(f"\nGenerating sentiment data...")
    print(f"  Model: {args.model}")
    print(f"  Output: {output_dir}")

    sentiment_scores, cls_embeddings = generate_sentiment_data(
        summaries_dir=summaries_dir,
        stock_ids=TWII_STOCK_IDS,
        trading_dates=trading_dates,
        output_dir=output_dir,
        model_name=args.model,
        batch_size=args.batch_size
    )

    print("\n=== Generation Complete ===")
    print(f"sentiment_scores.npy: {sentiment_scores.shape}")
    print(f"cls_embeddings.npy: {cls_embeddings.shape}")

    # Print summary statistics
    print("\n=== Sentiment Statistics ===")
    print(f"Negative (-1): {np.sum(sentiment_scores == -1)} ({100*np.mean(sentiment_scores == -1):.1f}%)")
    print(f"Neutral (0):   {np.sum(sentiment_scores == 0)} ({100*np.mean(sentiment_scores == 0):.1f}%)")
    print(f"Positive (1):  {np.sum(sentiment_scores == 1)} ({100*np.mean(sentiment_scores == 1):.1f}%)")


if __name__ == "__main__":
    main()
