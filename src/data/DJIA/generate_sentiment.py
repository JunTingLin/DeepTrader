"""
Generate sentiment data for DJIA stocks.

This script processes news summaries from summaries_v2/ and generates:
1. sentiment_scores.npy: (num_stocks, num_days) with values -1, 0, 1
2. cls_embeddings.npy: (num_stocks, num_days, 768)

Usage:
    python generate_sentiment.py [--start-date DATE] [--end-date DATE]

Example:
    python generate_sentiment.py --start-date 2013-01-01 --end-date 2026-02-28
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
DEFAULT_TRADING_DATES_CSV = './market_data/DIA.csv'

# Default model (Chinese summaries, same as TWII)
DEFAULT_MODEL = 'yiyanghkust/finbert-tone-chinese'

# DJIA 30 stocks list (2026-03-10 updated)
# Order must match the order in stocks_data.npy (deeptrader_data_us_mp_fill.py)
DJIA_STOCKS = [
    "AAPL",  # Apple
    "AXP",   # American Express
    "AMGN",  # Amgen
    "AMZN",  # Amazon
    "BA",    # Boeing
    "CAT",   # Caterpillar
    "CRM",   # Salesforce.com
    "CSCO",  # Cisco Systems
    "CVX",   # Chevron
    "DIS",   # Walt Disney
    "GS",    # Goldman Sachs
    "HD",    # The Home Depot
    "HON",   # Honeywell International
    "IBM",   # IBM
    "JNJ",   # Johnson & Johnson
    "JPM",   # JPMorgan Chase
    "KO",    # Coca-Cola
    "MCD",   # McDonald's
    "MMM",   # 3M
    "MRK",   # Merck Sharp & Dohme
    "MSFT",  # Microsoft
    "NKE",   # Nike
    "NVDA",  # NVIDIA
    "SHW",   # Sherwin-Williams
    "PG",    # Procter & Gamble
    "TRV",   # Travelers
    "UNH",   # UnitedHealth Group
    "V",     # Visa
    "VZ",    # Verizon
    "WMT",   # Walmart Inc
]


def get_trading_dates_from_csv(csv_path: str, start_date: str, end_date: str) -> np.ndarray:
    """
    Get trading dates from DIA.csv (same source as deeptrader_data_us_mp_fill.py).

    Args:
        csv_path: Path to DIA.csv
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)

    Returns:
        Array of trading dates
    """
    print(f"Loading trading dates from: {csv_path}")

    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df = df.set_index('Date')
    df = df.sort_index()

    # Filter by date range
    df = df[start_date:end_date]

    trading_dates = df.index.to_pydatetime()
    trading_dates = np.array(trading_dates)

    return trading_dates


def main():
    parser = argparse.ArgumentParser(description='Generate sentiment data for DJIA stocks')
    parser.add_argument('--summaries-dir', default=DEFAULT_SUMMARIES_DIR,
                        help=f'Path to summaries directory (default: {DEFAULT_SUMMARIES_DIR})')
    parser.add_argument('--trading-dates-csv', default=DEFAULT_TRADING_DATES_CSV,
                        help=f'Path to DIA.csv for trading dates (default: {DEFAULT_TRADING_DATES_CSV})')
    parser.add_argument('--start-date', default='2013-01-01',
                        help='Start date for trading days (default: 2013-01-01)')
    parser.add_argument('--end-date', default='2026-02-28',
                        help='End date for trading days (default: 2026-02-28)')
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

    trading_dates_csv = args.trading_dates_csv
    if not os.path.isabs(trading_dates_csv):
        trading_dates_csv = os.path.join(script_dir, trading_dates_csv)

    # Output to current directory (same level as this script)
    output_dir = script_dir

    # Verify summaries directory exists
    if not os.path.exists(summaries_dir):
        print(f"Error: Summaries directory not found: {summaries_dir}")
        sys.exit(1)

    # Verify trading dates CSV exists
    if not os.path.exists(trading_dates_csv):
        print(f"Error: Trading dates CSV not found: {trading_dates_csv}")
        print("Run market_data/deeptrader_market.py first to generate DIA.csv")
        sys.exit(1)

    # Get trading dates from DIA.csv (same source as deeptrader_data_us_mp_fill.py)
    trading_dates = get_trading_dates_from_csv(
        csv_path=trading_dates_csv,
        start_date=args.start_date,
        end_date=args.end_date
    )
    print(f"  Found {len(trading_dates)} trading days")
    print(f"  Range: {trading_dates[0]} to {trading_dates[-1]}")

    # Verify stock directories exist
    print(f"\nVerifying stock directories in: {summaries_dir}")
    missing_stocks = []
    for stock_id in DJIA_STOCKS:
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
        stock_ids=DJIA_STOCKS,
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
