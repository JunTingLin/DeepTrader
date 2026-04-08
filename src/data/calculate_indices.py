"""
Calculate train/val/test indices for hyper.json based on real trading dates.

Usage:
    python src/data/calculate_indices.py <trading_dates.npy> [--train-start DATE] [--train-end DATE]
        [--val-start DATE] [--val-end DATE] [--test-start DATE] [--test-end DATE]

Example:
    python src/data/calculate_indices.py src/data/TWII/feature5-sc47-2013-2025-finlab/trading_dates.npy --train-start 2014-01-01 --train-end 2018-12-31 --val-start 2019-01-01 --val-end 2021-12-31 --test-start 2024-01-01 --test-end 2025-12-31
"""

import argparse
import numpy as np
import pandas as pd


def calculate_indices(trading_dates_file, train_start, train_end, val_start, val_end, test_start, test_end):
    """Calculate indices for hyper.json"""
    trading_dates = np.load(trading_dates_file, allow_pickle=True)
    trading_dates = pd.DatetimeIndex(trading_dates)

    print(f"Total trading days: {len(trading_dates)}")
    print(f"Date range: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print()

    # Use provided train_start or default to first date
    if train_start is None:
        train_start = trading_dates[0].strftime('%Y-%m-%d')
    if val_start is None:
        val_start = (pd.Timestamp(train_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    if test_start is None:
        test_start = (pd.Timestamp(val_end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

    intervals = {
        "Training": (train_start, train_end),
        "Validation": (val_start, val_end),
        "Test": (test_start, test_end),
    }

    results = {}
    for interval_name, (start_date, end_date) in intervals.items():
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)
        if start_dt > end_dt:
            raise ValueError(f"{interval_name} start date {start_date} is after end date {end_date}")

        mask = (trading_dates >= start_dt) & (trading_dates <= end_dt)
        interval_dates = trading_dates[mask]

        if len(interval_dates) == 0:
            print(f"{interval_name}: No trading days found in range")
            continue

        start_idx = trading_dates.get_loc(interval_dates[0])
        end_idx = trading_dates.get_loc(interval_dates[-1])
        total_days = len(interval_dates)

        results[interval_name] = {
            'start_idx': start_idx,
            'end_idx': end_idx,
            'total_days': total_days
        }

        print(f"{interval_name}:")
        print(f"  Start Index = {start_idx}")
        print(f"  End Index = {end_idx}")
        print(f"  Total Trading Days = {total_days}")
        print()

    # Output configuration format
    if 'Training' in results and 'Validation' in results and 'Test' in results:
        train_idx = results['Training']['start_idx']
        train_idx_end = results['Training']['end_idx'] + 1  # exclusive end
        val_idx = results['Validation']['start_idx']
        val_idx_end = results['Validation']['end_idx'] + 1  # exclusive end
        test_idx = results['Test']['start_idx']
        test_idx_end = results['Test']['end_idx'] + 1  # exclusive end

        print("=" * 50)
        print("For hyper.json:")
        print("=" * 50)
        print(f'"train_idx": {train_idx},')
        print(f'"train_idx_end": {train_idx_end},')
        print(f'"val_idx": {val_idx},')
        print(f'"val_idx_end": {val_idx_end},')
        print(f'"test_idx": {test_idx},')
        print(f'"test_idx_end": {test_idx_end},')

        print()
        print("=" * 50)
        print("For src/plot/config.py (MARKET_CONFIGS):")
        print("=" * 50)
        print(f"'train_start': {train_idx},")
        print(f"'train_end': {train_idx_end},")
        print(f"'val_start': {val_idx},")
        print(f"'val_end': {val_idx_end},")
        print(f"'test_start': {test_idx},")
        print(f"'test_end': {test_idx_end},")

    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate train/val/test indices for hyper.json')
    parser.add_argument('trading_dates_file', help='Path to trading_dates.npy file')
    parser.add_argument('--train-start', default=None, help='Training start date (default: first date in file)')
    parser.add_argument('--train-end', default='2020-12-31', help='Training end date (default: 2020-12-31)')
    parser.add_argument('--val-start', default=None, help='Validation start date (default: day after train-end)')
    parser.add_argument('--val-end', default='2023-12-31', help='Validation end date (default: 2023-12-31)')
    parser.add_argument('--test-start', default=None, help='Test start date (default: day after val-end)')
    parser.add_argument('--test-end', default='2025-12-31', help='Test end date (default: 2025-12-31)')

    args = parser.parse_args()

    calculate_indices(
        args.trading_dates_file,
        args.train_start,
        args.train_end,
        args.val_start,
        args.val_end,
        args.test_start,
        args.test_end
    )


if __name__ == '__main__':
    main()
