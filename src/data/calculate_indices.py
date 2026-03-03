"""
Calculate train/val/test indices for hyper.json based on real trading dates.

Usage:
    python src/data/calculate_indices.py <trading_dates.npy> [--train-end DATE] [--val-end DATE] [--test-end DATE]

Example:
    python src/data/calculate_indices.py src/data/TWII/feature5-sc29-2016-2025-ror-open-td-score-embed-finlab/trading_dates.npy --train-end 2020-12-31 --val-end 2023-12-31 --test-end 2025-12-31
"""

import argparse
import numpy as np
import pandas as pd


def calculate_indices(trading_dates_file, train_end, val_end, test_end):
    """Calculate indices for hyper.json"""
    trading_dates = np.load(trading_dates_file, allow_pickle=True)
    trading_dates = pd.DatetimeIndex(trading_dates)

    print(f"Total trading days: {len(trading_dates)}")
    print(f"Date range: {trading_dates[0].date()} to {trading_dates[-1].date()}")
    print()

    intervals = {
        "Training": (trading_dates[0].strftime('%Y-%m-%d'), train_end),
        "Validation": (train_end, val_end),
        "Test": (val_end, test_end),
    }

    results = {}
    for interval_name, (start_date, end_date) in intervals.items():
        start_dt = pd.Timestamp(start_date)
        end_dt = pd.Timestamp(end_date)

        # For Training, include start_date; for others, exclude start_date (it's the previous period's end)
        if interval_name == "Training":
            mask = (trading_dates >= start_dt) & (trading_dates <= end_dt)
        else:
            mask = (trading_dates > start_dt) & (trading_dates <= end_dt)

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
        train_idx_end = results['Training']['end_idx'] + 1  # exclusive end
        val_idx = train_idx_end
        test_idx = results['Validation']['end_idx'] + 1  # exclusive end
        test_idx_end = results['Test']['end_idx'] + 1  # exclusive end

        print("=" * 50)
        print("For hyper.json:")
        print("=" * 50)
        print(f'"train_idx": 0,')
        print(f'"train_idx_end": {train_idx_end},')
        print(f'"val_idx": {val_idx},')
        print(f'"test_idx": {test_idx},')
        print(f'"test_idx_end": {test_idx_end},')

        print()
        print("=" * 50)
        print("For src/plot/config.py (MARKET_CONFIGS):")
        print("=" * 50)
        print(f"'train_end': {train_idx_end},")
        print(f"'val_end': {test_idx},")
        print(f"'test_end': {test_idx_end},")

    return results


def main():
    parser = argparse.ArgumentParser(description='Calculate train/val/test indices for hyper.json')
    parser.add_argument('trading_dates_file', help='Path to trading_dates.npy file')
    parser.add_argument('--train-end', default='2020-12-31', help='Training end date (default: 2020-12-31, 5 years)')
    parser.add_argument('--val-end', default='2023-12-31', help='Validation end date (default: 2023-12-31, 3 years)')
    parser.add_argument('--test-end', default='2025-12-31', help='Test end date (default: 2025-12-31, 2 years)')

    args = parser.parse_args()

    calculate_indices(
        args.trading_dates_file,
        args.train_end,
        args.val_end,
        args.test_end
    )


if __name__ == '__main__':
    main()
