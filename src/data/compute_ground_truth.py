"""
Compute Ground Truth for MSU and ASU Stage 1 Pretraining

This script computes ground truth labels for:
1. MSU (Market State Unit): Market-level trend prediction
2. ASU (Asset Selection Unit): Asset-level return prediction (TODO)

Features:
- Supports multiple data sources (fake data, DJIA, TWII, etc.)
- Feature index selection for MSU and ASU
- Configurable output naming (e.g., MSU_train_ground_truth.json)
- Robust error handling
"""

import numpy as np
import json
import os
import argparse
from pathlib import Path


# ============================================================================
# MSU Ground Truth Computation
# ============================================================================

def extract_weekly_data(daily_data, window_len_weeks):
    """
    Extract weekly data from daily data (every 5 days)

    Args:
        daily_data: [window_days] 1D array
        window_len_weeks: Number of weeks (13)

    Returns:
        weekly_data: [window_len_weeks] 1D array, or None if insufficient data
    """
    if len(daily_data) < window_len_weeks * 5:
        return None

    # Take every 5th day starting from day 4 (0-indexed: day 4, 9, 14, ...)
    weekly_data = daily_data[4::5][:window_len_weeks]
    return weekly_data


def normalize_weekly_data(weekly_data):
    """
    Z-score normalization

    Args:
        weekly_data: [window_len_weeks]

    Returns:
        normed: [window_len_weeks]
        mean: scalar
        std: scalar
    """
    mean = np.mean(weekly_data)
    std = np.std(weekly_data)

    # Avoid division by zero
    std = std if std != 0 else 1.0

    normed = (weekly_data - mean) / std

    return normed, mean, std


def compute_msu_ground_truth_for_step(market_data, idx, window_len_weeks, trade_len_days, feature_idx=0):
    """
    Compute MSU ground truth for one prediction step.

    Args:
        market_data: Market price data [T, num_features]
        idx: Current index (cursor position)
        window_len_weeks: Input window length in weeks (13)
        trade_len_days: Prediction horizon in days (21)
        feature_idx: Which feature to use (default: 0)

    Returns:
        dict with MSU ground truth, or None if invalid
    """
    # Step 1: Extract input window (what MSU sees)
    window_days = window_len_weeks * 5
    input_start = idx - window_days + 1
    input_end = idx + 1

    if input_start < 0 or input_end > len(market_data):
        return None

    # Extract single feature column
    prices = market_data[:, feature_idx]
    input_daily = prices[input_start:input_end]

    input_weekly = extract_weekly_data(input_daily, window_len_weeks)
    if input_weekly is None:
        return None

    # Step 2: Normalize input window (for reference, not used for ground truth)
    input_normed, input_mean, input_std = normalize_weekly_data(input_weekly)

    # Step 3: Get prediction window (future 21 days)
    pred_start = idx
    pred_end = idx + 1 + trade_len_days

    if pred_end > len(prices):
        return None

    pred_prices = prices[pred_start:pred_end]

    # Step 4: Compute raw returns
    raw_returns = np.diff(pred_prices) / pred_prices[:-1]
    raw_mu = np.mean(raw_returns)  # Average daily return
    raw_sigma = np.std(raw_returns)
    raw_total_return = (pred_prices[-1] - pred_prices[0]) / pred_prices[0]

    return {
        # Ground truth: raw_mu (average daily return)
        'raw_mu': float(raw_mu),
        'raw_sigma': float(raw_sigma),
        'raw_total_return': float(raw_total_return),

        # Index information
        'input_start': int(input_start),
        'input_end': int(input_end),
        'predict_start': int(pred_start),
        'predict_end': int(pred_end),
    }


def compute_msu_ground_truth_batch(market_data, start_idx, end_idx, window_len_weeks,
                                   trade_len_days, step=21, feature_idx=0):
    """
    Compute MSU ground truth for a batch of steps.

    Args:
        market_data: Market price data [T, num_features]
        start_idx: Start index
        end_idx: End index
        window_len_weeks: Input window length in weeks (13)
        trade_len_days: Prediction horizon in days (21)
        step: Sliding window step (1 for training, 21 for val/test)
        feature_idx: Which feature to use (default: 0)

    Returns:
        list of ground truth records
    """
    print(f"Computing MSU ground truth for indices [{start_idx}, {end_idx}] with step={step}")

    # Adjust start_idx to ensure we have enough historical data
    window_days = window_len_weeks * 5
    min_valid_idx = window_days - 1
    if start_idx < min_valid_idx:
        print(f"Note: Adjusting start_idx from {start_idx} to {min_valid_idx} (need {window_days} days of history)")
        current_idx = min_valid_idx
    else:
        current_idx = start_idx

    ground_truths = []
    step_num = 0

    while True:
        step_num += 1

        gt = compute_msu_ground_truth_for_step(market_data, current_idx, window_len_weeks,
                                               trade_len_days, feature_idx)

        if gt is None:
            break

        gt['step'] = step_num
        gt['idx'] = current_idx
        ground_truths.append(gt)

        if step_num <= 10 or step_num % 100 == 0:
            print(f"Step {step_num:4d}: idx={current_idx:4d}, raw_mu={gt['raw_mu']:9.6f}")

        # Move to next period
        current_idx += step

        # Check if we've reached the end
        if current_idx + trade_len_days > end_idx:
            break

    print(f"Generated {len(ground_truths)} MSU ground truth samples")
    return ground_truths


def add_msu_trend_label(ground_truths):
    """
    Add trend_label for MSU Stage 1 binary classification.

    Args:
        ground_truths: list of ground truth records

    Returns:
        Updated ground_truths with 'trend_label' field
    """
    for gt in ground_truths:
        # Binary label: 1.0 if uptrend, 0.0 otherwise
        gt['trend_label'] = 1.0 if gt['raw_mu'] > 0 else 0.0

    # Print statistics
    trend_labels = [gt['trend_label'] for gt in ground_truths]
    uptrend_count = sum(trend_labels)
    print(f"MSU Trend labels: {uptrend_count}/{len(trend_labels)} uptrend ({100*uptrend_count/len(trend_labels):.1f}%)")

    return ground_truths


def normalize_msu_rho(ground_truths):
    """
    Normalize raw_mu to rho using min-max normalization.

    Args:
        ground_truths: list of ground truth records

    Returns:
        Updated ground_truths with 'rho' field, min_raw_mu, max_raw_mu
    """
    raw_mu_values = np.array([gt['raw_mu'] for gt in ground_truths])
    min_raw_mu = raw_mu_values.min()
    max_raw_mu = raw_mu_values.max()

    print(f"MSU raw_mu range: [{min_raw_mu:.10f}, {max_raw_mu:.10f}]")

    for gt in ground_truths:
        rho = (gt['raw_mu'] - min_raw_mu) / (max_raw_mu - min_raw_mu)
        gt['rho'] = float(rho)

    return ground_truths, float(min_raw_mu), float(max_raw_mu)


# ============================================================================
# ASU Ground Truth Computation (TODO)
# ============================================================================

def compute_asu_ground_truth_batch(stocks_data, start_idx, end_idx, window_len_weeks,
                                   trade_len_days, step=21, feature_idx=0):
    """
    Compute ASU ground truth for a batch of steps.

    Args:
        stocks_data: Stock price data [T, num_stocks, num_features]
        start_idx: Start index
        end_idx: End index
        window_len_weeks: Input window length in weeks (13)
        trade_len_days: Prediction horizon in days (21)
        step: Sliding window step (1 for training, 21 for val/test)
        feature_idx: Which feature to use (default: 0)

    Returns:
        list of ground truth records

    TODO: Implement ASU ground truth computation
    - For each stock, compute future returns
    - Normalize using min-max or ranking
    - Return normalized scores for each stock
    """
    print(f"[WARNING] ASU ground truth computation not yet implemented!")
    print(f"          Will return empty list for now.")
    return []


# ============================================================================
# Main Workflow
# ============================================================================

def save_ground_truth(ground_truths, min_val, max_val, config, output_path):
    """
    Save ground truth to JSON file.

    Args:
        ground_truths: list of ground truth records
        min_val: minimum value (min_raw_mu for MSU)
        max_val: maximum value (max_raw_mu for MSU)
        config: configuration dict
        output_path: output file path
    """
    output = {
        'ground_truth_records': ground_truths,
        'rho_record': [gt.get('rho') for gt in ground_truths],
        'raw_mu_record': [gt.get('raw_mu') for gt in ground_truths],
        'raw_total_return_record': [gt.get('raw_total_return') for gt in ground_truths],
        'trend_label_record': [gt.get('trend_label') for gt in ground_truths],
        'normalization': f'min-max normalization of raw_mu within {config["split"].upper()} period',
        'method': 'direct_raw_returns',
        'min_raw_mu': min_val,
        'max_raw_mu': max_val,
        'metadata': {
            'module': config.get('module', 'msu'),
            'window_len': config['window_len'],
            'window_len_days': config['window_len'] * 5,
            'window_format': 'weekly averages (input only)',
            'trade_len': config['trade_len'],
            'start_idx': config['start_idx'],
            'end_idx': config['end_idx'],
            'step': config['step'],
            'num_steps': len(ground_truths),
            'feature_idx': config.get('feature_idx', 0),
            'ground_truth_description': 'raw_mu = average daily return over next 21 days',
            'rho_description': f'rho = (raw_mu - min_raw_mu) / (max_raw_mu - min_raw_mu) [computed within {config["split"].upper()} period]',
            'trend_label_description': 'trend_label = 1.0 if raw_mu > 0 else 0.0 (for MSU Stage 1 binary classification)'
        }
    }

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✅ Saved to: {output_path}")


def compute_ground_truth_main(args):
    """
    Main function to compute ground truth.

    Args:
        args: argparse arguments
    """
    print("="*80)
    print(f"Computing Ground Truth for {args.module.upper()}")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Module: {args.module}")
    print(f"Feature index: {args.feature_idx}")
    print(f"Window length: {args.window_len} weeks (= {args.window_len * 5} days)")
    print(f"Trade length: {args.trade_len} days")
    print()

    # Load data
    if args.module == 'msu':
        data_path = os.path.join(args.data_dir, args.market_data_file)
        print(f"Loading market data from: {data_path}")
        data = np.load(data_path)
        print(f"Data shape: {data.shape}")
        print(f"Using feature index: {args.feature_idx}")
        print()
    elif args.module == 'asu':
        data_path = os.path.join(args.data_dir, args.stocks_data_file)
        print(f"Loading stocks data from: {data_path}")
        data = np.load(data_path)
        print(f"Data shape: {data.shape}")
        print(f"Using feature index: {args.feature_idx}")
        print()

    # Compute ground truth for each split
    for split in ['train', 'val', 'test']:
        print("="*80)
        print(f"Computing {split.upper()} set")
        print("="*80)

        # Get indices
        if split == 'train':
            start_idx = args.train_idx
            end_idx = args.train_idx_end
            step = args.train_step
        elif split == 'val':
            start_idx = args.val_idx
            end_idx = args.test_idx
            step = args.val_step
        else:  # test
            start_idx = args.test_idx
            end_idx = args.test_idx_end
            step = args.test_step

        print(f"Indices: [{start_idx}, {end_idx}], step={step}")
        print()

        # Compute ground truth
        if args.module == 'msu':
            ground_truths = compute_msu_ground_truth_batch(
                data, start_idx, end_idx, args.window_len, args.trade_len,
                step=step, feature_idx=args.feature_idx
            )
            ground_truths = add_msu_trend_label(ground_truths)
            ground_truths, min_val, max_val = normalize_msu_rho(ground_truths)
        elif args.module == 'asu':
            ground_truths = compute_asu_ground_truth_batch(
                data, start_idx, end_idx, args.window_len, args.trade_len,
                step=step, feature_idx=args.feature_idx
            )
            min_val = None
            max_val = None

        # Save
        config = {
            'module': args.module,
            'split': split,
            'window_len': args.window_len,
            'trade_len': args.trade_len,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'step': step,
            'feature_idx': args.feature_idx,
        }

        output_filename = f"{args.module.upper()}_{split}_ground_truth.json"
        output_path = os.path.join(args.data_dir, output_filename)
        save_ground_truth(ground_truths, min_val, max_val, config, output_path)
        print()

    print("="*80)
    print("✅ All ground truth computed successfully!")
    print("="*80)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compute ground truth for MSU and ASU Stage 1 pretraining')

    # Module selection
    parser.add_argument('--module', type=str, default='msu', choices=['msu', 'asu'],
                        help='Which module to compute ground truth for (msu or asu)')

    # Data paths
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing data files (default: same directory as this script)')
    parser.add_argument('--market_data_file', type=str, default='market_data.npy',
                        help='Market data filename for MSU (default: market_data.npy)')
    parser.add_argument('--stocks_data_file', type=str, default='stocks_data.npy',
                        help='Stocks data filename for ASU (default: stocks_data.npy)')

    # Feature selection
    parser.add_argument('--feature_idx', type=int, default=0,
                        help='Feature index to use (default: 0, first feature)')

    # Window parameters (required)
    parser.add_argument('--window_len', type=int, required=True,
                        help='Window length in weeks (e.g., 13)')
    parser.add_argument('--trade_len', type=int, required=True,
                        help='Trade length in days (e.g., 21)')

    # Data split indices (required)
    parser.add_argument('--train_idx', type=int, required=True,
                        help='Training start index (e.g., 0)')
    parser.add_argument('--train_idx_end', type=int, required=True,
                        help='Training end index (e.g., 1304)')
    parser.add_argument('--val_idx', type=int, required=True,
                        help='Validation start index (e.g., 1304)')
    parser.add_argument('--test_idx', type=int, required=True,
                        help='Test start index (e.g., 2087)')
    parser.add_argument('--test_idx_end', type=int, required=True,
                        help='Test end index (e.g., 2673)')

    # Step sizes for sliding window
    parser.add_argument('--train_step', type=int, default=1,
                        help='Step size for training sliding window (default: 1)')
    parser.add_argument('--val_step', type=int, default=21,
                        help='Step size for validation sliding window (default: 21)')
    parser.add_argument('--test_step', type=int, default=21,
                        help='Step size for test sliding window (default: 21)')

    args = parser.parse_args()

    # If data_dir not specified, use the directory of this script
    if args.data_dir is None:
        args.data_dir = os.path.dirname(os.path.abspath(__file__))

    compute_ground_truth_main(args)
