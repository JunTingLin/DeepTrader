"""
Compute CORRECT ground truth using the exact same method as MSU sees the data.

Key corrections:
1. Window is 13 WEEKLY averages (not 91 or 65 daily values)
2. Each week = average of 5 consecutive days
3. Normalize over 13 weekly values
4. Use INPUT window statistics to normalize prediction window (NOT prediction's own stats)
"""
import numpy as np
import json
import os

def extract_weekly_data(prices, start_idx, window_len_weeks):
    """
    Extract and convert to weekly data exactly as portfolio_env does.

    Args:
        prices: Raw daily price data
        start_idx: Starting index
        window_len_weeks: Window length in weeks (13)

    Returns:
        weekly_data: (window_len_weeks, features) array of weekly averages
    """
    # Extract window_len * 5 days
    num_days = window_len_weeks * 5
    end_idx = start_idx + num_days

    if end_idx > len(prices):
        return None

    daily_data = prices[start_idx:end_idx]

    # Reshape to (weeks, 5 days, features)
    weekly_data = daily_data.reshape(window_len_weeks, 5, -1)

    # Average over 5 days per week
    weekly_avg = np.mean(weekly_data, axis=1)

    return weekly_avg

def normalize_weekly_data(weekly_data):
    """
    Normalize weekly data using z-score over the window dimension.
    Exactly matches __normalize_market() in portfolio_env.py line 252.

    Args:
        weekly_data: (window_len, features) array

    Returns:
        normed: Normalized data
        mean: Mean used for normalization
        std: Std used for normalization
    """
    # axis=0 because shape is (window_len, features)
    mean = weekly_data.mean(axis=0, keepdims=True)
    std = weekly_data.std(axis=0, keepdims=True)

    # Avoid division by zero
    std = np.where(std == 0, 1, std)

    normed = (weekly_data - mean) / std

    return normed, mean, std

def compute_ground_truth_for_step(prices, idx, window_len_weeks, trade_len_days):
    """
    Compute ground truth for one prediction step.

    Args:
        prices: Raw price data
        idx: Current index (cursor position in portfolio_env)
        window_len_weeks: Input window length in weeks (13)
        trade_len_days: Prediction horizon in days (21)

    Returns:
        dict with mu, sigma, rho
    """
    # Step 1: Extract input window (what MSU sees)
    # MSU sees window_len * 5 = 65 days of market data (13 weeks)
    input_start = idx - window_len_weeks * 5 + 1  # 65 days
    input_end = idx + 1

    input_weekly = extract_weekly_data(prices, input_start, window_len_weeks)
    if input_weekly is None:
        return None

    # Step 2: Normalize input window
    input_normed, input_mean, input_std = normalize_weekly_data(input_weekly)

    # Step 3: Get prediction window (future 21 days)
    # IMPORTANT: Agent sees data up to idx (cursor), then holds positions from idx+1 to idx+21
    # But the FIRST daily return is from idx to idx+1, so we need prices from idx onwards
    # future_return = [r_{idx+1}, r_{idx+2}, ..., r_{idx+21}]
    # where r_{idx+1} = (P_{idx+1} - P_{idx}) / P_{idx}
    # So the total return is from P_{idx} to P_{idx+21}
    pred_start = idx  # Include cursor day for calculating the first return
    pred_end = idx + 1 + trade_len_days

    if pred_end > len(prices):
        return None

    pred_prices = prices[pred_start:pred_end].flatten()

    # Step 4: Normalize prediction using INPUT window statistics (KEY!)
    # This is what MSU would need to predict
    pred_normed = (pred_prices - input_mean[0][0]) / input_std[0][0]

    # Step 5: Compute returns from normalized prices
    pred_returns = np.diff(pred_normed) / (pred_normed[:-1] + 1e-10)

    # Ground truth mu and sigma
    gt_mu = np.mean(pred_returns)
    gt_sigma = np.std(pred_returns)

    # Rho = clamp(mu, 0, 1) as per original DeepTrader paper
    # In test/val mode, rho is simply mu clamped to [0, 1]
    rho = np.clip(gt_mu, 0.0, 1.0)

    # Also compute raw (non-normalized) statistics for reference
    raw_returns = np.diff(pred_prices) / pred_prices[:-1]
    raw_mu = np.mean(raw_returns)
    raw_sigma = np.std(raw_returns)
    raw_total_return = (pred_prices[-1] - pred_prices[0]) / pred_prices[0]

    return {
        # Ground truth (what MSU should predict)
        'mu': float(gt_mu),
        'sigma': float(gt_sigma),
        'rho': float(rho),

        # Normalization statistics (for debugging/verification)
        'input_mean': float(input_mean[0][0]),
        'input_std': float(input_std[0][0]),

        # Raw data statistics (for reference)
        'raw_mu': float(raw_mu),
        'raw_sigma': float(raw_sigma),
        'raw_total_return': float(raw_total_return),

        # Index information (for debugging)
        'input_start': int(input_start),  # 65 days (what MSU actually sees)
        'input_end': int(input_end),
        'predict_start': int(pred_start),
        'predict_end': int(pred_end),
    }

def compute_all_ground_truth(data_file, window_len_weeks, trade_len_days,
                             start_idx, end_idx, step=21):
    """
    Compute ground truth for all test/validation periods.
    """
    prices = np.load(data_file)
    print(f"Loaded price data shape: {prices.shape}")
    print(f"Price range: {prices.min():.2f} to {prices.max():.2f}")

    ground_truths = []
    current_idx = start_idx
    step_num = 0

    while True:
        step_num += 1

        gt = compute_ground_truth_for_step(prices, current_idx, window_len_weeks, trade_len_days)

        if gt is None:
            break

        gt['step'] = step_num
        ground_truths.append(gt)

        print(f"Step {step_num:2d}: idx={current_idx:4d}, "
              f"mu={gt['mu']:7.4f}, sigma={gt['sigma']:7.4f}, rho={gt['rho']:7.4f} | "
              f"input: mean={gt['input_mean']:5.2f}, std={gt['input_std']:5.2f}")

        # Move to next period
        current_idx += step

        # Check if we've reached the end
        # We can continue as long as the prediction window doesn't exceed end_idx
        if current_idx + trade_len_days > end_idx:
            break

    return ground_truths

def main():
    # Load configuration
    # Script is in src/data/fake/, config is in src/
    script_dir = os.path.dirname(__file__)
    config_path = os.path.join(script_dir, '..', '..', 'hyper.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    window_len = config['window_len']  # 13 weeks
    trade_len = config['trade_len']    # 21 days
    test_idx = config['test_idx']      # 2087
    test_idx_end = config['test_idx_end']  # 2673
    val_idx = config['val_idx']        # 1304

    # market_data.npy is in the same directory as this script
    data_file = os.path.join(script_dir, 'market_data.npy')

    print("="*80)
    print("Computing Ground Truth")
    print("="*80)
    print(f"Data file: {data_file}")
    print(f"Window length: {window_len} weeks (= {window_len*5} days, averaged to {window_len} weekly values)")
    print(f"Prediction horizon: {trade_len} days")
    print(f"Normalization: Z-score over {window_len} weekly values")
    print()

    # Compute ground truth for test set
    print("Computing TEST set ground truth...")
    print(f"Test indices: {test_idx} to {test_idx_end}")
    print("-"*80)
    test_ground_truth = compute_all_ground_truth(
        data_file, window_len, trade_len, test_idx, test_idx_end
    )

    print()
    print("Computing VALIDATION set ground truth...")
    print(f"Validation indices: {val_idx} to {test_idx}")
    print("-"*80)
    val_ground_truth = compute_all_ground_truth(
        data_file, window_len, trade_len, val_idx, test_idx
    )

    # Save results in the same directory as this script
    output_dir = script_dir

    # Test ground truth
    test_output = {
        'ground_truth_records': test_ground_truth,
        'rho_record': [gt['rho'] for gt in test_ground_truth],
        'mu_record': [gt['mu'] for gt in test_ground_truth],
        'sigma_record': [gt['sigma'] for gt in test_ground_truth],
        'distribution_type': 'normal',
        'normalization': 'z-score over 13 weekly values, using INPUT window statistics',
        'method': 'weekly_average_then_normalize',
        'metadata': {
            'window_len': window_len,
            'window_len_days': window_len * 5,
            'window_format': 'weekly averages',
            'trade_len': trade_len,
            'start_idx': test_idx,
            'end_idx': test_idx_end,
            'num_steps': len(test_ground_truth)
        }
    }

    test_output_file = os.path.join(output_dir, 'test_ground_truth.json')
    with open(test_output_file, 'w') as f:
        json.dump(test_output, f, indent=2)
    print(f"\n✅ Test ground truth saved to: {test_output_file}")

    # Validation ground truth
    val_output = {
        'ground_truth_records': val_ground_truth,
        'rho_record': [gt['rho'] for gt in val_ground_truth],
        'mu_record': [gt['mu'] for gt in val_ground_truth],
        'sigma_record': [gt['sigma'] for gt in val_ground_truth],
        'distribution_type': 'normal',
        'normalization': 'z-score over 13 weekly values, using INPUT window statistics',
        'method': 'weekly_average_then_normalize',
        'metadata': {
            'window_len': window_len,
            'window_len_days': window_len * 5,
            'window_format': 'weekly averages',
            'trade_len': trade_len,
            'start_idx': val_idx,
            'end_idx': test_idx,
            'num_steps': len(val_ground_truth)
        }
    }

    val_output_file = os.path.join(output_dir, 'val_ground_truth.json')
    with open(val_output_file, 'w') as f:
        json.dump(val_output, f, indent=2)
    print(f"✅ Validation ground truth saved to: {val_output_file}")

    # Print summary statistics
    print("\n" + "="*80)
    print("Summary Statistics")
    print("="*80)

    print("\nTEST SET:")
    test_mus = [gt['mu'] for gt in test_ground_truth]
    test_sigmas = [gt['sigma'] for gt in test_ground_truth]
    test_rhos = [gt['rho'] for gt in test_ground_truth]

    print(f"  Mu    - Mean: {np.mean(test_mus):7.4f}, Std: {np.std(test_mus):7.4f}, "
          f"Range: [{np.min(test_mus):7.4f}, {np.max(test_mus):7.4f}]")
    print(f"  Sigma - Mean: {np.mean(test_sigmas):7.4f}, Std: {np.std(test_sigmas):7.4f}, "
          f"Range: [{np.min(test_sigmas):7.4f}, {np.max(test_sigmas):7.4f}]")
    print(f"  Rho   - Mean: {np.mean(test_rhos):7.4f}, Std: {np.std(test_rhos):7.4f}, "
          f"Range: [{np.min(test_rhos):7.4f}, {np.max(test_rhos):7.4f}]")

    print("\nVALIDATION SET:")
    val_mus = [gt['mu'] for gt in val_ground_truth]
    val_sigmas = [gt['sigma'] for gt in val_ground_truth]
    val_rhos = [gt['rho'] for gt in val_ground_truth]

    print(f"  Mu    - Mean: {np.mean(val_mus):7.4f}, Std: {np.std(val_mus):7.4f}, "
          f"Range: [{np.min(val_mus):7.4f}, {np.max(val_mus):7.4f}]")
    print(f"  Sigma - Mean: {np.mean(val_sigmas):7.4f}, Std: {np.std(val_sigmas):7.4f}, "
          f"Range: [{np.min(val_sigmas):7.4f}, {np.max(val_sigmas):7.4f}]")
    print(f"  Rho   - Mean: {np.mean(val_rhos):7.4f}, Std: {np.std(val_rhos):7.4f}, "
          f"Range: [{np.min(val_rhos):7.4f}, {np.max(val_rhos):7.4f}]")

    print("\n" + "="*80)
    print("✅ Done!")
    print("="*80)

if __name__ == '__main__':
    main()
