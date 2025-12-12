"""
Generate Random Rho Ground Truth Files
---------------------------------------
This script generates JSON files with random rho values (0-1) for each step.
Used for testing random rho allocation strategy.

Usage:
    python data/generate_random_rho.py \
      --data_dir data/DJIA/feature34-Inter-P532 \
      --window_len 13 \
      --trade_len 21 \
      --val_idx 1304 \
      --test_idx 2087 \
      --test_idx_end 2673 \
      --seed 42
"""

import argparse
import json
import os
import numpy as np


def generate_random_rho_file(data_dir, period_idx, period_idx_end, window_len, trade_len, seed, period='val'):
    """Generate random rho ground truth file for a specific period

    Args:
        data_dir: Output directory for the JSON file
        period_idx: Start index of the period
        period_idx_end: End index of the period
        window_len: Window length in weeks (e.g., 13)
        trade_len: Trade length in days (e.g., 21)
        seed: Random seed for reproducibility
        period: 'val' or 'test'

    Returns:
        Path to the generated file
    """

    # Set seed
    np.random.seed(seed)

    print(f"\n{'='*60}")
    print(f"Generating Random Rho File - {period.upper()} (seed={seed})")
    print(f"{'='*60}")

    # Calculate number of steps
    num_steps = (period_idx_end - period_idx) // trade_len

    print(f"Period: {period}")
    print(f"Period range: {period_idx} to {period_idx_end}")
    print(f"Trade length: {trade_len}")
    print(f"Number of steps: {num_steps}")

    # Generate random rho values (uniform distribution between 0 and 1)
    random_rhos = np.random.uniform(0.0, 1.0, size=num_steps)

    print(f"\nRandom rho statistics:")
    print(f"  Min: {random_rhos.min():.4f}")
    print(f"  Max: {random_rhos.max():.4f}")
    print(f"  Mean: {random_rhos.mean():.4f}")
    print(f"  Std: {random_rhos.std():.4f}")

    # Create ground truth records
    ground_truth_records = []
    for step in range(num_steps):
        cursor = period_idx + step * trade_len

        record = {
            "raw_mu": None,  # Not applicable for random
            "raw_sigma": None,  # Not applicable for random
            "raw_total_return": None,  # Not applicable for random
            "input_start": cursor - window_len * 5,
            "input_end": cursor + 1,
            "predict_start": cursor,
            "predict_end": cursor + trade_len + 1,
            "step": step + 1,
            "idx": cursor,
            "trend_label": None,  # Not applicable for random
            "rho": float(random_rhos[step])
        }

        ground_truth_records.append(record)

    # Create output JSON
    output_data = {
        "ground_truth_records": ground_truth_records,
        "rho_record": [float(gt["rho"]) for gt in ground_truth_records]
    }

    # Save to file
    os.makedirs(data_dir, exist_ok=True)

    output_filename = f"random_rho_{period}_seed{seed}.json"
    output_path = os.path.join(data_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Random rho file saved to: {output_path}")
    print(f"{'='*60}\n")

    return output_path


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate Random Rho Ground Truth Files')

    # Data paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory to save random rho files (e.g., data/DJIA/feature34-Inter-P532)')

    # Window parameters (required)
    parser.add_argument('--window_len', type=int, required=True,
                        help='Window length in weeks (e.g., 13)')
    parser.add_argument('--trade_len', type=int, required=True,
                        help='Trade length in days (e.g., 21)')

    # Data split indices (required)
    parser.add_argument('--val_idx', type=int, required=True,
                        help='Validation start index (e.g., 1304)')
    parser.add_argument('--test_idx', type=int, required=True,
                        help='Test start index (e.g., 2087)')
    parser.add_argument('--test_idx_end', type=int, required=True,
                        help='Test end index (e.g., 2673)')

    # Random seed
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')

    # Period selection
    parser.add_argument('--period', type=str, default='both',
                        choices=['val', 'test', 'both'],
                        help='Which period to generate: "val", "test", or "both" (default: both)')

    args = parser.parse_args()

    print("="*80)
    print("Generating Random Rho Ground Truth Files")
    print("="*80)
    print(f"Data directory: {args.data_dir}")
    print(f"Window length: {args.window_len} weeks (= {args.window_len * 5} days)")
    print(f"Trade length: {args.trade_len} days")
    print(f"Random seed: {args.seed}")
    print()

    # Generate files
    if args.period in ['val', 'both']:
        val_path = generate_random_rho_file(
            args.data_dir,
            args.val_idx,
            args.test_idx,  # val ends where test begins
            args.window_len,
            args.trade_len,
            args.seed,
            period='val'
        )
        print(f"✓ Validation file: {val_path}")

    if args.period in ['test', 'both']:
        test_path = generate_random_rho_file(
            args.data_dir,
            args.test_idx,
            args.test_idx_end,
            args.window_len,
            args.trade_len,
            args.seed,
            period='test'
        )
        print(f"✓ Test file: {test_path}")

    print("\n" + "="*80)
    print("✅ Random rho generation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
