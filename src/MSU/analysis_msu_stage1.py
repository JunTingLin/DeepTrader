"""
MSU Stage 1 Analysis (Masked Reconstruction)

This script reads val_results.json and test_results.json from evaluation,
analyzes reconstruction quality, and generates visualizations.

Key questions this script answers:
1. Is the model learning? (compare to baselines)
2. Which features are easy/hard to reconstruct?
3. Are masked positions being reconstructed accurately?

Usage:
    python MSU/analysis_msu_stage1.py --results_dir ./MSU/checkpoints/msu_stage1_masked/1205/184826
    python MSU/analysis_msu_stage1.py --results_dir ./MSU/checkpoints/msu_stage1_masked/1205/184826 --plot
"""
import argparse
import json
import os
import sys
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_results(results_path):
    """Load results JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_baselines(data_dir, split='test', mask_ratio=0.3):
    """
    Compute baseline for MASKED positions only using TRAINING set statistics (no data leakage).

    Only computes zero baseline for masked positions, since:
    1. Masked reconstruction is the main learning objective
    2. Unmasked positions should naturally stay close to 0 (MSE ≈ 0)
    3. Zero/mean/per-feature baselines are equivalent due to normalization

    Returns:
        dict: Baseline metrics (masked only)
    """
    from MSU.msu_dataset_stage1 import MSUDataset

    # 1. Load training set to compute statistics (no data leakage)
    print(f"\n  Loading TRAIN set to compute statistics...")
    train_dataset = MSUDataset(data_dir=data_dir, split='train')
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

    all_train_data = []
    for market_input in tqdm(train_loader, desc='  Loading train', leave=False):
        all_train_data.append(market_input)

    all_train_data = torch.cat(all_train_data, dim=0)  # [N_train, 13, 27]

    # Compute training statistics
    train_overall_mean = all_train_data.mean().item()

    print(f"  Train samples: {all_train_data.shape[0]}")
    print(f"  Train overall mean: {train_overall_mean:.10f}")

    # 2. Evaluate baseline on specified split (val/test)
    print(f"\n  Computing baseline for {split} split...")
    dataset = MSUDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    # Accumulator for masked baseline only
    total_mse_zero_masked = 0
    n_batches = 0

    for market_input in tqdm(dataloader, desc='  Evaluating', leave=False):
        batch_size, seq_len, num_features = market_input.shape

        # Create random mask (same as training)
        num_masked = int(seq_len * mask_ratio)  # Must match training (int not round)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for i in range(batch_size):
            masked_indices = torch.randperm(seq_len)[:num_masked]
            mask[i, masked_indices] = True

        mask_expanded = mask.unsqueeze(-1).expand_as(market_input)

        # Ground truth for masked positions only
        ground_truth_masked = market_input[mask_expanded]

        # Baseline: Predict Zero for masked positions
        pred_zero_masked = torch.zeros_like(ground_truth_masked)
        mse_zero_masked = F.mse_loss(pred_zero_masked, ground_truth_masked)

        # Accumulate
        total_mse_zero_masked += mse_zero_masked.item()
        n_batches += 1

    return {
        # Masked positions baseline (zero prediction)
        'mse_baseline_masked': float(total_mse_zero_masked / n_batches),

        # Training statistics (for reference)
        'train_overall_mean': float(train_overall_mean),
        'train_n_samples': int(all_train_data.shape[0]),
    }


def analyze_reconstruction_quality(results, baselines):
    """
    Analyze reconstruction quality and compare to baseline (masked only).

    Note: Does NOT include learning assessment (that's generated at print time).
          JSON should only contain objective metrics.

    Args:
        results (dict): Results from eval_msu_stage1.py
        baselines (dict): Baseline metrics (masked only)

    Returns:
        dict: Analysis metrics (objective only)
    """
    # Model performance
    model_mse_masked = results['mse_masked']
    model_mse_unmasked = results.get('mse_unmasked', None)
    model_mse_all = results.get('mse_all', None)

    # Compute improvement for masked positions only
    baseline_mse_masked = baselines['mse_baseline_masked']
    improvement_pct = (baseline_mse_masked - model_mse_masked) / baseline_mse_masked * 100

    # Analyze per-feature reconstruction
    mse_per_feature = np.array(results['mse_per_feature'])

    analysis = {
        'split': results['split'],
        'n_samples': results['n_samples'],
        'mask_ratio': results['mask_ratio'],
        'model_performance': {
            'mse_masked': float(model_mse_masked),
            'mse_unmasked': float(model_mse_unmasked) if model_mse_unmasked is not None else None,
            'mse_all': float(model_mse_all) if model_mse_all is not None else None,
            'mae_masked': float(results.get('mae_masked', None)),
        },
        'baseline': {
            'mse_masked': float(baseline_mse_masked),
        },
        'improvement': {
            'masked_pct': float(improvement_pct),
        },
        'per_feature_analysis': {
            'mean_mse': float(mse_per_feature.mean()),
            'std_mse': float(mse_per_feature.std()),
            'min_mse': float(mse_per_feature.min()),
            'max_mse': float(mse_per_feature.max()),
            'easiest_5_features': np.argsort(mse_per_feature)[:5].tolist(),
            'hardest_5_features': np.argsort(mse_per_feature)[-5:][::-1].tolist(),
        },
        # Keep raw baseline data for reference
        'baseline_details': baselines,
    }

    return analysis


def print_analysis(analysis):
    """
    Print analysis in a readable format.

    Note: Learning assessment is generated dynamically here (not stored in JSON).
    """
    print(f"\n{'='*80}")
    print(f"Reconstruction Analysis: {analysis['split'].upper()}")
    print(f"{'='*80}")

    # Dataset info
    print(f"\nDataset Info:")
    print(f"  Total samples: {analysis['n_samples']}")
    print(f"  Mask ratio:    {analysis['mask_ratio']:.1%}")

    # Model performance
    perf = analysis['model_performance']
    print(f"\nModel Performance:")
    print(f"  MSE (masked):   {perf['mse_masked']:.6f}")
    if perf['mse_unmasked'] is not None:
        print(f"  MSE (unmasked): {perf['mse_unmasked']:.6f}")
        print(f"  MSE (overall):  {perf['mse_all']:.6f}")
        print(f"  MAE (masked):   {perf['mae_masked']:.6f}")
        print(f"  Ratio (M/U):    {perf['mse_masked']/perf['mse_unmasked']:.1f}x")

    # Baseline comparison (masked only)
    baseline = analysis['baseline']
    improvement = analysis['improvement']

    print(f"\nBaseline Comparison:")
    print(f"  {'Metric':<20} {'Model':>12} {'Baseline':>12} {'Improvement':>12}")
    print(f"  {'-'*60}")
    print(f"  {'Masked':<20} {perf['mse_masked']:>12.6f} {baseline['mse_masked']:>12.6f} {improvement['masked_pct']:>11.1f}%")

    # Show unmasked MSE (no baseline needed - should be close to 0)
    if perf['mse_unmasked'] is not None:
        print(f"  {'Unmasked':<20} {perf['mse_unmasked']:>12.6f} {'~0.0':>12} {'(near perfect)':>12}")

    print(f"\n  Note: Baseline uses zero prediction on normalized data (mean≈0)")
    print(f"        Unmasked positions should naturally stay close to original (MSE≈0)")

    # Learning assessment (based on masked improvement)
    if 'masked_pct' in improvement:
        best_improvement_pct = improvement['masked_pct']

        print(f"\nLearning Assessment:")
        if best_improvement_pct > 50:
            status = "EXCELLENT"
            emoji = "✅"
            msg = "Model is learning VERY well! Significant improvement over baselines."
        elif best_improvement_pct > 30:
            status = "GOOD"
            emoji = "✅"
            msg = "Model is learning well. Noticeable improvement over baselines."
        elif best_improvement_pct > 10:
            status = "MODEST"
            emoji = "⚠️"
            msg = "Model shows some learning, but improvement is modest."
        else:
            status = "POOR"
            emoji = "❌"
            msg = "Model is NOT learning effectively. Consider adjusting hyperparameters."

        print(f"  Status: {status}")
        print(f"  {emoji} {msg}")

    # Per-feature analysis
    per_feat = analysis['per_feature_analysis']
    print(f"\nPer-Feature MSE Statistics (masked positions):")
    print(f"  Mean: {per_feat['mean_mse']:.6f}")
    print(f"  Std:  {per_feat['std_mse']:.6f}")
    print(f"  Min:  {per_feat['min_mse']:.6f} (Feature {per_feat['easiest_5_features'][0]})")
    print(f"  Max:  {per_feat['max_mse']:.6f} (Feature {per_feat['hardest_5_features'][0]})")

    print(f"\n  Top 5 Easiest Features to Reconstruct:")
    for feat_idx in per_feat['easiest_5_features']:
        print(f"    Feature {feat_idx:2d}")

    print(f"\n  Top 5 Hardest Features to Reconstruct:")
    for feat_idx in per_feat['hardest_5_features']:
        print(f"    Feature {feat_idx:2d}")

    print(f"{'='*80}\n")


def compare_splits(all_analyses):
    """Print comparison table across splits"""
    if len(all_analyses) < 2:
        return

    print(f"\n{'='*80}")
    print(f"Comparison Across Splits")
    print(f"{'='*80}")

    # Header
    splits = [a['split'] for a in all_analyses]
    header = f"{'Metric':<40} " + " ".join([f"{s:>12}" for s in splits])
    print(header)
    print("-" * len(header))

    # Model Performance
    print(f"\n{'Model Performance:':<40}")

    model_mses_masked = [a['model_performance']['mse_masked'] for a in all_analyses]
    print(f"{'  MSE (Masked)':<40} " + " ".join([f"{m:12.6f}" for m in model_mses_masked]))

    model_mses_unmasked = [a['model_performance']['mse_unmasked'] for a in all_analyses]
    print(f"{'  MSE (Unmasked)':<40} " + " ".join([f"{m:12.6f}" for m in model_mses_unmasked]))

    model_mses_all = [a['model_performance']['mse_all'] for a in all_analyses]
    print(f"{'  MSE (Overall)':<40} " + " ".join([f"{m:12.6f}" for m in model_mses_all]))

    # Baselines (masked only)
    print(f"\n{'Baseline (Zero):':<40}")

    if 'baseline' in all_analyses[0] and 'mse_masked' in all_analyses[0]['baseline']:
        baseline_masked = [a['baseline']['mse_masked'] for a in all_analyses]
        print(f"{'  MSE (Masked)':<40} " + " ".join([f"{m:12.6f}" for m in baseline_masked]))

    # Improvements
    print(f"\n{'Improvement (vs Baseline):':<40}")

    if 'improvement' in all_analyses[0] and 'masked_pct' in all_analyses[0]['improvement']:
        impr_masked = [a['improvement']['masked_pct'] for a in all_analyses]
        print(f"{'  Masked':<40} " + " ".join([f"{i:+12.1f}%" for i in impr_masked]))

    # Ratio
    print(f"\n{'Ratio (Masked/Unmasked):':<40}")
    ratios = [a['model_performance']['mse_masked'] / a['model_performance']['mse_unmasked'] for a in all_analyses]
    print(f"{'  Ratio':<40} " + " ".join([f"{r:12.1f}x" for r in ratios]))

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze MSU Stage 1 masked reconstruction results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing val_results.json and test_results.json')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data directory (for computing baselines). If not provided, looks for config.json')
    parser.add_argument('--splits', type=str, default='test,val',
                        help='Comma-separated splits to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for analysis (default: <results_dir>/analysis.json)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots (TODO: not implemented yet)')

    args = parser.parse_args()

    print("="*80)
    print("MSU Stage 1 Reconstruction Analysis")
    print("="*80)
    print(f"Results directory: {args.results_dir}")

    # Load config to get data_dir if not provided
    if args.data_dir is None:
        config_file = os.path.join(args.results_dir, 'config.json')
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                config = json.load(f)
                args.data_dir = config.get('data_dir', './data/DJIA/feature34-Inter-P532')
                print(f"Data directory:    {args.data_dir} (from config.json)")
        else:
            print("Warning: config.json not found, using default data_dir")
            args.data_dir = './data/DJIA/feature34-Inter-P532'
    else:
        print(f"Data directory:    {args.data_dir}")

    all_analyses = []
    all_baselines = {}
    splits = args.splits.split(',')

    # Process each split
    for split in splits:
        split = split.strip()
        results_file = os.path.join(args.results_dir, f'{split}_results.json')

        if not os.path.exists(results_file):
            print(f"\nWarning: {results_file} not found, skipping...")
            continue

        print(f"\n{'='*80}")
        print(f"Processing {split.upper()} split...")
        print(f"{'='*80}")

        # Load evaluation results
        results = load_results(results_file)
        mask_ratio = results.get('mask_ratio', 0.3)

        # Compute baselines for this split
        baselines = compute_baselines(args.data_dir, split=split, mask_ratio=mask_ratio)
        all_baselines[split] = baselines

        # Analyze reconstruction quality
        analysis = analyze_reconstruction_quality(results, baselines)
        all_analyses.append(analysis)

        # Print analysis
        print_analysis(analysis)

    # Compare splits
    if len(all_analyses) > 1:
        compare_splits(all_analyses)

    # Save analysis
    if args.output is None:
        args.output = os.path.join(args.results_dir, 'analysis.json')

    output_data = {
        'analyses': all_analyses,
        'baselines': all_baselines,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Analysis saved to: {args.output}")
    print(f"{'='*80}\n")

    if args.plot:
        print("Note: Plotting functionality is not implemented yet.")
        print("You can add visualization code using matplotlib here in the future.")


if __name__ == '__main__':
    main()
