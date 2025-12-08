"""
MSU Stage 1 Analysis (Masked Reconstruction)

This script reads val_results.json and test_results.json from evaluation,
analyzes reconstruction quality, and generates visualizations.

Key questions this script answers:
1. Is the model learning? (compare to baselines)
2. Which features are easy/hard to reconstruct?
3. Are masked positions being reconstructed accurately?

Usage:
    python plot/msu_stage1.py --results_dir ./checkpoints/msu_stage1_masked/1205/184826
    python plot/msu_stage1.py --results_dir ./checkpoints/msu_stage1_masked/1205/184826 --plot
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
    Compute baseline MSE: what if we predict zeros or global mean?

    Returns:
        dict: Baseline metrics
    """
    from msu_dataset_stage1 import MSUDataset

    print(f"\n  Computing baselines for {split} split...")

    dataset = MSUDataset(data_dir=data_dir, split=split)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    all_data = []
    all_masked_data = []

    # Collect data
    for market_input in tqdm(dataloader, desc='  Collecting data', leave=False):
        batch_size, seq_len, num_features = market_input.shape

        # Create random mask
        num_masked = int(seq_len * mask_ratio)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        for i in range(batch_size):
            masked_indices = torch.randperm(seq_len)[:num_masked]
            mask[i, masked_indices] = True

        mask_expanded = mask.unsqueeze(-1).expand_as(market_input)

        all_data.append(market_input.numpy())
        all_masked_data.append(market_input[mask_expanded].numpy())

    all_data = np.concatenate([d.reshape(-1, d.shape[-1]) for d in all_data])  # [N, 27]
    all_masked_data = np.concatenate(all_masked_data)  # [M] flattened

    # Compute global mean per feature
    global_mean = all_data.mean(axis=0)  # [27]

    # Note: all_masked_data is flattened, need to compare to overall mean scalar
    overall_mean = all_data.mean()  # scalar

    # Baseline 1: Predict zeros
    mse_zero = np.mean(all_masked_data ** 2)

    # Baseline 2: Predict overall mean (scalar)
    mse_mean = np.mean((all_masked_data - overall_mean) ** 2)

    return {
        'mse_zero_baseline': float(mse_zero),
        'mse_mean_baseline': float(mse_mean),
        'global_mean': global_mean.tolist(),
    }


def analyze_reconstruction_quality(results, baselines):
    """
    Analyze reconstruction quality and compare to baselines.

    Args:
        results (dict): Results from eval_msu_stage1.py
        baselines (dict): Baseline metrics

    Returns:
        dict: Analysis metrics
    """
    model_mse = results['mse_masked']
    baseline_zero = baselines['mse_zero_baseline']
    baseline_mean = baselines['mse_mean_baseline']

    # Compute improvement percentages
    improvement_vs_zero = (baseline_zero - model_mse) / baseline_zero * 100
    improvement_vs_mean = (baseline_mean - model_mse) / baseline_mean * 100

    # Determine if model is learning
    if improvement_vs_mean > 20:
        learning_status = "excellent"
        learning_msg = "✅ Model is learning VERY well! Significant improvement over baselines."
    elif improvement_vs_mean > 10:
        learning_status = "good"
        learning_msg = "✅ Model is learning! Noticeable improvement over baselines."
    elif improvement_vs_mean > 0:
        learning_status = "modest"
        learning_msg = "⚠️  Model shows some learning, but improvement is modest."
    else:
        learning_status = "poor"
        learning_msg = "❌ Model is NOT learning effectively. Consider adjusting hyperparameters."

    # Analyze per-feature reconstruction
    mse_per_feature = np.array(results['mse_per_feature'])
    easiest_5 = np.argsort(mse_per_feature)[:5].tolist()
    hardest_5 = np.argsort(mse_per_feature)[-5:][::-1].tolist()

    analysis = {
        'split': results['split'],
        'n_samples': results['n_samples'],
        'mask_ratio': results['mask_ratio'],
        'mse_comparison': {
            'model_mse_masked': model_mse,
            'baseline_zero': baseline_zero,
            'baseline_mean': baseline_mean,
            'improvement_vs_zero_pct': float(improvement_vs_zero),
            'improvement_vs_mean_pct': float(improvement_vs_mean),
        },
        'learning_assessment': {
            'status': learning_status,
            'message': learning_msg,
        },
        'per_feature_analysis': {
            'mean_mse': float(mse_per_feature.mean()),
            'std_mse': float(mse_per_feature.std()),
            'min_mse': float(mse_per_feature.min()),
            'max_mse': float(mse_per_feature.max()),
            'easiest_5_features': easiest_5,
            'hardest_5_features': hardest_5,
        },
        'raw_metrics': {
            'mse_all': results.get('mse_all', None),
            'mse_unmasked': results.get('mse_unmasked', None),
            'mae_masked': results.get('mae_masked', None),
        }
    }

    return analysis


def print_analysis(analysis):
    """Print analysis in a readable format"""
    print(f"\n{'='*80}")
    print(f"Reconstruction Analysis: {analysis['split'].upper()}")
    print(f"{'='*80}")

    # Dataset info
    print(f"\nDataset Info:")
    print(f"  Total samples: {analysis['n_samples']}")
    print(f"  Mask ratio:    {analysis['mask_ratio']:.1%}")

    # MSE comparison
    mse_comp = analysis['mse_comparison']
    print(f"\nMSE on Masked Positions:")
    print(f"  Zero Baseline:    {mse_comp['baseline_zero']:.6f}")
    print(f"  Mean Baseline:    {mse_comp['baseline_mean']:.6f}")
    print(f"  Model:            {mse_comp['model_mse_masked']:.6f}")

    print(f"\nModel Improvement:")
    print(f"  vs Zero Baseline: {mse_comp['improvement_vs_zero_pct']:+.1f}%")
    print(f"  vs Mean Baseline: {mse_comp['improvement_vs_mean_pct']:+.1f}%")

    # Learning assessment
    assess = analysis['learning_assessment']
    print(f"\n{assess['message']}")

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

    # Raw metrics
    raw = analysis['raw_metrics']
    if raw['mse_unmasked'] is not None:
        print(f"\nAdditional Metrics:")
        print(f"  MSE (all positions):      {raw['mse_all']:.6f}")
        print(f"  MSE (unmasked only):      {raw['mse_unmasked']:.6f}")
        print(f"  MAE (masked only):        {raw['mae_masked']:.6f}")
        print(f"  Ratio (masked/unmasked):  {mse_comp['model_mse_masked']/raw['mse_unmasked']:.2f}x")

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
    header = f"{'Metric':<30} " + " ".join([f"{s:>12}" for s in splits])
    print(header)
    print("-" * len(header))

    # Model MSE
    model_mses = [a['mse_comparison']['model_mse_masked'] for a in all_analyses]
    print(f"{'Model MSE (masked)':<30} " + " ".join([f"{m:12.6f}" for m in model_mses]))

    # Baseline MSEs
    zero_baselines = [a['mse_comparison']['baseline_zero'] for a in all_analyses]
    print(f"{'Zero Baseline MSE':<30} " + " ".join([f"{m:12.6f}" for m in zero_baselines]))

    mean_baselines = [a['mse_comparison']['baseline_mean'] for a in all_analyses]
    print(f"{'Mean Baseline MSE':<30} " + " ".join([f"{m:12.6f}" for m in mean_baselines]))

    # Improvement percentages
    improvements = [a['mse_comparison']['improvement_vs_mean_pct'] for a in all_analyses]
    print(f"{'Improvement vs Mean (%)':<30} " + " ".join([f"{i:+12.1f}" for i in improvements]))

    # Learning status
    statuses = [a['learning_assessment']['status'] for a in all_analyses]
    print(f"{'Learning Status':<30} " + " ".join([f"{s:>12}" for s in statuses]))

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
