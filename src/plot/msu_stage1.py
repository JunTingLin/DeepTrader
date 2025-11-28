"""
MSU Stage 1 Analysis

This script reads val_results.json and test_results.json from evaluation,
computes essential classification metrics, and optionally generates visualizations.

Usage:
    python msu_stage1.py --results_dir ./checkpoints/msu_stage1/1127/215051
    python msu_stage1.py --results_dir ./checkpoints/msu_stage1/1127/215051 --plot
"""
import argparse
import json
import os
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix
)


def load_results(results_path):
    """Load results JSON file"""
    with open(results_path, 'r') as f:
        return json.load(f)


def compute_metrics(results):
    """
    Compute essential classification metrics from results.

    Args:
        results (dict): Results from eval_msu_stage1.py

    Returns:
        dict: Essential metrics
    """
    preds = np.array(results['predictions']['preds'])
    labels = np.array(results['predictions']['labels'])
    probs = np.array(results['predictions']['probs'])

    # Overall metrics
    accuracy = accuracy_score(labels, preds)

    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, preds, average=None, zero_division=0
    )

    # Confusion matrix
    conf_matrix = confusion_matrix(labels, preds)
    tn, fp, fn, tp = conf_matrix.ravel()

    metrics = {
        'split': results['split'],
        'n_samples': results['n_samples'],
        'class_distribution': {
            'n_positive': results['n_positive'],
            'n_negative': results['n_negative'],
            'positive_pct': results['n_positive'] / results['n_samples'] * 100,
            'negative_pct': results['n_negative'] / results['n_samples'] * 100
        },
        'loss': results.get('loss', None),
        'accuracy': float(accuracy),
        'confusion_matrix': {
            'TN': int(tn),
            'FP': int(fp),
            'FN': int(fn),
            'TP': int(tp),
            'matrix': conf_matrix.tolist()
        },
        'per_class_metrics': {
            'class_0_negative': {
                'precision': float(precision[0]),
                'recall': float(recall[0]),
                'f1_score': float(f1[0]),
                'support': int(support[0])
            },
            'class_1_positive': {
                'precision': float(precision[1]),
                'recall': float(recall[1]),
                'f1_score': float(f1[1]),
                'support': int(support[1])
            }
        }
    }

    return metrics


def print_metrics(metrics):
    """Print metrics in a readable format"""
    print(f"\n{'='*80}")
    print(f"Analysis Results: {metrics['split'].upper()}")
    print(f"{'='*80}")

    # Dataset info
    print(f"\nClass Distribution:")
    print(f"  Total samples: {metrics['n_samples']}")
    print(f"  Positive (1): {metrics['class_distribution']['n_positive']} ({metrics['class_distribution']['positive_pct']:.2f}%)")
    print(f"  Negative (0): {metrics['class_distribution']['n_negative']} ({metrics['class_distribution']['negative_pct']:.2f}%)")

    # Overall performance
    print(f"\nOverall Performance:")
    if metrics['loss'] is not None:
        print(f"  Loss:     {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']*100:.2f}%")

    # Confusion matrix
    cm = metrics['confusion_matrix']
    print(f"\nConfusion Matrix:")
    print(f"                 Predicted")
    print(f"                 Neg    Pos")
    print(f"  Actual  Neg  {cm['TN']:5d}  {cm['FP']:5d}")
    print(f"          Pos  {cm['FN']:5d}  {cm['TP']:5d}")

    # Per-class metrics
    print(f"\nPer-Class Metrics:")
    print(f"  Class 0 (Negative):")
    cls0 = metrics['per_class_metrics']['class_0_negative']
    print(f"    Precision: {cls0['precision']:.4f}")
    print(f"    Recall:    {cls0['recall']:.4f}")
    print(f"    F1-Score:  {cls0['f1_score']:.4f}")
    print(f"    Support:   {cls0['support']}")

    print(f"\n  Class 1 (Positive):")
    cls1 = metrics['per_class_metrics']['class_1_positive']
    print(f"    Precision: {cls1['precision']:.4f}")
    print(f"    Recall:    {cls1['recall']:.4f}")
    print(f"    F1-Score:  {cls1['f1_score']:.4f}")
    print(f"    Support:   {cls1['support']}")

    print(f"{'='*80}\n")


def compare_splits(all_metrics):
    """Print comparison table across splits"""
    if len(all_metrics) < 2:
        return

    print(f"\n{'='*80}")
    print(f"Comparison Across Splits")
    print(f"{'='*80}")

    # Header
    splits = [m['split'] for m in all_metrics]
    header = f"{'Metric':<25} " + " ".join([f"{s:>10}" for s in splits])
    print(header)
    print("-" * len(header))

    # Accuracy
    accs = [m['accuracy'] * 100 for m in all_metrics]
    print(f"{'Accuracy (%)':<25} " + " ".join([f"{a:10.2f}" for a in accs]))

    # Loss
    if all(m['loss'] is not None for m in all_metrics):
        losses = [m['loss'] for m in all_metrics]
        print(f"{'Loss':<25} " + " ".join([f"{l:10.4f}" for l in losses]))

    # Class 0 F1
    cls0_f1s = [m['per_class_metrics']['class_0_negative']['f1_score'] for m in all_metrics]
    print(f"{'Class 0 F1-Score':<25} " + " ".join([f"{f:10.4f}" for f in cls0_f1s]))

    # Class 1 F1
    cls1_f1s = [m['per_class_metrics']['class_1_positive']['f1_score'] for m in all_metrics]
    print(f"{'Class 1 F1-Score':<25} " + " ".join([f"{f:10.4f}" for f in cls1_f1s]))

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Analyze MSU Stage 1 evaluation results')
    parser.add_argument('--results_dir', type=str, required=True,
                        help='Directory containing val_results.json and test_results.json')
    parser.add_argument('--splits', type=str, default='val,test',
                        help='Comma-separated splits to analyze')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for metrics (default: <results_dir>/metrics.json)')
    parser.add_argument('--plot', action='store_true',
                        help='Generate visualization plots (TODO: not implemented yet)')

    args = parser.parse_args()

    print(f"Analyzing results from: {args.results_dir}")

    all_metrics = []
    splits = args.splits.split(',')

    # Process each split
    for split in splits:
        split = split.strip()
        results_file = os.path.join(args.results_dir, f'{split}_results.json')

        if not os.path.exists(results_file):
            print(f"Warning: {results_file} not found, skipping...")
            continue

        print(f"\nProcessing {split} results...")
        results = load_results(results_file)
        metrics = compute_metrics(results)
        all_metrics.append(metrics)

        # Print metrics
        print_metrics(metrics)

    # Compare splits
    if len(all_metrics) > 1:
        compare_splits(all_metrics)

    # Save metrics
    if args.output is None:
        args.output = os.path.join(args.results_dir, 'metrics.json')

    output_data = {
        'splits': all_metrics,
        'timestamp': __import__('datetime').datetime.now().isoformat()
    }

    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"Metrics saved to: {args.output}")

    if args.plot:
        print("\nNote: Plotting functionality is not implemented yet.")
        print("You can add visualization code here in the future.")


if __name__ == '__main__':
    main()
