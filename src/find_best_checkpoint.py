"""
Find the best checkpoint for test period by evaluating all saved checkpoints.

Usage:
    python src/find_best_checkpoint.py --prefix ./src/outputs/0307/225821
    python src/find_best_checkpoint.py --prefix ./src/outputs/0307/225821 --metric CR
    python src/find_best_checkpoint.py --prefix ./src/outputs/0307/225821 --metric ARR --top 5
"""

import argparse
import json
import os
import re
import subprocess
import sys
from typing import Dict, List, Tuple


def get_checkpoint_files(prefix: str, include_periodic: bool = True) -> List[str]:
    """Get all checkpoint files sorted by epoch.

    Args:
        prefix: Experiment output directory
        include_periodic: If True, include epoch-{n}.pkl files. If False, only best_cr-{n}.pkl
    """
    model_dir = os.path.join(prefix, 'model_file')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = []
    for f in os.listdir(model_dir):
        if not f.endswith('.pkl'):
            continue
        # Include best_cr checkpoints
        if 'best_cr' in f:
            files.append(f)
        # Include periodic checkpoints (epoch-{n}.pkl)
        elif include_periodic and f.startswith('epoch-'):
            files.append(f)

    # Sort by epoch number
    def get_epoch(filename):
        match = re.search(r'(\d+)', filename)
        return int(match.group(1)) if match else 0

    files.sort(key=get_epoch)
    return files


def run_test(prefix: str, model_path: str) -> Dict:
    """Run test.py with a specific checkpoint and return results."""
    cmd = [
        sys.executable,
        'src/test.py',
        '--prefix', prefix,
        '--model_path', model_path
    ]

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )

        # Read the test results from JSON
        json_file = os.path.join(prefix, 'json_file', 'test_results.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
        else:
            return None
    except Exception as e:
        print(f"Error running test: {e}")
        return None


def evaluate_checkpoints(prefix: str, metric: str = 'CR', top_n: int = 5, include_periodic: bool = True) -> List[Tuple[str, float, Dict]]:
    """Evaluate all checkpoints and return sorted results."""
    checkpoints = get_checkpoint_files(prefix, include_periodic=include_periodic)
    print(f"Found {len(checkpoints)} checkpoints to evaluate")
    print("=" * 60)

    results = []

    for i, checkpoint in enumerate(checkpoints):
        model_path = os.path.join(prefix, 'model_file', checkpoint)
        print(f"[{i+1}/{len(checkpoints)}] Testing {checkpoint}...", end=' ')

        test_result = run_test(prefix, model_path)

        if test_result and 'performance_metrics' in test_result:
            metrics = test_result['performance_metrics']
            metric_value = metrics.get(metric, 0)
            final_wealth = test_result['summary']['final_wealth']

            results.append((checkpoint, metric_value, metrics))
            print(f"{metric}={metric_value:.4f}, Final Wealth={final_wealth:.4f}")
        else:
            print("Failed")

    # Sort by metric (higher is better for most metrics, except MDD and AVOL)
    reverse = metric not in ['MDD', 'AVOL']
    results.sort(key=lambda x: x[1], reverse=reverse)

    return results[:top_n]


def main():
    parser = argparse.ArgumentParser(description='Find the best checkpoint for test period')
    parser.add_argument('--prefix', type=str, required=True, help='Experiment output directory')
    parser.add_argument('--metric', type=str, default='CR',
                        choices=['CR', 'ARR', 'MDD', 'AVOL', 'ASR', 'DDR'],
                        help='Metric to optimize (default: CR)')
    parser.add_argument('--top', type=int, default=5, help='Show top N results (default: 5)')
    parser.add_argument('--periodic', action='store_true', default=True,
                        help='Include periodic checkpoints (epoch-{n}.pkl) (default: True)')
    parser.add_argument('--best-only', action='store_true',
                        help='Only evaluate best_cr checkpoints, exclude periodic ones')

    args = parser.parse_args()
    include_periodic = args.periodic and not args.best_only

    print(f"Evaluating checkpoints in: {args.prefix}")
    print(f"Optimizing for: {args.metric}")
    print(f"Include periodic checkpoints: {include_periodic}")
    print("=" * 60)

    try:
        top_results = evaluate_checkpoints(args.prefix, args.metric, args.top, include_periodic)

        print("\n" + "=" * 60)
        print(f"TOP {len(top_results)} CHECKPOINTS (by {args.metric}):")
        print("=" * 60)

        for rank, (checkpoint, metric_value, metrics) in enumerate(top_results, 1):
            print(f"\n#{rank}: {checkpoint}")
            print(f"    {args.metric}: {metric_value:.4f}")
            print(f"    All metrics: CR={metrics['CR']:.4f}, ARR={metrics['ARR']:.4f}, "
                  f"MDD={metrics['MDD']:.4f}, ASR={metrics['ASR']:.4f}")

        if top_results:
            best_checkpoint = top_results[0][0]
            best_path = os.path.join(args.prefix, 'model_file', best_checkpoint)
            print("\n" + "=" * 60)
            print(f"BEST CHECKPOINT: {best_checkpoint}")
            print(f"To use this checkpoint:")
            print(f"  python src/test.py --prefix {args.prefix} --model_path {best_path}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
