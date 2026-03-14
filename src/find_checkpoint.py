"""
Find checkpoints by ARR (Annualized Rate of Return) - both best and worst.

Usage:
    python src/find_checkpoint.py --prefix ./src/outputs/0313/015755
    python src/find_checkpoint.py --prefix ./src/outputs/0313/015755 --top 10
    python src/find_checkpoint.py --prefix ./src/outputs/0313/015755 --mode val  # Fast, no test
"""

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class CheckpointResult:
    """Store validation and test results for a checkpoint."""
    epoch: int
    checkpoint_name: str
    # Validation metrics
    val_arr: float = 0.0
    val_asr: float = 0.0
    val_cr: float = 0.0
    val_mdd: float = 0.0
    val_ddr: float = 0.0
    # Test metrics
    test_arr: float = 0.0
    test_asr: float = 0.0
    test_cr: float = 0.0
    test_mdd: float = 0.0
    test_ddr: float = 0.0
    # Flags
    has_val: bool = False
    has_test: bool = False

    @property
    def combined_arr(self) -> float:
        """Val ARR + Test ARR"""
        return self.val_arr + self.test_arr


def parse_validation_results(prefix: str) -> Dict[int, Dict]:
    """Parse validation results from training log."""
    log_file = os.path.join(prefix, 'log_file', 'logger.log')
    if not os.path.exists(log_file):
        return {}

    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    pattern = r'Epoch (\d+) - Validation: Final wealth: [\d.]+, ARR: ([\d.]+)%, ASR: ([\d.]+), AVOL: [\d.]+, MDD: ([\d.]+)%, CR: ([\d.]+), DDR: ([\d.]+)'

    results = {}
    for match in re.finditer(pattern, content):
        epoch = int(match.group(1))
        results[epoch] = {
            'arr': float(match.group(2)),
            'asr': float(match.group(3)),
            'mdd': float(match.group(4)),
            'cr': float(match.group(5)),
            'ddr': float(match.group(6)),
        }
    return results


def get_checkpoint_files(prefix: str) -> List[str]:
    """Get all checkpoint files (epoch-*.pkl and best_cr-*.pkl)."""
    model_dir = os.path.join(prefix, 'model_file')
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = []
    for f in os.listdir(model_dir):
        if f.endswith('.pkl') and (f.startswith('epoch-') or 'best_cr' in f):
            files.append(f)

    files.sort(key=lambda x: int(re.search(r'(\d+)', x).group(1)))
    return files


def run_test(prefix: str, model_path: str) -> Optional[Dict]:
    """Run test.py and return results."""
    cmd = [sys.executable, 'src/test.py', '--prefix', prefix, '--model_path', model_path]

    try:
        subprocess.run(cmd, capture_output=True, text=True,
                      cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        json_file = os.path.join(prefix, 'json_file', 'test_results.json')
        if os.path.exists(json_file):
            with open(json_file, 'r') as f:
                return json.load(f)
    except Exception as e:
        print(f"Error: {e}")
    return None


def evaluate_checkpoints(prefix: str, run_tests: bool = True) -> List[CheckpointResult]:
    """Evaluate all checkpoints."""
    val_results = parse_validation_results(prefix)

    checkpoints = get_checkpoint_files(prefix)
    print(f"Found {len(checkpoints)} checkpoints (epoch-*.pkl + best_cr-*.pkl)")
    print("=" * 100)

    results = []
    for i, ckpt in enumerate(checkpoints):
        epoch = int(re.search(r'(\d+)', ckpt).group(1))
        r = CheckpointResult(epoch=epoch, checkpoint_name=ckpt)

        # Add validation results
        if epoch in val_results:
            v = val_results[epoch]
            r.val_arr = v['arr']
            r.val_asr = v['asr']
            r.val_cr = v['cr']
            r.val_mdd = v['mdd']
            r.val_ddr = v['ddr']
            r.has_val = True

        # Run test
        if run_tests:
            print(f"[{i+1}/{len(checkpoints)}] {ckpt}...", end=' ', flush=True)
            model_path = os.path.join(prefix, 'model_file', ckpt)
            test_result = run_test(prefix, model_path)

            if test_result and 'performance_metrics' in test_result:
                m = test_result['performance_metrics']
                r.test_arr = m['ARR'] * 100  # Convert to %
                r.test_asr = m['ASR']
                r.test_cr = m['CR']
                r.test_mdd = m['MDD'] * 100  # Convert to %
                r.test_ddr = m['DDR']
                r.has_test = True
                print(f"ARR={r.test_arr:.2f}%")
            else:
                print("Failed")

        results.append(r)

    return results


def print_table(results: List[CheckpointResult], title: str, sort_key: str, top_n: int, ascending: bool = False):
    """Print ranking table.

    Args:
        ascending: If True, show worst (lowest) first. If False, show best (highest) first.
    """
    # Sort
    reverse = not ascending  # reverse=True means descending (best first)

    if sort_key == 'val_arr':
        sorted_r = sorted([r for r in results if r.has_val], key=lambda x: x.val_arr, reverse=reverse)
    elif sort_key == 'test_arr':
        sorted_r = sorted([r for r in results if r.has_test], key=lambda x: x.test_arr, reverse=reverse)
    else:  # combined
        sorted_r = sorted([r for r in results if r.has_val and r.has_test], key=lambda x: x.combined_arr, reverse=reverse)

    icon = "💩" if ascending else "🏆"
    print(f"\n{'='*120}")
    print(f"{icon} {title} (Top {top_n})")
    print("="*120)
    print(f"{'#':<3} {'Checkpoint':<18} │ {'Val ARR%':>9} {'Val ASR':>8} {'Val CR':>8} {'Val MDD%':>9} │ {'Test ARR%':>10} {'Test ASR':>9} {'Test CR':>8} {'Test MDD%':>10}")
    print("-"*120)

    for i, r in enumerate(sorted_r[:top_n], 1):
        val_arr = f"{r.val_arr:.2f}" if r.has_val else "N/A"
        val_asr = f"{r.val_asr:.3f}" if r.has_val else "N/A"
        val_cr = f"{r.val_cr:.3f}" if r.has_val else "N/A"
        val_mdd = f"{r.val_mdd:.2f}" if r.has_val else "N/A"
        test_arr = f"{r.test_arr:.2f}" if r.has_test else "N/A"
        test_asr = f"{r.test_asr:.3f}" if r.has_test else "N/A"
        test_cr = f"{r.test_cr:.3f}" if r.has_test else "N/A"
        test_mdd = f"{r.test_mdd:.2f}" if r.has_test else "N/A"

        print(f"{i:<3} {r.checkpoint_name:<18} │ {val_arr:>9} {val_asr:>8} {val_cr:>8} {val_mdd:>9} │ {test_arr:>10} {test_asr:>9} {test_cr:>8} {test_mdd:>10}")


def main():
    parser = argparse.ArgumentParser(description='Find checkpoints by ARR (best and worst)')
    parser.add_argument('--prefix', type=str, required=True, help='Experiment directory')
    parser.add_argument('--top', type=int, default=5, help='Top N results (default: 5)')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'val'],
                        help='all: run tests, val: only show validation (fast)')

    args = parser.parse_args()
    run_tests = args.mode == 'all'

    print(f"Prefix: {args.prefix}")
    print(f"Mode: {args.mode} {'(fast, no test runs)' if not run_tests else ''}")

    results = evaluate_checkpoints(args.prefix, run_tests=run_tests)

    if args.mode == 'val':
        print_table(results, "Best Validation ARR", 'val_arr', args.top, ascending=False)
        print_table(results, "Worst Validation ARR", 'val_arr', args.top, ascending=True)
    else:
        # Best
        print_table(results, "Best Validation ARR", 'val_arr', args.top, ascending=False)
        print_table(results, "Best Test ARR", 'test_arr', args.top, ascending=False)
        print_table(results, "Best Combined ARR (Val + Test)", 'combined', args.top, ascending=False)

        # Worst
        print_table(results, "Worst Validation ARR", 'val_arr', args.top, ascending=True)
        print_table(results, "Worst Test ARR", 'test_arr', args.top, ascending=True)
        print_table(results, "Worst Combined ARR (Val + Test)", 'combined', args.top, ascending=True)

        # Summary
        complete = [r for r in results if r.has_val and r.has_test]
        if complete:
            best_val = max(complete, key=lambda x: x.val_arr)
            best_test = max(complete, key=lambda x: x.test_arr)
            best_combined = max(complete, key=lambda x: x.combined_arr)
            worst_val = min(complete, key=lambda x: x.val_arr)
            worst_test = min(complete, key=lambda x: x.test_arr)
            worst_combined = min(complete, key=lambda x: x.combined_arr)

            print(f"\n{'='*120}")
            print("📋 SUMMARY")
            print("="*120)
            print(f"🏆 Best Val ARR:       {best_val.checkpoint_name:<18} Val={best_val.val_arr:.2f}%, Test={best_val.test_arr:.2f}%")
            print(f"🏆 Best Test ARR:      {best_test.checkpoint_name:<18} Val={best_test.val_arr:.2f}%, Test={best_test.test_arr:.2f}%")
            print(f"🏆 Best Combined ARR:  {best_combined.checkpoint_name:<18} Val={best_combined.val_arr:.2f}%, Test={best_combined.test_arr:.2f}%, Sum={best_combined.combined_arr:.2f}%")
            print("-"*120)
            print(f"💩 Worst Val ARR:      {worst_val.checkpoint_name:<18} Val={worst_val.val_arr:.2f}%, Test={worst_val.test_arr:.2f}%")
            print(f"💩 Worst Test ARR:     {worst_test.checkpoint_name:<18} Val={worst_test.val_arr:.2f}%, Test={worst_test.test_arr:.2f}%")
            print(f"💩 Worst Combined ARR: {worst_combined.checkpoint_name:<18} Val={worst_combined.val_arr:.2f}%, Test={worst_combined.test_arr:.2f}%, Sum={worst_combined.combined_arr:.2f}%")


if __name__ == '__main__':
    main()
