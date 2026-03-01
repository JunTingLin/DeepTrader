"""
Script to update existing training_summary.json files to the new format.
Adds val_metrics_avg/std, test_metrics_avg/std, and additional metrics to cycle_results.
"""

import json
import os
import numpy as np
from glob import glob


def compute_avg_std(values):
    """Compute average and std, handling infinity and None values."""
    finite_values = [v for v in values if v is not None and np.isfinite(v)]
    if len(finite_values) == 0:
        return None, None
    avg = float(np.mean(finite_values))
    std = float(np.std(finite_values)) if len(finite_values) > 1 else 0.0
    return avg, std


def update_training_summary(output_dir):
    """Update a single training_summary.json to the new format."""
    json_dir = os.path.join(output_dir, 'json_file')
    summary_path = os.path.join(json_dir, 'training_summary.json')

    if not os.path.exists(summary_path):
        print(f"  [SKIP] training_summary.json not found in {output_dir}")
        return False

    # Read existing summary
    with open(summary_path, 'r', encoding='utf-8') as f:
        summary = json.load(f)

    # Check if already updated
    if 'val_metrics_avg' in summary:
        print(f"  [SKIP] Already updated: {output_dir}")
        return False

    print(f"  [UPDATE] {output_dir}")

    cycle_results = summary.get('cycle_results', [])

    # Update each cycle with additional metrics from individual result files
    for cycle_data in cycle_results:
        cycle_num = cycle_data['cycle']

        # Try to read val_results_cycleN.json for additional validation metrics
        val_file = os.path.join(json_dir, f'val_results_cycle{cycle_num}.json')
        if os.path.exists(val_file):
            with open(val_file, 'r', encoding='utf-8') as f:
                val_results = json.load(f)

            perf = val_results.get('performance_metrics', {})
            if 'val_ARR' not in cycle_data:
                cycle_data['val_ARR'] = perf.get('ARR')
            if 'val_MDD' not in cycle_data:
                cycle_data['val_MDD'] = perf.get('MDD')
            if 'val_ASR' not in cycle_data:
                cycle_data['val_ASR'] = perf.get('ASR')
            if 'val_AVOL' not in cycle_data:
                cycle_data['val_AVOL'] = perf.get('AVOL')
            if 'val_DDR' not in cycle_data:
                cycle_data['val_DDR'] = perf.get('DDR')

        # Try to read test_results_cycleN.json for additional test metrics
        test_file = os.path.join(json_dir, f'test_results_cycle{cycle_num}.json')
        if os.path.exists(test_file):
            with open(test_file, 'r', encoding='utf-8') as f:
                test_results = json.load(f)

            perf = test_results.get('performance_metrics', {})
            if 'test_ASR' not in cycle_data:
                cycle_data['test_ASR'] = perf.get('ASR')
            if 'test_AVOL' not in cycle_data:
                cycle_data['test_AVOL'] = perf.get('AVOL')
            if 'test_DDR' not in cycle_data:
                cycle_data['test_DDR'] = perf.get('DDR')

    # Compute averages and standard deviations
    val_metrics_keys = ['CR', 'ARR', 'MDD', 'ASR', 'AVOL', 'DDR']
    test_metrics_keys = ['CR', 'ARR', 'MDD', 'ASR', 'AVOL', 'DDR']

    val_metrics_avg = {}
    val_metrics_std = {}
    test_metrics_avg = {}
    test_metrics_std = {}

    for key in val_metrics_keys:
        values = [r.get(f'val_{key}') for r in cycle_results]
        avg, std = compute_avg_std(values)
        val_metrics_avg[key] = avg
        val_metrics_std[key] = std

    for key in test_metrics_keys:
        values = [r.get(f'test_{key}') for r in cycle_results]
        avg, std = compute_avg_std(values)
        test_metrics_avg[key] = avg
        test_metrics_std[key] = std

    # Add new fields to summary
    summary['val_metrics_avg'] = val_metrics_avg
    summary['val_metrics_std'] = val_metrics_std
    summary['test_metrics_avg'] = test_metrics_avg
    summary['test_metrics_std'] = test_metrics_std

    # Update cycle_results
    summary['cycle_results'] = cycle_results

    # Backup original file
    backup_path = summary_path + '.backup'
    if not os.path.exists(backup_path):
        with open(backup_path, 'w', encoding='utf-8') as f:
            with open(summary_path, 'r', encoding='utf-8') as original:
                f.write(original.read())

    # Write updated summary
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"    - Added val_metrics_avg/std")
    print(f"    - Added test_metrics_avg/std")
    print(f"    - Updated {len(cycle_results)} cycles with additional metrics")

    return True


def main():
    # Directories to update
    output_dirs = [
        './src/outputs_sliding/0226/014505',
        './src/outputs_sliding/0226/120847',
        './src/outputs_sliding/0226/200228',
    ]

    print("Updating training_summary.json files to new format...")
    print("=" * 60)

    updated_count = 0
    for output_dir in output_dirs:
        if os.path.exists(output_dir):
            if update_training_summary(output_dir):
                updated_count += 1
        else:
            print(f"  [NOT FOUND] {output_dir}")

    print("=" * 60)
    print(f"Updated {updated_count} files.")

    # Also update any other outputs_sliding directories if they exist
    print("\nScanning for other outputs_sliding directories...")
    other_dirs = glob('./src/outputs_sliding/*/*')
    for output_dir in other_dirs:
        if output_dir not in output_dirs and os.path.isdir(output_dir):
            json_dir = os.path.join(output_dir, 'json_file')
            if os.path.exists(os.path.join(json_dir, 'training_summary.json')):
                if update_training_summary(output_dir):
                    updated_count += 1

    print(f"\nTotal updated: {updated_count}")


if __name__ == '__main__':
    main()
