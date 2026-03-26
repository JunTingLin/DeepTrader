#!/usr/bin/env python3
"""
Analyze MSU Output rho values from experiment results.
Extracts min, max, avg for both val and test sets.
"""

import json
import os
from pathlib import Path

def analyze_experiment(exp_path: str) -> dict:
    """Analyze a single experiment's rho values."""
    exp_path = Path(exp_path)
    val_json = exp_path / "json_file" / "val_results.json"
    test_json = exp_path / "json_file" / "test_results.json"

    results = {"path": str(exp_path)}

    # Analyze validation results
    if val_json.exists():
        with open(val_json, 'r') as f:
            val_data = json.load(f)
        rho_val = val_data.get('rho_record', [])
        if rho_val:
            results['val'] = {
                'min': min(rho_val),
                'max': max(rho_val),
                'avg': sum(rho_val) / len(rho_val),
                'count': len(rho_val)
            }
    else:
        results['val'] = None

    # Analyze test results
    if test_json.exists():
        with open(test_json, 'r') as f:
            test_data = json.load(f)
        rho_test = test_data.get('rho_record', [])
        if rho_test:
            results['test'] = {
                'min': min(rho_test),
                'max': max(rho_test),
                'avg': sum(rho_test) / len(rho_test),
                'count': len(rho_test)
            }
    else:
        results['test'] = None

    return results

def main():
    # Experiment paths
    experiments = [
        "src/outputs/0309/022652",
        "src/outputs/0316/010439",
        "src/outputs/0316/010452",
        "src/outputs/0316/010828",
        "src/outputs/0316/133004",
        "src/outputs/0316/172509",
    ]

    # Get project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent.parent

    all_results = []

    print("=" * 80)
    print("MSU Output Rho Analysis")
    print("=" * 80)

    for exp in experiments:
        exp_path = project_root / exp
        results = analyze_experiment(exp_path)
        all_results.append(results)

        print(f"\n📁 Experiment: {exp}")
        print("-" * 60)

        if results['val']:
            v = results['val']
            print(f"  Validation (n={v['count']}):")
            print(f"    Min: {v['min']:.6f}")
            print(f"    Max: {v['max']:.6f}")
            print(f"    Avg: {v['avg']:.6f}")
        else:
            print("  Validation: No data")

        if results['test']:
            t = results['test']
            print(f"  Test (n={t['count']}):")
            print(f"    Min: {t['min']:.6f}")
            print(f"    Max: {t['max']:.6f}")
            print(f"    Avg: {t['avg']:.6f}")
        else:
            print("  Test: No data")

    # Summary table
    print("\n" + "=" * 80)
    print("Summary Table")
    print("=" * 80)
    print(f"{'Experiment':<20} | {'Val Min':>10} | {'Val Max':>10} | {'Val Avg':>10} | {'Test Min':>10} | {'Test Max':>10} | {'Test Avg':>10}")
    print("-" * 100)

    for results in all_results:
        exp_name = results['path'].split('/')[-2] + '/' + results['path'].split('/')[-1]
        v = results['val'] if results['val'] else {'min': float('nan'), 'max': float('nan'), 'avg': float('nan')}
        t = results['test'] if results['test'] else {'min': float('nan'), 'max': float('nan'), 'avg': float('nan')}
        print(f"{exp_name:<20} | {v['min']:>10.6f} | {v['max']:>10.6f} | {v['avg']:>10.6f} | {t['min']:>10.6f} | {t['max']:>10.6f} | {t['avg']:>10.6f}")

    # Save results to JSON
    output_file = project_root / "src/outputs/msu_rho_analysis.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✅ Results saved to: {output_file}")

if __name__ == "__main__":
    main()
