"""
Comprehensive correlation analysis with statistical testing.

This script provides a complete statistical analysis including:
1. Overall correlation (all 30 stocks) - Pearson & Spearman
2. Top-4 vs Bottom-4 group difference test - Extreme value identification

Usage:
    python src/plot/test_correlation_analysis.py <experiment_id> <period> [outputs_path]

Examples:
    python src/plot/test_correlation_analysis.py 0718/181011 test
    python src/plot/test_correlation_analysis.py 0718/181011 val
    python src/plot/test_correlation_analysis.py 0718/181011 test outputs
"""

import sys
import os
import json
import numpy as np
from scipy.stats import ttest_1samp

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from analysis import compute_correlation_metrics, print_correlation_test_results


def test_top_bottom_difference(experiment_id, period, outputs_base_path):
    """
    Test if Top-4 significantly outperforms Bottom-4.

    This is the KEY test to validate extreme value identification ability.
    Uses one-sample t-test.
    """
    # Load portfolio data
    json_filename = f'{period}_results.json'
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)

    if not os.path.exists(json_path):
        return None

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    portfolio_records = data.get('portfolio_records', [])

    # Collect step-wise differences
    differences = []
    top4_returns = []
    bottom4_returns = []

    for step_data in portfolio_records:
        scores = np.array(step_data['all_scores'])

        # Get returns
        ror_data = step_data['sim_info']['ror']
        if isinstance(ror_data[0], list):
            returns = np.array(ror_data[0]) - 1.0
        else:
            returns = np.array(ror_data) - 1.0

        # Top-4 and Bottom-4
        top_indices = np.argsort(scores)[-4:]
        bottom_indices = np.argsort(scores)[:4]

        top_mean = np.mean(returns[top_indices])
        bottom_mean = np.mean(returns[bottom_indices])

        top4_returns.append(top_mean)
        bottom4_returns.append(bottom_mean)
        differences.append(top_mean - bottom_mean)

    differences = np.array(differences)
    top4_returns = np.array(top4_returns)
    bottom4_returns = np.array(bottom4_returns)

    # Statistical test: one-sample t-test
    t_stat, p_value_t = ttest_1samp(differences, 0, alternative='greater')

    # Effect size
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)

    return {
        'n_steps': len(differences),
        'mean_diff': np.mean(differences),
        'std_diff': np.std(differences, ddof=1),
        'top4_mean': np.mean(top4_returns),
        'bottom4_mean': np.mean(bottom4_returns),
        't_stat': t_stat,
        'p_value_t': p_value_t,
        'cohens_d': cohens_d
    }


def main(experiment_id, period='test', outputs_base_path=None):
    """
    Main function to run correlation analysis with statistical testing.

    Args:
        experiment_id: Experiment ID (e.g., '0718/181011')
        period: 'val' or 'test' (default: 'test')
        outputs_base_path: Base path to outputs directory (default: None, uses config default)
    """
    # Import config to get default path if not specified
    from config import OUTPUTS_BASE_PATH

    if outputs_base_path is None:
        outputs_base_path = OUTPUTS_BASE_PATH

    print("\n" + "="*80)
    print("Correlation Analysis with Statistical Testing")
    print("="*80)

    # Compute correlation metrics with statistical testing
    print(f"\nAnalyzing experiment: {experiment_id}")
    print(f"Period: {period.upper()}")
    print(f"Base path: {outputs_base_path}")

    results = compute_correlation_metrics(
        experiment_id=experiment_id,
        outputs_base_path=outputs_base_path,
        period=period
    )

    if not results:
        print("❌ No results returned. Check if the experiment path exists.")
        print(f"   Expected path: {outputs_base_path}/{experiment_id}/json_file/")
        return

    # Print the detailed results
    print_correlation_test_results(results)

    # ========================================================================
    # Part 2: Long-4 vs Short-4 Difference Test
    # ========================================================================
    print("\n" + "="*80)
    print("PART 2: Long-4 vs Short-4 Group Difference Test")
    print("="*80)
    print("\nThis test checks if AI-selected long positions outperform short positions.")
    print("Long-4: 4 stocks AI selected to BUY (high scores)")
    print("Short-4: 4 stocks AI selected to SELL SHORT (low scores)")
    print("Even if overall correlation is weak, this test can still be significant.\n")

    group_results = test_top_bottom_difference(experiment_id, period, outputs_base_path)

    if group_results:
        print(f"Summary Statistics:")
        print("-" * 80)
        print(f"  Number of steps: {group_results['n_steps']}")
        print(f"  Long-4 mean return:  {100*group_results['top4_mean']:.2f}%")
        print(f"  Short-4 mean return: {100*group_results['bottom4_mean']:.2f}%")
        print(f"  Difference (Long - Short): {100*group_results['mean_diff']:.2f}% ± {100*group_results['std_diff']:.2f}%")

        print(f"\nStatistical Test (One-sample t-test):")
        print("-" * 80)
        print(f"  H₀: μ_diff ≤ 0  (Long-4 does not outperform Short-4)")
        print(f"  H₁: μ_diff > 0  (Long-4 outperforms Short-4)")
        print()
        print(f"  t-statistic: {group_results['t_stat']:.4f}")
        print(f"  p-value: {group_results['p_value_t']:.6f} {'✓ SIGNIFICANT' if group_results['p_value_t'] < 0.05 else '✗ Not significant'}")

        print(f"\nEffect Size:")
        print("-" * 80)
        print(f"  Cohen's d: {group_results['cohens_d']:.4f}")
        if abs(group_results['cohens_d']) < 0.2:
            effect_interpretation = "Negligible"
        elif abs(group_results['cohens_d']) < 0.5:
            effect_interpretation = "Small"
        elif abs(group_results['cohens_d']) < 0.8:
            effect_interpretation = "Medium"
        else:
            effect_interpretation = "Large"
        print(f"  Interpretation: {effect_interpretation} effect")

        print(f"\n{'='*80}")
        print("INTERPRETATION")
        print("="*80)

        if group_results['p_value_t'] < 0.001:
            print("✓✓✓ STRONG EVIDENCE that Long-4 outperforms Short-4 (p < 0.001)")
            print("    → AI-selected long positions significantly beat short positions")
            print("    → Model has strong selection ability")
        elif group_results['p_value_t'] < 0.05:
            print("✓ MODERATE EVIDENCE that Long-4 outperforms Short-4 (p < 0.05)")
            print("    → AI shows some selection ability")
        else:
            print("✗ INSUFFICIENT EVIDENCE that Long-4 outperforms Short-4 (p ≥ 0.05)")
            print("    → Cannot confirm AI has selection ability")
    else:
        print("⚠️  Could not perform Top-4 vs Bottom-4 analysis (data not found)")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("\nTwo complementary analyses:")
    print("1. Overall Correlation: Measures linear prediction accuracy across all stocks")
    print("2. Long-4 vs Short-4: Measures if AI-selected longs beat shorts")
    print("\nKey insight: A strategy can be profitable even with weak correlation,")
    print("if AI-selected long positions reliably outperform short positions.")
    print("="*80 + "\n")


if __name__ == "__main__":
    # Parse command line arguments
    if len(sys.argv) < 3:
        print("\n" + "="*80)
        print("ERROR: Missing required arguments")
        print("="*80)
        print("\nUsage:")
        print("  python src/plot/test_correlation_analysis.py <experiment_id> <period> [outputs_path]")
        print("\nExamples:")
        print("  python src/plot/test_correlation_analysis.py 0718/181011 test")
        print("  python src/plot/test_correlation_analysis.py 0718/181011 val")
        print("  python src/plot/test_correlation_analysis.py 0718/181011 test outputs")
        print("\nArguments:")
        print("  experiment_id: Experiment ID (e.g., '0718/181011')")
        print("  period:        'val' or 'test'")
        print("  outputs_path:  Base path to outputs directory (default: 'outputs')")
        print("\n" + "="*80 + "\n")
        sys.exit(1)

    experiment_id = sys.argv[1]
    period = sys.argv[2]
    outputs_base_path = sys.argv[3] if len(sys.argv) > 3 else None  # None means use config default

    # Validate period
    if period not in ['val', 'test']:
        print(f"\n❌ ERROR: period must be 'val' or 'test', got '{period}'")
        sys.exit(1)

    # Run the analysis
    main(experiment_id, period, outputs_base_path)
