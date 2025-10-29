"""
Test script for correlation analysis with statistical testing.

This script demonstrates how to use the enhanced compute_correlation_metrics function
with statistical testing (t-tests, confidence intervals, etc.).

Usage:
    python src/plot/test_correlation_analysis.py <experiment_id> <period> [outputs_path]

Examples:
    python src/plot/test_correlation_analysis.py 0718/181011 test
    python src/plot/test_correlation_analysis.py 0718/181011 val
    python src/plot/test_correlation_analysis.py 0718/181011 test outputs
"""

import sys
import os

# Add src directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from analysis import compute_correlation_metrics, print_correlation_test_results


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

    # You can also access individual values programmatically
    print("\n" + "="*80)
    print("PROGRAMMATIC ACCESS EXAMPLE")
    print("="*80)
    print(f"""
    # Access results programmatically:
    mean_spearman = {results['mean_step_spearman']:.4f}

    # Parametric test (t-test)
    spearman_p_value_t = {results['spearman_p_value_t_test']:.6f}
    is_significant_t = {results['spearman_significant_t_test']}

    # Non-parametric test (Wilcoxon)
    spearman_p_value_w = {results['spearman_p_value_wilcoxon']:.6f}
    is_significant_w = {results['spearman_significant_wilcoxon']}

    # 95% Confidence Interval for Spearman
    ci_lower = {results['spearman_ci_95_lower']:.4f}
    ci_upper = {results['spearman_ci_95_upper']:.4f}

    # Comparison
    pearson_spearman_diff = {results['mean_pearson_spearman_diff']:.4f}
    """)


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
