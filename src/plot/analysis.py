# -------------------------------
# Analysis Functions - Periodic Returns & Win Rates
# -------------------------------

import numpy as np
import pandas as pd
import sys
import os
import json
from scipy.stats import spearmanr
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import calculate_metrics
from config import config, TRADE_MODE, MARKET_DATA_PATH, MARKET_PRICE_INDEX, TRADE_LEN

def calculate_periodic_returns_df(df, period):
    """
    Calculate periodic returns using non-overlapping fixed periods:
    - 'ME': 1 cycle (1 * 21 days, ~monthly)
    - 'QE': 3 cycles (3 * 21 days, ~quarterly)
    - '6ME': 6 cycles (6 * 21 days, ~semi-annual)
    - 'YE': 12 cycles (12 * 21 days, ~annual)

    Args:
        df: DataFrame with datetime index (sampled every ~21 business days)
        period: 'ME', 'QE', '6ME', 'YE'
    """
    if len(df) == 0:
        return pd.DataFrame()

    # Map period to number of 21-day cycles
    cycle_map = {
        'ME': 1,    # Monthly: 1 cycle (21 days)
        'QE': 3,    # Quarterly: 3 cycles (63 days)
        '6ME': 6,   # Semi-annual: 6 cycles (126 days)
        'YE': 12    # Yearly: 12 cycles (252 days)
    }

    if period not in cycle_map:
        print(f"Unsupported period: {period}. Use 'ME', 'QE', '6ME', or 'YE'")
        return pd.DataFrame()

    cycles = cycle_map[period]

    try:
        returns_data = {}
        return_dates = []

        # Calculate returns for non-overlapping fixed periods
        start_idx = 0
        while start_idx + cycles < len(df):
            end_idx = start_idx + cycles

            # Get start and end values
            start_values = df.iloc[start_idx]
            end_values = df.iloc[end_idx]

            # Calculate returns: (end - start) / start
            period_returns = (end_values - start_values) / start_values

            # Use end date as the period identifier
            end_date = df.index[end_idx]

            # Store returns for this period
            for col in df.columns:
                if col not in returns_data:
                    returns_data[col] = []
                returns_data[col].append(period_returns[col])

            return_dates.append(end_date)

            # Move to next non-overlapping period
            start_idx = end_idx

        # Create DataFrame with returns
        if returns_data and return_dates:
            returns_df = pd.DataFrame(returns_data, index=return_dates)
            return returns_df.dropna()
        else:
            print(f"Not enough data for {period} calculation (need at least {cycles + 1} data points)")
            return pd.DataFrame()

    except Exception as e:
        print(f"Error in period return calculation: {e}")
        return pd.DataFrame()

def calculate_win_rate_df(returns_df):
    """
    Calculate win rate for each column (except the benchmark) as the ratio of periods
    where the column's return is higher than the benchmark's return.
    """
    benchmark_column = config['benchmark_column']
    cols = [col for col in returns_df.columns if col != benchmark_column]
    win_rates = {}
    total = len(returns_df)
    for col in cols:
        wins = (returns_df[col] > returns_df[benchmark_column]).sum()
        win_rates[col] = wins / total
    return pd.Series(win_rates)

def compute_metrics_df(df, series_list):
    """
    For each specified series in the DataFrame, compute performance metrics (ARR, AVOL, ASR, MDD, CR, DDR)
    using calculate_metrics, and return a DataFrame where rows are metrics and columns are strategies.
    Also prints detailed MDD period information (peak to trough) for each strategy.
    """
    metrics_dict = {}
    mdd_details = {}

    for col in series_list:
        wealth = df[col].values
        m = calculate_metrics(wealth.reshape(1, -1), TRADE_MODE)

        # Extract scalar values
        peak_idx = m['MDD_peak_idx'] if isinstance(m['MDD_peak_idx'], (int, np.integer)) else m['MDD_peak_idx'][0]
        trough_idx = m['MDD_trough_idx'] if isinstance(m['MDD_trough_idx'], (int, np.integer)) else m['MDD_trough_idx'][0]

        metrics_dict[col] = {
            'ARR': m['ARR'][0, 0] if isinstance(m['ARR'], np.ndarray) else m['ARR'],
            'AVOL': m['AVOL'][0, 0] if isinstance(m['AVOL'], np.ndarray) else m['AVOL'],
            'ASR': m['ASR'][0, 0] if isinstance(m['ASR'], np.ndarray) else m['ASR'],
            'MDD': m['MDD'],
            'CR': m['CR'][0, 0] if isinstance(m['CR'], np.ndarray) else m['CR'],
            'DDR': m['DDR'][0, 0] if isinstance(m['DDR'], np.ndarray) else m['DDR']
        }

        # Store MDD period details
        peak_date = df.index[peak_idx]
        trough_date = df.index[trough_idx]
        peak_wealth = wealth[peak_idx]
        trough_wealth = wealth[trough_idx]

        mdd_details[col] = {
            'peak_step': peak_idx + 1,  # Convert to 1-indexed for display
            'trough_step': trough_idx + 1,  # Convert to 1-indexed for display
            'peak_date': peak_date,
            'trough_date': trough_date,
            'peak_wealth': peak_wealth,
            'trough_wealth': trough_wealth,
            'drawdown': (peak_wealth - trough_wealth) / peak_wealth
        }

    # Print MDD details
    print("\n" + "="*80)
    print("Maximum Drawdown (MDD) Period Details")
    print("="*80)
    for col, details in mdd_details.items():
        print(f"\n{col}:")
        print(f"  Peak:    Step {details['peak_step']:3d} | {details['peak_date']} | Wealth: {details['peak_wealth']:.4f}")
        print(f"  Trough:  Step {details['trough_step']:3d} | {details['trough_date']} | Wealth: {details['trough_wealth']:.4f}")
        print(f"  Drawdown: {details['drawdown']:.2%} over {details['trough_step'] - details['peak_step']} steps")
    print("="*80 + "\n")

    return pd.DataFrame(metrics_dict)

def compute_single_step_correlation(scores, returns):
    """
    Compute correlation metrics for a single trading step.
    Used by plotting functions to get consistent correlation calculations.

    Args:
        scores: Array or list of scores for one step
        returns: Array or list of returns for one step

    Returns:
        dict: Dictionary containing Pearson and Spearman correlations
    """
    if len(scores) <= 1 or len(returns) <= 1:
        return {
            'pearson_corr': np.nan,
            'spearman_corr': np.nan,
            'valid_data_points': 0
        }

    # Convert to numpy arrays and remove NaN values
    scores = np.array(scores)
    returns = np.array(returns)
    valid_mask = ~(np.isnan(scores) | np.isnan(returns))

    valid_scores = scores[valid_mask]
    valid_returns = returns[valid_mask]

    if len(valid_scores) <= 1:
        return {
            'pearson_corr': np.nan,
            'spearman_corr': np.nan,
            'valid_data_points': len(valid_scores)
        }

    try:
        pearson_corr = np.corrcoef(valid_scores, valid_returns)[0, 1]
        spearman_corr, _ = spearmanr(valid_scores, valid_returns)

        return {
            'pearson_corr': pearson_corr if not np.isnan(pearson_corr) else np.nan,
            'spearman_corr': spearman_corr if not np.isnan(spearman_corr) else np.nan,
            'valid_data_points': len(valid_scores)
        }
    except Exception as e:
        print(f"Error computing correlations: {e}")
        return {
            'pearson_corr': np.nan,
            'spearman_corr': np.nan,
            'valid_data_points': 0
        }

def calculate_single_step_msu_metrics(rho, market_return):
    """
    Calculate MSU metrics for a single step.

    Args:
        rho: MSU allocation value for this step
        market_return: Market return for the next 21 days

    Returns:
        dict: Single step metrics
    """
    # Determine prediction
    predicted_bullish = rho > 0.5
    predicted_bearish = rho < 0.5
    predicted_neutral = rho == 0.5

    # Determine actual market movement
    actual_up = market_return > 0
    actual_down = market_return < 0
    actual_flat = market_return == 0

    # Calculate True Positives
    bullish_tp = predicted_bullish and actual_up
    bearish_tp = predicted_bearish and actual_down
    neutral_tp = predicted_neutral and actual_flat

    # Calculate precision (only if prediction was made)
    bullish_precision = 1.0 if predicted_bullish and bullish_tp else (0.0 if predicted_bullish else None)
    bearish_precision = 1.0 if predicted_bearish and bearish_tp else (0.0 if predicted_bearish else None)
    neutral_precision = 1.0 if predicted_neutral and neutral_tp else (0.0 if predicted_neutral else None)

    # Calculate recall (only if actual movement occurred)
    bullish_recall = 1.0 if actual_up and bullish_tp else (0.0 if actual_up else None)
    bearish_recall = 1.0 if actual_down and bearish_tp else (0.0 if actual_down else None)
    neutral_recall = 1.0 if actual_flat and neutral_tp else (0.0 if actual_flat else None)

    return {
        'bullish_precision': bullish_precision,
        'bearish_precision': bearish_precision,
        'neutral_precision': neutral_precision,
        'bullish_recall': bullish_recall,
        'bearish_recall': bearish_recall,
        'neutral_recall': neutral_recall,
        'predicted_bullish': predicted_bullish,
        'predicted_bearish': predicted_bearish,
        'predicted_neutral': predicted_neutral,
        'actual_up': actual_up,
        'actual_down': actual_down,
        'actual_flat': actual_flat
    }

def compute_correlation_metrics(experiment_id, outputs_base_path, period='test'):
    """
    Compute correlation metrics between scores and returns for all 30 stocks with statistical testing.

    Args:
        experiment_id: The experiment ID
        outputs_base_path: Base path to outputs directory
        period: 'val' or 'test'

    Returns:
        dict: Correlation metrics for all stocks including:
            - Mean Pearson/Spearman correlations
            - t-statistics and p-values (two-tailed and one-tailed)
            - 95% confidence intervals
            - Significance indicators
    """
    import json
    import os
    from scipy.stats import ttest_1samp, t as t_dist, wilcoxon
    from config import JSON_FILES, config, TRADE_LEN, STOCK_DATA_PATH, STOCK_PRICE_INDEX

    # Load JSON data
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    portfolio_records = results.get('portfolio_records', [])
    if not portfolio_records:
        return {}

    # Load stock price data to calculate returns
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"Warning: Stock data not found at {STOCK_DATA_PATH}")
        return {}

    stocks_data = np.load(STOCK_DATA_PATH)

    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end']

    # Collect scores and returns for all stocks
    all_scores = []
    all_returns = []
    step_correlations = []
    n_stocks = stocks_data.shape[0]

    for step_idx, record in enumerate(portfolio_records):
        # Get scores from all_scores field
        scores = np.array(record.get('all_scores', []))

        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN

        # Calculate returns for each stock (21-day forward returns)
        returns = []
        for stock_idx in range(min(len(scores), n_stocks)):
            if (decision_date_idx + 1 >= 0 and
                decision_date_idx + 1 < stocks_data.shape[1] and
                decision_date_idx + TRADE_LEN < stocks_data.shape[1]):
                current_price = stocks_data[stock_idx, decision_date_idx + 1, STOCK_PRICE_INDEX]
                future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, STOCK_PRICE_INDEX]
                if current_price > 0:
                    return_rate = (future_price - current_price) / current_price
                else:
                    return_rate = 0.0
            else:
                return_rate = 0.0
            returns.append(return_rate)

        returns = np.array(returns)

        # Use all stocks for correlation calculation
        if len(scores) > 1 and len(returns) > 1:
            # Compute correlation for this step
            if not np.isnan(scores).any() and not np.isnan(returns).any():
                step_pearson = np.corrcoef(scores, returns)[0, 1]
                step_spearman, _ = spearmanr(scores, returns)
            else:
                step_pearson = np.nan
                step_spearman = np.nan

            step_correlations.append({
                'step': step_idx + 1,
                'pearson_corr': step_pearson,
                'spearman_corr': step_spearman,
                'n_stocks': len(scores)
            })

            # Collect all data points for overall correlation
            all_scores.extend(scores)
            all_returns.extend(returns)

    # Compute overall correlations
    overall_pearson = np.nan
    overall_spearman = np.nan
    if len(all_scores) > 1 and not np.isnan(all_scores).any() and not np.isnan(all_returns).any():
        overall_pearson = np.corrcoef(all_scores, all_returns)[0, 1]
        overall_spearman, _ = spearmanr(all_scores, all_returns)

    # Compute mean step correlations with statistical testing
    valid_step_correlations = [sc for sc in step_correlations if not np.isnan(sc['pearson_corr'])]

    if valid_step_correlations:
        # Extract correlation arrays
        step_pearsons = np.array([sc['pearson_corr'] for sc in valid_step_correlations])
        step_spearmans = np.array([sc['spearman_corr'] for sc in valid_step_correlations])
        avg_stocks_per_step = np.mean([sc['n_stocks'] for sc in valid_step_correlations])
        n_steps = len(valid_step_correlations)

        # Mean correlations
        mean_pearson = np.mean(step_pearsons)
        mean_spearman = np.mean(step_spearmans)

        # Standard deviations
        std_pearson = np.std(step_pearsons, ddof=1)
        std_spearman = np.std(step_spearmans, ddof=1)

        # Standard errors
        se_pearson = std_pearson / np.sqrt(n_steps)
        se_spearman = std_spearman / np.sqrt(n_steps)

        # Statistical testing for Pearson
        # Parametric test (t-test): H0: ρ ≤ 0 vs H1: ρ > 0
        t_stat_pearson, p_value_pearson_t = ttest_1samp(step_pearsons, 0, alternative='greater')
        # Non-parametric test (Wilcoxon): H0: median ≤ 0 vs H1: median > 0
        try:
            w_stat_pearson, p_value_pearson_w = wilcoxon(step_pearsons, alternative='greater')
        except ValueError:
            # Wilcoxon fails if all values are the same
            w_stat_pearson, p_value_pearson_w = np.nan, np.nan
        # 95% Confidence Interval
        ci_95_pearson = t_dist.interval(0.95, df=n_steps-1, loc=mean_pearson, scale=se_pearson)

        # Statistical testing for Spearman
        # Parametric test (t-test): H0: ρ ≤ 0 vs H1: ρ > 0
        t_stat_spearman, p_value_spearman_t = ttest_1samp(step_spearmans, 0, alternative='greater')
        # Non-parametric test (Wilcoxon): H0: median ≤ 0 vs H1: median > 0
        try:
            w_stat_spearman, p_value_spearman_w = wilcoxon(step_spearmans, alternative='greater')
        except ValueError:
            # Wilcoxon fails if all values are the same
            w_stat_spearman, p_value_spearman_w = np.nan, np.nan
        # 95% Confidence Interval
        ci_95_spearman = t_dist.interval(0.95, df=n_steps-1, loc=mean_spearman, scale=se_spearman)

        # Compute comparison between Pearson and Spearman
        correlation_diff = np.abs(step_pearsons - step_spearmans)
        mean_correlation_diff = np.mean(correlation_diff)

    else:
        mean_pearson = np.nan
        mean_spearman = np.nan
        std_pearson = np.nan
        std_spearman = np.nan
        se_pearson = np.nan
        se_spearman = np.nan
        t_stat_pearson = np.nan
        p_value_pearson_t = np.nan
        w_stat_pearson = np.nan
        p_value_pearson_w = np.nan
        ci_95_pearson = (np.nan, np.nan)
        t_stat_spearman = np.nan
        p_value_spearman_t = np.nan
        w_stat_spearman = np.nan
        p_value_spearman_w = np.nan
        ci_95_spearman = (np.nan, np.nan)
        mean_correlation_diff = np.nan
        avg_stocks_per_step = 0
        n_steps = 0

    return {
        'experiment_id': experiment_id,
        'period': period,

        # Overall correlations (pooled data - for reference but less interpretable)
        'overall_pearson': overall_pearson,
        'overall_spearman': overall_spearman,

        # Mean step correlations (recommended metric)
        'mean_step_pearson': mean_pearson,
        'mean_step_spearman': mean_spearman,

        # Standard deviations
        'std_step_pearson': std_pearson,
        'std_step_spearman': std_spearman,

        # Standard errors
        'se_step_pearson': se_pearson,
        'se_step_spearman': se_spearman,

        # Pearson statistical testing (H0: ρ ≤ 0 vs H1: ρ > 0)
        'pearson_t_stat': t_stat_pearson,
        'pearson_p_value_t_test': p_value_pearson_t,
        'pearson_w_stat': w_stat_pearson,
        'pearson_p_value_wilcoxon': p_value_pearson_w,
        'pearson_ci_95_lower': ci_95_pearson[0],
        'pearson_ci_95_upper': ci_95_pearson[1],
        'pearson_significant_t_test': p_value_pearson_t < 0.05 if not np.isnan(p_value_pearson_t) else False,
        'pearson_significant_wilcoxon': p_value_pearson_w < 0.05 if not np.isnan(p_value_pearson_w) else False,

        # Spearman statistical testing (H0: ρ ≤ 0 vs H1: ρ > 0)
        'spearman_t_stat': t_stat_spearman,
        'spearman_p_value_t_test': p_value_spearman_t,
        'spearman_w_stat': w_stat_spearman,
        'spearman_p_value_wilcoxon': p_value_spearman_w,
        'spearman_ci_95_lower': ci_95_spearman[0],
        'spearman_ci_95_upper': ci_95_spearman[1],
        'spearman_significant_t_test': p_value_spearman_t < 0.05 if not np.isnan(p_value_spearman_t) else False,
        'spearman_significant_wilcoxon': p_value_spearman_w < 0.05 if not np.isnan(p_value_spearman_w) else False,

        # Comparison between Pearson and Spearman
        'mean_pearson_spearman_diff': mean_correlation_diff,

        # Sample information
        'total_data_points': len(all_scores),
        'avg_stocks_per_step': avg_stocks_per_step,
        'valid_steps': n_steps,
        'total_steps': len(step_correlations)
    }


def print_correlation_test_results(results):
    """
    Pretty print the correlation test results from compute_correlation_metrics.

    Args:
        results: dict returned from compute_correlation_metrics
    """
    exp_id = results.get('experiment_id', 'N/A')
    period = results.get('period', 'N/A')
    n_steps = results.get('valid_steps', 0)

    print("\n" + "="*80)
    print(f"Correlation Analysis Results: {exp_id} ({period.upper()} period)")
    print("="*80)

    print(f"\nSample Information:")
    print(f"  Valid Steps: {n_steps}")
    print(f"  Avg Stocks per Step: {results.get('avg_stocks_per_step', 0):.1f}")
    print(f"  Total Data Points (pooled): {results.get('total_data_points', 0)}")

    print("\n" + "-"*80)
    print("PEARSON CORRELATION (Linear Relationship)")
    print("-"*80)

    mean_p = results.get('mean_step_pearson', np.nan)
    std_p = results.get('std_step_pearson', np.nan)
    ci_low_p = results.get('pearson_ci_95_lower', np.nan)
    ci_high_p = results.get('pearson_ci_95_upper', np.nan)

    print(f"\nMean Step Pearson: {mean_p:.4f} ± {std_p:.4f}")
    print(f"95% CI: [{ci_low_p:.4f}, {ci_high_p:.4f}]")

    # Parametric test (t-test)
    t_stat_p = results.get('pearson_t_stat', np.nan)
    p_t_p = results.get('pearson_p_value_t_test', np.nan)
    sig_t_p = results.get('pearson_significant_t_test', False)

    print(f"\nParametric Test - t-test (H₀: ρ ≤ 0 vs H₁: ρ > 0):")
    print(f"  t-statistic: {t_stat_p:.4f}")
    print(f"  p-value: {p_t_p:.6f}")
    if sig_t_p:
        print(f"  Result: ✅ SIGNIFICANT (p < 0.05) - Positive correlation detected")
    else:
        print(f"  Result: ❌ NOT SIGNIFICANT (p ≥ 0.05) - No evidence of positive correlation")

    # Non-parametric test (Wilcoxon)
    w_stat_p = results.get('pearson_w_stat', np.nan)
    p_w_p = results.get('pearson_p_value_wilcoxon', np.nan)
    sig_w_p = results.get('pearson_significant_wilcoxon', False)

    print(f"\nNon-Parametric Test - Wilcoxon (H₀: median ≤ 0 vs H₁: median > 0):")
    print(f"  W-statistic: {w_stat_p:.4f}")
    print(f"  p-value: {p_w_p:.6f}")
    if sig_w_p:
        print(f"  Result: ✅ SIGNIFICANT (p < 0.05) - Positive correlation detected")
    else:
        print(f"  Result: ❌ NOT SIGNIFICANT (p ≥ 0.05) - No evidence of positive correlation")

    print("\n" + "-"*80)
    print("SPEARMAN CORRELATION (Monotonic Relationship)")
    print("-"*80)

    mean_s = results.get('mean_step_spearman', np.nan)
    std_s = results.get('std_step_spearman', np.nan)
    ci_low_s = results.get('spearman_ci_95_lower', np.nan)
    ci_high_s = results.get('spearman_ci_95_upper', np.nan)

    print(f"\nMean Step Spearman: {mean_s:.4f} ± {std_s:.4f}")
    print(f"95% CI: [{ci_low_s:.4f}, {ci_high_s:.4f}]")

    # Parametric test (t-test)
    t_stat_s = results.get('spearman_t_stat', np.nan)
    p_t_s = results.get('spearman_p_value_t_test', np.nan)
    sig_t_s = results.get('spearman_significant_t_test', False)

    print(f"\nParametric Test - t-test (H₀: ρ ≤ 0 vs H₁: ρ > 0):")
    print(f"  t-statistic: {t_stat_s:.4f}")
    print(f"  p-value: {p_t_s:.6f}")
    if sig_t_s:
        print(f"  Result: ✅ SIGNIFICANT (p < 0.05) - Positive correlation detected")
    else:
        print(f"  Result: ❌ NOT SIGNIFICANT (p ≥ 0.05) - No evidence of positive correlation")

    # Non-parametric test (Wilcoxon)
    w_stat_s = results.get('spearman_w_stat', np.nan)
    p_w_s = results.get('spearman_p_value_wilcoxon', np.nan)
    sig_w_s = results.get('spearman_significant_wilcoxon', False)

    print(f"\nNon-Parametric Test - Wilcoxon (H₀: median ≤ 0 vs H₁: median > 0):")
    print(f"  W-statistic: {w_stat_s:.4f}")
    print(f"  p-value: {p_w_s:.6f}")
    if sig_w_s:
        print(f"  Result: ✅ SIGNIFICANT (p < 0.05) - Positive correlation detected")
    else:
        print(f"  Result: ❌ NOT SIGNIFICANT (p ≥ 0.05) - No evidence of positive correlation")

    print("\n" + "-"*80)
    print("COMPARISON")
    print("-"*80)

    diff = results.get('mean_pearson_spearman_diff', np.nan)
    print(f"\nAverage |Pearson - Spearman| difference: {diff:.4f}")

    if diff < 0.05:
        print("✅ Pearson and Spearman are very similar")
        print("   → Linear relationship, minimal outliers")
        print("   → Either metric is appropriate")
    elif diff < 0.10:
        print("⚠️  Pearson and Spearman show moderate differences")
        print("   → Some non-linearity or outliers present")
        print("   → Spearman is more robust (recommended)")
    else:
        print("❌ Pearson and Spearman differ substantially")
        print("   → Strong non-linearity or significant outliers")
        print("   → Use Spearman (rank-based correlation)")

    print("\n" + "-"*80)
    print("INTERPRETATION GUIDE")
    print("-"*80)
    print("""
    Correlation Strength:
      |ρ| < 0.1  : Negligible
      0.1 ≤ |ρ| < 0.3 : Weak
      0.3 ≤ |ρ| < 0.5 : Moderate
      0.5 ≤ |ρ| < 0.7 : Strong
      |ρ| ≥ 0.7  : Very Strong

    Test Selection:
      - t-test: Parametric test, assumes normal distribution
      - Wilcoxon: Non-parametric test, no distribution assumption (more robust)
      - Recommendation: If both tests agree, result is robust

    Metric Selection:
      - Pearson: Measures linear relationship, sensitive to outliers
      - Spearman: Measures monotonic relationship, robust to outliers
      - Recommended: Use Spearman for financial data (often has outliers)

    Hypothesis:
      - H₀: ρ ≤ 0 (no positive correlation)
      - H₁: ρ > 0 (positive correlation exists)
      - We test if higher scores predict higher returns
    """)

    print("="*80 + "\n")


def compute_prediction_accuracy(experiment_id, outputs_base_path, period='test'):
    """
    Compute precision and recall for traded positions only (P@K, R@K).

    Args:
        experiment_id: The experiment ID
        outputs_base_path: Base path to outputs directory
        period: 'val' or 'test'

    Returns:
        dict: Prediction accuracy metrics for traded positions
    """
    import json
    import os
    from config import JSON_FILES, config, TRADE_LEN, STOCK_DATA_PATH, STOCK_PRICE_INDEX

    # Load JSON data
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)

    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return {}

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    portfolio_records = results.get('portfolio_records', [])
    if not portfolio_records:
        return {}

    # Load stock price data to calculate returns
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"Warning: Stock data not found at {STOCK_DATA_PATH}")
        return {}

    stocks_data = np.load(STOCK_DATA_PATH)

    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end']

    # Initialize counters for Overall precision calculation
    long_correct = 0  # TP for long (predicted long and actually went up)
    long_wrong = 0    # FP for long (predicted long but went down)
    long_flat = 0     # return = 0
    short_correct = 0 # TP for short (predicted short and actually went down)
    short_wrong = 0   # FP for short (predicted short but went up)
    short_flat = 0    # return = 0

    # Initialize counters for recall calculation (keeping for compatibility, but will use Mean Step only)
    total_actual_up = 0    # All stocks that actually went up (for compatibility)
    total_actual_down = 0  # All stocks that actually went down (for compatibility)
    total_actual_flat = 0  # All stocks that stayed flat (for compatibility)

    # Initialize for Mean Step calculation
    step_precisions = []  # Store each step's precision metrics

    for step_idx, record in enumerate(portfolio_records):
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN

        if (decision_date_idx + 1 >= 0 and
            decision_date_idx + 1 < stocks_data.shape[1] and
            decision_date_idx + TRADE_LEN < stocks_data.shape[1]):

            # Initialize step-level counters first (before using them)
            step_long_correct = 0
            step_long_wrong = 0
            step_long_flat = 0
            step_short_correct = 0
            step_short_wrong = 0
            step_short_flat = 0
            step_actual_up = 0
            step_actual_down = 0
            step_actual_flat = 0

            # Calculate actual returns for all stocks (for recall denominator)
            scores = np.array(record.get('all_scores', []))
            n_stocks = min(len(scores), stocks_data.shape[0])
            for stock_idx in range(n_stocks):
                current_price = stocks_data[stock_idx, decision_date_idx + 1, STOCK_PRICE_INDEX]
                future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, STOCK_PRICE_INDEX]
                if current_price > 0:
                    return_rate = (future_price - current_price) / current_price
                    if return_rate > 0:
                        total_actual_up += 1
                        step_actual_up += 1
                    elif return_rate < 0:
                        total_actual_down += 1
                        step_actual_down += 1
                    else:
                        total_actual_flat += 1
                        step_actual_flat += 1

            # Evaluate only traded positions for precision
            for pos in record.get('long_positions', []):
                stock_idx = pos.get('stock_index')
                if stock_idx is not None and stock_idx < stocks_data.shape[0]:
                    current_price = stocks_data[stock_idx, decision_date_idx + 1, STOCK_PRICE_INDEX]
                    future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, STOCK_PRICE_INDEX]
                    if current_price > 0:
                        return_rate = (future_price - current_price) / current_price
                        if return_rate > 0:
                            long_correct += 1
                            step_long_correct += 1
                        elif return_rate < 0:
                            long_wrong += 1
                            step_long_wrong += 1
                        else:
                            long_flat += 1
                            step_long_flat += 1

            for pos in record.get('short_positions', []):
                stock_idx = pos.get('stock_index')
                if stock_idx is not None and stock_idx < stocks_data.shape[0]:
                    current_price = stocks_data[stock_idx, decision_date_idx + 1, STOCK_PRICE_INDEX]
                    future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, STOCK_PRICE_INDEX]
                    if current_price > 0:
                        return_rate = (future_price - current_price) / current_price
                        if return_rate < 0:
                            short_correct += 1
                            step_short_correct += 1
                        elif return_rate > 0:
                            short_wrong += 1
                            step_short_wrong += 1
                        else:
                            short_flat += 1
                            step_short_flat += 1

            # Calculate step-level precision metrics
            step_long_total = step_long_correct + step_long_wrong + step_long_flat
            step_short_total = step_short_correct + step_short_wrong + step_short_flat
            step_total = step_long_total + step_short_total

            if step_total > 0:  # Only include steps with actual positions
                step_long_precision = step_long_correct / step_long_total if step_long_total > 0 else 0.0
                step_short_precision = step_short_correct / step_short_total if step_short_total > 0 else 0.0
                step_overall_precision = (step_long_correct + step_short_correct) / step_total

                # Calculate step-level recall (standard IR definition)
                # Find actual top K performers for this step
                step_returns = []
                for stock_idx in range(n_stocks):
                    current_price = stocks_data[stock_idx, decision_date_idx + 1, STOCK_PRICE_INDEX]
                    future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, STOCK_PRICE_INDEX]
                    if current_price > 0:
                        return_rate = (future_price - current_price) / current_price
                        step_returns.append((stock_idx, return_rate))
                    else:
                        step_returns.append((stock_idx, 0.0))

                # Sort by return rate (descending for top performers)
                step_returns.sort(key=lambda x: x[1], reverse=True)

                # Get predicted stock indices
                predicted_long_indices = set([pos.get('stock_index') for pos in record.get('long_positions', [])])
                predicted_short_indices = set([pos.get('stock_index') for pos in record.get('short_positions', [])])

                # Find actual top/bottom performers among stocks that moved in the right direction
                actual_up_stocks = [(x[0], x[1]) for x in step_returns if x[1] > 0]  # Only stocks that went up
                actual_down_stocks = [(x[0], x[1]) for x in step_returns if x[1] < 0]  # Only stocks that went down

                # Get top K from actual up stocks and bottom K from actual down stocks
                actual_up_stocks.sort(key=lambda x: x[1], reverse=True)  # Sort by return (high to low)
                actual_down_stocks.sort(key=lambda x: x[1])  # Sort by return (low to high, most negative first)

                # Take top K that actually went up, and worst K that actually went down
                actual_top_k_indices = set([x[0] for x in actual_up_stocks[:step_long_total]]) if len(actual_up_stocks) >= step_long_total else set([x[0] for x in actual_up_stocks])
                actual_bottom_k_indices = set([x[0] for x in actual_down_stocks[:step_short_total]]) if len(actual_down_stocks) >= step_short_total else set([x[0] for x in actual_down_stocks])

                # Calculate intersection (how many we got right in terms of selection)
                long_intersection = len(predicted_long_indices & actual_top_k_indices)
                short_intersection = len(predicted_short_indices & actual_bottom_k_indices)

                # Recall@K: intersection / available targets
                step_long_recall = long_intersection / len(actual_top_k_indices) if len(actual_top_k_indices) > 0 else 0.0
                step_short_recall = short_intersection / len(actual_bottom_k_indices) if len(actual_bottom_k_indices) > 0 else 0.0

                # Overall recall
                total_available_targets = len(actual_top_k_indices) + len(actual_bottom_k_indices)
                step_overall_recall = (long_intersection + short_intersection) / total_available_targets if total_available_targets > 0 else 0.0

                step_precisions.append({
                    'step': step_idx + 1,
                    'long_precision': step_long_precision,
                    'short_precision': step_short_precision,
                    'overall_precision': step_overall_precision,
                    'long_recall': step_long_recall,
                    'short_recall': step_short_recall,
                    'overall_recall': step_overall_recall,
                    'long_positions': step_long_total,
                    'short_positions': step_short_total,
                    'total_positions': step_total,
                    'actual_up': step_actual_up,
                    'actual_down': step_actual_down,
                    'actual_flat': step_actual_flat
                })

    # Calculate totals (for compatibility with existing code structure)
    total_long_predicted = long_correct + long_wrong + long_flat
    total_short_predicted = short_correct + short_wrong + short_flat
    total_predicted = total_long_predicted + total_short_predicted

    # We don't need overall precision/recall anymore, only Mean Step versions
    long_precision = 0.0  # Not used
    short_precision = 0.0  # Not used
    overall_precision = 0.0  # Not used
    long_recall = 0.0  # Not used
    short_recall = 0.0  # Not used
    overall_recall = 0.0  # Not used

    # Calculate Mean Step metrics
    if step_precisions:
        mean_step_long_precision = np.mean([sp['long_precision'] for sp in step_precisions])
        mean_step_short_precision = np.mean([sp['short_precision'] for sp in step_precisions])
        mean_step_overall_precision = np.mean([sp['overall_precision'] for sp in step_precisions])
        mean_step_long_recall = np.mean([sp['long_recall'] for sp in step_precisions])
        mean_step_short_recall = np.mean([sp['short_recall'] for sp in step_precisions])
        mean_step_overall_recall = np.mean([sp['overall_recall'] for sp in step_precisions])
        avg_positions_per_step = np.mean([sp['total_positions'] for sp in step_precisions])
        avg_long_positions_per_step = np.mean([sp['long_positions'] for sp in step_precisions])
        avg_short_positions_per_step = np.mean([sp['short_positions'] for sp in step_precisions])
        avg_actual_up_per_step = np.mean([sp['actual_up'] for sp in step_precisions])
        avg_actual_down_per_step = np.mean([sp['actual_down'] for sp in step_precisions])
        valid_steps = len(step_precisions)
    else:
        mean_step_long_precision = 0.0
        mean_step_short_precision = 0.0
        mean_step_overall_precision = 0.0
        mean_step_long_recall = 0.0
        mean_step_short_recall = 0.0
        mean_step_overall_recall = 0.0
        avg_positions_per_step = 0.0
        avg_long_positions_per_step = 0.0
        avg_short_positions_per_step = 0.0
        avg_actual_up_per_step = 0.0
        avg_actual_down_per_step = 0.0
        valid_steps = 0

    return {
        'experiment_id': experiment_id,
        'period': period,
        # Precision
        'long_precision': long_precision,
        'short_precision': short_precision,
        'overall_precision': overall_precision,
        # Recall
        'long_recall': long_recall,
        'short_recall': short_recall,
        'overall_recall': overall_recall,
        # Detailed counts for precision
        'long_correct': long_correct,
        'long_wrong': long_wrong,
        'long_flat': long_flat,
        'short_correct': short_correct,
        'short_wrong': short_wrong,
        'short_flat': short_flat,
        'total_long_predicted': total_long_predicted,
        'total_short_predicted': total_short_predicted,
        'total_predicted': total_predicted,
        # Detailed counts for recall
        'total_actual_up': total_actual_up,
        'total_actual_down': total_actual_down,
        'total_actual_flat': total_actual_flat,
        # Mean Step metrics
        'mean_step_long_precision': mean_step_long_precision,
        'mean_step_short_precision': mean_step_short_precision,
        'mean_step_overall_precision': mean_step_overall_precision,
        'mean_step_long_recall': mean_step_long_recall,
        'mean_step_short_recall': mean_step_short_recall,
        'mean_step_overall_recall': mean_step_overall_recall,
        'avg_long_positions_per_step': avg_long_positions_per_step,
        'avg_short_positions_per_step': avg_short_positions_per_step,
        'avg_positions_per_step': avg_positions_per_step,
        'avg_actual_up_per_step': avg_actual_up_per_step,
        'avg_actual_down_per_step': avg_actual_down_per_step,
        'valid_steps': valid_steps
    }


def calculate_msu_market_accuracy(experiment_path, period='val'):
    """
    Calculate MSU market prediction accuracy.

    For each step, check if:
    - Market return > 0 and rho > 0.5: Correct prediction (bullish)
    - Market return = 0 and rho = 0.5: Correct prediction (neutral)
    - Market return < 0 and rho < 0.5: Correct prediction (bearish)

    Returns:
        dict: MSU market prediction accuracy metrics
    """
    # Load configuration - try multiple possible paths
    config_paths = [
        f"{experiment_path}/config.json",
        f"{experiment_path}/log_file/hyper.json"
    ]

    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            break

    if config is None:
        print(f"Warning: Config not found in any of: {config_paths}")
        return {}

    # Load portfolio records - try multiple possible file names
    if period == 'val':
        json_paths = [
            f"{experiment_path}/json_file/val_results.json",
            f"{experiment_path}/json_file/val_results_msu_original.json"
        ]
    else:
        json_paths = [
            f"{experiment_path}/json_file/test_results.json",
            f"{experiment_path}/json_file/test_results_msu_original.json"
        ]

    json_path = None
    for path in json_paths:
        if os.path.exists(path):
            json_path = path
            break

    if json_path is None:
        print(f"Warning: JSON file not found in any of: {json_paths}")
        return {}

    with open(json_path, 'r') as f:
        data = json.load(f)

    portfolio_records = data.get('portfolio_records', [])
    rho_record = data.get('rho_record', [])

    if not portfolio_records:
        return {}

    if not rho_record:
        print(f"Warning: No rho_record found in {json_path}")
        return {}

    # Load market data to calculate market returns
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: Market data not found at {MARKET_DATA_PATH}")
        return {}

    market_data = np.load(MARKET_DATA_PATH)

    # Get date range for the period and trade length
    trade_len = config.get('trade_len', 21)  # Default to 21 days

    if period == 'val':
        date_start_idx = config.get('train_idx_end', config.get('train_end', 0))
    else:  # test
        date_start_idx = config.get('test_idx', config.get('val_end', 0))

    # Track predictions
    correct_predictions = 0
    total_predictions = 0
    step_accuracies = []

    # Track predictions and actual market movements
    # For Precision: Count predictions
    predicted_bullish = 0  # rho > 0.5
    predicted_bearish = 0  # rho < 0.5
    predicted_neutral = 0  # rho = 0.5

    # For Recall: Count actual market movements
    actual_up = 0    # market_return > 0
    actual_down = 0  # market_return < 0
    actual_flat = 0  # market_return = 0

    # Confusion matrix elements
    bullish_tp = 0  # Predicted bullish, market went up (True Positive)
    bearish_tp = 0  # Predicted bearish, market went down
    neutral_tp = 0  # Predicted neutral, market flat

    for step_idx, record in enumerate(portfolio_records):
        # Get rho value for this step from rho_record array
        if step_idx < len(rho_record):
            rho = rho_record[step_idx]
        else:
            print(f"Warning: No rho value for step {step_idx}, using default 0.5")
            rho = 0.5

        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * trade_len

        # Calculate market return for the next trade_len days
        if (decision_date_idx + 1 >= 0 and
            decision_date_idx + 1 < market_data.shape[0] and
            decision_date_idx + trade_len < market_data.shape[0]):

            current_market = market_data[decision_date_idx + 1, MARKET_PRICE_INDEX]
            future_market = market_data[decision_date_idx + trade_len, MARKET_PRICE_INDEX]

            if current_market > 0:
                market_return = (future_market - current_market) / current_market
            else:
                continue  # Skip if invalid price
        else:
            continue  # Skip if out of bounds

        # Track predictions
        if rho > 0.5:
            predicted_bullish += 1
        elif rho < 0.5:
            predicted_bearish += 1
        else:  # rho == 0.5
            predicted_neutral += 1

        # Track actual market movements
        if market_return > 0:
            actual_up += 1
        elif market_return < 0:
            actual_down += 1
        else:
            actual_flat += 1

        # Check prediction accuracy (for confusion matrix)
        is_correct = False

        if rho > 0.5:  # Predicted bullish
            if market_return > 0:
                bullish_tp += 1
                is_correct = True
        elif rho < 0.5:  # Predicted bearish
            if market_return < 0:
                bearish_tp += 1
                is_correct = True
        else:  # Predicted neutral (rho = 0.5)
            if market_return == 0:
                neutral_tp += 1
                is_correct = True

        if is_correct:
            correct_predictions += 1
        total_predictions += 1

        # Store step accuracy
        step_accuracies.append({
            'step': step_idx + 1,
            'rho': rho,
            'market_return': market_return,
            'is_correct': is_correct
        })

    # Calculate overall accuracy
    overall_accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0.0

    # Calculate Precision for each class
    bullish_precision = bullish_tp / predicted_bullish if predicted_bullish > 0 else 0.0
    bearish_precision = bearish_tp / predicted_bearish if predicted_bearish > 0 else 0.0
    neutral_precision = neutral_tp / predicted_neutral if predicted_neutral > 0 else 0.0

    # Calculate Recall for each class
    bullish_recall = bullish_tp / actual_up if actual_up > 0 else 0.0
    bearish_recall = bearish_tp / actual_down if actual_down > 0 else 0.0
    neutral_recall = neutral_tp / actual_flat if actual_flat > 0 else 0.0

    return {
        'period': period,
        'overall_accuracy': overall_accuracy,
        'correct_predictions': correct_predictions,
        'total_predictions': total_predictions,
        # Precision metrics
        'bullish_precision': bullish_precision,
        'bearish_precision': bearish_precision,
        'neutral_precision': neutral_precision,
        'predicted_bullish': predicted_bullish,
        'predicted_bearish': predicted_bearish,
        'predicted_neutral': predicted_neutral,
        # Recall metrics
        'bullish_recall': bullish_recall,
        'bearish_recall': bearish_recall,
        'neutral_recall': neutral_recall,
        'actual_up': actual_up,
        'actual_down': actual_down,
        'actual_flat': actual_flat,
        # True positives for each class
        'bullish_tp': bullish_tp,
        'bearish_tp': bearish_tp,
        'neutral_tp': neutral_tp,
        'step_accuracies': step_accuracies
    }