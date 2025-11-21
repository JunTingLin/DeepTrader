"""
Comprehensive Analysis: Compare Stock A, Stock B, Market, and DeepTrader
"""
import numpy as np
import json
from pathlib import Path

# Configuration
EXPERIMENT_ID = '1114/035244'
OUTPUTS_BASE_PATH = '../outputs'
TRADE_LEN = 21  # 21 trading days = 1 month

def calculate_metrics(prices, period_name="", is_wealth=False):
    """
    Calculate ASR, ARR, MDD, AVOL, Cumulative Wealth, Total Return

    Args:
        prices: numpy array of daily prices or wealth series
        period_name: name of the period for display
        is_wealth: if True, prices is already a wealth series (no need to sample/normalize)

    Returns:
        dict with all metrics
    """
    if is_wealth:
        # Already a wealth series (from DeepTrader)
        wealth = prices
        trade_ror = wealth[1:] / wealth[:-1] - 1
    else:
        # Replicate data_processor.py's simple sampling method
        # Step 1: Convert daily prices to cumulative wealth
        daily_return = np.zeros(len(prices))
        daily_return[0] = 0.0
        daily_return[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        market_wealth_daily = (1.0 + daily_return).cumprod()

        # Step 2: Sample every TRADE_LEN days (cursor days: 0, 21, 42, ...)
        # This matches data_processor.py line 45:
        # market_series_val = market_wealth_val.iloc[::TRADE_LEN][:n_val_complete + 1]
        n_complete = len(prices) // TRADE_LEN
        sampled_wealth = market_wealth_daily[::TRADE_LEN][:n_complete + 1]

        # Normalize to start at 1.0 (matches data_processor.py line 46)
        wealth = sampled_wealth / sampled_wealth[0]

        # Calculate monthly returns from sampled wealth
        trade_ror = wealth[1:] / wealth[:-1] - 1

    # ARR: Simple annualization (mean monthly return × 12)
    AT = np.mean(trade_ror)
    arr = AT * 12  # Ny = 12 for monthly

    # AVOL: Annualized Volatility (std × sqrt(12))
    VT = np.std(trade_ror, ddof=0)  # ddof=0 to match official
    avol = VT * np.sqrt(12)

    # ASR: Annualized Sharpe Ratio
    asr = arr / avol if avol > 0 else 0

    # MDD: Maximum Drawdown (from wealth series)
    running_max = np.maximum.accumulate(wealth)
    drawdown = (running_max - wealth) / running_max
    mdd = np.max(drawdown)

    # Total Return
    total_return = wealth[-1] - 1

    return {
        'ASR': asr,
        'ARR': arr,
        'MDD': mdd,
        'AVOL': avol,
        'Cum_Wealth': wealth[-1],
        'Total_Return': total_return,
        'monthly_returns': trade_ror,
        'wealth_series': wealth
    }


def load_fake_data():
    """Load fake data for Stock A, Stock B, and Market"""
    # Load stocks data from numpy file
    stocks_data_path = Path('../data/fake/stocks_data.npy')
    stocks_data = np.load(stocks_data_path)  # Shape: (num_stocks, num_days, features)

    # Extract close prices
    # stocks_data shape: (stocks, days, features)
    stock_a_prices = stocks_data[0, :, 0]  # Stock 0, all days, price
    stock_b_prices = stocks_data[1, :, 0]  # Stock 1, all days, price

    # Load market data from market_data.npy (to match main.py)
    market_data_path = Path('../data/fake/market_data.npy')
    market_data = np.load(market_data_path)  # Shape: (num_days, features)
    market_prices = market_data[:, 0]  # Extract price column

    # Split according to config.py indices
    # train_end: 1304, val_end: 2087, test_end: 2673
    train_end = 1304
    val_end = 2087
    test_end = 2673

    stock_a_val = stock_a_prices[train_end:val_end]
    stock_b_val = stock_b_prices[train_end:val_end]
    market_val = market_prices[train_end:val_end]

    stock_a_test = stock_a_prices[val_end:test_end]
    stock_b_test = stock_b_prices[val_end:test_end]
    market_test = market_prices[val_end:test_end]

    return {
        'val': {
            'stock_a': stock_a_val,
            'stock_b': stock_b_val,
            'market': market_val
        },
        'test': {
            'stock_a': stock_a_test,
            'stock_b': stock_b_test,
            'market': market_test
        }
    }


def load_deeptrader_results():
    """Load DeepTrader results from JSON"""
    val_path = Path(OUTPUTS_BASE_PATH) / EXPERIMENT_ID / 'json_file' / 'val_results.json'
    test_path = Path(OUTPUTS_BASE_PATH) / EXPERIMENT_ID / 'json_file' / 'test_results.json'

    with open(val_path, 'r') as f:
        val_results = json.load(f)

    with open(test_path, 'r') as f:
        test_results = json.load(f)

    return {
        'val': np.array(val_results['agent_wealth']).flatten(),
        'test': np.array(test_results['agent_wealth']).flatten()
    }


def print_comparison_table(metrics_dict, period_name):
    """Print formatted comparison table"""
    print("\n" + "="*120)
    print(f"{period_name} Period")
    print("="*120)

    # Header
    print(f"{'Target':<15} {'ASR':>8} {'ARR':>10} {'MDD':>10} {'AVOL':>10} {'Cum_Wealth':>12} {'Total_Return':>12}")
    print("-" * 120)

    # Print each target
    for name, metrics in metrics_dict.items():
        print(f"{name:<15} {metrics['ASR']:>8.2f} {metrics['ARR']*100:>9.2f}% "
              f"{metrics['MDD']*100:>9.2f}% {metrics['AVOL']*100:>9.2f}% "
              f"{metrics['Cum_Wealth']:>12.4f} {metrics['Total_Return']*100:>11.2f}%")

    print("="*120)


def print_monthly_returns(metrics_dict, period_name):
    """Print monthly returns for all targets"""
    print("\n" + "="*120)
    print(f"{period_name} Period - Monthly Returns")
    print("="*120)

    # Get the minimum number of months across all targets
    n_months = min(len(metrics['monthly_returns']) for metrics in metrics_dict.values())

    # Header
    print(f"{'Month':<8}", end='')
    for name in metrics_dict.keys():
        print(f"{name:>15}", end='')
    print()
    print("-" * 120)

    # Print each month
    for i in range(n_months):
        print(f"Month {i+1:<2}", end='')
        for name, metrics in metrics_dict.items():
            if i < len(metrics['monthly_returns']):
                ret = metrics['monthly_returns'][i] * 100
                print(f"{ret:>14.4f}%", end='')
            else:
                print(f"{'N/A':>15}", end='')
        print()

    # Print verification (product of (1+returns) should equal Cum_Wealth)
    print("\n" + "-" * 120)
    print("Verification: Product of (1 + monthly_return) should equal Cum_Wealth")
    print("-" * 120)

    for name, metrics in metrics_dict.items():
        cumulative = np.prod(1 + metrics['monthly_returns'])
        print(f"{name:<15}: Product = {cumulative:.6f}, Cum_Wealth = {metrics['Cum_Wealth']:.6f}, "
              f"Diff = {abs(cumulative - metrics['Cum_Wealth']):.8f}")

    print("="*120)


def main():
    print("\n" + "="*120)
    print("Comprehensive Analysis: Stock A, Stock B, Market, DeepTrader")
    print("="*120)

    # Load data
    fake_data = load_fake_data()
    deeptrader_wealth = load_deeptrader_results()

    # Analyze each period
    for period in ['val', 'test']:
        # Calculate metrics for each target
        metrics = {
            'Stock A': calculate_metrics(fake_data[period]['stock_a'], period, is_wealth=False),
            'Stock B': calculate_metrics(fake_data[period]['stock_b'], period, is_wealth=False),
            'Market': calculate_metrics(fake_data[period]['market'], period, is_wealth=False),
            'DeepTrader': calculate_metrics(deeptrader_wealth[period], period, is_wealth=True)
        }

        # Print comparison table
        period_name = period.upper()
        print_comparison_table(metrics, period_name)

        # Print monthly returns
        print_monthly_returns(metrics, period_name)


if __name__ == '__main__':
    main()
