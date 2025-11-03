"""
Oracle Portfolio Testing Script
--------------------------------
This script tests an oracle (perfect foresight) portfolio strategy on val or test period:
- Selects the top G stocks with highest future returns for long positions (equal weight)
- Selects the top G stocks with lowest future returns for short positions (equal weight)
- Achieves 100% precision and recall by definition
- Uses the same PortfolioSim to calculate returns
- Generates val_results_oracle.json or test_results_oracle.json

This provides the theoretical upper bound for portfolio performance.
"""

import argparse
import json
import os
import numpy as np
from utils.parse_config import ConfigParser
from utils.functions import convert_to_native_type, calculate_metrics, setup_seed
from environment.portfolio_env import PortfolioSim


def oracle_portfolio_test(func_args, period='test'):
    """Run oracle (perfect foresight) portfolio test on val or test period

    Args:
        func_args: Configuration arguments
        period: 'val' or 'test' (default: 'test')
    """

    print("\n" + "="*60)
    print(f"Oracle Portfolio Testing - {period.upper()}")
    print("Perfect Foresight Strategy (Theoretical Upper Bound)")
    print("="*60)

    # Load data
    data_prefix = func_args.data_prefix
    print(f"Loading data from {data_prefix}...")

    stocks_data = np.load(data_prefix + 'stocks_data.npy')
    rate_of_return = np.load(data_prefix + 'ror.npy')  # Shape: (num_stocks, num_days)
    market_history = np.load(data_prefix + 'market_data.npy')

    num_stocks = stocks_data.shape[0]
    print(f"Number of stocks: {num_stocks}")

    # Get period indices based on period argument
    if period == 'val':
        period_idx = func_args.val_idx
        period_idx_end = func_args.test_idx  # val ends where test begins
    else:  # test
        period_idx = func_args.test_idx
        period_idx_end = func_args.test_idx_end

    trade_len = func_args.trade_len
    G = func_args.G  # Number of stocks to long/short

    # Calculate number of trading steps in the period
    num_steps = (period_idx_end - period_idx) // trade_len
    print(f"{period.upper()} period: {period_idx} to {period_idx_end}")
    print(f"Number of trading steps: {num_steps}")
    print(f"G (long/short stocks): {G}")

    # Initialize PortfolioSim
    allow_short = getattr(func_args, 'allow_short', True)
    fee = getattr(func_args, 'fee', 0.0025)
    time_cost = getattr(func_args, 'time_cost', 0.0)

    sim = PortfolioSim(
        num_assets=num_stocks,
        fee=fee,
        time_cost=time_cost,
        allow_short=allow_short
    )

    print(f"Portfolio settings: allow_short={allow_short}, fee={fee}, G={G}, trade_len={trade_len}")

    # Initialize tracking variables
    agent_wealth = [1.0]  # Start with $1
    portfolio_records = []

    # Reset simulator
    sim.reset(batch_num=1)

    print("\n" + "="*60)
    print("Running oracle portfolio simulation...")
    print("="*60)

    # Run through the period
    for step in range(num_steps):
        # Calculate cursor
        cursor = period_idx + step * trade_len

        # Get future returns for this period (21-day cumulative return)
        future_return = rate_of_return[:, cursor + 1:cursor + 1 + trade_len]  # (num_stocks, trade_len)
        cumulative_returns = np.prod((future_return + 1), axis=-1) - 1  # (num_stocks,) - cumulative return

        # Oracle selection: choose stocks with highest/lowest future returns
        # Create list of (stock_idx, return) tuples
        stock_returns = [(i, cumulative_returns[i]) for i in range(num_stocks)]

        # Sort by return (descending)
        stock_returns_sorted = sorted(stock_returns, key=lambda x: x[1], reverse=True)

        # Select top G for long (highest returns)
        long_stocks = sorted([stock_returns_sorted[i][0] for i in range(G)])

        # Select bottom G for short (lowest returns)
        short_stocks = sorted([stock_returns_sorted[-(i+1)][0] for i in range(G)])

        # Calculate equal weight (1/G for each position)
        equal_weight = 1.0 / G

        # Construct weight vector
        # Shape: (batch=1, num_assets*2) where first half is long, second half is short
        weights = np.zeros((1, num_stocks * 2))

        # Set long position weights
        long_weights = [equal_weight] * G
        for idx, stock_idx in enumerate(long_stocks):
            weights[0, stock_idx] = long_weights[idx]

        # Set short position weights
        short_weights = [equal_weight] * G
        for idx, stock_idx in enumerate(short_stocks):
            weights[0, num_stocks + stock_idx] = short_weights[idx]

        # Get rate of return for simulation
        ror = cumulative_returns + 1  # Convert back to (1 + return) format
        ror = ror[np.newaxis, :]  # (1, num_stocks)

        # Set rho (MSU allocation)
        rho = np.array([0.5])

        # Execute step
        reward, info, done = sim._step(weights, ror, rho)

        # Record results
        current_wealth = sim.v[0]
        agent_wealth.append(current_wealth)

        # Calculate prediction window (holding period)
        predict_start = cursor + 1
        predict_end = cursor + trade_len + 1

        # Create fake scores (use actual returns as "scores" for visualization)
        # Normalize returns to [0, 1] range for visualization
        min_ret = cumulative_returns.min()
        max_ret = cumulative_returns.max()
        if max_ret > min_ret:
            normalized_scores = (cumulative_returns - min_ret) / (max_ret - min_ret)
        else:
            normalized_scores = np.ones(num_stocks) * 0.5

        # Construct portfolio record
        portfolio_record = {
            'step': step + 1,
            'predict_start': predict_start,  # Holding period start
            'predict_end': predict_end,      # Holding period end
            'long_indices': [long_stocks],  # (1, G)
            'long_weights': [long_weights],  # (1, G)
            'short_indices': [short_stocks],  # (1, G)
            'short_weights': [short_weights],  # (1, G)
            'all_scores': np.array([normalized_scores]),  # (1, num_stocks) - normalized returns as scores
            'sim_info': info
        }

        portfolio_records.append(portfolio_record)

        if (step + 1) % 5 == 0 or step == num_steps - 1:
            long_returns = [cumulative_returns[i] for i in long_stocks]
            short_returns = [cumulative_returns[i] for i in short_stocks]
            print(f"Step {step+1}/{num_steps}: Wealth = ${current_wealth:.4f}, RoR = {info['rate_of_return'][0]:.4%}")
            print(f"  Long (top {G}): {long_stocks} | Avg Return: {np.mean(long_returns):.2%}")
            print(f"  Short (bottom {G}): {short_stocks} | Avg Return: {np.mean(short_returns):.2%}")

    print("\n" + "="*60)
    print("Simulation complete!")
    print("="*60)

    # Calculate metrics
    agent_wealth_array = np.array([agent_wealth])
    metrics = calculate_metrics(agent_wealth_array, func_args.trade_mode)

    print("\nPerformance Metrics:")
    print(f"  Final Wealth: ${agent_wealth[-1]:.4f}")
    print(f"  Total Return: {(agent_wealth[-1] - 1.0) * 100:.2f}%")
    print(f"  ARR (Annualized Return Rate): {metrics['ARR'].item():.4f}")
    print(f"  MDD (Maximum Drawdown): {metrics['MDD'].item():.4f}")
    print(f"  AVOL (Annualized Volatility): {metrics['AVOL'].item():.4f}")
    print(f"  ASR (Annualized Sharpe Ratio): {metrics['ASR'].item():.4f}")
    print(f"  CR (Calmar Ratio): {metrics['CR'].item():.4f}")

    # Convert portfolio records to JSON format
    json_portfolio_records = []
    for i, portfolio_info in enumerate(portfolio_records):
        step_data = {
            'step': portfolio_info['step'],
            'predict_start': portfolio_info['predict_start'],
            'predict_end': portfolio_info['predict_end']
        }

        # Process long positions
        batch_idx = 0
        long_positions = []
        for pos_idx in range(len(portfolio_info['long_indices'][batch_idx])):
            stock_idx = int(portfolio_info['long_indices'][batch_idx][pos_idx])
            weight = float(portfolio_info['long_weights'][batch_idx][pos_idx])
            score = float(portfolio_info['all_scores'][batch_idx][stock_idx])
            long_positions.append({
                'stock_index': stock_idx,
                'weight': weight,
                'score': score
            })
        step_data['long_positions'] = long_positions

        # Process short positions
        short_positions = []
        for pos_idx in range(len(portfolio_info['short_indices'][batch_idx])):
            stock_idx = int(portfolio_info['short_indices'][batch_idx][pos_idx])
            weight = float(portfolio_info['short_weights'][batch_idx][pos_idx])
            score = float(portfolio_info['all_scores'][batch_idx][stock_idx])
            short_positions.append({
                'stock_index': stock_idx,
                'weight': weight,
                'score': score
            })
        step_data['short_positions'] = short_positions

        # Add all scores
        step_data['all_scores'] = portfolio_info['all_scores'][batch_idx].tolist()

        # Add sim_info (simplified)
        if 'sim_info' in portfolio_info:
            sim_info = portfolio_info['sim_info']
            step_data['sim_info'] = {
                'LongPosition_return': convert_to_native_type(sim_info.get('LongPosition_return')),
                'rate_of_return': convert_to_native_type(sim_info.get('rate_of_return')),
                'ror': convert_to_native_type(sim_info.get('market_fluctuation')),
                'long_returns': convert_to_native_type(sim_info.get('long_returns')),
            }

            if 'ShortPosition_return' in sim_info:
                step_data['sim_info'].update({
                    'ShortPosition_return': convert_to_native_type(sim_info.get('ShortPosition_return')),
                    'short_returns': convert_to_native_type(sim_info.get('short_returns')),
                })

        json_portfolio_records.append(step_data)

    # Prepare output JSON
    results = {
        'agent_wealth': agent_wealth_array.tolist(),
        'rho_record': [0.5] * num_steps,  # Fixed at 0.5 for oracle
        'mu_record': [None] * num_steps,  # Not applicable
        'sigma_record': [None] * num_steps,  # Not applicable
        'portfolio_records': json_portfolio_records,
        'performance_metrics': {
            'ARR': convert_to_native_type(metrics['ARR']),
            'MDD': convert_to_native_type(metrics['MDD']),
            'AVOL': convert_to_native_type(metrics['AVOL']),
            'ASR': convert_to_native_type(metrics['ASR']),
            'DDR': convert_to_native_type(metrics['DDR']),
            'CR': convert_to_native_type(metrics['CR'])
        },
        'summary': {
            'strategy': 'oracle',
            'period': period,
            'G': G,
            'total_steps': num_steps,
            'final_wealth': convert_to_native_type(agent_wealth[-1]),
            'total_return': convert_to_native_type(agent_wealth[-1] - 1.0)
        }
    }

    # Save to JSON
    PREFIX = func_args.prefix
    json_save_dir = os.path.join(PREFIX, 'json_file')
    os.makedirs(json_save_dir, exist_ok=True)

    output_filename = f'{period}_results_oracle.json'
    output_path = os.path.join(json_save_dir, output_filename)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to: {output_path}")
    print("="*60)

    return results


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Oracle Portfolio Testing (Perfect Foresight)')
    parser.add_argument('-c', '--config', type=str, default=None,
                        help='config file path (default: None)')
    parser.add_argument('--prefix', type=str, required=True,
                        help='path to experiment directory (required)')
    parser.add_argument('--period', type=str, default='test', choices=['val', 'test'],
                        help='evaluation period: "val" or "test" (default: test)')

    opts = parser.parse_args()

    # Load config
    if opts.config is not None:
        # Use custom config if provided
        with open(opts.config) as f:
            options = json.load(f)
            func_args = ConfigParser(options)
    else:
        # Load config from experiment directory
        config_path = os.path.join(opts.prefix, 'log_file', 'hyper.json')

        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found at {config_path}")

        with open(config_path) as f:
            options = json.load(f)
            func_args = ConfigParser(options)

    # Override with command line arguments
    func_args.prefix = opts.prefix

    # Run oracle test
    oracle_portfolio_test(func_args, period=opts.period)


if __name__ == '__main__':
    main()
