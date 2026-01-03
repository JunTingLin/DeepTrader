"""
Test ASU Supervised Model and Generate Results JSON

Generate test results compatible with DeepTrader for src/plot/main.py evaluation
Support --period val/test, using the same timestamp as training
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from environment.portfolio_env import PortfolioSim

from model.ASU import ASU
from ASU.asu_dataset import ASUDataset


def load_checkpoint_and_create_model(checkpoint_path, device):
    """
    Load checkpoint and create model

    Returns:
        model, args_dict, timestamp
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Read training parameters from checkpoint
    args_dict = checkpoint['args']
    timestamp = checkpoint.get('timestamp', 'unknown')
    model_type = args_dict.get('model_type', 'ASU_GCN')

    # Determine model type
    is_te2d = (model_type == 'ASU_TE2D' or 'TE2D' in str(checkpoint_path))

    # Create ASU model
    model = ASU(
        num_nodes=args_dict.get('num_stocks', 30),
        in_features=args_dict.get('num_features', 34),
        hidden_dim=args_dict.get('hidden_dim', 128),
        window_len=args_dict.get('window_len', 13),
        dropout=args_dict.get('dropout', 0.5),
        kernel_size=args_dict.get('kernel_size', 2),
        layers=args_dict.get('num_blocks', 4),
        supports=None,
        spatial_bool=args_dict.get('spatial_bool', False),
        addaptiveadj=args_dict.get('addaptiveadj', True),
        aptinit=None,
        transformer_asu_bool=is_te2d,  # 根據模型類型設置
        num_assets=args_dict.get('num_stocks', 30)
    ).to(device)

    # Load weights
    model.load_state_dict(checkpoint['asu_state_dict'])
    model.eval()

    print(f"✓ Loaded model from {checkpoint_path}")
    print(f"  Model type: {model_type}")
    print(f"  Timestamp: {timestamp}")
    print(f"  Best Val IC: {checkpoint.get('val_ic', 'N/A'):.4f}")

    return model, args_dict, timestamp


def create_test_dataloader(args_dict, period='test', batch_size=1):
    """
    Create test dataloader

    Args:
        args_dict: 訓練參數字典
        period: 'val' or 'test'
        batch_size: 通常為 1（逐步測試）

    Returns:
        dataloader, num_stocks, num_features
    """
    data_dir = args_dict['data_dir']
    stocks_data_path = os.path.join(data_dir, 'stocks_data.npy')
    ror_path = os.path.join(data_dir, 'ror.npy')

    window_len = args_dict['window_len']
    horizon = args_dict['horizon']

    if period == 'val':
        start_idx = args_dict['val_idx']
        end_idx = args_dict['test_idx']
    else:  # test
        start_idx = args_dict['test_idx']
        end_idx = args_dict['test_idx_end']

    # Use stride=21 (trade_len) to match DeepTrader's trading period
    dataset = ASUDataset(
        stocks_data_path, ror_path,
        start_idx=start_idx,
        end_idx=end_idx,
        window_len=window_len,
        horizon=horizon,
        stride=21  # Evaluate every 21 days (trade_len)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    num_stocks = dataset.get_num_stocks()
    num_features = dataset.get_num_features()

    return dataloader, num_stocks, num_features


def generate_portfolio_decisions(scores, mask, k=4, allow_short=True):
    """
    Generate portfolio decisions based on scores

    Args:
        scores: [num_stocks] - 股票評分
        mask: [num_stocks] - True for invalid stocks
        k: Select top K for long, bottom K for short
        allow_short: Whether to allow short positions

    Returns:
        long_indices, short_indices
    """
    # Set masked stocks to -inf
    scores_copy = scores.copy()
    scores_copy[mask] = -np.inf

    # Sort
    sorted_indices = np.argsort(scores_copy)[::-1]  # Descending order

    # Select top-K for long
    long_indices = sorted_indices[:k]
    long_indices = long_indices[scores_copy[long_indices] > -np.inf]  # Remove masked stocks

    if allow_short:
        # Select bottom-K for short
        short_indices = sorted_indices[-k:]
        short_indices = short_indices[scores_copy[short_indices] > -np.inf]
        # Reverse order (from worst to second worst)
        short_indices = short_indices[::-1]
    else:
        short_indices = np.array([])

    return long_indices, short_indices


def test_and_generate_json(model, dataloader, device, output_path, k=4, allow_short=True,
                           rate_of_return=None, trade_len=21, num_stocks=30, rho=0.0):
    """
    Test model and generate JSON results with portfolio simulation

    Format compatible with src/outputs/random/01/json_file/test_results_random.json
    Uses portfolio_records format for src/plot/main.py evaluation

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: torch device
        output_path: JSON output path
        k: Number of stocks to select for long/short
        allow_short: Whether to allow short positions
        rate_of_return: RoR data for portfolio simulation (num_stocks, num_days)
        trade_len: Trading period length (default: 21)
        num_stocks: Number of stocks (default: 30)
        rho: MSU allocation parameter (default: 0.0 for ASU-only)
    """
    model.eval()

    # Initialize PortfolioSim
    sim = PortfolioSim(
        num_assets=num_stocks,
        fee=0.0025,
        time_cost=0.0,
        allow_short=allow_short
    )
    sim.reset(batch_num=1)

    agent_wealth = [1.0]  # Start with $1
    rho_record = []
    portfolio_records = []

    print("\n" + "="*80)
    print("Testing model and generating portfolio decisions...")
    print(f"Portfolio settings: k={k}, allow_short={allow_short}, rho={rho}, trade_len={trade_len}")
    print("="*80)

    with torch.no_grad():
        for step_idx, (stocks_window, future_ror, mask, time_idx) in enumerate(dataloader):
            stocks_window = stocks_window.to(device)  # [1, num_stocks, window_len, features]
            mask_tensor = mask.to(device)  # [1, num_stocks]

            # Predict scores
            scores = model(stocks_window, mask_tensor)  # [1, num_stocks]

            # Convert to numpy
            scores_np = scores[0].cpu().numpy()  # [num_stocks]
            mask_np = mask[0].cpu().numpy()  # [num_stocks]
            future_ror_np = future_ror[0].cpu().numpy()  # [num_stocks]

            # Generate portfolio decisions
            long_indices, short_indices = generate_portfolio_decisions(
                scores_np, mask_np, k=k, allow_short=allow_short
            )

            # Calculate equal weights
            equal_weight = 1.0 / k if k > 0 else 0.0

            # Construct weight vector for PortfolioSim
            # Shape: (1, num_stocks*2) where first half is long, second half is short
            weights = np.zeros((1, num_stocks * 2))

            # Set long position weights
            for stock_idx in long_indices:
                weights[0, stock_idx] = equal_weight

            # Set short position weights
            for stock_idx in short_indices:
                weights[0, num_stocks + stock_idx] = equal_weight

            # Get RoR for PortfolioSim (cumulative return over trade_len days)
            cursor = int(time_idx[0])
            if rate_of_return is not None:
                # Get future returns from rate_of_return data
                future_return_data = rate_of_return[:, cursor + 1:cursor + 1 + trade_len]
                ror_sim = np.prod((future_return_data + 1), axis=-1)  # Cumulative return
                ror_sim = ror_sim[np.newaxis, :]  # (1, num_stocks)
            else:
                # Fallback: use future_ror from dataset (already cumulative)
                ror_sim = (1.0 + future_ror_np)[np.newaxis, :]  # (1, num_stocks)

            # Execute portfolio simulation step
            rho_array = np.array([rho])
            reward, info, done = sim._step(weights, ror_sim, rho_array)

            # Record wealth
            current_wealth = sim.v[0]
            agent_wealth.append(current_wealth)
            rho_record.append(rho)

            # Build portfolio_records in src/plot/main.py format
            # Sort by score for ranking
            long_scores = [(idx, scores_np[idx]) for idx in long_indices]
            long_scores.sort(key=lambda x: x[1], reverse=True)

            short_scores = [(idx, scores_np[idx]) for idx in short_indices]
            short_scores.sort(key=lambda x: x[1])  # Ascending for short (worst first)

            # Long positions
            long_positions = []
            long_weights = []
            for rank, (idx, score) in enumerate(long_scores, start=1):
                long_positions.append({
                    'stock_index': int(idx),
                    'weight': equal_weight,
                    'score': float(score),
                    'rank': rank,
                    'future_return': float(future_ror_np[idx])
                })
                long_weights.append(equal_weight)

            # Short positions
            short_positions = []
            short_weights = []
            for rank, (idx, score) in enumerate(short_scores, start=1):
                short_positions.append({
                    'stock_index': int(idx),
                    'weight': equal_weight,
                    'score': float(score),
                    'rank': rank,
                    'future_return': float(future_ror_np[idx])
                })
                short_weights.append(equal_weight)

            # Build sim_info (compatible with DeepTrader's plot functions)
            # ROR: future returns as ratios (1 + return_rate)
            # Note: DeepTrader uses nested list format [[stock1, stock2, ...]] for batch compatibility
            ror_ratios = [(1.0 + future_ror_np).tolist()]  # Wrap in list for batch dimension
            long_returns = [float(future_ror_np[idx]) for idx in long_indices]
            short_returns = [float(future_ror_np[idx]) for idx in short_indices]

            portfolio_records.append({
                'step': int(step_idx),
                'time_idx': int(time_idx[0]),
                'long_positions': long_positions,
                'short_positions': short_positions,
                'all_scores': scores_np.tolist(),  # All 30 stocks' scores for scatter plots
                'sim_info': {
                    'ror': ror_ratios,  # [[stock1_ratio, stock2_ratio, ...]] - nested list format
                    'long_returns': long_returns,  # Long positions' returns
                    'short_returns': short_returns  # Short positions' returns
                }
            })

            # Print progress
            if (step_idx + 1) % 5 == 0 or step_idx == len(dataloader) - 1:
                print(f"  Step {step_idx + 1}/{len(dataloader)}: Wealth = ${current_wealth:.4f}")

    # Build final JSON in DeepTrader format
    results = {
        'agent_wealth': [agent_wealth],  # [[step0, step1, ...]]
        'rho_record': rho_record,
        'portfolio_records': portfolio_records,
        'experiment_type': 'asu_supervised',
        'allow_short': allow_short,
        'k': k,
        'num_steps': len(portfolio_records)
    }

    # Save JSON
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Generated {len(portfolio_records)} portfolio decisions")
    print(f"✓ Saved to: {output_path}")

    # Compute statistics
    avg_long = np.mean([len(rec['long_positions']) for rec in portfolio_records])
    avg_short = np.mean([len(rec['short_positions']) for rec in portfolio_records])
    final_wealth = agent_wealth[-1]
    total_return = (final_wealth - 1.0) * 100

    print(f"\nPortfolio Statistics:")
    print(f"  Average long positions: {avg_long:.1f}")
    print(f"  Average short positions: {avg_short:.1f}")
    print(f"  Total steps: {len(portfolio_records)}")
    print(f"  Final wealth: ${final_wealth:.4f}")
    print(f"  Total return: {total_return:+.2f}%")

    return results


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Load model
    checkpoint_path = Path(args.checkpoint_path)
    model, args_dict, timestamp = load_checkpoint_and_create_model(checkpoint_path, device)

    # Load rate_of_return data for portfolio simulation
    print(f"\nLoading rate of return data...")
    data_prefix = args_dict.get('data_dir', 'src/data/DJIA/feature34-Inter-P532') + '/'
    rate_of_return = np.load(data_prefix + 'ror.npy')  # (num_stocks, num_days)
    print(f"RoR data shape: {rate_of_return.shape}")

    # Create test data
    print(f"\nLoading {args.period} data...")
    dataloader, num_stocks, num_features = create_test_dataloader(
        args_dict, period=args.period, batch_size=1
    )

    # Generate output path (using same timestamp)
    if timestamp != 'unknown':
        output_dir = Path(args.output_dir) / f"asu_supervised_{timestamp}"
    else:
        output_dir = Path(args.output_dir) / f"asu_supervised_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save copy of training parameters in log_file (following DeepTrader convention)
    log_dir = output_dir / 'log_file'
    log_dir.mkdir(parents=True, exist_ok=True)
    with open(log_dir / 'hyper.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Test and generate JSON
    json_dir = output_dir / 'json_file'
    json_dir.mkdir(parents=True, exist_ok=True)
    json_output_path = json_dir / f'{args.period}_results.json'

    # Get trade_len and rho from args
    trade_len = args_dict.get('horizon', 21)  # Use horizon as trade_len
    rho = getattr(args, 'rho', 0.0)  # Default rho=0 for ASU-only

    results = test_and_generate_json(
        model, dataloader, device, json_output_path,
        k=args.k, allow_short=args.allow_short,
        rate_of_return=rate_of_return,
        trade_len=trade_len,
        num_stocks=num_stocks,
        rho=rho
    )

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"Results JSON: {json_output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Test ASU Supervised Model')

    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help='Path to checkpoint file (best_asu.pth)')
    parser.add_argument('--period', type=str, default='test', choices=['val', 'test'],
                        help='Test on validation or test set')
    parser.add_argument('--output_dir', type=str, default='./src/outputs',
                        help='Output directory for results')
    parser.add_argument('--k', type=int, default=4,
                        help='Number of stocks to select (top-K long, bottom-K short)')
    parser.add_argument('--allow_short', action='store_true', default=True,
                        help='Allow short positions')
    parser.add_argument('--rho', type=float, default=0.0,
                        help='MSU allocation parameter (default: 0.0 for ASU-only)')
    parser.add_argument('--use_gpu', action='store_true', default=True)

    args = parser.parse_args()
    main(args)
