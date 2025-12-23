# -------------------------------
# Stock Price Trend Plotting Functions
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import json
import os
# spearmanr is now imported and used in analysis.py
from config import (
    config, START_DATE, END_DATE, TRADE_LEN,
    STOCK_DATA_PATH, STOCK_PRICE_INDEX, MARKET_DATA_PATH, MARKET_PRICE_INDEX,
    GROUND_TRUTH_PREFIX
)


def plot_stock_price_trends(experiment_id, outputs_base_path, stock_symbols, period='test', save_plots=True):
    """
    Plot stock price trends with long/short position markers for each stock.
    Green triangles up = Long positions, Red triangles down = Short positions.
    Save plots as PNG files.
    """
    # Load portfolio data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    portfolio_records = results.get('portfolio_records', [])
    if not portfolio_records:
        print(f"No portfolio records found for {experiment_id}")
        return
    
    # Load stock price data
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"Warning: Stock data not found at {STOCK_DATA_PATH}")
        return
    
    stocks_data = np.load(STOCK_DATA_PATH)
    print(f"Loaded stock data with shape: {stocks_data.shape}")
    
    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
        date_end_idx = config['val_end']
    else:  # test
        date_start_idx = config['val_end'] 
        date_end_idx = config['test_end']
    
    # Create output directory for stock price plots
    output_dir = f'src/plot/plot_outputs/{experiment_id}/stock_price_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    period_dates = full_dates[date_start_idx:date_end_idx]
    
    # Process each stock
    for stock_idx, stock_symbol in enumerate(stock_symbols):
        if stock_idx >= stocks_data.shape[0]:
            print(f"Warning: Stock index {stock_idx} exceeds data shape")
            continue
            
        # Extract price data for this stock and period
        stock_prices = stocks_data[stock_idx, date_start_idx:date_end_idx, STOCK_PRICE_INDEX]
        
        # Create position markers
        long_positions = []
        short_positions = []
        
        for step_idx, record in enumerate(portfolio_records):
            # Calculate the date index for this step
            step_date_idx = step_idx * TRADE_LEN
            if step_date_idx >= len(period_dates):
                break
                
            # Check if this stock is in long positions
            for pos in record['long_positions']:
                if pos['stock_index'] == stock_idx:
                    long_positions.append((step_date_idx, stock_prices[step_date_idx], pos['weight']))
            
            # Check if this stock is in short positions  
            for pos in record['short_positions']:
                if pos['stock_index'] == stock_idx:
                    short_positions.append((step_date_idx, stock_prices[step_date_idx], pos['weight']))
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot price trend
        plt.plot(period_dates, stock_prices, 'b-', linewidth=1, alpha=0.8, label='Open Price')
        
        # Draw vertical lines for each trading step (like heatmap x-axis)
        n_steps = len(portfolio_records)
        for step_idx in range(n_steps + 1):  # n_steps + 1 lines for n_steps intervals
            step_date_idx = step_idx * TRADE_LEN
            if step_date_idx < len(period_dates):
                step_date = period_dates[step_date_idx]
                plt.axvline(x=step_date, color='gray', linestyle='--', alpha=0.5, linewidth=0.8, zorder=2)
        
        # Add step labels on x-axis
        step_dates = []
        step_labels = []
        for step_idx in range(n_steps):
            step_date_idx = step_idx * TRADE_LEN
            if step_date_idx < len(period_dates):
                step_dates.append(period_dates[step_date_idx])
                step_labels.append(f'S{step_idx}')
        
        # Plot position markers
        if long_positions:
            long_dates = [period_dates[pos[0]] for pos in long_positions]
            long_prices = [pos[1] for pos in long_positions]
            long_weights = [pos[2] for pos in long_positions]
            
            # Scale marker size by weight (weight is typically 0-0.25)
            marker_sizes = [w * 400 for w in long_weights]  # Scale to visible size
            plt.scatter(long_dates, long_prices, marker='^', c='green', s=marker_sizes, 
                       alpha=0.8, label=f'Long Positions ({len(long_positions)})', zorder=5)
        
        if short_positions:
            short_dates = [period_dates[pos[0]] for pos in short_positions]
            short_prices = [pos[1] for pos in short_positions]
            short_weights = [pos[2] for pos in short_positions]
            
            # Scale marker size by weight
            marker_sizes = [w * 400 for w in short_weights]
            plt.scatter(short_dates, short_prices, marker='v', c='red', s=marker_sizes,
                       alpha=0.8, label=f'Short Positions ({len(short_positions)})', zorder=5)
        
        # Formatting
        plt.title(f'{stock_symbol} - Price Trend with Trading Steps ({period.upper()}, {n_steps} steps)', fontsize=14)
        plt.xlabel('Trading Steps (21 days per step)', fontsize=12)
        plt.ylabel('Open Price ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Set custom x-axis ticks showing step numbers and dates
        if len(step_dates) > 0:
            # Show every 3rd step to avoid crowding
            tick_interval = max(1, len(step_dates) // 10)
            tick_indices = range(0, len(step_dates), tick_interval)
            tick_dates = [step_dates[i] for i in tick_indices]
            tick_labels = [f'{step_dates[i].strftime("%Y-%m-%d")}' for i in tick_indices]
            
            plt.xticks(tick_dates, tick_labels, rotation=0, fontsize=10)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{output_dir}/{stock_symbol}_{period}_price_trend.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()  # Free memory


def plot_step_analysis(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plots=True):
    """
    Plot analysis for all trading steps showing the 8 selected stocks (4 long + 4 short) for each step.
    Each subplot shows past 70 days + future 21 days of open prices with decision point marked.

    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        stock_symbols: List of all stock symbols
        sample_dates: DatetimeIndex of trading decision dates (df_val.index or df_test.index)
        period: 'val' or 'test'
        save_plots: Whether to save the plots
    """
    # Load portfolio data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    portfolio_records = results.get('portfolio_records', [])
    rho_record = results.get('rho_record', [])

    if not portfolio_records:
        print(f"No portfolio records found for {experiment_id}")
        return

    if len(rho_record) != len(portfolio_records):
        print(f"Warning: rho_record length ({len(rho_record)}) != portfolio_records length ({len(portfolio_records)})")
        # Fill missing rho values with 0.5 (neutral)
        while len(rho_record) < len(portfolio_records):
            rho_record.append(0.5)

    # Load stock price data
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"Warning: Stock data not found at {STOCK_DATA_PATH}")
        return

    stocks_data = np.load(STOCK_DATA_PATH)
    print(f"Loaded stock data with shape: {stocks_data.shape}")

    # Load market data for MSU metrics calculation
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: Market data not found at {MARKET_DATA_PATH}")
        market_data = None
    else:
        market_data = np.load(MARKET_DATA_PATH)

    # Get date range for the period
    if period == 'val':
        date_start_idx = config.get('train_idx_end', config.get('train_end', 0))
    else:  # test
        date_start_idx = config.get('test_idx', config.get('val_end', 0))
    
    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    
    # Create output directory
    output_dir = f'src/plot/plot_outputs/{experiment_id}/step_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all trading steps (use portfolio_records length as the limit)
    for step_idx in range(len(portfolio_records)):
        
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN
        
        # Define the analysis window: past 70 days + future 21 days
        window_start_idx = decision_date_idx - 70
        window_end_idx = decision_date_idx + TRADE_LEN
        
        # Check bounds
        if window_start_idx < 0 or window_end_idx >= len(full_dates):
            print(f"Step {step_idx} analysis window is out of bounds, skipping...")
            continue
        
        # Get dates for the analysis window
        analysis_dates = full_dates[window_start_idx:window_end_idx + 1]
        decision_date = full_dates[decision_date_idx]
        
        # Get the portfolio positions for this step
        step_record = portfolio_records[step_idx]
        long_positions = step_record['long_positions']
        short_positions = step_record['short_positions']
        
        # Create dictionaries for quick lookup of selected stocks
        long_stocks = {pos['stock_index']: pos['weight'] for pos in long_positions}
        short_stocks = {pos['stock_index']: pos['weight'] for pos in short_positions}
        
        # Plot all 30 stocks (assuming stock indices 0-29)
        total_stocks = len(stock_symbols)
        n_cols = 6  # Fixed 6 columns for good layout
        n_rows = (total_stocks + n_cols - 1) // n_cols
        
        # Create the plot with layout for all stocks
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
        
        # Handle different subplot configurations
        if n_rows == 1:
            axes = [axes]
        
        # Calculate ASU metrics for this step
        asu_metrics_text = ""
        try:
            # Get all scores for this step
            all_scores = step_record.get('all_scores', [])

            if all_scores and len(all_scores) >= len(stock_symbols):
                # Get future returns from sim_info (already calculated correctly in validate.py)
                sim_info = step_record.get('sim_info', {})
                ror_array = sim_info.get('ror', None)

                if ror_array is not None:
                    # Flatten nested list structure if needed
                    if isinstance(ror_array, list) and len(ror_array) > 0:
                        if isinstance(ror_array[0], list):
                            flat_ror = [item for sublist in ror_array for item in sublist]
                        else:
                            flat_ror = ror_array
                    else:
                        flat_ror = ror_array

                    # Convert ror to return rate percentage
                    # ror = 1.08 means +8%, ror = 0.98 means -2%
                    returns = []
                    scores = []

                    for stock_idx in range(min(len(stock_symbols), len(all_scores), len(flat_ror))):
                        score = all_scores[stock_idx]
                        return_rate = (flat_ror[stock_idx] - 1.0) * 100  # Convert to percentage
                        scores.append(score)
                        returns.append(return_rate)
                else:
                    returns = []
                    scores = []

                if len(scores) > 0 and len(returns) > 0:
                    # Convert to numpy arrays
                    scores_array = np.array(scores)
                    returns_array = np.array(returns)

                    K = 4  # 4 long, 4 short

                    # Sort by scores to get predicted positions
                    sorted_indices = np.argsort(scores_array)
                    predicted_long_indices = set(sorted_indices[-K:])
                    predicted_short_indices = set(sorted_indices[:K])

                    # Calculate precision
                    long_correct = sum(1 for idx in predicted_long_indices if returns_array[idx] > 0)
                    short_correct = sum(1 for idx in predicted_short_indices if returns_array[idx] < 0)

                    p_l4 = long_correct / K if K > 0 else 0.0
                    p_s4 = short_correct / K if K > 0 else 0.0

                    # Calculate recall
                    positive_returns = [idx for idx in range(len(returns_array)) if returns_array[idx] > 0]
                    negative_returns = [idx for idx in range(len(returns_array)) if returns_array[idx] < 0]

                    r_l4 = 0.0
                    r_s4 = 0.0

                    if len(positive_returns) > 0:
                        positive_returns.sort(key=lambda x: returns_array[x], reverse=True)
                        actual_top_k = set(positive_returns[:min(K, len(positive_returns))])
                        long_intersection = len(predicted_long_indices & actual_top_k)
                        r_l4 = long_intersection / len(actual_top_k) if len(actual_top_k) > 0 else 0.0

                    if len(negative_returns) > 0:
                        negative_returns.sort(key=lambda x: returns_array[x])
                        actual_bottom_k = set(negative_returns[:min(K, len(negative_returns))])
                        short_intersection = len(predicted_short_indices & actual_bottom_k)
                        r_s4 = short_intersection / len(actual_bottom_k) if len(actual_bottom_k) > 0 else 0.0

                    asu_metrics_text = f' | ASU: P_L@4={p_l4:.3f} P_S@4={p_s4:.3f} R_L@4={r_l4:.3f} R_S@4={r_s4:.3f}'
        except Exception as e:
            print(f"Warning: Could not calculate ASU metrics for step {step_idx+1}: {e}")
            asu_metrics_text = " | ASU: Metrics N/A"

        fig.suptitle(f'Step {step_idx+1} Analysis - ALL STOCKS - {period.upper()} - Decision Date: {decision_date.strftime("%Y-%m-%d")}{asu_metrics_text}',
                     fontsize=16, fontweight='bold')
        
        # Plot all stocks by stock index order
        for stock_idx in range(total_stocks):
            try:
                if stock_idx >= len(stock_symbols):
                    continue
                    
                stock_symbol = stock_symbols[stock_idx]
                
                # Extract price data for this stock in the analysis window with bounds checking
                if (stock_idx < stocks_data.shape[0] and 
                    window_start_idx >= 0 and 
                    window_end_idx < stocks_data.shape[1] and
                    STOCK_PRICE_INDEX < stocks_data.shape[2]):
                    stock_prices = stocks_data[stock_idx, window_start_idx:window_end_idx + 1, STOCK_PRICE_INDEX]
                else:
                    print(f"Warning: Invalid indices for stock {stock_idx} at step {step_idx}")
                    continue
                
                # Check for valid price data
                if len(stock_prices) == 0 or len(analysis_dates) == 0:
                    print(f"Warning: Empty data for stock {stock_idx} at step {step_idx}")
                    continue
                    
                if len(stock_prices) != len(analysis_dates):
                    print(f"Warning: Data length mismatch for stock {stock_idx}: prices={len(stock_prices)}, dates={len(analysis_dates)}")
                    continue
                
                # Determine subplot position (by stock index order)
                row = stock_idx // n_cols
                col = stock_idx % n_cols
                
                if row < len(axes) and col < len(axes[row]):
                    ax = axes[row][col]
                    ax.plot(analysis_dates, stock_prices, 'b-', linewidth=1.5, alpha=0.8)
                    
                    # Mark the decision point with vertical line
                    ax.axvline(x=decision_date, color='red', linestyle='--', linewidth=2, alpha=0.8)
                    
                    # Add background colors for past vs future
                    ax.axvspan(analysis_dates[0], decision_date, facecolor='lightblue', alpha=0.1, label='Past 70 days')
                    
                    # Determine position type and styling
                    if stock_idx in long_stocks:
                        # This stock is selected for LONG
                        pos_type = 'LONG'
                        weight = long_stocks[stock_idx]
                        title_color = 'green'
                        future_color = 'lightgreen'
                        ax.axvspan(decision_date, analysis_dates[-1], facecolor=future_color, alpha=0.1, label='Future 21 days')
                        title = f'{stock_symbol} (IDX: {stock_idx}) - {pos_type}\nWeight: {weight:.3f}'
                    elif stock_idx in short_stocks:
                        # This stock is selected for SHORT
                        pos_type = 'SHORT'
                        weight = short_stocks[stock_idx]
                        title_color = 'red'
                        future_color = 'lightcoral'
                        ax.axvspan(decision_date, analysis_dates[-1], facecolor=future_color, alpha=0.1, label='Future 21 days')
                        title = f'{stock_symbol} (IDX: {stock_idx}) - {pos_type}\nWeight: {weight:.3f}'
                    else:
                        # This stock is NOT selected
                        pos_type = 'NOT SELECTED'
                        title_color = 'gray'
                        future_color = 'lightgray'
                        ax.axvspan(decision_date, analysis_dates[-1], facecolor=future_color, alpha=0.1, label='Future 21 days')
                        title = f'{stock_symbol} (IDX: {stock_idx}) - {pos_type}'
                    
                    # Formatting with stock index in title
                    ax.set_title(title, fontsize=10, fontweight='bold', color=title_color)
                    ax.set_ylabel('Open Price ($)', fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.tick_params(axis='x', rotation=45, labelsize=8)
                    ax.tick_params(axis='y', labelsize=8)
                    
                    # Add x-label for bottom row
                    if row == n_rows - 1:
                        ax.set_xlabel('Date', fontsize=8)
                        
            except Exception as e:
                print(f"Error plotting stock {stock_idx} at step {step_idx}: {e}")
                continue
        
        # Hide unused subplots if any
        for row in range(n_rows):
            for col in range(n_cols):
                stock_idx = row * n_cols + col
                if stock_idx >= total_stocks:
                    if row < len(axes) and col < len(axes[row]):
                        axes[row][col].set_visible(False)
        
        # Add legend
        if len(axes) > 0 and len(axes[0]) > 0:
            handles, labels = axes[0][0].get_legend_handles_labels()
            if handles:
                fig.legend(handles, labels, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Make room for suptitle
        
        if save_plots:
            filename = f'{output_dir}/step_{step_idx+1:02d}_{period}_analysis.png'
            try:
                plt.savefig(filename, dpi=150, bbox_inches='tight')
                print(f"Saved: {filename}")
            except Exception as e:
                print(f"Error saving {filename}: {e}")
            finally:
                plt.close(fig)  # Close specific figure
                plt.close('all')  # Ensure all figures are closed


def plot_msu_step_analysis(experiment_id, outputs_base_path, sample_dates, period='test', save_plots=True):
    """
    Plot MSU step analysis showing past 65 days + future 21 days of market open prices
    with the AI's rho decision for each trading step.
    
    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        sample_dates: DatetimeIndex of trading decision dates 
        period: 'val' or 'test'
        save_plots: Whether to save the plots
    """
    # Load portfolio data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    portfolio_records = results.get('portfolio_records', [])
    rho_record = results.get('rho_record', [])
    
    if not portfolio_records:
        print(f"No portfolio records found for {experiment_id}")
        return
    
    if len(rho_record) != len(portfolio_records):
        print(f"Warning: rho_record length ({len(rho_record)}) != portfolio_records length ({len(portfolio_records)})")
        return
    
    # Load market data
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: Market data not found at {MARKET_DATA_PATH}")
        return
    
    market_data = np.load(MARKET_DATA_PATH)
    print(f"Loaded market data with shape: {market_data.shape}")
    
    # Use market open price index from config
    market_open_index = MARKET_PRICE_INDEX
    
    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end'] 
    
    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    
    # Create output directory
    output_dir = f'src/plot/plot_outputs/{experiment_id}/msu_step_analysis'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all trading steps (use portfolio_records length as the limit)
    for step_idx in range(len(portfolio_records)):
        
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN
        
        # Define the MSU analysis window: past 65 days + future 21 days
        window_start_idx = decision_date_idx - 65
        window_end_idx = decision_date_idx + TRADE_LEN
        
        # Check bounds
        if window_start_idx < 0 or window_end_idx >= len(full_dates):
            print(f"Step {step_idx} MSU analysis window is out of bounds, skipping...")
            continue
        
        # Get dates and market prices for the analysis window
        analysis_dates = full_dates[window_start_idx:window_end_idx + 1]
        decision_date = full_dates[decision_date_idx]
        
        # Extract market open prices for the analysis window
        market_prices = market_data[window_start_idx:window_end_idx + 1, market_open_index]
        
        # Get the rho value for this step
        rho_value = rho_record[step_idx]
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot market price trend
        plt.plot(analysis_dates, market_prices, 'b-', linewidth=1.5, alpha=0.8, label='Market Open')
        
        # Mark the decision point with vertical line
        plt.axvline(x=decision_date, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Decision Point')
        
        # Add background colors for past vs future
        plt.axvspan(analysis_dates[0], decision_date, facecolor='lightblue', alpha=0.1, label='Past 65 days (MSU input)')
        plt.axvspan(decision_date, analysis_dates[-1], facecolor='lightgreen', alpha=0.1, label='Future 21 days (Trading period)')
        
        # Formatting
        plt.title(f'MSU Step {step_idx+1} Analysis - {period.upper()} - ρ = {rho_value:.4f}\\n'
                  f'Decision Date: {decision_date.strftime("%Y-%m-%d")}',
                  fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Market Open Price', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{output_dir}/step_{step_idx+1:02d}_{period}_msu_analysis.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()


def plot_step_score_scatter(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plots=True):
    """
    Plot scatter plots for each trading step showing Score vs Future 21-day Return Rate.
    Each point represents one stock. If prediction is accurate, points should align along
    a positive diagonal (high score = high return, low score = low return).

    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        stock_symbols: List of all stock symbols
        sample_dates: DatetimeIndex of trading decision dates
        period: 'val' or 'test'
        save_plots: Whether to save the plots
    """
    # Load portfolio data
    from config import JSON_FILES
    import json

    # Load config for date indices
    config_paths = [
        f"{outputs_base_path}/{experiment_id}/config.json",
        f"{outputs_base_path}/{experiment_id}/log_file/hyper.json"
    ]

    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            break

    if config is None:
        print(f"Warning: No config file found for {experiment_id}")
        return
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    portfolio_records = results.get('portfolio_records', [])
    if not portfolio_records:
        print(f"No portfolio records found for {experiment_id}")
        return

    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_idx_end']
    else:  # test
        date_start_idx = config['test_idx']

    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)

    # Create output directory
    output_dir = f'src/plot/plot_outputs/{experiment_id}/score_scatter_plots'
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all trading steps
    for step_idx in range(len(portfolio_records)):
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN
        decision_date = full_dates[decision_date_idx]
        
        # Get the portfolio positions for this step
        step_record = portfolio_records[step_idx]
        all_scores = step_record.get('all_scores', [])
        long_positions = step_record['long_positions']
        short_positions = step_record['short_positions']

        if not all_scores or len(all_scores) < len(stock_symbols):
            print(f"Warning: Step {step_idx} missing score data, skipping...")
            continue

        # Get ror from sim_info
        sim_info = step_record.get('sim_info', {})
        ror_array = sim_info.get('ror', None)

        if ror_array is None:
            print(f"Warning: Step {step_idx} missing ror data, skipping...")
            continue

        # Handle nested list structure (flatten if needed)
        if isinstance(ror_array, list) and len(ror_array) > 0:
            if isinstance(ror_array[0], list):
                # Flatten nested list: [[1.08, 0.97, ...]] -> [1.08, 0.97, ...]
                flat_ror = [item for sublist in ror_array for item in sublist]
            else:
                flat_ror = ror_array
        else:
            flat_ror = ror_array

        # Create sets for quick lookup of selected positions
        long_stocks = set([pos['stock_index'] for pos in long_positions])
        short_stocks = set([pos['stock_index'] for pos in short_positions])

        # Prepare data for scatter plot
        scores = []
        returns = []
        colors = []
        markers = []
        labels = []

        for stock_idx in range(len(stock_symbols)):
            if stock_idx < len(all_scores) and stock_idx < len(flat_ror):
                score = all_scores[stock_idx]

                # Get return rate from ror (convert to percentage)
                # ror = 1.08 means +8%, ror = 0.98 means -2%
                return_rate = (flat_ror[stock_idx] - 1.0) * 100

                scores.append(score)
                returns.append(return_rate)
                labels.append(stock_symbols[stock_idx])

                # Color and marker based on position type
                if stock_idx in long_stocks:
                    colors.append('green')
                    markers.append('^')  # triangle up
                elif stock_idx in short_stocks:
                    colors.append('red')
                    markers.append('v')  # triangle down
                else:
                    colors.append('gray')
                    markers.append('o')  # circle
        
        if not scores:
            continue
        
        # Create the scatter plot
        plt.figure(figsize=(12, 8))
        
        # Plot each point with its specific color and marker
        for i in range(len(scores)):
            plt.scatter(scores[i], returns[i], c=colors[i], marker=markers[i], 
                       s=80, alpha=0.7, edgecolors='black', linewidth=0.5)
        
        # Add trend line (linear regression)
        z = np.polyfit(scores, returns, 1)
        p = np.poly1d(z)
        x_trend = np.linspace(0, 1, 100)
        plt.plot(x_trend, p(x_trend), "b--", alpha=0.5, linewidth=2, label='Trend Line')
        
        # Add horizontal and vertical reference lines
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        plt.axvline(x=0.5, color='black', linestyle='-', alpha=0.3, linewidth=0.8)
        
        # Calculate ASU metrics for this step
        asu_metrics_text = ""
        if len(scores) == len(returns) and len(scores) > 0:
            try:
                # Convert to numpy arrays for calculation
                scores_array = np.array(scores)
                returns_array = np.array(returns)

                # Calculate metrics for K=4 (long top 4, short bottom 4)
                K = 4

                # Sort by scores to get predicted long/short positions
                sorted_indices = np.argsort(scores_array)
                predicted_long_indices = set(sorted_indices[-K:])  # Top K scores (long)
                predicted_short_indices = set(sorted_indices[:K])  # Bottom K scores (short)

                # Calculate precision: How many predicted positions moved in expected direction
                long_correct = sum(1 for idx in predicted_long_indices if returns_array[idx] > 0)
                short_correct = sum(1 for idx in predicted_short_indices if returns_array[idx] < 0)

                p_l4 = long_correct / K if K > 0 else 0.0  # Precision Long@4
                p_s4 = short_correct / K if K > 0 else 0.0  # Precision Short@4

                # Calculate recall: How many of actual top/bottom performers we caught
                # Sort by actual returns to get actual top/bottom performers
                sorted_by_returns = np.argsort(returns_array)

                # Filter to only consider stocks that actually moved in the right direction
                positive_returns = [idx for idx in range(len(returns_array)) if returns_array[idx] > 0]
                negative_returns = [idx for idx in range(len(returns_array)) if returns_array[idx] < 0]

                if len(positive_returns) > 0:
                    # Get actual top performers among stocks that went up
                    positive_returns.sort(key=lambda x: returns_array[x], reverse=True)
                    actual_top_k = set(positive_returns[:min(K, len(positive_returns))])
                    long_intersection = len(predicted_long_indices & actual_top_k)
                    r_l4 = long_intersection / len(actual_top_k) if len(actual_top_k) > 0 else 0.0
                else:
                    r_l4 = 0.0

                if len(negative_returns) > 0:
                    # Get actual bottom performers among stocks that went down
                    negative_returns.sort(key=lambda x: returns_array[x])
                    actual_bottom_k = set(negative_returns[:min(K, len(negative_returns))])
                    short_intersection = len(predicted_short_indices & actual_bottom_k)
                    r_s4 = short_intersection / len(actual_bottom_k) if len(actual_bottom_k) > 0 else 0.0
                else:
                    r_s4 = 0.0

                asu_metrics_text = (f' | ASU: P_L@4={p_l4:.3f} P_S@4={p_s4:.3f} '
                                   f'R_L@4={r_l4:.3f} R_S@4={r_s4:.3f}')
            except Exception as e:
                print(f"Warning: Could not calculate ASU metrics for step {step_idx+1}: {e}")
                asu_metrics_text = " | ASU: Metrics N/A"

        # Formatting
        plt.title(f'Step {step_idx+1} Score vs Future Return - {period.upper()}\n'
                  f'Decision Date: {decision_date.strftime("%Y-%m-%d")} '
                  f'({len(stock_symbols)} stocks){asu_metrics_text}', fontsize=14, fontweight='bold')
        plt.xlabel('ASU Score (0 = Worst, 1 = Best)', fontsize=12)
        plt.ylabel('Future 21-day Return Rate (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Set axis limits
        plt.xlim(-0.05, 1.05)
        y_margin = (max(returns) - min(returns)) * 0.1
        plt.ylim(min(returns) - y_margin, max(returns) + y_margin)
        
        # Create custom legend
        legend_elements = []
        
        # Add Trend Line to legend
        legend_elements.append(plt.Line2D([0], [0], color='blue', linestyle='--', linewidth=2, 
                                        label='Trend Line'))
        
        if long_stocks:
            legend_elements.append(plt.scatter([], [], c='green', marker='^', s=80, 
                                             label=f'Long Positions ({len(long_stocks)})'))
        if short_stocks:
            legend_elements.append(plt.scatter([], [], c='red', marker='v', s=80, 
                                             label=f'Short Positions ({len(short_stocks)})'))
        legend_elements.append(plt.scatter([], [], c='gray', marker='o', s=80, 
                                         label=f'Not Selected ({len(stock_symbols) - len(long_stocks) - len(short_stocks)})'))
        
        plt.legend(handles=legend_elements, loc='upper left')
        
        # Calculate correlation using unified function from analysis module
        from analysis import compute_single_step_correlation

        corr_metrics = compute_single_step_correlation(scores, returns)

        if not np.isnan(corr_metrics['pearson_corr']):
            # Statistics text with proper line breaks
            stats_text = f'Pearson (values): {corr_metrics["pearson_corr"]:.3f}\n'
            stats_text += f'Spearman (values→ranks): {corr_metrics["spearman_corr"]:.3f}'
        else:
            stats_text = 'Insufficient data for correlation'
        
        plt.text(0.98, 0.98, stats_text, 
                transform=plt.gca().transAxes, fontsize=9, 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                verticalalignment='top', horizontalalignment='right')
        
        # Add stock labels for selected positions (optional, might be crowded)
        # for i in range(len(scores)):
        #     if colors[i] != 'gray':  # Only label selected stocks
        #         plt.annotate(labels[i], (scores[i], returns[i]), 
        #                     xytext=(5, 5), textcoords='offset points', 
        #                     fontsize=8, alpha=0.8)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{output_dir}/step_{step_idx+1:02d}_{period}_score_scatter.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()


def plot_all_steps_score_scatter(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plots=True):
    """
    Plot a combined scatter plot for all trading steps showing Score vs Future 21-day Return Rate.
    This gives an overview of prediction accuracy across all time steps.
    
    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        stock_symbols: List of all stock symbols
        sample_dates: DatetimeIndex of trading decision dates
        period: 'val' or 'test'
    """
    # Load portfolio data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    portfolio_records = results.get('portfolio_records', [])
    if not portfolio_records:
        print(f"No portfolio records found for {experiment_id}")
        return
    
    # Load stock price data
    if not os.path.exists(STOCK_DATA_PATH):
        print(f"Warning: Stock data not found at {STOCK_DATA_PATH}")
        return
    
    stocks_data = np.load(STOCK_DATA_PATH)
    
    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end']
    
    # Collect all data points across all steps
    all_scores = []
    all_returns = []
    all_colors = []
    all_markers = []
    
    # Note: Step correlations are now calculated in compute_correlation_metrics_with_strategy
    
    # Process each trading step
    for step_idx in range(len(portfolio_records)):
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + step_idx * TRADE_LEN
        
        # Get the portfolio positions for this step
        step_record = portfolio_records[step_idx]
        all_step_scores = step_record.get('all_scores', [])
        long_positions = step_record['long_positions']
        short_positions = step_record['short_positions']
        
        if not all_step_scores or len(all_step_scores) < len(stock_symbols):
            continue
        
        # Create sets for quick lookup
        long_stocks = set([pos['stock_index'] for pos in long_positions])
        short_stocks = set([pos['stock_index'] for pos in short_positions])

        # Get future returns from sim_info (already calculated correctly in validate.py)
        sim_info = step_record.get('sim_info', {})
        ror_array = sim_info.get('ror', None)

        if ror_array is None:
            continue

        # Flatten nested list structure if needed
        if isinstance(ror_array, list) and len(ror_array) > 0:
            if isinstance(ror_array[0], list):
                flat_ror = [item for sublist in ror_array for item in sublist]
            else:
                flat_ror = ror_array
        else:
            flat_ror = ror_array

        # Collect data for this step
        step_scores = []
        step_returns = []

        for stock_idx in range(min(len(stock_symbols), len(all_step_scores), len(flat_ror))):
            score = all_step_scores[stock_idx]
            # Convert ror to return rate percentage (ror = 1.08 means +8%)
            return_rate = (flat_ror[stock_idx] - 1.0) * 100

            all_scores.append(score)
            all_returns.append(return_rate)
            step_scores.append(score)
            step_returns.append(return_rate)

            # Color and marker based on position type
            if stock_idx in long_stocks:
                all_colors.append('green')
                all_markers.append('^')
            elif stock_idx in short_stocks:
                all_colors.append('red')
                all_markers.append('v')
            else:
                all_colors.append('gray')
                all_markers.append('o')
        # Step correlations are now calculated in compute_correlation_metrics_with_strategy
    
    if not all_scores:
        print(f"No valid data found for {period}")
        return
    
    # Create the combined scatter plot
    plt.figure(figsize=(14, 10))
    
    # Create separate scatter plots for each position type to get proper legend
    long_indices = [i for i, c in enumerate(all_colors) if c == 'green']
    short_indices = [i for i, c in enumerate(all_colors) if c == 'red']
    neutral_indices = [i for i, c in enumerate(all_colors) if c == 'gray']
    
    if neutral_indices:
        plt.scatter([all_scores[i] for i in neutral_indices], 
                   [all_returns[i] for i in neutral_indices], 
                   c='gray', marker='o', s=30, alpha=0.5, 
                   label=f'Not Selected ({len(neutral_indices)})', zorder=1)
    
    if long_indices:
        plt.scatter([all_scores[i] for i in long_indices], 
                   [all_returns[i] for i in long_indices], 
                   c='green', marker='^', s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'Long Positions ({len(long_indices)})', zorder=2)
    
    if short_indices:
        plt.scatter([all_scores[i] for i in short_indices], 
                   [all_returns[i] for i in short_indices], 
                   c='red', marker='v', s=60, alpha=0.7, edgecolors='black', linewidth=0.5,
                   label=f'Short Positions ({len(short_indices)})', zorder=2)
    
    # Add reference lines
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1)
    plt.axvline(x=0.5, color='black', linestyle='-', alpha=0.3, linewidth=1)
    
    # Add trend line (linear regression)
    z = np.polyfit(all_scores, all_returns, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(0, 1, 100)
    plt.plot(x_trend, p(x_trend), "b--", alpha=0.5, linewidth=2, label='Trend Line')
    
    # Formatting
    plt.title(f'All Steps Combined: Score vs Future Return - {period.upper()}\n'
              f'{len(portfolio_records)} trading steps, {len(all_scores)} data points',
              fontsize=16, fontweight='bold')
    plt.xlabel('ASU Score (0 = Worst, 1 = Best)', fontsize=14)
    plt.ylabel('Future 21-day Return Rate (%)', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    plt.xlim(-0.05, 1.05)
    y_margin = (max(all_returns) - min(all_returns)) * 0.1
    plt.ylim(min(all_returns) - y_margin, max(all_returns) + y_margin)
    
    # Calculate overall correlations using unified function from analysis module
    from analysis import compute_correlation_metrics

    # Get correlation metrics for ALL strategy (all 30 stocks)
    corr_metrics = compute_correlation_metrics(
        experiment_id, outputs_base_path, period
    )

    if corr_metrics and not np.isnan(corr_metrics['overall_pearson']):
        # Statistics text with proper line breaks
        stats_text = 'OVERALL CORRELATIONS:\n'
        stats_text += f'Pearson (values): {corr_metrics["overall_pearson"]:.3f}\n'
        stats_text += f'Spearman (values→ranks): {corr_metrics["overall_spearman"]:.3f}\n'
        stats_text += '\n'  # Empty line
        stats_text += 'MEAN STEP CORRELATIONS:\n'
        stats_text += f'Pearson (values): {corr_metrics["mean_step_pearson"]:.3f}\n'
        stats_text += f'Spearman (values→ranks): {corr_metrics["mean_step_spearman"]:.3f}'
    else:
        stats_text = 'Insufficient data for correlation analysis'
    
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, fontsize=11, 
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
             verticalalignment='top')
    
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()

    if save_plots:
        # Save the plot
        output_dir = f'src/plot/plot_outputs/{experiment_id}/score_scatter_plots'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/all_steps_{period}_score_scatter.png'
        plt.savefig(filename, dpi=200, bbox_inches='tight')
        print(f"Saved combined scatter plot: {filename}")
        plt.close()


def plot_rho_with_market_trend(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot DJIA market trend with predicted mu, sigma, rho, and ground truth rho.

    Layout from top to bottom:
    1. DJIA opening price trend line
    2. Predicted Mu (μ) values as bar chart
    3. Predicted Sigma (σ) values as bar chart
    4. Predicted Rho (ρ) values as bar chart
    5. Ground Truth Rho (ρ) values as bar chart (if available)

    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        sample_dates: List of sample dates
        period: 'val' or 'test'
        save_plot: Whether to save the plot
    """
    # Load records from JSON
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)

    mu_record = results.get('mu_record', [])
    sigma_record = results.get('sigma_record', [])
    rho_record = results.get('rho_record', [])

    if not rho_record:
        print(f"No rho records found for {experiment_id}")
        return

    mu_array = np.array(mu_record) if mu_record else None
    sigma_array = np.array(sigma_record) if sigma_record else None
    rho_array = np.array(rho_record)
    n_steps = len(rho_array)

    # Load ground truth rho records
    # Format: MSU_{period}_ground_truth_step{TRADE_LEN}.json
    # Use GROUND_TRUTH_PREFIX from config to allow switching between datasets
    ground_truth_path = f'{GROUND_TRUTH_PREFIX}/MSU_{period}_ground_truth_step{TRADE_LEN}.json'
    ground_truth_rho = None

    if os.path.exists(ground_truth_path):
        with open(ground_truth_path, 'r', encoding='utf-8') as f:
            ground_truth_data = json.load(f)

        # Extract rho from ground_truth_records
        if 'ground_truth_records' in ground_truth_data:
            ground_truth_records = ground_truth_data['ground_truth_records']
            ground_truth_rho = np.array([record['rho'] for record in ground_truth_records])
            print(f"Loaded ground truth rho with {len(ground_truth_rho)} steps from {ground_truth_path}")
        else:
            print(f"Warning: 'ground_truth_records' not found in {ground_truth_path}")
            ground_truth_rho = None

        # Verify alignment
        if ground_truth_rho is not None and len(ground_truth_rho) != n_steps:
            print(f"Warning: Ground truth rho length ({len(ground_truth_rho)}) != predicted rho length ({n_steps})")
            # Truncate or pad to match
            if len(ground_truth_rho) > n_steps:
                ground_truth_rho = ground_truth_rho[:n_steps]
            else:
                # Pad with zeros if ground truth is shorter
                ground_truth_rho = np.pad(ground_truth_rho, (0, n_steps - len(ground_truth_rho)), constant_values=0)
    else:
        print(f"Warning: Ground truth file not found at {ground_truth_path}")
        ground_truth_rho = None

    # Load market data (DJIA closing prices)
    if not os.path.exists(MARKET_DATA_PATH):
        print(f"Warning: Market data not found at {MARKET_DATA_PATH}")
        return

    market_data = np.load(MARKET_DATA_PATH)

    # Get the date range for this period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end']

    # Extract DJIA opening prices: initial + after each step (n_steps + 1 points total)
    # This matches plot_market_profit_heatmap logic: wealth has n_steps values
    djia_open_prices = []

    # Point 0: Initial state (before first action)
    cursor_initial = date_start_idx
    if cursor_initial < market_data.shape[0]:
        djia_open_prices.append(market_data[cursor_initial, MARKET_PRICE_INDEX])
    else:
        djia_open_prices.append(np.nan)

    # Points 1 to n_steps: After each action period
    for step in range(n_steps):
        cursor = date_start_idx + (step + 1) * TRADE_LEN
        if cursor < market_data.shape[0]:
            djia_open = market_data[cursor, MARKET_PRICE_INDEX]
            djia_open_prices.append(djia_open)
        else:
            djia_open_prices.append(np.nan)

    djia_open_prices = np.array(djia_open_prices)
    n_djia_points = len(djia_open_prices)  # Should be n_steps + 1

    # Verify data alignment
    print(f"DEBUG: n_steps (rho)={n_steps}, djia_points={n_djia_points}, sample_dates={len(sample_dates) if sample_dates is not None else None}")

    # Create figure with subplots
    # Layout: opening price, mu, sigma, rho, ground_truth_rho
    num_subplots = 1  # Always have opening price
    if mu_array is not None:
        num_subplots += 1
    if sigma_array is not None:
        num_subplots += 1
    num_subplots += 1  # Always have predicted rho
    if ground_truth_rho is not None:
        num_subplots += 1

    # Create height ratios: first panel (opening price) is taller
    height_ratios = [2] + [1] * (num_subplots - 1)

    fig, axes = plt.subplots(num_subplots, 1, figsize=(16, 4 * num_subplots),
                             gridspec_kw={'height_ratios': height_ratios})

    # Make sure axes is iterable
    if num_subplots == 1:
        axes = [axes]

    # Assign axes to specific panels
    ax_idx = 0
    ax1 = axes[ax_idx]  # Opening price
    ax_idx += 1

    ax2 = axes[ax_idx] if mu_array is not None else None  # Mu
    if mu_array is not None:
        ax_idx += 1

    ax3 = axes[ax_idx] if sigma_array is not None else None  # Sigma
    if sigma_array is not None:
        ax_idx += 1

    ax4 = axes[ax_idx]  # Predicted Rho (always present)
    ax_idx += 1

    ax5 = axes[ax_idx] if ground_truth_rho is not None else None  # Ground truth Rho
    if ground_truth_rho is not None:
        ax_idx += 1

    # === Panel 1: DJIA opening price trend (n_steps + 1 points) ===
    djia_x_positions = np.arange(n_djia_points)  # 0, 1, 2, ..., n_steps
    ax1.plot(djia_x_positions, djia_open_prices,
             color='darkblue', linewidth=2, marker='o', markersize=3, label='DJIA Open')
    ax1.set_ylabel('DJIA Opening Price', fontsize=12, fontweight='bold')
    ax1.set_title(f'Market Trend and MSU Parameters (μ, σ, ρ) - {period.upper()}',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='upper left')

    # Format y-axis for prices
    ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:,.0f}'))

    # Positions for bar charts (n_steps bars)
    bar_x_positions = np.arange(n_steps)

    # === Panel 2: Predicted Mu bar chart (if available) ===
    if mu_array is not None and ax2 is not None:
        ax2.bar(bar_x_positions, mu_array, color='lightsalmon', edgecolor='darkorange',
                linewidth=0.5, alpha=0.8, width=0.8)
        ax2.set_ylabel('Predicted Mu (μ)', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3, axis='y')

        # Add horizontal reference line at 0
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='μ=0 (Neutral)')
        ax2.legend(loc='upper right', fontsize=9)

    # === Panel 3: Predicted Sigma bar chart (if available) ===
    if sigma_array is not None and ax3 is not None:
        ax3.bar(bar_x_positions, sigma_array, color='lightgreen', edgecolor='darkgreen',
                linewidth=0.5, alpha=0.8, width=0.8)
        ax3.set_ylabel('Predicted Sigma (σ)', fontsize=12, fontweight='bold')
        # Sigma typically ranges from 0 to some reasonable max, adjust as needed
        ax3.set_ylim(0, max(sigma_array) * 1.1 if len(sigma_array) > 0 else 1.0)
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.legend(loc='upper right', fontsize=9)

    # === Panel 4: Predicted Rho bar chart (always present) ===
    ax4.bar(bar_x_positions, rho_array, color='skyblue', edgecolor='steelblue',
            linewidth=0.5, alpha=0.8, width=0.8)
    ax4.set_ylabel('Predicted Rho (ρ)', fontsize=12, fontweight='bold')
    ax4.set_ylim(0, 1.0)
    ax4.grid(True, alpha=0.3, axis='y')

    # Add horizontal reference lines
    ax4.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='ρ=0.5 (Balanced)')
    ax4.legend(loc='upper right', fontsize=9)

    # === Panel 5: Ground Truth Rho bar chart (if available) ===
    if ground_truth_rho is not None and ax5 is not None:
        ax5.bar(bar_x_positions, ground_truth_rho, color='lightcoral', edgecolor='firebrick',
                linewidth=0.5, alpha=0.8, width=0.8)
        ax5.set_ylabel('Ground Truth Rho (ρ)', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Decision Date (Action Day)', fontsize=12, fontweight='bold')
        ax5.set_ylim(0, 1.0)
        ax5.grid(True, alpha=0.3, axis='y')

        # Add horizontal reference lines
        ax5.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='ρ=0.5 (Balanced)')
        ax5.legend(loc='upper right', fontsize=9)

    # Set x-axis labels (actual decision dates from sample_dates)
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Use same x-axis label format as other heatmaps
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)

        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]

        # Determine bottom-most panel and set labels
        bottom_ax = ax5 if ax5 is not None else ax4

        # Set x-axis ticks and labels for all panels
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])  # Hide labels on top panel

        if ax2 is not None:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([])

        if ax3 is not None:
            ax3.set_xticks(xticks)
            ax3.set_xticklabels([])

        ax4.set_xticks(xticks)
        if bottom_ax == ax4:
            ax4.set_xticklabels(xlabels, rotation=45, ha='right')
        else:
            ax4.set_xticklabels([])

        if ax5 is not None:
            ax5.set_xticks(xticks)
            ax5.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback: use step numbers
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)

        xlabels = [f'Step {i+1}' for i in xticks]

        # Determine bottom-most panel
        bottom_ax = ax5 if ax5 is not None else ax4

        # Set x-axis ticks and labels for all panels
        ax1.set_xticks(xticks)
        ax1.set_xticklabels([])

        if ax2 is not None:
            ax2.set_xticks(xticks)
            ax2.set_xticklabels([])

        if ax3 is not None:
            ax3.set_xticks(xticks)
            ax3.set_xticklabels([])

        ax4.set_xticks(xticks)
        if bottom_ax == ax4:
            ax4.set_xticklabels(xlabels)
        else:
            ax4.set_xticklabels([])

        if ax5 is not None:
            ax5.set_xticks(xticks)
            ax5.set_xticklabels(xlabels)

    # Set x-axis limits: -0.5 to n_steps+0.5 to show all DJIA points (including the extra one)
    ax1.set_xlim(-0.5, n_steps + 0.5)
    if ax2 is not None:
        ax2.set_xlim(-0.5, n_steps + 0.5)
    if ax3 is not None:
        ax3.set_xlim(-0.5, n_steps + 0.5)
    ax4.set_xlim(-0.5, n_steps + 0.5)
    if ax5 is not None:
        ax5.set_xlim(-0.5, n_steps + 0.5)

    plt.tight_layout()

    if save_plot:
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)
        filename = f'{output_dir}/rho_market_trend_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()