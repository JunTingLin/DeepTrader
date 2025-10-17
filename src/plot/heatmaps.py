# -------------------------------
# Heatmap Plotting Functions
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_portfolio_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test',
                          save_plot=True, show_scores=False, show_ranks=False):
    """
    Plot portfolio positions as single heatmap (positive=long, negative=short).

    Args:
        experiment_id: Experiment ID
        outputs_base_path: Base path to outputs
        stock_symbols: List of stock symbols
        sample_dates: List of sample dates
        period: 'test' or 'val'
        save_plot: Whether to save the plot
        show_scores: Whether to show score numbers on cells (0-100, no % symbol)
        show_ranks: Whether to show rank numbers on cells (1-30, highest score = rank 1)
    """
    # Load JSON data
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
    
    # Prepare data matrix
    n_stocks = len(stock_symbols)
    n_steps = len(portfolio_records)
    
    # Single matrix: positive for long, negative for short
    position_matrix = np.zeros((n_stocks, n_steps))

    # Matrices for scores and ranks if needed
    score_matrix = np.zeros((n_stocks, n_steps)) if show_scores else None
    rank_matrix = np.zeros((n_stocks, n_steps)) if show_ranks else None

    for i, record in enumerate(portfolio_records):
        # Long positions (positive values)
        for pos in record['long_positions']:
            idx = pos['stock_index']
            if idx < n_stocks:
                position_matrix[idx, i] = pos['weight']

        # Short positions (negative values)
        for pos in record['short_positions']:
            idx = pos['stock_index']
            if idx < n_stocks:
                position_matrix[idx, i] = -pos['weight']

        # Process scores and ranks if needed
        if show_scores or show_ranks:
            all_scores = record.get('all_scores', [])
            if all_scores and len(all_scores) >= n_stocks:
                # Create list of (score, stock_idx) for ranking
                score_data = [(all_scores[stock_idx], stock_idx) for stock_idx in range(n_stocks)]
                # Sort by score descending (highest score gets rank 1)
                score_data_sorted = sorted(score_data, key=lambda x: x[0], reverse=True)

                for stock_idx in range(n_stocks):
                    if show_scores:
                        # Store original score (keep as decimal)
                        score_matrix[stock_idx, i] = all_scores[stock_idx]

                    if show_ranks:
                        # Find rank for this stock (1-based)
                        rank = next((rank for rank, (score, idx) in enumerate(score_data_sorted, 1) if idx == stock_idx), 0)
                        rank_matrix[stock_idx, i] = rank
    
    # Create figure - adjust width for text display if needed
    cell_width = 1.2 if (show_scores or show_ranks) else 0.4
    # Adjust height based on number of stocks for better aspect ratio
    height = max(10, min(16, n_stocks * 0.5))
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), height))
    
    # Plot heatmap with diverging colormap
    # Use auto aspect for better readability when showing text
    aspect_setting = 'auto' if (show_scores or show_ranks) else 'equal'

    im = ax.imshow(position_matrix, aspect=aspect_setting, cmap='RdYlGn', interpolation='nearest',
                   vmin=-0.3, vmax=0.3, origin='lower')

    # Add text annotations if requested
    if show_scores or show_ranks:
        for i in range(n_stocks):
            for j in range(n_steps):
                text_parts = []
                if show_scores and score_matrix is not None:
                    score_val = score_matrix[i, j]
                    if score_val > 0:
                        text_parts.append(f'{score_val:.2f}')

                if show_ranks and rank_matrix is not None:
                    rank_val = int(rank_matrix[i, j])
                    if rank_val > 0:
                        text_parts.append(f'#{rank_val}')

                if text_parts:
                    # Choose text color based on background
                    position_val = position_matrix[i, j]
                    text_color = 'white' if abs(position_val) > 0.15 else 'black'

                    # Join text parts with newline if both present
                    text = '\n'.join(text_parts)

                    ax.text(j, i, text, ha='center', va='center',
                           color=text_color, fontsize=9, fontweight='bold')
    
    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    # Update title based on display options
    title_parts = [f'Portfolio Positions - {period.upper()}']
    if show_scores:
        title_parts.append('Scores (0.00-1.00)')
    if show_ranks:
        title_parts.append('Ranks (#1-30)')

    title = ' | '.join(title_parts)
    ax.set_title(title, fontsize=14)
    
    # Set y-axis labels with both index and symbol
    ax.set_yticks(range(n_stocks))
    y_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Set x-axis labels - show more dates with wider cells
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, n_steps // 20)  # Show more labels
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, n_steps // 20)
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Weight (Green=Long, Red=Short)', fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_stocks) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        # Create output directory for this experiment
        output_dir = f'plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{output_dir}/portfolio_positions_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def plot_future_return_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plot=True):
    """
    Plot future 21-day return rates as heatmap with same layout as portfolio_heatmap.
    Color depth represents future return rate, text shows the actual RR value.

    Args:
        experiment_id: Experiment ID
        outputs_base_path: Base path to outputs
        stock_symbols: List of stock symbols
        sample_dates: List of sample dates
        period: 'test' or 'val'
        save_plot: Whether to save the plot
    """
    # Load JSON data
    from config import JSON_FILES, config, TRADE_LEN, STOCK_DATA_PATH, CLOSE_PRICE_INDEX
    import pandas as pd
    from config import START_DATE, END_DATE

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
    else:  # test
        date_start_idx = config['val_end']

    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)

    # Prepare data matrix
    n_stocks = len(stock_symbols)
    n_steps = len(portfolio_records)

    # Matrix for future return rates
    return_matrix = np.zeros((n_stocks, n_steps))

    for i, record in enumerate(portfolio_records):
        # Calculate the decision date for this step
        decision_date_idx = date_start_idx + i * TRADE_LEN

        # Calculate 21-day future return for each stock
        for stock_idx in range(n_stocks):
            if (stock_idx < stocks_data.shape[0] and
                decision_date_idx + 1 >= 0 and
                decision_date_idx + TRADE_LEN < stocks_data.shape[1]):

                current_price = stocks_data[stock_idx, decision_date_idx + 1, CLOSE_PRICE_INDEX]  # t+1
                future_price = stocks_data[stock_idx, decision_date_idx + TRADE_LEN, CLOSE_PRICE_INDEX]  # t+21

                if current_price > 0:
                    future_return_rate = ((future_price - current_price) / current_price) * 100
                    return_matrix[stock_idx, i] = future_return_rate
                else:
                    return_matrix[stock_idx, i] = 0.0
            else:
                return_matrix[stock_idx, i] = 0.0

    # Create figure - use same dimensions as portfolio_heatmap
    cell_width = 1.2  # Same as portfolio_heatmap with text
    height = max(10, min(16, n_stocks * 0.5))
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), height))

    # Plot heatmap with diverging colormap (same as portfolio positions)
    # Use RdYlGn: Red=negative returns, Green=positive returns
    vmax = max(abs(return_matrix.min()), abs(return_matrix.max()))
    if vmax == 0:
        vmax = 0.01  # Avoid division by zero

    im = ax.imshow(return_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add text annotations showing return rate values
    for i in range(n_stocks):
        for j in range(n_steps):
            return_val = return_matrix[i, j]
            if abs(return_val) > 0.001:  # Only show non-zero values
                # Choose text color based on background
                text_color = 'white' if abs(return_val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{return_val:.1f}%', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Formatting - same style as portfolio_heatmap
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title(f'Future 21-day Return Rates - {period.upper()}', fontsize=14)

    # Set y-axis labels with both index and symbol (same as portfolio_heatmap)
    ax.set_yticks(range(n_stocks))
    y_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]
    ax.set_yticklabels(y_labels, fontsize=8)

    # Set x-axis labels - show more dates with wider cells
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, n_steps // 20)  # Show more labels
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, n_steps // 20)
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, pad=0.01)
    cbar.set_label('Future 21-day Return Rate (%) (Green=Positive, Red=Negative)', fontsize=10)

    # Add grid (same as portfolio_heatmap)
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_stocks) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory for this experiment
        output_dir = f'plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        filename = f'{output_dir}/future_return_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def plot_profit_heatmap(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot step-wise profit heatmap (single row showing profit for each trading step).
    Each step shows the return from previous step: (wealth[i] - wealth[i-1]) / wealth[i-1]
    """
    # Load JSON data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    agent_wealth = results.get('agent_wealth', [])
    if not agent_wealth:
        print(f"No agent wealth found for {experiment_id}")
        return
    
    # Convert to numpy array and flatten if nested
    wealth_array = np.array(agent_wealth).flatten()
    n_steps = len(wealth_array)
    
    # Calculate step-wise returns (skip first step as it's always 1.0)
    returns = []
    for i in range(1, n_steps):
        step_return = ((wealth_array[i] - wealth_array[i-1]) / wealth_array[i-1]) * 100  # Convert to percentage
        returns.append(step_return)
    
    # Convert to matrix for heatmap (1 row, n_steps-1 columns)
    profit_matrix = np.array(returns).reshape(1, -1)

    # Create figure - use same width as other heatmaps
    cell_width = 1.2  # Same as portfolio_heatmap
    fig, ax = plt.subplots(figsize=(min(30, (n_steps-1) * cell_width), 4))
    
    # Plot heatmap with diverging colormap (red=loss, green=profit)
    vmax = max(abs(profit_matrix.min()), abs(profit_matrix.max()))
    if vmax == 0:
        vmax = 0.01  # Set minimum range to avoid division by zero
    
    im = ax.imshow(profit_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add text annotations showing profit values
    for j in range(profit_matrix.shape[1]):
        profit_val = profit_matrix[0, j]
        # Choose text color based on background
        text_color = 'white' if abs(profit_val) > vmax * 0.5 else 'black'
        ax.text(j, 0, f'{profit_val:.1f}%', ha='center', va='center',
               color=text_color, fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Step Returns', fontsize=12)
    ax.set_title(f'Step-wise Profit/Loss - {period}', fontsize=14)
    
    # Remove y-axis ticks (only one row)
    ax.set_yticks([])
    
    # Set x-axis labels - show more dates with wider cells
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, (n_steps-1) // 20)  # Show more labels
        xticks = range(0, n_steps-1, step_interval)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, (n_steps-1) // 20)
        xticks = range(0, n_steps-1, step_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])
    
    # Add horizontal colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.50, shrink=0.8)
    cbar.set_label('Return (Green=Profit, Red=Loss)', fontsize=10)
    
    # Add grid
    ax.set_xticks(np.arange(n_steps-1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        # Create output directory for this experiment
        output_dir = f'plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{output_dir}/profit_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def plot_rho_heatmap(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot rho (long/short allocation parameter) as single row heatmap.
    Rho ranges from 0.0 (100% short) to 1.0 (100% long).
    """
    # Load JSON data
    from config import JSON_FILES
    json_filename = JSON_FILES[f'{period}_results']
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', json_filename)
    if not os.path.exists(json_path):
        print(f"Warning: {json_path} not found")
        return
    
    with open(json_path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    rho_record = results.get('rho_record', [])
    if not rho_record:
        print(f"No rho records found for {experiment_id}")
        return
    
    # Convert to numpy array and reshape to 1 row
    rho_array = np.array(rho_record)
    n_steps = len(rho_array)
    rho_matrix = rho_array.reshape(1, n_steps)
    
    # Create figure - use same width as other heatmaps
    cell_width = 1.2  # Same as portfolio_heatmap
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), 4))
    
    # Plot heatmap with consistent colormap (red=short, white=neutral, green=long)
    im = ax.imshow(rho_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=0.0, vmax=1.0, origin='lower')

    # Add text annotations showing rho values
    for j in range(rho_matrix.shape[1]):
        rho_val = rho_matrix[0, j]
        # Choose text color based on background (middle values need better contrast)
        text_color = 'black' if 0.3 <= rho_val <= 0.7 else 'white'
        ax.text(j, 0, f'{rho_val:.2f}', ha='center', va='center',
               color=text_color, fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Rho (ρ)', fontsize=12)
    ax.set_title(f'MSU Rho Values - Long/Short Allocation - {period.upper()}', fontsize=14)
    
    # Set y-axis (only one row)
    ax.set_yticks([0])
    ax.set_yticklabels(['ρ'], fontsize=12)
    
    # Set x-axis labels - show more dates with wider cells
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, n_steps // 20)  # Show more labels
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, n_steps // 20)
        xticks = range(0, n_steps, step_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])
    
    
    # Add horizontal colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.50, shrink=0.8)
    cbar.set_label('Rho Value (0.0=100% Short/Red, 0.5=Balanced/Yellow, 1.0=100% Long/Green)', fontsize=10)
    
    # Add grid for better readability
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)
    
    plt.tight_layout()
    
    if save_plot:
        # Create output directory for this experiment
        output_dir = f'plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f'{output_dir}/rho_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()


def plot_market_profit_heatmap(df_market, period='val', save_plot=True):
    """
    Plot market (DowJones) step-wise profit heatmap showing monthly returns.
    Similar to plot_profit_heatmap but for market benchmark.

    Args:
        df_market: DataFrame with market benchmark data (DowJones column)
        period: 'val' or 'test'
        save_plot: Whether to save the plot
    """
    from config import config

    # Get market column name from config
    market_col = config['benchmark_column']

    if market_col not in df_market.columns:
        print(f"Warning: Market column '{market_col}' not found in DataFrame")
        return

    # Get market wealth values
    market_wealth = df_market[market_col].values
    n_steps = len(market_wealth)

    # Calculate step-wise returns (skip first step as it's always 1.0)
    returns = []
    for i in range(1, n_steps):
        step_return = ((market_wealth[i] - market_wealth[i-1]) / market_wealth[i-1]) * 100  # Convert to percentage
        returns.append(step_return)

    # Convert to matrix for heatmap (1 row, n_steps-1 columns)
    profit_matrix = np.array(returns).reshape(1, -1)

    # Create figure - use same width as other heatmaps
    cell_width = 1.2  # Same as other heatmaps
    fig, ax = plt.subplots(figsize=(min(30, (n_steps-1) * cell_width), 4))

    # Plot heatmap with diverging colormap (red=loss, green=profit)
    vmax = max(abs(profit_matrix.min()), abs(profit_matrix.max()))
    if vmax == 0:
        vmax = 0.01  # Set minimum range to avoid division by zero

    im = ax.imshow(profit_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add text annotations showing market profit values
    for j in range(profit_matrix.shape[1]):
        profit_val = profit_matrix[0, j]
        # Choose text color based on background
        text_color = 'white' if abs(profit_val) > vmax * 0.5 else 'black'
        ax.text(j, 0, f'{profit_val:.1f}%', ha='center', va='center',
               color=text_color, fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel(f'{market_col}', fontsize=12)
    ax.set_title(f'Market ({market_col}) Step-wise Profit/Loss - {period.upper()}', fontsize=14)

    # Remove y-axis ticks (only one row)
    ax.set_yticks([])

    # Set x-axis labels - show more dates with wider cells
    if hasattr(df_market, 'index') and len(df_market.index) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, (n_steps-1) // 20)  # Show more labels
        xticks = range(0, n_steps-1, step_interval)
        ax.set_xticks(xticks)
        xlabels = [df_market.index[i].strftime('%Y-%m-%d') for i in xticks if i < len(df_market.index)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, (n_steps-1) // 20)
        xticks = range(0, n_steps-1, step_interval)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])

    # Add horizontal colorbar
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=0.50, shrink=0.8)
    cbar.set_label('Return (Green=Profit, Red=Loss)', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_steps-1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory
        output_dir = f'plot_outputs'
        os.makedirs(output_dir, exist_ok=True)

        filename = f'{output_dir}/market_profit_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()