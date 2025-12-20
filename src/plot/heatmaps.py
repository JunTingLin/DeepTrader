# -------------------------------
# Heatmap Plotting Functions
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
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
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/portfolio_positions_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/portfolio_positions_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        row_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]

        df_positions = pd.DataFrame(
            position_matrix,
            index=row_labels,
            columns=columns
        )
        df_positions.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")

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

    # Matrix for future return rates
    return_matrix = np.zeros((n_stocks, n_steps))

    # Fill matrix from JSON ror data
    for step_idx, record in enumerate(portfolio_records):
        if 'sim_info' not in record or not record['sim_info']:
            continue

        sim_info = record['sim_info']
        ror_array = sim_info.get('ror', None)

        if ror_array is None:
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

        # Convert ror to return rate percentage
        # ror = 1.08 means +8%, ror = 0.98 means -2%
        for stock_idx in range(min(n_stocks, len(flat_ror))):
            return_rate = (flat_ror[stock_idx] - 1.0) * 100  # Convert to percentage
            return_matrix[stock_idx, step_idx] = return_rate

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
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/future_return_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/future_return_heatmap_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        row_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]

        df_future_return = pd.DataFrame(
            return_matrix,
            index=row_labels,
            columns=columns
        )
        df_future_return.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")

def plot_profit_heatmap(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot step-wise profit heatmap with 3 rows:
    Row 1: Overall returns (%)
    Row 2: Long position returns (%)
    Row 3: Short position returns (%)
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

    # Get both portfolio records and agent wealth
    portfolio_records = results.get('portfolio_records', [])
    agent_wealth = results.get('agent_wealth', [])

    if not portfolio_records or not agent_wealth:
        print(f"No portfolio records or agent wealth found for {experiment_id}")
        return

    # Convert wealth to numpy array and flatten if nested
    wealth_array = np.array(agent_wealth).flatten()
    n_steps = len(portfolio_records)

    # Initialize return arrays
    long_returns = []
    short_returns = []
    overall_returns = []

    # Calculate step-wise returns from sim_info if available
    for i, record in enumerate(portfolio_records):
        if 'sim_info' in record and record['sim_info']:
            sim_info = record['sim_info']

            # Long position return (convert to percentage)
            long_return = sim_info.get('LongPosition_return', 0) * 100
            long_returns.append(long_return)

            # Short position return (convert to percentage, if available)
            short_return = sim_info.get('ShortPosition_return', 0) * 100
            short_returns.append(short_return)

            # Overall return (use JSON rate_of_return for accuracy)
            overall_return = sim_info.get('rate_of_return', 0) * 100
            overall_returns.append(overall_return)
        else:
            # Fallback: calculate from wealth only
            long_returns.append(0)
            short_returns.append(0)
            if i + 1 < len(wealth_array):
                overall_return = ((wealth_array[i + 1] - wealth_array[i]) / wealth_array[i]) * 100
            else:
                overall_return = 0
            overall_returns.append(overall_return)

    # Convert to matrix for heatmap (3 rows, n_steps columns)
    # Order with origin='lower': index 0 is bottom, index 2 is top
    # So: Short (bottom/index 0) -> Long (middle/index 1) -> Overall (top/index 2)
    profit_matrix = np.array([
        short_returns,
        long_returns,
        overall_returns
    ])

    # Create figure - adjust height for 3 rows
    cell_width = 1.2
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), 6))

    # Plot heatmap with diverging colormap (red=loss, green=profit)
    vmax = max(abs(profit_matrix.min()), abs(profit_matrix.max()))
    if vmax == 0:
        vmax = 0.01  # Set minimum range to avoid division by zero

    im = ax.imshow(profit_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add text annotations showing profit values for all rows
    for i in range(profit_matrix.shape[0]):  # 3 rows
        for j in range(profit_matrix.shape[1]):  # n_steps columns
            profit_val = profit_matrix[i, j]
            # Choose text color based on background
            text_color = 'white' if abs(profit_val) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{profit_val:.1f}%', ha='center', va='center',
                   color=text_color, fontsize=8, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Position Type', fontsize=12)
    ax.set_title(f'Long/Short/Overall Position Returns - {period.upper()}', fontsize=14)

    # Set y-axis labels for the 3 rows (bottom to top with origin='lower')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(['Short Return', 'Long Return', 'Overall Return'])
    
    # Set x-axis labels - show more dates with wider cells
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Show more dates since we have wider cells now
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        # Always include the last step
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        # Fallback to step numbers
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        # Always include the last step
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])

    # Add horizontal colorbar - adjust pad based on number of rows (3 rows)
    # More rows = smaller pad needed (colorbar closer to plot)
    colorbar_pad = 0.20  # For 3 rows
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=colorbar_pad, shrink=0.8)
    cbar.set_label('Return (Green=Profit, Red=Loss)', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_steps-1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory for this experiment
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/profit_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/profit_heatmap_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        df_profit = pd.DataFrame(
            profit_matrix,
            index=['Short Return (%)', 'Long Return (%)', 'Overall Return (%)'],
            columns=columns
        )
        df_profit.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")


def plot_precision_analysis_heatmap(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot precision analysis heatmap showing monthly precision for long and short positions:
    - Row 1 (P_L@4): Long Precision - shows "correct/G" (e.g., 3/4) for each month
    - Row 2 (P_S@4): Short Precision - shows "correct/G" (e.g., 2/4) for each month
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

    n_steps = len(portfolio_records)

    # Initialize count arrays for precision calculation
    long_correct_counts = []  # Count of long positions that went up
    long_total_counts = []    # Total count of long positions
    short_correct_counts = [] # Count of short positions that went down
    short_total_counts = []   # Total count of short positions

    # Analyze each trading step
    for i, record in enumerate(portfolio_records):
        if 'sim_info' not in record or not record['sim_info']:
            # Fallback to zeros if no sim_info
            long_correct_counts.append(0)
            long_total_counts.append(0)
            short_correct_counts.append(0)
            short_total_counts.append(0)
            continue

        sim_info = record['sim_info']
        ror_array = sim_info.get('ror', [])  # Individual stock returns

        if not ror_array:
            long_correct_counts.append(0)
            long_total_counts.append(0)
            short_correct_counts.append(0)
            short_total_counts.append(0)
            continue

        # Convert ror to return rates (ror format: 1.02 = +2%, 0.98 = -2%)
        # Flatten nested list structure
        flat_ror = [item for sublist in ror_array for item in sublist]
        returns = [(r - 1) for r in flat_ror]

        # Analyze long positions
        long_positions = record.get('long_positions', [])
        long_correct = 0
        long_total = len(long_positions)

        for pos in long_positions:
            stock_idx = pos.get('stock_index')
            if stock_idx is not None and stock_idx < len(returns):
                stock_return = returns[stock_idx]
                if stock_return > 0:  # Prediction correct: long position went up
                    long_correct += 1

        long_correct_counts.append(long_correct)
        long_total_counts.append(long_total)

        # Analyze short positions
        short_positions = record.get('short_positions', [])
        short_correct = 0
        short_total = len(short_positions)

        for pos in short_positions:
            stock_idx = pos.get('stock_index')
            if stock_idx is not None and stock_idx < len(returns):
                stock_return = returns[stock_idx]
                if stock_return < 0:  # Prediction correct: short position went down
                    short_correct += 1

        short_correct_counts.append(short_correct)
        short_total_counts.append(short_total)

    # Create precision matrix (2 rows: long precision, short precision)
    # We'll use precision rate (0.0 to 1.0) for coloring
    long_precision_values = []
    short_precision_values = []

    for i in range(n_steps):
        # Long precision
        if long_total_counts[i] > 0:
            long_precision_values.append(long_correct_counts[i] / long_total_counts[i])
        else:
            long_precision_values.append(0)

        # Short precision
        if short_total_counts[i] > 0:
            short_precision_values.append(short_correct_counts[i] / short_total_counts[i])
        else:
            short_precision_values.append(0)

    precision_matrix = np.array([
        long_precision_values,
        short_precision_values
    ])

    # Create figure
    cell_width = 1.2
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), 5))

    # Plot heatmap with green colormap (0.0 to 1.0 range)
    im = ax.imshow(precision_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=0.0, vmax=1.0, origin='lower')

    # Add text annotations showing "correct/total" format
    for i in range(2):  # 2 rows
        for j in range(n_steps):  # n_steps columns
            if i == 0:  # Long precision row
                correct = long_correct_counts[j]
                total = long_total_counts[j]
            else:  # Short precision row
                correct = short_correct_counts[j]
                total = short_total_counts[j]

            # Display as "correct/total" format (e.g., "3/4")
            text = f'{correct}/{total}' if total > 0 else '0/0'

            # Choose text color based on precision value
            precision_val = precision_matrix[i, j]
            text_color = 'white' if precision_val < 0.5 or precision_val > 0.8 else 'black'

            ax.text(j, i, text, ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps (Monthly)', fontsize=12)
    ax.set_ylabel('Position Type', fontsize=12)
    ax.set_title(f'Monthly Precision Analysis (P@4) - {period.upper()}', fontsize=14)

    # Set y-axis labels for the 2 rows
    ax.set_yticks([0, 1])
    ax.set_yticklabels(['P_L@4 (Long Precision)', 'P_S@4 (Short Precision)'])

    # Set x-axis labels
    if sample_dates is not None and len(sample_dates) >= n_steps:
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)
        ax.set_xticks(xticks)
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    else:
        step_interval = max(1, n_steps // 20)
        xticks = list(range(0, n_steps, step_interval))
        if xticks[-1] != n_steps - 1:
            xticks.append(n_steps - 1)
        ax.set_xticks(xticks)
        ax.set_xticklabels([f'Step {i+1}' for i in xticks])

    # Add colorbar
    colorbar_pad = 0.25
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=colorbar_pad, shrink=0.8)
    cbar.set_label('Precision (0.0=0%, 0.5=50%, 1.0=100%)', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    # Save plot
    if save_plot:
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/precision_analysis_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/precision_analysis_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        # Create CSV with both precision rates and counts
        df_precision = pd.DataFrame(
            precision_matrix,
            index=['P_L@4 (Long Precision)', 'P_S@4 (Short Precision)'],
            columns=columns
        )

        # Add count rows
        df_long_counts = pd.DataFrame(
            [[f"{long_correct_counts[i]}/{long_total_counts[i]}" for i in range(n_steps)]],
            index=['Long Correct/Total'],
            columns=columns
        )
        df_short_counts = pd.DataFrame(
            [[f"{short_correct_counts[i]}/{short_total_counts[i]}" for i in range(n_steps)]],
            index=['Short Correct/Total'],
            columns=columns
        )

        df_combined = pd.concat([df_precision, df_long_counts, df_short_counts])
        df_combined.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")


def plot_individual_stock_returns_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates,
                                          period='test', save_plot=True):
    """
    Plot individual stock returns heatmap showing return rate for each stock at each step.
    - Color represents return rate (green=positive, red=negative)
    - Green border for long positions, red border for short positions

    Args:
        experiment_id: Experiment ID
        outputs_base_path: Base path to outputs
        stock_symbols: List of stock symbols
        sample_dates: List of sample dates
        period: 'test' or 'val'
        save_plot: Whether to save the plot
    """
    import matplotlib.patches as patches

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

    # Matrix for return rates (percentage)
    returns_matrix = np.zeros((n_stocks, n_steps))
    # Track position type: 1=long, -1=short, 0=no position
    position_type_matrix = np.zeros((n_stocks, n_steps))

    # Fill the matrix with individual stock returns
    for step_idx, record in enumerate(portfolio_records):
        if 'sim_info' not in record or not record['sim_info']:
            continue

        sim_info = record['sim_info']

        # Process long returns
        long_returns = sim_info.get('long_returns', None)
        if long_returns is not None:
            # Handle nested list structure (flatten if needed)
            if isinstance(long_returns, list) and len(long_returns) > 0:
                if isinstance(long_returns[0], list):
                    flat_long_returns = [item for sublist in long_returns for item in sublist]
                else:
                    flat_long_returns = long_returns
            else:
                flat_long_returns = long_returns

            # Fill long positions
            for stock_idx in range(min(n_stocks, len(flat_long_returns))):
                if abs(flat_long_returns[stock_idx]) > 0.0001:  # Has long position
                    returns_matrix[stock_idx, step_idx] = flat_long_returns[stock_idx] * 100
                    position_type_matrix[stock_idx, step_idx] = 1  # Mark as long

        # Process short returns
        short_returns = sim_info.get('short_returns', None)
        if short_returns is not None:
            # Handle nested list structure (flatten if needed)
            if isinstance(short_returns, list) and len(short_returns) > 0:
                if isinstance(short_returns[0], list):
                    flat_short_returns = [item for sublist in short_returns for item in sublist]
                else:
                    flat_short_returns = short_returns
            else:
                flat_short_returns = short_returns

            # Fill short positions
            for stock_idx in range(min(n_stocks, len(flat_short_returns))):
                if abs(flat_short_returns[stock_idx]) > 0.0001:  # Has short position
                    returns_matrix[stock_idx, step_idx] = flat_short_returns[stock_idx] * 100
                    position_type_matrix[stock_idx, step_idx] = -1  # Mark as short

    # Create figure - same dimensions as portfolio_heatmap
    cell_width = 1.2
    height = max(10, min(16, n_stocks * 0.5))
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), height))

    # Plot heatmap with diverging colormap (red=negative, green=positive)
    vmax = max(abs(returns_matrix.min()), abs(returns_matrix.max()))
    if vmax == 0:
        vmax = 0.01

    im = ax.imshow(returns_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add borders and text annotations
    for i in range(n_stocks):
        for j in range(n_steps):
            position_type = position_type_matrix[i, j]

            # Add border for positions
            if position_type == 1:  # Long position - green border
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2.5,
                                        edgecolor='darkgreen', facecolor='none')
                ax.add_patch(rect)
            elif position_type == -1:  # Short position - red border
                rect = patches.Rectangle((j-0.5, i-0.5), 1, 1, linewidth=2.5,
                                        edgecolor='darkred', facecolor='none')
                ax.add_patch(rect)

            # Add text for return values
            return_val = returns_matrix[i, j]
            if abs(return_val) > 0.01:  # Only show non-zero values
                # Choose text color based on background
                text_color = 'white' if abs(return_val) > vmax * 0.5 else 'black'
                ax.text(j, i, f'{return_val:.1f}%', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Formatting - same style as portfolio_heatmap
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title(f'Individual Stock Returns (Green Border=Long, Red Border=Short) - {period.upper()}', fontsize=14)

    # Set y-axis labels with both index and symbol (same as portfolio_heatmap)
    ax.set_yticks(range(n_stocks))
    y_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]
    ax.set_yticklabels(y_labels, fontsize=8)

    # Set x-axis labels - show dates with same style as portfolio_heatmap
    if sample_dates is not None and len(sample_dates) >= n_steps:
        step_interval = max(1, n_steps // 20)
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
    cbar.set_label('Return Rate (%) (Green=Positive, Red=Negative)', fontsize=10)

    # Add grid (same as portfolio_heatmap)
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.set_yticks(np.arange(n_stocks) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory for this experiment
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/individual_returns_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/individual_returns_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        # Row labels: stock index and symbol
        row_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]

        df_returns = pd.DataFrame(
            returns_matrix,
            index=row_labels,
            columns=columns
        )
        df_returns.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")


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

    # Add horizontal colorbar - adjust pad based on number of rows (1 row)
    colorbar_pad = 0.35  # For 1 row
    cbar = fig.colorbar(im, ax=ax, orientation='horizontal', pad=colorbar_pad, shrink=0.8)
    cbar.set_label('Return (Green=Profit, Red=Loss)', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_steps-1) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory
        output_dir = f'src/plot/plot_outputs'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/market_profit_heatmap_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/market_profit_heatmap_{period}.csv'
        # Create DataFrame with proper labels
        if hasattr(df_market, 'index') and len(df_market.index) >= n_steps:
            columns = [df_market.index[i].strftime('%Y-%m-%d') for i in range(n_steps-1)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps-1)]

        from config import config
        market_col = config['benchmark_column']

        df_market_profit = pd.DataFrame(
            profit_matrix,
            index=[f'{market_col} Return (%)'],
            columns=columns
        )
        df_market_profit.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")


def plot_selection_quality_heatmap(experiment_id, outputs_base_path, sample_dates, period='test', save_plot=True):
    """
    Plot selection quality heatmap comparing agent's stock selection with ideal selection.

    Shows 4 rows for each trading step:
    - Row 1: Sum of ROR for top 4 best performing stocks (ideal long)
    - Row 2: Sum of ROR for agent's actual long positions (4 stocks)
    - Row 3: Sum of ROR for bottom 4 worst performing stocks (ideal short, absolute value)
    - Row 4: Sum of ROR for agent's actual short positions (4 stocks, absolute value)

    Args:
        experiment_id: Experiment ID
        outputs_base_path: Base path to outputs
        sample_dates: List of sample dates
        period: 'test' or 'val'
        save_plot: Whether to save the plot
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

    n_steps = len(portfolio_records)

    # Initialize 4-row matrix
    selection_matrix = np.zeros((4, n_steps))

    # Process each step
    for i, record in enumerate(portfolio_records):
        sim_info = record.get('sim_info', {})

        # Get ROR from sim_info (flatten in case it's 2D)
        # ROR is a ratio (e.g., 1.05 = +5% return), convert to return rate
        ror_ratio = np.array(sim_info['ror']).flatten()  # 30 stocks' future returns as ratios
        ror_percent = (ror_ratio - 1) * 100  # Convert to percentage returns

        # Get long_returns and short_returns (already in correct format)
        long_returns = np.array(sim_info.get('long_returns', [])).flatten()
        short_returns = np.array(sim_info.get('short_returns', [])).flatten()

        # Extract stock indices from long/short positions
        long_positions = record.get('long_positions', [])
        short_positions = record.get('short_positions', [])
        long_stocks = [pos['stock_index'] for pos in long_positions]
        short_stocks = [pos['stock_index'] for pos in short_positions]

        # Row 1: Top 4 best performing stocks (ideal long)
        top4_indices = np.argsort(ror_percent)[-4:]  # Highest 4 returns
        top4_sum = np.sum(ror_percent[top4_indices])
        selection_matrix[0, i] = top4_sum

        # Row 2: Agent's actual long positions (use long_returns)
        if long_stocks and len(long_returns) > 0:
            long_return_sum = np.sum([long_returns[idx] * 100 for idx in long_stocks if idx < len(long_returns)])
            selection_matrix[1, i] = long_return_sum

        # Row 3: Bottom 4 worst performing stocks (ideal short)
        # For short: profit = -(ror - 1) = (1 - ror), so if stock drops 10%, short gains 10%
        bottom4_indices = np.argsort(ror_percent)[:4]  # Lowest 4 returns (most negative)
        bottom4_short_profit = -np.sum(ror_percent[bottom4_indices])  # Negate to get short profit
        selection_matrix[2, i] = bottom4_short_profit

        # Row 4: Agent's actual short positions (use short_returns)
        if short_stocks and len(short_returns) > 0:
            short_return_sum = np.sum([short_returns[idx] * 100 for idx in short_stocks if idx < len(short_returns)])
            selection_matrix[3, i] = short_return_sum

    # Create figure
    cell_width = 1.2
    fig, ax = plt.subplots(figsize=(min(30, n_steps * cell_width), 6))

    # Plot heatmap with diverging colormap
    vmax = max(abs(selection_matrix.min()), abs(selection_matrix.max()))
    if vmax == 0:
        vmax = 0.01

    im = ax.imshow(selection_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=-vmax, vmax=vmax, origin='lower')

    # Add text annotations
    for i in range(4):
        for j in range(n_steps):
            value = selection_matrix[i, j]
            # Choose text color based on background
            text_color = 'white' if abs(value) > vmax * 0.5 else 'black'
            ax.text(j, i, f'{value:.1f}%', ha='center', va='center',
                   color=text_color, fontsize=9, fontweight='bold')

    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Selection Quality', fontsize=12)
    ax.set_title(f'Stock Selection Quality (Long/Short vs Ideal) - {period.upper()}', fontsize=14)

    # Set y-axis labels
    ax.set_yticks(range(4))
    y_labels = [
        'Ideal Long (Top 4)',
        'Agent Long (4 stocks)',
        'Ideal Short (Bottom 4)',
        'Agent Short (4 stocks)'
    ]
    ax.set_yticklabels(y_labels, fontsize=10)

    # Set x-axis labels - show dates
    if sample_dates is not None and len(sample_dates) >= n_steps:
        step_interval = max(1, n_steps // 20)
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
    cbar.set_label('Return Rate (%) (Green=Positive, Red=Negative)', fontsize=10)

    # Add grid
    ax.set_xticks(np.arange(n_steps) - 0.5, minor=True)
    ax.set_yticks(np.arange(4) - 0.5, minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5, alpha=0.3)

    plt.tight_layout()

    if save_plot:
        # Create output directory for this experiment
        output_dir = f'src/plot/plot_outputs/{experiment_id}'
        os.makedirs(output_dir, exist_ok=True)

        # Save PNG
        filename = f'{output_dir}/selection_quality_{period}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

        # Save CSV
        csv_filename = f'{output_dir}/selection_quality_{period}.csv'
        # Create DataFrame with proper labels
        if sample_dates is not None and len(sample_dates) >= n_steps:
            columns = [sample_dates[i].strftime('%Y-%m-%d') for i in range(n_steps)]
        else:
            columns = [f'Step {i+1}' for i in range(n_steps)]

        df_selection = pd.DataFrame(
            selection_matrix,
            index=[
                'Ideal Long (Top 4) %',
                'Agent Long (4 stocks) %',
                'Ideal Short (Bottom 4) %',
                'Agent Short (4 stocks) %'
            ],
            columns=columns
        )
        df_selection.to_csv(csv_filename)
        print(f"Saved: {csv_filename}")