# -------------------------------
# Heatmap Plotting Functions
# -------------------------------

import numpy as np
import matplotlib.pyplot as plt
import json
import os

def plot_portfolio_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plot=True):
    """
    Plot portfolio positions as single heatmap (positive=long, negative=short).
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
    
    # Create figure
    fig, ax = plt.subplots(figsize=(min(20, n_steps * 0.30), 10))
    
    # Plot heatmap with diverging colormap
    im = ax.imshow(position_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', 
                   vmin=-0.3, vmax=0.3, origin='lower')
    
    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title(f'Portfolio Positions - {period}', fontsize=14)
    
    # Set y-axis labels with both index and symbol
    ax.set_yticks(range(n_stocks))
    y_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Set x-axis labels
    step_interval = max(1, n_steps // 10)
    xticks = range(0, n_steps, step_interval)
    ax.set_xticks(xticks)
    if sample_dates is not None and len(sample_dates) >= n_steps:
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    
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
        step_return = (wealth_array[i] - wealth_array[i-1]) / wealth_array[i-1]
        returns.append(step_return)
    
    # Convert to matrix for heatmap (1 row, n_steps-1 columns)
    profit_matrix = np.array(returns).reshape(1, -1)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(min(20, (n_steps-1) * 0.3), 3))
    
    # Plot heatmap with diverging colormap (red=loss, green=profit)
    vmax = max(abs(profit_matrix.min()), abs(profit_matrix.max()))
    if vmax == 0:
        vmax = 0.01  # Set minimum range to avoid division by zero
    
    im = ax.imshow(profit_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', 
                   vmin=-vmax, vmax=vmax, origin='lower')
    
    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Step Returns', fontsize=12)
    ax.set_title(f'Step-wise Profit/Loss - {period}', fontsize=14)
    
    # Remove y-axis ticks (only one row)
    ax.set_yticks([])
    
    # Set x-axis labels (align with portfolio heatmap dates)
    step_interval = max(1, (n_steps-1) // 10)
    xticks = range(0, n_steps-1, step_interval)
    ax.set_xticks(xticks)
    if sample_dates is not None and len(sample_dates) >= n_steps:
        # Remove +1 to align dates with portfolio heatmap
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    
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
    
    # Create figure  
    fig, ax = plt.subplots(figsize=(min(20, n_steps * 0.30), 3))
    
    # Plot heatmap with consistent colormap (red=short, white=neutral, green=long)
    im = ax.imshow(rho_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest',
                   vmin=0.0, vmax=1.0, origin='lower')
    
    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Rho (ρ)', fontsize=12)
    ax.set_title(f'MSU Rho Values - Long/Short Allocation - {period.upper()}', fontsize=14)
    
    # Set y-axis (only one row)
    ax.set_yticks([0])
    ax.set_yticklabels(['ρ'], fontsize=12)
    
    # Set x-axis labels (align with other heatmaps)
    step_interval = max(1, n_steps // 10)
    xticks = range(0, n_steps, step_interval)
    ax.set_xticks(xticks)
    if sample_dates is not None and len(sample_dates) >= n_steps:
        xlabels = [sample_dates[i].strftime('%Y-%m-%d') for i in xticks if i < len(sample_dates)]
        ax.set_xticklabels(xlabels, rotation=45, ha='right')
    
    
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