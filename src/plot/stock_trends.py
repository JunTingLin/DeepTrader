# -------------------------------
# Stock Price Trend Plotting Functions
# -------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
from config import (
    config, START_DATE, END_DATE, TRADE_LEN,
    STOCK_DATA_PATH, CLOSE_PRICE_INDEX
)

def plot_stock_price_trends(experiment_id, outputs_base_path, stock_symbols, period='test', save_plots=True):
    """
    Plot stock price trends with long/short position markers for each stock.
    Green triangles up = Long positions, Red triangles down = Short positions.
    Save plots as PNG files.
    """
    # Load portfolio data
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', f'{period}_results.json')
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
    output_dir = f'plot_outputs/{experiment_id}/stock_price_plots'
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
        stock_prices = stocks_data[stock_idx, date_start_idx:date_end_idx, CLOSE_PRICE_INDEX]
        
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
        plt.plot(period_dates, stock_prices, 'b-', linewidth=1, alpha=0.8, label='Close Price')
        
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
        plt.ylabel('Close Price ($)', fontsize=12)
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
        else:
            plt.show()