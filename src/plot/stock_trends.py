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
    STOCK_DATA_PATH, CLOSE_PRICE_INDEX, MARKET_DATA_PATH, MARKET_CLOSE_INDEX
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

def plot_step_analysis(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test', save_plots=True):
    """
    Plot analysis for all trading steps showing the 8 selected stocks (4 long + 4 short) for each step.
    Each subplot shows past 70 days + future 21 days of close prices with decision point marked.
    
    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        stock_symbols: List of all stock symbols
        sample_dates: DatetimeIndex of trading decision dates (df_val.index or df_test.index)
        period: 'val' or 'test'
        save_plots: Whether to save the plots
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
    else:  # test
        date_start_idx = config['val_end']
    
    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    
    # Create output directory
    output_dir = f'plot_outputs/{experiment_id}/step_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all trading steps
    for step_idx in range(len(sample_dates)):
        if step_idx >= len(portfolio_records):
            print(f"Warning: Step {step_idx} not found. Available portfolio records: 0-{len(portfolio_records)-1}")
            continue
        
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
        
        if len(long_positions) != 4 or len(short_positions) != 4:
            print(f"Warning: Step {step_idx} has {len(long_positions)} long and {len(short_positions)} short positions")
        
        # Create the plot with 2x4 subplots (top row: long, bottom row: short)
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        fig.suptitle(f'Step {step_idx} Analysis - {period.upper()} - Decision Date: {decision_date.strftime("%Y-%m-%d")}', 
                     fontsize=16, fontweight='bold')
        
        # Plot long positions (top row)
        for i, pos in enumerate(long_positions[:4]):  # Ensure max 4 positions
            stock_idx = pos['stock_index']
            weight = pos['weight']
            
            if stock_idx >= len(stock_symbols):
                continue
                
            stock_symbol = stock_symbols[stock_idx]
            
            # Extract price data for this stock in the analysis window
            stock_prices = stocks_data[stock_idx, window_start_idx:window_end_idx + 1, CLOSE_PRICE_INDEX]
            
            # Plot in the top row (long positions)
            ax = axes[0, i] if i < 4 else None
            if ax is not None:
                ax.plot(analysis_dates, stock_prices, 'b-', linewidth=1.5, alpha=0.8)
                
                # Mark the decision point with vertical line
                ax.axvline(x=decision_date, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add background colors for past vs future
                ax.axvspan(analysis_dates[0], decision_date, facecolor='lightblue', alpha=0.1, label='Past 70 days')
                ax.axvspan(decision_date, analysis_dates[-1], facecolor='lightgreen', alpha=0.1, label='Future 21 days')
                
                # Formatting
                ax.set_title(f'{stock_symbol} (LONG)\nWeight: {weight:.3f}', fontsize=12, fontweight='bold', color='green')
                ax.set_ylabel('Close Price ($)', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Plot short positions (bottom row)
        for i, pos in enumerate(short_positions[:4]):  # Ensure max 4 positions
            stock_idx = pos['stock_index']
            weight = pos['weight']
            
            if stock_idx >= len(stock_symbols):
                continue
                
            stock_symbol = stock_symbols[stock_idx]
            
            # Extract price data for this stock in the analysis window
            stock_prices = stocks_data[stock_idx, window_start_idx:window_end_idx + 1, CLOSE_PRICE_INDEX]
            
            # Plot in the bottom row (short positions)
            ax = axes[1, i] if i < 4 else None
            if ax is not None:
                ax.plot(analysis_dates, stock_prices, 'b-', linewidth=1.5, alpha=0.8)
                
                # Mark the decision point with vertical line
                ax.axvline(x=decision_date, color='red', linestyle='--', linewidth=2, alpha=0.8)
                
                # Add background colors for past vs future
                ax.axvspan(analysis_dates[0], decision_date, facecolor='lightblue', alpha=0.1, label='Past 70 days')
                ax.axvspan(decision_date, analysis_dates[-1], facecolor='lightcoral', alpha=0.1, label='Future 21 days')
                
                # Formatting
                ax.set_title(f'{stock_symbol} (SHORT)\nWeight: {weight:.3f}', fontsize=12, fontweight='bold', color='red')
                ax.set_ylabel('Close Price ($)', fontsize=10)
                ax.set_xlabel('Date', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.tick_params(axis='x', rotation=45)
        
        # Hide unused subplots
        for i in range(len(long_positions), 4):
            if i < 4:
                axes[0, i].set_visible(False)
        
        for i in range(len(short_positions), 4):
            if i < 4:
                axes[1, i].set_visible(False)
        
        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='upper right', fontsize=10)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)  # Make room for suptitle
        
        if save_plots:
            filename = f'{output_dir}/step_{step_idx:02d}_{period}_analysis.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()


def plot_msu_step_analysis(experiment_id, outputs_base_path, sample_dates, period='test', save_plots=True):
    """
    Plot MSU step analysis showing past 65 days + future 21 days of market close prices
    with the AI's rho decision for each trading step.
    
    Args:
        experiment_id: Experiment identifier
        outputs_base_path: Base path to outputs
        sample_dates: DatetimeIndex of trading decision dates 
        period: 'val' or 'test'
        save_plots: Whether to save the plots
    """
    # Load portfolio data
    json_path = os.path.join(outputs_base_path, experiment_id, 'json_file', f'{period}_results.json')
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
    
    # Use market close price index from config
    market_close_index = MARKET_CLOSE_INDEX
    
    # Get date range for the period
    if period == 'val':
        date_start_idx = config['train_end']
    else:  # test
        date_start_idx = config['val_end'] 
    
    # Generate business day range for the entire dataset
    full_dates = pd.bdate_range(start=START_DATE, end=END_DATE)
    
    # Create output directory
    output_dir = f'plot_outputs/{experiment_id}/msu_step_analysis'
    os.makedirs(output_dir, exist_ok=True)
    
    # Iterate through all trading steps
    for step_idx in range(len(sample_dates)):
        if step_idx >= len(portfolio_records):
            print(f"Warning: Step {step_idx} not found. Available portfolio records: 0-{len(portfolio_records)-1}")
            continue
        
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
        
        # Extract market close prices for the analysis window
        market_prices = market_data[window_start_idx:window_end_idx + 1, market_close_index]
        
        # Get the rho value for this step
        rho_value = rho_record[step_idx]
        
        # Create the plot
        plt.figure(figsize=(15, 8))
        
        # Plot market price trend
        plt.plot(analysis_dates, market_prices, 'b-', linewidth=1.5, alpha=0.8, label='Market Close')
        
        # Mark the decision point with vertical line
        plt.axvline(x=decision_date, color='red', linestyle='--', linewidth=2, alpha=0.8, label='Decision Point')
        
        # Add background colors for past vs future
        plt.axvspan(analysis_dates[0], decision_date, facecolor='lightblue', alpha=0.1, label='Past 65 days (MSU input)')
        plt.axvspan(decision_date, analysis_dates[-1], facecolor='lightgreen', alpha=0.1, label='Future 21 days (Trading period)')
        
        # Formatting
        plt.title(f'MSU Step {step_idx} Analysis - {period.upper()} - ρ = {rho_value:.4f}\\n'
                  f'Decision Date: {decision_date.strftime("%Y-%m-%d")}', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Market Close Price', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_plots:
            filename = f'{output_dir}/step_{step_idx:02d}_{period}_msu_analysis.png'
            plt.savefig(filename, dpi=150, bbox_inches='tight')
            print(f"Saved: {filename}")
            plt.close()
        else:
            plt.show()