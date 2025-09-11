# -------------------------------
# Analysis Functions - Periodic Returns & Win Rates
# -------------------------------

import numpy as np
import pandas as pd
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import calculate_metrics
from config import config, TRADE_MODE

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
    """
    metrics_dict = {}
    for col in series_list:
        wealth = df[col].values
        m = calculate_metrics(wealth.reshape(1, -1), TRADE_MODE)
        metrics_dict[col] = {
            'ARR': m['ARR'][0, 0] if isinstance(m['ARR'], np.ndarray) else m['ARR'],
            'AVOL': m['AVOL'][0, 0] if isinstance(m['AVOL'], np.ndarray) else m['AVOL'],
            'ASR': m['ASR'][0, 0] if isinstance(m['ASR'], np.ndarray) else m['ASR'],
            'MDD': m['MDD'],
            'CR': m['CR'][0, 0] if isinstance(m['CR'], np.ndarray) else m['CR'],
            'DDR': m['DDR'][0, 0] if isinstance(m['DDR'], np.ndarray) else m['DDR']
        }
    return pd.DataFrame(metrics_dict)