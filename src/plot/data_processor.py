# -------------------------------
# Data Processing Functions
# -------------------------------

import numpy as np
import pandas as pd
from data_loader import (
    load_agent_wealth, get_business_day_segments, 
    get_market_data, compute_cumulative_wealth
)
from config import config, TRADE_LEN

def process_data():
    """
    Process all data into validation and testing DataFrames.
    1. Generate full business days and split into training, validation, and testing segments.
    2. Load market data for the full period, then extract the validation and test segments,
       computing their cumulative wealth independently (starting at 1).
    3. Load the agent wealth data arrays, which cover the 
       respective validation and testing periods.
    4. Create DataFrames (df_val and df_test) with the sample dates as index and columns for each 
       agent series and the benchmark.
    """
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Load market data for the entire period
    df_market_full = get_market_data(full_days)
    
    # Extract validation segment
    df_market_val = df_market_full.loc[val_days]
    market_wealth_val = compute_cumulative_wealth(df_market_val)
    
    # Extract testing segment
    df_market_test = df_market_full.loc[test_days]
    market_wealth_test = compute_cumulative_wealth(df_market_test)
    
    # Sample dates for validation and testing segments
    val_sample_dates = val_days[::TRADE_LEN]
    test_sample_dates = test_days[::TRADE_LEN]
    
    # Sample the market cumulative wealth for validation and testing, and rebase to 1
    market_series_val = market_wealth_val.iloc[::TRADE_LEN].copy()
    market_series_val = market_series_val / market_series_val.iloc[0]
    
    market_series_test = market_wealth_test.iloc[::TRADE_LEN].copy()
    market_series_test = market_series_test / market_series_test.iloc[0]
    
    # Load combined agent wealth data
    agent_wealth = load_agent_wealth()
    
    # Process validation data
    n_val = len(val_sample_dates)
    val_data = {}
    for key in agent_wealth:
        if key.startswith('val_'):
            # Assume the agent array covers the entire validation period and has n_val points
            agent_val = agent_wealth[key][:n_val]
            agent_val = agent_val / agent_val[0]
            val_data[key] = agent_val
    
    # Process test data
    n_test = len(test_sample_dates)
    test_data = {}
    for key in agent_wealth:
        if key.startswith('test_'):
            # Assume the agent array covers the entire testing period and has n_test points
            agent_test = agent_wealth[key][:n_test]
            agent_test = agent_test / agent_test[0]
            test_data[key] = agent_test
    
    # Create DataFrames with sample dates as index and include market benchmark
    df_val = pd.DataFrame(val_data, index=val_sample_dates)
    df_val[config['benchmark_column']] = market_series_val.values
    
    df_test = pd.DataFrame(test_data, index=test_sample_dates)
    df_test[config['benchmark_column']] = market_series_test.values
    
    return df_val, df_test, full_days, train_days, val_days, test_days