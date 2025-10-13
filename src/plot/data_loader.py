# -------------------------------
# Data Loading Functions
# -------------------------------

import numpy as np
import pandas as pd
import json
import os
from config import OUTPUTS_BASE_PATH, START_DATE, END_DATE, WEALTH_MODE, config

def load_agent_wealth():
    """
    Load agent wealth arrays from JSON files (both validation and test).
    Based on EXPERIMENT_IDS list containing full date/time paths.
    """
    from config import EXPERIMENT_IDS, JSON_FILES

    agent_wealth = {}

    for i, exp_path in enumerate(EXPERIMENT_IDS, 1):
        # Construct file paths for JSON files using dynamic configuration
        val_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'json_file', JSON_FILES['val_results'])
        test_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'json_file', JSON_FILES['test_results'])
        
        try:
            # Load validation data from JSON
            if os.path.exists(val_json_path):
                with open(val_json_path, 'r', encoding='utf-8') as f:
                    val_results = json.load(f)
                val_data = np.array(val_results['agent_wealth']).flatten()
                agent_wealth[f'val_{i}'] = val_data
                print(f"Successfully loaded validation data for experiment {exp_path}")
            
            # Load test data from JSON
            if os.path.exists(test_json_path):
                with open(test_json_path, 'r', encoding='utf-8') as f:
                    test_results = json.load(f)
                test_data = np.array(test_results['agent_wealth']).flatten()
                agent_wealth[f'test_{i}'] = test_data
                print(f"Successfully loaded test data for experiment {exp_path}")
            
        except Exception as e:
            print(f"Warning: Could not load experiment {exp_path}: {e}")
            continue
    
    return agent_wealth

def get_business_day_segments():
    """
    Generate full business day date range from START_DATE to END_DATE,
    and split into training, validation, and testing segments based on market config.
    """
    full_days = pd.bdate_range(start=START_DATE, end=END_DATE)
    total_days = len(full_days)
    print(f"Total business days: {total_days}")
    
    train_days = full_days[0:config['train_end']]
    val_days = full_days[config['train_end']:config['val_end']]
    test_days = full_days[config['val_end']:config['test_end']]
    
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")
    print(f"Test days: {len(test_days)}")
    
    return full_days, train_days, val_days, test_days

def get_market_data(full_days, file_path=None):
    """
    Load market data from a local CSV file, filter for full_days, 
    reindex to the full business day range, and fill missing values.
    """
    if file_path is None:
        file_path = config['market_file']
    
    df = pd.read_csv(file_path)
    df['Date'] = df['Date'].str.split(' ').str[0]
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    full_days = pd.DatetimeIndex(full_days).tz_localize(None)
    
    df = df.loc[full_days[0]:full_days[-1]]
    df = df.reindex(full_days)
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def compute_cumulative_wealth(df_market, wealth_mode=WEALTH_MODE):
    """
    Compute daily cumulative wealth using a Buy & Hold strategy from market Close prices.
    Rebase the series so that it starts at 1.
    """
    if wealth_mode == 'inter':
        market_open = df_market["Open"].copy()
        daily_return = market_open.pct_change().fillna(0.0)
    elif wealth_mode == 'intra':
        daily_return = ((df_market["Close"] - df_market["Open"]) / df_market["Open"]).fillna(0.0)
    else:
        raise ValueError("Invalid wealth_mode. Use 'inter' or 'intra'.")
    wealth_daily = (1.0 + daily_return).cumprod()
    wealth_rebased = wealth_daily / wealth_daily.iloc[0]
    return wealth_rebased

def resample_series(series, step):
    """Resample a series at regular intervals."""
    return series.iloc[::step].values

def resample_dates(dates, step):
    """Resample dates at regular intervals."""
    return dates[::step]