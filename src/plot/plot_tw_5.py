import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import calculate_metrics # ../utils/functions.py

# -------------------------------
# Constants
# -------------------------------
TRADE_MODE = "M"    # "M": Monthly mode (12 trading periods per year)
TRADE_LEN = 21      # Sampling interval: 21 business days per sample
START_DATE = "2015-01-01"
END_DATE = "2025-03-31"

# -------------------------------
# Data Loading Functions
# -------------------------------
def load_agent_wealth():
    """
    Load and flatten agent wealth arrays for validation and test.
    """
    # Validation data
    val_w_MSU_dynamic_1 = np.load(r'..\outputs\0523\012851\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_2 = np.load(r'..\outputs\0523\120028\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_3 = np.load(r'..\outputs\0523\181814\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_4 = np.load(r'..\outputs\0319\144237\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_5 = np.load(r'..\outputs\0319\144329\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_6 = np.load(r'..\outputs\0319\204039\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_dynamic_7 = np.load(r'..\outputs\0319\204051\npy_file\agent_wealth_val.npy').flatten()

    # Test data
    test_w_MSU_dynamic_1 = np.load(r'..\outputs\0523\012851\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_2 = np.load(r'..\outputs\0523\120028\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_3 = np.load(r'..\outputs\0523\181814\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_4 = np.load(r'..\outputs\0319\144237\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_5 = np.load(r'..\outputs\0319\144329\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_6 = np.load(r'..\outputs\0319\204039\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_dynamic_7 = np.load(r'..\outputs\0319\204051\npy_file\agent_wealth_test.npy').flatten()

    return {
        'val_w_MSU_dynamic_1': val_w_MSU_dynamic_1,
        'val_w_MSU_dynamic_2': val_w_MSU_dynamic_2,
        'val_w_MSU_dynamic_3': val_w_MSU_dynamic_3,
        'val_w_MSU_dynamic_4': val_w_MSU_dynamic_4,
        'val_w_MSU_dynamic_5': val_w_MSU_dynamic_5,
        'val_w_MSU_dynamic_6': val_w_MSU_dynamic_6,
        'val_w_MSU_dynamic_7': val_w_MSU_dynamic_7,

        'test_w_MSU_dynamic_1': test_w_MSU_dynamic_1,
        'test_w_MSU_dynamic_2': test_w_MSU_dynamic_2,
        'test_w_MSU_dynamic_3': test_w_MSU_dynamic_3,
        'test_w_MSU_dynamic_4': test_w_MSU_dynamic_4,
        'test_w_MSU_dynamic_5': test_w_MSU_dynamic_5,
        'test_w_MSU_dynamic_6': test_w_MSU_dynamic_6,
        'test_w_MSU_dynamic_7': test_w_MSU_dynamic_7
    }


def get_business_day_segments():
    """
    Generate full business day date range from START_DATE to END_DATE,
    and split into:
      - Training: indices 0 to 1303 (1304 days)
      - Validation: indices 1304 to 2086 (783 days)
      - Testing: indices 2087 to 2672 (586 days)
    """
    full_days = pd.bdate_range(start=START_DATE, end=END_DATE)
    total_days = len(full_days)
    print(f"Total business days: {total_days}")
    
    train_days = full_days[0:1304]
    val_days   = full_days[1304:2087]
    test_days  = full_days[2087:2673]
    
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")
    print(f"Test days: {len(test_days)}")
    
    return full_days, train_days, val_days, test_days

def get_twii_data(full_days, file_path="^TWII.csv"):
    """
    Load TWII data from a local CSV file, filter for full_days, 
    reindex to the full business day range, and fill missing values.
    """
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

def compute_cumulative_wealth(df_twii):
    """
    Compute daily cumulative wealth using a Buy & Hold strategy from TWII Close prices.
    Rebase the series so that it starts at 1.
    """
    twii_close = df_twii["Close"].copy()
    daily_return = twii_close.pct_change().fillna(0.0)
    wealth_daily = (1.0 + daily_return).cumprod()
    wealth_rebased = wealth_daily / wealth_daily.iloc[0]
    return wealth_rebased

def resample_series(series, step):
    """
    Resample a series at regular intervals.
    """
    return series.iloc[::step].values

def resample_dates(dates, step):
    """
    Resample dates at regular intervals.
    """
    return dates[::step]

# -------------------------------
# Data Processing
# -------------------------------
def process_data():
    """
    Process all data into validation and testing DataFrames.
    1. Generate full business days and split into training, validation, and testing segments.
    2. Download TWII data for the full period, then extract the validation segment (indices 2109 to 4129)
       and test segment (indices 4130 to 6259), computing their cumulative wealth independently (starting at 1).
    3. Load the agent wealth data arrays (val_w_MSU_dynamic, test_w_MSU_dynamic, etc.), which cover the 
       respective validation and testing periods.
    4. Create DataFrames (df_val and df_test) with the sample dates as index and columns for each 
       agent series and '0050.TW'.
    """
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Download TWII data for the entire period
    df_twii_full = get_twii_data(full_days)
    
    # Extract validation segment: indices 2109 to 4129
    df_twii_val = df_twii_full.loc[val_days]
    twii_wealth_val = compute_cumulative_wealth(df_twii_val)
    
    # Extract testing segment: indices 4130 to 6259
    df_twii_test = df_twii_full.loc[test_days]
    twii_wealth_test = compute_cumulative_wealth(df_twii_test)
    
    # Sample dates for validation and testing segments
    val_sample_dates = val_days[::TRADE_LEN]
    test_sample_dates = test_days[::TRADE_LEN]
    
    # Sample the TWII cumulative wealth for validation and testing, and rebase to 1
    twii_series_val = twii_wealth_val.iloc[::TRADE_LEN].copy()
    twii_series_val = twii_series_val / twii_series_val.iloc[0]
    
    twii_series_test = twii_wealth_test.iloc[::TRADE_LEN].copy()
    twii_series_test = twii_series_test / twii_series_test.iloc[0]
    
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
    
    # Create a validation DataFrame with sample dates as index and include TWII as '0050.TW'
    df_val = pd.DataFrame(val_data, index=val_sample_dates)
    df_val['0050.TW'] = twii_series_val.values
    
    # Create a testing DataFrame with sample dates as index and include TWII as '0050.TW'
    df_test = pd.DataFrame(test_data, index=test_sample_dates)
    df_test['0050.TW'] = twii_series_test.values
    
    return df_val, df_test, full_days, train_days, val_days, test_days

# -------------------------------
# Plotting Functions (using original axvspan and color settings)
# -------------------------------
def plot_results(df_val, df_test, train_days, val_days, test_days):
    """
    Plot cumulative wealth with background shading for Training, Validation, and Testing periods.
    Uses original settings for axvspan and color.
    """
    plt.figure(figsize=(14, 7))
    
    # Background shading for segments: training, validation, and testing
    plt.axvspan(train_days[0], train_days[-1], facecolor='gray', alpha=0.1, label='Training Period')
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Testing Period')
    
    # Plot TWII wealth for validation and testing
    plt.plot(df_val.index, df_val['0050.TW'], color='r', linestyle='-', marker='o', label='TWII')
    plt.plot(df_test.index, df_test['0050.TW'], color='r', linestyle='-', marker='o', label=None)
    
    # Plot agent wealth for validation segment
    plt.plot(df_val.index, df_val['val_w_MSU_dynamic_1'], color='b', linestyle='-', marker='o', label='1')
    plt.plot(df_val.index, df_val['val_w_MSU_dynamic_2'], color='darkblue', linestyle='-', marker='o', label='2')
    plt.plot(df_val.index, df_val['val_w_MSU_dynamic_3'], color='c', linestyle='-.', marker='o', label='3')
    # plt.plot(df_val.index, df_val['val_w_MSU_dynamic_4'], color='steelblue', linestyle='-', marker='o', label='4')
    # plt.plot(df_val.index, df_val['val_w_MSU_dynamic_5'], color='limegreen', linestyle='-', marker='o', label='5')
    # plt.plot(df_val.index, df_val['val_w_MSU_dynamic_6'], color='g', linestyle='-', marker='o', label='6')
    # plt.plot(df_val.index, df_val['val_w_MSU_dynamic_7'], color='lawngreen', linestyle='-', marker='o', label='7')
    
    # Plot agent wealth for testing segment
    plt.plot(df_test.index, df_test['test_w_MSU_dynamic_1'], color='b', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_w_MSU_dynamic_2'], color='darkblue', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_w_MSU_dynamic_3'], color='c', linestyle='-.', marker='o', label=None)
    # plt.plot(df_test.index, df_test['test_w_MSU_dynamic_4'], color='steelblue', linestyle='-', marker='o', label=None)
    # plt.plot(df_test.index, df_test['test_w_MSU_dynamic_5'], color='limegreen', linestyle='-', marker='o', label=None)
    # plt.plot(df_test.index, df_test['test_w_MSU_dynamic_6'], color='g', linestyle='-', marker='o', label=None)
    # plt.plot(df_test.index, df_test['test_w_MSU_dynamic_7'], color='lawngreen', linestyle='-', marker='o', label=None)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. TWII", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()


# -------------------------------
# Periodic Returns & Win Rate Functions
# -------------------------------
def calculate_periodic_returns_df(df, period):
    """
    Resample the DataFrame using the specified period (e.g., 'ME', 'QE', '6ME', 'YE')
    by taking the last value of each period, then compute period-over-period returns.
    """
    resampled = df.resample(period).last()
    returns = resampled.pct_change().dropna()
    return returns

def calculate_win_rate_df(returns_df, benchmark_column='0050.TW'):
    """
    Calculate win rate for each column (except the benchmark) as the ratio of periods
    where the column's return is higher than the benchmark's return.
    """
    cols = [col for col in returns_df.columns if col != benchmark_column]
    win_rates = {}
    total = len(returns_df)
    for col in cols:
        wins = (returns_df[col] > returns_df[benchmark_column]).sum()
        win_rates[col] = wins / total
    return pd.Series(win_rates)

def compute_metrics_df(df, series_list):
    """
    For each specified series in the DataFrame, compute performance metrics (APR, AVOL, ASR, MDD, CR, DDR)
    using calculate_metrics, and return a DataFrame where rows are metrics and columns are strategies.
    """
    metrics_dict = {}
    for col in series_list:
        wealth = df[col].values
        m = calculate_metrics(wealth.reshape(1, -1), TRADE_MODE)
        metrics_dict[col] = {
            'APR': m['APR'][0, 0] if isinstance(m['APR'], np.ndarray) else m['APR'],
            'AVOL': m['AVOL'][0, 0] if isinstance(m['AVOL'], np.ndarray) else m['AVOL'],
            'ASR': m['ASR'][0, 0] if isinstance(m['ASR'], np.ndarray) else m['ASR'],
            'MDD': m['MDD'],
            'CR': m['CR'][0, 0] if isinstance(m['CR'], np.ndarray) else m['CR'],
            'DDR': m['DDR'][0, 0] if isinstance(m['DDR'], np.ndarray) else m['DDR']
        }
    return pd.DataFrame(metrics_dict)

# -------------------------------
# Main
# -------------------------------
def main():
    # Process data into validation and testing DataFrames
    df_val, df_test, full_days, train_days, val_days, test_days = process_data()
    
    print("Validation Data (first 5 rows):")
    print(df_val.head())
    
    print("\nTesting Data (first 5 rows):")
    print(df_test.head())
    
    # Plot cumulative wealth with background shading (Training vs Validation vs Testing)
    plot_results(df_val, df_test, train_days, val_days, test_days)
    
    # Compute periodic returns and win rates for validation period
    period_codes = ['ME', 'QE', '6ME', 'YE']
    print("\nPeriodic Returns and Win Rates (Validation):")
    for period in period_codes:
        returns_val = calculate_periodic_returns_df(df_val, period)
        win_rate_val = calculate_win_rate_df(returns_val, benchmark_column='0050.TW')
        print(f"\nValidation Period: {period}")
        if period == 'YE':
            print("Returns:")
            print(returns_val)
        print("Win Rates:")
        print(win_rate_val)
    
    # Compute periodic returns and win rates for testing period
    print("\nPeriodic Returns and Win Rates (Testing):")
    for period in period_codes:
        returns_test = calculate_periodic_returns_df(df_test, period)
        win_rate_test = calculate_win_rate_df(returns_test, benchmark_column='0050.TW')
        print(f"\nTesting Period: {period}")
        if period == 'YE':
            print("Returns:")
            print(returns_test)
        print("Win Rates:")
        print(win_rate_test)
    
    # Compute performance metrics for validation columns
    metrics_val = compute_metrics_df(df_val, df_val.columns)
    print("\nValidation Metrics:")
    print(metrics_val)
    
    # Compute performance metrics for testing columns
    metrics_test = compute_metrics_df(df_test, df_test.columns)
    print("\nTesting Metrics:")
    print(metrics_test)

if __name__ == "__main__":
    main()