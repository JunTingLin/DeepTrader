import numpy as np
import glob
import re
import os
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.functions import calculate_metrics

# -------------------------------
# Constants
# -------------------------------
TRADE_MODE = "M"    # "M": Monthly mode (12 trading periods per year)
TRADE_LEN = 21      # Sampling interval: 21 business days per sample
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

# -------------------------------
# Data Loading Functions
# -------------------------------
def merge_agent_wealth_series(pattern):
    """
    Merge agent test wealth files across cycles for a specific series.
    Each file is expected to have shape (1, 2), with the first value fixed at 1 and the second
    being the final asset value for that cycle.
    The cumulative series is computed by multiplying sequentially, starting from 1.
    """
    
    # Use a raw string for the pattern to avoid escape issues
    file_list = sorted(glob.glob(pattern), key=lambda x: int(re.search(r'cycle(\d+)', x).group(1)))
    
    if not file_list:
        print(f"No files found for pattern: {pattern}")
        return None
    
    cumulative = [1.0]
    for f in file_list:
        try:
            data = np.load(f)  # Expected shape is (1,2)
            factor = data[0, -1]
            cumulative.append(cumulative[-1] * factor)
        except Exception as e:
            print(f"Error processing file {f}: {e}")
            
    result = np.array(cumulative).reshape(1, -1)
    return result.flatten()  # Flatten to match the expected format


def load_agent_wealth():
    """
    Load and merge agent wealth arrays for testing.
    Returns a dictionary with keys being the strategy names and values being the flattened wealth arrays.
    """
    
    # Merge wealth series for each strategy
    w_MSU_dynamic = merge_agent_wealth_series(r"..\outputs_2\0304\000154\npy_file\agent_wealth_test_cycle*.npy")
    w_MSU_rho0    = merge_agent_wealth_series(r"..\outputs_2\0304\023512\npy_file\agent_wealth_test_cycle*.npy")
    w_MSU_rho05   = merge_agent_wealth_series(r"..\outputs_2\0304\023629\npy_file\agent_wealth_test_cycle*.npy")
    w_MSU_rho1    = merge_agent_wealth_series(r"..\outputs_2\0304\023707\npy_file\agent_wealth_test_cycle*.npy")
    wo_MSU_rho0   = merge_agent_wealth_series(r"..\outputs_2\0304\033635\npy_file\agent_wealth_test_cycle*.npy")
    wo_MSU_rho05  = merge_agent_wealth_series(r"..\outputs_2\0304\033659\npy_file\agent_wealth_test_cycle*.npy")
    wo_MSU_rho1   = merge_agent_wealth_series(r"..\outputs_2\0304\033717\npy_file\agent_wealth_test_cycle*.npy")

    return {
        'w_MSU_dynamic': w_MSU_dynamic,
        'w_MSU_rho0': w_MSU_rho0,
        'w_MSU_rho05': w_MSU_rho05,
        'w_MSU_rho1': w_MSU_rho1,
        'wo_MSU_rho0': wo_MSU_rho0,
        'wo_MSU_rho05': wo_MSU_rho05,
        'wo_MSU_rho1': wo_MSU_rho1,
    }


def get_business_day_segments():
    """
    Generate full business day date range from START_DATE to END_DATE,
    and split into:
      - Training: indices 0 to 2042 (2043 days)
      - Validation: indices 2043 to 4150 (2108 days)
      - Testing: indices 4151 to 6259 (2109 days)
    """
    full_days = pd.bdate_range(start=START_DATE, end=END_DATE)
    total_days = len(full_days)
    print(f"Total business days: {total_days}")
    
    train_days = full_days[0:2043]
    val_days   = full_days[2043:4151]
    test_days  = full_days[4151:6260]
    
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")
    print(f"Test days: {len(test_days)}")
    
    return full_days, train_days, val_days, test_days

def get_djia_data(full_days, file_path="^DJI.csv"):
    """
    Load DJIA data from a local CSV file, filter for full_days, 
    reindex to the full business day range, and fill missing values.
    """
    df = pd.read_csv(file_path, parse_dates=["Date"], index_col="Date")
    full_days = pd.DatetimeIndex(full_days)
    df = df.loc[full_days[0]:full_days[-1]]
    df = df.reindex(full_days)
    df.replace(0, np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df

def compute_cumulative_wealth(df_djia):
    """
    Compute daily cumulative wealth using a Buy & Hold strategy from DJIA Close prices.
    Rebase the series so that it starts at 1.
    """
    djia_close = df_djia["Close"].copy()
    daily_return = djia_close.pct_change().fillna(0.0)
    wealth_daily = (1.0 + daily_return).cumprod()
    wealth_rebased = wealth_daily / wealth_daily.iloc[0]
    return wealth_rebased

# -------------------------------
# Data Processing
# -------------------------------
def process_data():
    """
    Process all data into a testing DataFrame.
    1. Generate full business days and split into training and testing segments.
    2. Download DJIA data for the full period, then extract the testing segment (indices 2043 to 6259)
       and compute its cumulative wealth independently (starting at 1).
    3. Load the agent wealth data arrays (w_MSU_dynamic, etc.), which are merged from cycle files.
    4. Create a DataFrame (df_test) with the sample dates as index and columns for each agent series
       (prefixed with "test_") and 'DowJones'.
    """
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Load DJIA data for the entire period
    df_djia_full = get_djia_data(full_days)
    
    # Extract testing segment: indices 2043 to 6259
    df_djia_test = df_djia_full.loc[test_days]
    djia_wealth_test = compute_cumulative_wealth(df_djia_test)
    
    # Sample dates for testing segment
    test_sample_dates = test_days[::TRADE_LEN]
    
    # Sample the DJIA cumulative wealth for testing and rebase to 1
    djia_series_test = djia_wealth_test.iloc[::TRADE_LEN].copy()
    djia_series_test = djia_series_test / djia_series_test.iloc[0]
    
    # Load combined agent wealth data (merged from cycle files)
    agent_wealth = load_agent_wealth()
    n_test = len(test_sample_dates)
    test_data = {}
    for key, series in agent_wealth.items():
        if series is not None:
            # Make sure we have the right number of points
            agent_test = series[:n_test]
            # Rebase to start at 1
            agent_test = agent_test / agent_test[0]
            # Prefix with "test_" for clarity
            test_data["test_" + key] = agent_test
    
    # Create a testing DataFrame with sample dates as index and include DJIA as 'DowJones'
    df_test = pd.DataFrame(test_data, index=test_sample_dates)
    df_test['DowJones'] = djia_series_test.values
    
    return df_test, full_days, train_days, test_days

# -------------------------------
# Plotting Functions
# -------------------------------
def plot_results(df_test, test_days):
    """
    Plot cumulative wealth for Testing period only.
    No background shading for training period.
    """
    plt.figure(figsize=(14, 7))
    
    # Background shading only for testing segment
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Testing Period')
    
    # Plot DJIA wealth
    plt.plot(df_test.index, df_test['DowJones'], color='r', linestyle='-', marker='o', label='DJIA')
    
    # Plot agent wealth for testing segment
    plt.plot(df_test.index, df_test['test_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(df_test.index, df_test['test_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(df_test.index, df_test['test_w_MSU_rho05'], color='c', linestyle='-.', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(df_test.index, df_test['test_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(df_test.index, df_test['test_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(df_test.index, df_test['test_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(df_test.index, df_test['test_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. DJIA", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_yearly_results(df_test, test_days):
    """
    Plot yearly rebased cumulative wealth. Each year's first value is rebased to 1.
    Background shading is applied for the Testing period.
    """
    def rebase_yearly_series(s):
        rebased = s.copy()
        for year, group in s.groupby(s.index.year):
            rebased.loc[group.index] = group / group.iloc[0]
        return rebased

    df_test_yearly = df_test.copy()
    for col in df_test_yearly.columns:
        df_test_yearly[col] = rebase_yearly_series(df_test_yearly[col])
    
    plt.figure(figsize=(12, 6))
    
    # Background shading for testing segment
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Testing Period')
    
    # Plot DJIA
    plt.plot(df_test_yearly.index, df_test_yearly['DowJones'], color='r', linestyle='-', marker='o', label='DJIA')
    
    # Plot agent series for testing
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho05'], color='c', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth (Yearly Rebased)", fontsize=14)
    plt.title("DeepTrader vs. DJIA (Yearly Rebased)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper center')
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

def calculate_win_rate_df(returns_df, benchmark_column='DowJones'):
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
    # Process data into a single testing DataFrame (from index 2043 to 6259)
    df_test, full_days, train_days, test_days = process_data()
    print("Testing Data (first 5 rows):")
    print(df_test.head())
    
    # Plot cumulative wealth (Testing period only)
    plot_results(df_test, test_days)
    plot_yearly_results(df_test, test_days)
    
    # Compute periodic returns and win rates for multiple period codes: ME, QE, 6ME, YE
    period_codes = ['ME', 'QE', '6ME', 'YE']
    print("\nPeriodic Returns and Win Rates (Testing):")
    for period in period_codes:
        returns_test = calculate_periodic_returns_df(df_test, period)
        win_rate_test = calculate_win_rate_df(returns_test, benchmark_column='DowJones')
        print(f"\nPeriod: {period}")
        if period == 'YE':
            print("Returns:")
            print(returns_test)
        print(returns_test)
        print("Win Rates:")
        print(win_rate_test)
    
    # Compute performance metrics for all columns (including DowJones)
    metrics_test = compute_metrics_df(df_test, df_test.columns)
    print("\nTesting Metrics:")
    print(metrics_test)

if __name__ == "__main__":
    main()