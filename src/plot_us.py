import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from utils.functions import calculate_metrics


TRADE_MODE = "M"    # "M": Monthly mode, i.e. 12 trading periods per year
TRADE_LEN = 21      # Sampling interval: 21 business days per sample
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"


def load_agent_wealth():
    """
    Load and flatten agent wealth arrays for validation and test.
    """
    val_w_MSU_dynamic = np.load(r'outputs\0207\035156\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_rho0    = np.load(r'outputs\0207\195947\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_rho05   = np.load(r'outputs\0208\012246\npy_file\agent_wealth_val.npy').flatten()
    val_w_MSU_rho1    = np.load(r'outputs\0207\152525\npy_file\agent_wealth_val.npy').flatten()
    val_wo_MSU_rho0   = np.load(r'outputs\0209\112527\npy_file\agent_wealth_val.npy').flatten()
    val_wo_MSU_rho05  = np.load(r'outputs\0209\162424\npy_file\agent_wealth_val.npy').flatten()
    val_wo_MSU_rho1   = np.load(r'outputs\0209\033149\npy_file\agent_wealth_val.npy').flatten()
    
    test_w_MSU_dynamic = np.load(r'outputs\0207\035156\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_rho0    = np.load(r'outputs\0207\195947\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_rho05   = np.load(r'outputs\0208\012246\npy_file\agent_wealth_test.npy').flatten()
    test_w_MSU_rho1    = np.load(r'outputs\0207\152525\npy_file\agent_wealth_test.npy').flatten()
    test_wo_MSU_rho0   = np.load(r'outputs\0209\112527\npy_file\agent_wealth_test.npy').flatten()
    test_wo_MSU_rho05  = np.load(r'outputs\0209\162424\npy_file\agent_wealth_test.npy').flatten()
    test_wo_MSU_rho1   = np.load(r'outputs\0209\033149\npy_file\agent_wealth_test.npy').flatten()
    
    return {
        'val_w_MSU_dynamic': val_w_MSU_dynamic,
        'val_w_MSU_rho0': val_w_MSU_rho0,
        'val_w_MSU_rho05': val_w_MSU_rho05,
        'val_w_MSU_rho1': val_w_MSU_rho1,
        'val_wo_MSU_rho0': val_wo_MSU_rho0,
        'val_wo_MSU_rho05': val_wo_MSU_rho05,
        'val_wo_MSU_rho1': val_wo_MSU_rho1,
        'test_w_MSU_dynamic': test_w_MSU_dynamic,
        'test_w_MSU_rho0': test_w_MSU_rho0,
        'test_w_MSU_rho05': test_w_MSU_rho05,
        'test_w_MSU_rho1': test_w_MSU_rho1,
        'test_wo_MSU_rho0': test_wo_MSU_rho0,
        'test_wo_MSU_rho05': test_wo_MSU_rho05,
        'test_wo_MSU_rho1': test_wo_MSU_rho1
    }

def get_business_day_segments():
    """
    Generate full business day date range from START_DATE to END_DATE,
    and split into:
      - Training: indices 0 to 2042 (2043 days)
      - Validation: indices 2043 to 4150 (2108 days)
      - Test: indices 4151 to 6259 (2109 days)
    """
    full_days = pd.bdate_range(start=START_DATE, end=END_DATE)
    train_days = full_days[0:2043]
    val_days   = full_days[2043:4151]
    test_days  = full_days[4151:6260]
    print(f"Total business days: {len(full_days)}")
    print(f"Training days: {len(train_days)}")
    print(f"Validation days: {len(val_days)}")
    print(f"Test days: {len(test_days)}")
    return full_days, train_days, val_days, test_days

def download_djia_data(full_days):
    """
    Download DJIA data for the period covering full_days,
    reindex to the full business day range and fill missing values.
    """
    djia_ticker = yf.Ticker("^DJI")
    df = djia_ticker.history(start=full_days[0].strftime("%Y-%m-%d"),
                             end=full_days[-1].strftime("%Y-%m-%d"))
    df.index = df.index.tz_localize(None)
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
    Process all data into DataFrames.
    1. Generate full business days and split into training, validation, and test segments.
    2. Download DJIA data, compute daily cumulative wealth, and sample validation and test segments
       every TRADE_LEN days, then rebase each segment to start at 1.
    3. Load agent wealth data, rebase each agent series (agent data are already sampled every TRADE_LEN days).
    4. Create DataFrames (df_val and df_test) with the sample dates as index and columns as agent series and 'DowJones'.
    """
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Download DJIA data for the entire period
    df_djia_full = download_djia_data(full_days)

    # Split DJIA data into validation and test segments
    df_djia_val = df_djia_full.loc[val_days]
    df_djia_test = df_djia_full.loc[test_days]

    # Compute cumulative wealth separately for validation and test segments
    djia_wealth_val = compute_cumulative_wealth(df_djia_val)
    djia_wealth_test = compute_cumulative_wealth(df_djia_test)
    
    # Sample dates for validation and test segments
    val_sample_dates = val_days[::TRADE_LEN]
    test_sample_dates = test_days[::TRADE_LEN]
    
    # Sample the DJIA cumulative wealth for each segment and rebase to 1
    djia_series_val = djia_wealth_val.iloc[::TRADE_LEN].copy()
    djia_series_val = djia_series_val / djia_series_val.iloc[0]
    
    djia_series_test = djia_wealth_test.iloc[::TRADE_LEN].copy()
    djia_series_test = djia_series_test / djia_series_test.iloc[0]
    
    # Load agent data and rebase (each agent series is assumed to have the same number of points as sample dates)
    agent_wealth = load_agent_wealth()
    val_series_names = ['val_w_MSU_dynamic', 'val_w_MSU_rho0', 'val_w_MSU_rho05', 'val_w_MSU_rho1',
                        'val_wo_MSU_rho0', 'val_wo_MSU_rho05', 'val_wo_MSU_rho1']
    test_series_names = ['test_w_MSU_dynamic', 'test_w_MSU_rho0', 'test_w_MSU_rho05', 'test_w_MSU_rho1',
                         'test_wo_MSU_rho0', 'test_wo_MSU_rho05', 'test_wo_MSU_rho1']
    
    val_data = {name: agent_wealth[name] / agent_wealth[name][0] for name in val_series_names}
    test_data = {name: agent_wealth[name] / agent_wealth[name][0] for name in test_series_names}
    
    # Create DataFrames with sample dates as index; include DJIA as 'DowJones'
    df_val = pd.DataFrame(val_data, index=val_sample_dates)
    df_val['DowJones'] = djia_series_val.values
    df_test = pd.DataFrame(test_data, index=test_sample_dates)
    df_test['DowJones'] = djia_series_test.values
    
    return df_val, df_test, full_days, train_days, val_days, test_days

# -------------------------------
# Plotting Functions (using original axvspan and color settings)
# -------------------------------
def plot_results(df_val, df_test, train_days, val_days, test_days):
    """
    Plot cumulative wealth with background shading for Training, Validation, and Test periods.
    Uses original settings for axvspan and color.
    """
    plt.figure(figsize=(14, 7))
    
    # Background shading for segments
    plt.axvspan(train_days[0], train_days[-1], facecolor='gray', alpha=0.1, label='Training Period')
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Test Period')
    
    # Plot DJIA wealth
    plt.plot(df_val.index, df_val['DowJones'], color='r', linestyle='-', marker='o', label='DJIA')
    plt.plot(df_test.index, df_test['DowJones'], color='r', linestyle='-', marker='o', label=None)
    
    # Plot agent wealth for validation segment
    plt.plot(df_val.index, df_val['val_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(df_val.index, df_val['val_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(df_val.index, df_val['val_w_MSU_rho05'], color='c', linestyle='-.', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(df_val.index, df_val['val_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(df_val.index, df_val['val_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(df_val.index, df_val['val_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(df_val.index, df_val['val_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    
    # Plot agent wealth for test segment
    plt.plot(df_test.index, df_test['test_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_w_MSU_rho05'], color='c', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label=None)
    plt.plot(df_test.index, df_test['test_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label=None)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. DJIA", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_yearly_results(df_val, df_test, val_days, test_days):
    """
    Plot yearly rebased cumulative wealth. Each year's first value is rebased to 1.
    Background shading is applied for Validation and Test periods.
    """
    def rebase_yearly_series(s):
        rebased = s.copy()
        for year, group in s.groupby(s.index.year):
            rebased.loc[group.index] = group / group.iloc[0]
        return rebased

    df_val_yearly = df_val.copy()
    df_test_yearly = df_test.copy()
    for col in df_val_yearly.columns:
        df_val_yearly[col] = rebase_yearly_series(df_val_yearly[col])
    for col in df_test_yearly.columns:
        df_test_yearly[col] = rebase_yearly_series(df_test_yearly[col])
    
    plt.figure(figsize=(12, 6))
    
    # Background shading for Validation and Test segments
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Test Period')
    
    # Plot DJIA
    plt.plot(df_val_yearly.index, df_val_yearly['DowJones'], color='r', linestyle='-', marker='o', label='DJIA')
    plt.plot(df_test_yearly.index, df_test_yearly['DowJones'], color='r', linestyle='-', marker='o', label=None)
    
    # Plot agent series for Validation segment
    plt.plot(df_val_yearly.index, df_val_yearly['val_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_w_MSU_rho05'], color='c', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(df_val_yearly.index, df_val_yearly['val_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    
    # Plot agent series for Test segment
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho05'], color='c', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label=None)
    plt.plot(df_test_yearly.index, df_test_yearly['test_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label=None)
    
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
    Calculate win rate for each agent column as the ratio of periods
    where the agent's return is higher than the benchmark's return.
    """
    agent_columns = [col for col in returns_df.columns if col != benchmark_column]
    win_rates = {}
    total = len(returns_df)
    for col in agent_columns:
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



# Process data into DataFrames
df_val, df_test, full_days, train_days, val_days, test_days = process_data()

# Compute periodic returns and win rates for multiple period codes
period_codes = ['ME', 'QE', '6ME', 'YE']
print("\nPeriodic Returns and Win Rates (Validation):")
for period in period_codes:
    returns_val = calculate_periodic_returns_df(df_val, period)
    win_rate_val = calculate_win_rate_df(returns_val, benchmark_column='DowJones')
    print(f"\nPeriod: {period}")
    if period == 'YE':
        print("Returns:")
        print(returns_val)
    print("Win Rates:")
    print(win_rate_val)

print("\nPeriodic Returns and Win Rates (Test):")
for period in period_codes:
    returns_test = calculate_periodic_returns_df(df_test, period)
    win_rate_test = calculate_win_rate_df(returns_test, benchmark_column='DowJones')
    print(f"\nPeriod: {period}")
    if period == 'YE':
        print("Returns:")
        print(returns_test)
    print("Win Rates:")
    print(win_rate_test)

# Compute performance metrics for all columns (including DowJones)
# For validation, use all columns present in df_val; similarly for test.
metrics_val = compute_metrics_df(df_val, df_val.columns)
metrics_test = compute_metrics_df(df_test, df_test.columns)
print("\nValidation Metrics:")
print(metrics_val)
print("\nTest Metrics:")
print(metrics_test)

# Plot cumulative wealth with background shading
plot_results(df_val, df_test, train_days, val_days, test_days)
plot_yearly_results(df_val, df_test, val_days, test_days)
