import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import json

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
WEALTH_MODE = 'inter' # 'inter' or 'intra' for TWII daily returns

# -------------------------------
# Experiment Configuration
# -------------------------------
# Outputs base path
OUTPUTS_BASE_PATH = '../outputs'

# Define experiment IDs
EXPERIMENT_IDS = [
    '0718/141038',
    '0718/141055',
    '0718/213952',
    '0718/214006',
    '0719/140312',
    '0719/140324',
    '0719/230025',
    '0719/230035',
    '0720/104851',
    '0720/104859'
]

# -------------------------------
# Plotting Style Configuration
# -------------------------------
AGENT_COLORS = ['b', 'darkblue', 'c', 'steelblue', 'limegreen', 'g', 'lawngreen', 'purple', 'orange', 'brown']
AGENT_LINESTYLES = ['-', '-', '-.', '-', '-', '-', '-', '--', '--', ':']
AGENT_MARKERS = ['o'] * 10  # Use 'o' marker for all agents
AGENT_LABELS = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6', 'Agent 7', 'Agent 8', 'Agent 9', 'Agent 10']


# -------------------------------
# Data Loading Functions
# -------------------------------
def load_agent_wealth():
    """
    Load agent wealth arrays from JSON files (both validation and test).
    Based on EXPERIMENT_IDS list containing full date/time paths.
    """
    agent_wealth = {}
    
    for i, exp_path in enumerate(EXPERIMENT_IDS, 1):
        # Construct file paths for JSON files
        val_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'val_results', 'val_results.json')
        test_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'test_results', 'test_results.json')
        
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

def get_twii_data(full_days, file_path="0050.TW.csv"):
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

def compute_cumulative_wealth(df_twii, wealth_mode=WEALTH_MODE):
    """
    Compute daily cumulative wealth using a Buy & Hold strategy from TWII Close prices.
    Rebase the series so that it starts at 1.
    """
    if wealth_mode == 'inter':
        twii_open = df_twii["Open"].copy()
        daily_return = twii_open.pct_change().fillna(0.0)
    elif wealth_mode == 'intra':
        daily_return = ((df_twii["Close"] - df_twii["Open"]) / df_twii["Open"]).fillna(0.0)
    else:
        raise ValueError("Invalid wealth_mode. Use 'inter' or 'intra'.")
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
    2. Download TWII data for the full period, then extract the validation segment (indices 1304 to 2086)
       and test segment (indices 2087 to 2672), computing their cumulative wealth independently (starting at 1).
    3. Load the agent wealth data arrays (val_w_MSU_dynamic, test_w_MSU_dynamic, etc.), which cover the 
       respective validation and testing periods.
    4. Create DataFrames (df_val and df_test) with the sample dates as index and columns for each 
       agent series and '0050.TW'.
    """
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Download TWII data for the entire period
    df_twii_full = get_twii_data(full_days)
    
    # Extract validation segment: indices 1304 to 2086
    df_twii_val = df_twii_full.loc[val_days]
    twii_wealth_val = compute_cumulative_wealth(df_twii_val)
    
    # Extract testing segment: indices 2087 to 2672
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
    
    # Get agent columns (exclude 'DowJones')
    val_agent_cols = [col for col in df_val.columns if col.startswith('val_')]
    test_agent_cols = [col for col in df_test.columns if col.startswith('test_')]

    # Plot agent wealth for validation and testing segments automatically
    for i, (val_col, test_col) in enumerate(zip(val_agent_cols, test_agent_cols)):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        linestyle = AGENT_LINESTYLES[i % len(AGENT_LINESTYLES)]
        marker = AGENT_MARKERS[i % len(AGENT_MARKERS)]
        label = AGENT_LABELS[i % len(AGENT_LABELS)]
        
        # Plot validation (with label) and testing (without label)
        plt.plot(df_val.index, df_val[val_col], color=color, linestyle=linestyle, 
                marker=marker, label=label)
        plt.plot(df_test.index, df_test[test_col], color=color, linestyle=linestyle, 
                marker=marker, label=None)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. TWII", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_yearly_results(df_val, df_test, val_days, test_days):
    """
    Plot yearly rebased cumulative wealth. Each year's first value is rebased to 1.
    Background shading is applied for the Validation and Testing periods.
    """
    def rebase_yearly_series(s):
        rebased = s.copy()
        for year, group in s.groupby(s.index.year):
            rebased.loc[group.index] = group / group.iloc[0]
        return rebased

    # Create copies of the dataframes
    df_val_yearly = df_val.copy()
    df_test_yearly = df_test.copy()
    
    # Rebase each column yearly
    for col in df_val_yearly.columns:
        df_val_yearly[col] = rebase_yearly_series(df_val_yearly[col])
    
    for col in df_test_yearly.columns:
        df_test_yearly[col] = rebase_yearly_series(df_test_yearly[col])
    
    plt.figure(figsize=(12, 6))
    
    # Background shading for validation and testing segments
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Testing Period')
    
    # Plot TWII yearly rebased for validation and testing
    plt.plot(df_val_yearly.index, df_val_yearly['0050.TW'], color='r', linestyle='-', marker='o', label='TWII')
    plt.plot(df_test_yearly.index, df_test_yearly['0050.TW'], color='r', linestyle='-', marker='o', label=None)
    
    # Get agent columns (exclude 'DowJones')
    val_agent_cols = [col for col in df_val_yearly.columns if col.startswith('val_')]
    test_agent_cols = [col for col in df_test_yearly.columns if col.startswith('test_')]
    
    # Plot agent yearly rebased for validation and testing automatically
    for i, (val_col, test_col) in enumerate(zip(val_agent_cols, test_agent_cols)):
        color = AGENT_COLORS[i % len(AGENT_COLORS)]
        linestyle = AGENT_LINESTYLES[i % len(AGENT_LINESTYLES)]
        marker = AGENT_MARKERS[i % len(AGENT_MARKERS)]
        label = AGENT_LABELS[i % len(AGENT_LABELS)]
        
        # Plot validation (with label) and testing (without label)
        plt.plot(df_val_yearly.index, df_val_yearly[val_col], color=color, linestyle=linestyle, 
                marker=marker, label=label)
        plt.plot(df_test_yearly.index, df_test_yearly[test_col], color=color, linestyle=linestyle, 
                marker=marker, label=None)

    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth (Yearly Rebased)", fontsize=14)
    plt.title("DeepTrader vs. TWII (Yearly Rebased)", fontsize=16)
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

# -------------------------------
# Portfolio Visualization
# -------------------------------
def plot_portfolio_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test'):
    """
    Plot portfolio positions as single heatmap (positive=long, negative=short).
    """
    # Load JSON data
    json_path = os.path.join(outputs_base_path, experiment_id, f'{period}_results', f'{period}_results.json')
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
    fig, ax = plt.subplots(figsize=(min(20, n_steps * 0.15), 10))
    
    # Plot heatmap with diverging colormap
    im = ax.imshow(position_matrix, aspect='auto', cmap='RdYlGn', interpolation='nearest', 
                   vmin=-0.3, vmax=0.3, origin='lower')
    
    # Formatting
    ax.set_xlabel('Trading Steps', fontsize=12)
    ax.set_ylabel('Stocks', fontsize=12)
    ax.set_title(f'Portfolio Positions - {experiment_id} ({period})', fontsize=14)
    
    # Set y-axis labels with both index and symbol
    ax.set_yticks(range(n_stocks))
    y_labels = [f"{i}: {stock_symbols[i]}" for i in range(n_stocks)]
    ax.set_yticklabels(y_labels, fontsize=8)
    
    # Set x-axis labels
    step_interval = max(1, n_steps // 10)
    xticks = range(0, n_steps, step_interval)
    ax.set_xticks(xticks)
    if len(sample_dates) >= n_steps:
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
    plt.show()

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
    # Plot yearly rebased cumulative wealth
    plot_yearly_results(df_val, df_test, val_days, test_days)
    
    # Load Taiwan stock symbols from Excel
    
    df_symbols = pd.read_excel('../data/TWII/0050.xlsx')
    tw_symbols = df_symbols.iloc[:, 0].tolist() if not df_symbols.empty else [f'Stock_{i}' for i in range(49)]
    
    # Plot portfolio visualizations for each experiment
    print("\n=== Portfolio Visualizations ===")
    for exp_id in EXPERIMENT_IDS:
        print(f"\nVisualizing portfolio for {exp_id}...")
        # Test period visualizations - using heatmap style
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, tw_symbols, df_test.index, 'test')
        # Validation period visualizations (optional)
        # plot_portfolio_scatter(exp_id, OUTPUTS_BASE_PATH, tw_symbols, df_val.index, 'val')
    
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