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
# Market Configurations
# -------------------------------
# Stock symbols for different markets
DJIA_STOCKS = [
    "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
    "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
    "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
]

TWII_STOCKS = [
    "1101.TW", "1216.TW", "1301.TW", "1303.TW", "2002.TW", "2207.TW", "2301.TW", "2303.TW", "2308.TW", "2317.TW",
    "2327.TW", "2330.TW", "2345.TW", "2357.TW", "2379.TW", "2382.TW", "2383.TW", "2395.TW", "2412.TW", "2454.TW",
    "2603.TW", "2609.TW", "2615.TW", "2880.TW", "2881.TW", "2882.TW", "2883.TW", "2884.TW", "2885.TW", "2886.TW",
    "2887.TW", "2890.TW", "2891.TW", "2892.TW", "2912.TW", "3008.TW", "3017.TW", "3034.TW", "3045.TW", "3231.TW",
    "3661.TW", "3711.TW", "4904.TW", "4938.TW", "5871.TW", "5876.TW", "5880.TW", "6446.TW", "6505.TW"
]

MARKET_CONFIGS = {
    'US': {
        'name': 'US',
        'start_date': "2015-01-01",
        'end_date': "2025-08-31",
        'market_file': "^DJI.csv",
        'stock_symbols': DJIA_STOCKS,
        'benchmark_column': 'DowJones',
        'benchmark_label': 'DJIA',
        'title': 'DeepTrader vs. DJIA',
        'train_end': 1304,
        'val_end': 2087,
        'test_end': 2782,
        'experiment_ids': [
            '0906/023903',
        ],
        'plot_ylim': None
    },
    'TW': {
        'name': 'Taiwan',
        'start_date': "2015-01-01",
        'end_date': "2025-03-31",
        'market_file': "0050.TW.csv",
        'stock_symbols': TWII_STOCKS,
        'benchmark_column': '0050.TW',
        'benchmark_label': 'TWII',
        'title': 'DeepTrader vs. TWII',
        'train_end': 1304,
        'val_end': 2087,
        'test_end': 2673,
        'experiment_ids': [
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
        ],
        'plot_ylim': None
    }
}

# -------------------------------
# Configuration Selection
# -------------------------------
# Change this to 'TW' or 'US' to switch markets
CURRENT_MARKET = 'US'

# Get current market configuration
config = MARKET_CONFIGS[CURRENT_MARKET]

# -------------------------------
# Constants (from configuration)
# -------------------------------
TRADE_MODE = "M"    # "M": Monthly mode (12 trading periods per year)
TRADE_LEN = 21      # Sampling interval: 21 business days per sample
START_DATE = config['start_date']
END_DATE = config['end_date']
WEALTH_MODE = 'inter'  # 'inter' or 'intra' for daily returns

# -------------------------------
# Experiment Configuration
# -------------------------------
OUTPUTS_BASE_PATH = '../outputs'
EXPERIMENT_IDS = config['experiment_ids']

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
def get_stock_symbols():
    """
    Get stock symbols for the current market.
    """
    return config['stock_symbols']

def load_agent_wealth():
    """
    Load agent wealth arrays from JSON files (both validation and test).
    Based on EXPERIMENT_IDS list containing full date/time paths.
    """
    agent_wealth = {}
    
    for i, exp_path in enumerate(EXPERIMENT_IDS, 1):
        # Construct file paths for JSON files
        val_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'json_file', 'val_results.json')
        test_json_path = os.path.join(OUTPUTS_BASE_PATH, exp_path, 'json_file', 'test_results.json')
        
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

# -------------------------------
# Plotting Functions
# -------------------------------
def plot_results(df_val, df_test, train_days, val_days, test_days):
    """
    Plot cumulative wealth with background shading for Training, Validation, and Testing periods.
    """
    plt.figure(figsize=(14, 7))
    
    # Background shading for segments: training, validation, and testing
    plt.axvspan(train_days[0], train_days[-1], facecolor='gray', alpha=0.1, label='Training Period')
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Testing Period')
    
    # Plot market benchmark for validation and testing
    benchmark_col = config['benchmark_column']
    benchmark_label = config['benchmark_label']
    plt.plot(df_val.index, df_val[benchmark_col], color='r', linestyle='-', marker='o', label=benchmark_label)
    plt.plot(df_test.index, df_test[benchmark_col], color='r', linestyle='-', marker='o', label=None)
    
    # Get agent columns
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
    plt.title(config['title'], fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    
    # Apply y-limit if specified in config
    if config['plot_ylim'] is not None:
        plt.ylim(config['plot_ylim'])
    
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
    
    # Plot market benchmark yearly rebased for validation and testing
    benchmark_col = config['benchmark_column']
    benchmark_label = config['benchmark_label']
    plt.plot(df_val_yearly.index, df_val_yearly[benchmark_col], color='r', linestyle='-', marker='o', label=benchmark_label)
    plt.plot(df_test_yearly.index, df_test_yearly[benchmark_col], color='r', linestyle='-', marker='o', label=None)
    
    # Get agent columns
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
    plt.title(f"{config['title']} (Yearly Rebased)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper center')
    
    # Apply y-limit if specified in config
    if config['plot_ylim'] is not None:
        plt.ylim(config['plot_ylim'])
        
    plt.tight_layout()
    plt.show()


# -------------------------------
# Periodic Returns & Win Rate Functions
# -------------------------------
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

# -------------------------------
# Portfolio Visualization
# -------------------------------
def plot_portfolio_heatmap(experiment_id, outputs_base_path, stock_symbols, sample_dates, period='test'):
    """
    Plot portfolio positions as single heatmap (positive=long, negative=short).
    """
    # Load JSON data
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
    ax.set_title(f'Portfolio Positions - {period}', fontsize=14)
    
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
    print(f"Using {config['name']} market configuration")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Benchmark: {config['benchmark_label']}")
    
    # Load stock symbols
    symbols = get_stock_symbols()
    print(f"Loaded {len(symbols)} stock symbols")
    
    # Process data into validation and testing DataFrames
    df_val, df_test, _, train_days, val_days, test_days = process_data()
    
    print("Validation Data (first 5 rows):")
    print(df_val.head())
    
    print("\nTesting Data (first 5 rows):")
    print(df_test.head())
    
    # Plot cumulative wealth with background shading (Training vs Validation vs Testing)
    plot_results(df_val, df_test, train_days, val_days, test_days)
    plot_yearly_results(df_val, df_test, val_days, test_days)
    
    # Plot portfolio visualizations for each experiment
    print("\n=== Portfolio Visualizations ===")
    for exp_id in EXPERIMENT_IDS:
        print(f"\nVisualizing portfolio for {exp_id}...")
        # Test period visualization
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test')
        # Validation period visualization
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val')
    
    # Compute periodic returns and win rates for validation period
    period_codes = ['ME', 'QE', '6ME', 'YE']
    print("\nPeriodic Returns and Win Rates (Validation):")
    for period in period_codes:
        returns_val = calculate_periodic_returns_df(df_val, period)
        win_rate_val = calculate_win_rate_df(returns_val)
        print(f"\nValidation Period: {period}")
        # if period == 'YE':
        print("Returns:")
        print(returns_val)
        print("Win Rates:")
        print(win_rate_val)
    
    # Compute periodic returns and win rates for testing period
    print("\nPeriodic Returns and Win Rates (Testing):")
    for period in period_codes:
        returns_test = calculate_periodic_returns_df(df_test, period)
        win_rate_test = calculate_win_rate_df(returns_test)
        print(f"\nTesting Period: {period}")
        # if period == 'YE':
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

# -------------------------------
# Utility function to switch markets
# -------------------------------
def set_market(market_code):
    """
    Switch to a different market configuration.
    
    Args:
        market_code (str): 'TW' for Taiwan market or 'US' for US market
    """
    global CURRENT_MARKET, config, START_DATE, END_DATE, EXPERIMENT_IDS
    
    if market_code not in MARKET_CONFIGS:
        raise ValueError(f"Invalid market code. Use one of: {list(MARKET_CONFIGS.keys())}")
    
    CURRENT_MARKET = market_code
    config = MARKET_CONFIGS[CURRENT_MARKET]
    START_DATE = config['start_date']
    END_DATE = config['end_date']
    EXPERIMENT_IDS = config['experiment_ids']
    
    print(f"Switched to {config['name']} market")

if __name__ == "__main__":
    main()