import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from utils.functions import calculate_metrics

TRADE_MODE = "M"
TRADE_LEN = 21
TARGET_DIR = "."
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"
# VAL_NUM_POINTS = 101
# TEST_NUM_POINTS = 101

def load_agent_wealth():
    """
    Load and flatten agent wealth arrays for validation and test.
    Expected shape of each loaded array is (1, 101).
    """
    val_w_MSU_dynamic = np.load('agent_wealth_val_w_MSU_dynamic.npy').flatten()
    val_w_MSU_rho0    = np.load('agent_wealth_val_w_MSU_rho0.npy').flatten()
    val_w_MSU_rho05   = np.load('agent_wealth_val_w_MSU_rho05.npy').flatten()
    val_w_MSU_rho1    = np.load('agent_wealth_val_w_MSU_rho1.npy').flatten()
    val_wo_MSU_rho0   = np.load('agent_wealth_val_wo_MSU_rho0.npy').flatten()
    val_wo_MSU_rho05  = np.load('agent_wealth_val_wo_MSU_rho05.npy').flatten()
    val_wo_MSU_rho1   = np.load('agent_wealth_val_wo_MSU_rho1.npy').flatten()
    
    test_w_MSU_dynamic = np.load('agent_wealth_test_w_MSU_dynamic.npy').flatten()
    test_w_MSU_rho0    = np.load('agent_wealth_test_w_MSU_rho0.npy').flatten()
    test_w_MSU_rho05   = np.load('agent_wealth_test_w_MSU_rho05.npy').flatten()
    test_w_MSU_rho1    = np.load('agent_wealth_test_w_MSU_rho1.npy').flatten()
    test_wo_MSU_rho0   = np.load('agent_wealth_test_wo_MSU_rho0.npy').flatten()
    test_wo_MSU_rho05  = np.load('agent_wealth_test_wo_MSU_rho05.npy').flatten()
    test_wo_MSU_rho1   = np.load('agent_wealth_test_wo_MSU_rho1.npy').flatten()
    
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
    Generate the full business day date range (6260 days) and split into:
      - Training: indices 0 to 2042 (2043 days)
      - Validation: indices 2043 to 4150 (2108 days)
      - Test: indices 4151 to 6259 (2109 days)
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

def download_djia_data(full_days):
    """
    Download DJIA data for the period covering full_days,
    reindex to the full business day range and fill missing values.
    """
    djia_ticker = yf.Ticker("^DJI")
    df_djia = djia_ticker.history(start=full_days[0].strftime("%Y-%m-%d"),
                                  end=full_days[-1].strftime("%Y-%m-%d"))
    df_djia.index = df_djia.index.tz_localize(None)
    df_djia = df_djia.reindex(full_days)
    df_djia.replace(0, np.nan, inplace=True)
    df_djia.ffill(inplace=True)
    df_djia.bfill(inplace=True)
    return df_djia

def compute_cumulative_wealth(df_djia):
    """
    Compute cumulative wealth from DJIA Close prices using a Buy & Hold strategy.
    Rebase the series so that it starts at 1.
    """
    djia_close = df_djia["Close"].copy()
    daily_return = djia_close.pct_change().fillna(0.0)
    wealth_daily = (1.0 + daily_return).cumprod()
    wealth_rebased = wealth_daily / wealth_daily.iloc[0]
    return wealth_rebased

def resample_series(series, step):
    return series.iloc[::step].values

def resample_dates(dates, step):
    return dates[::step]

def rebase_yearly(series):
    """
    Rebase a time series so that each year's first value becomes 1.
    """
    rebased = series.copy()
    for year, group in series.groupby(series.index.year):
        rebased.loc[group.index] = group / group.iloc[0]
    return rebased

def plot_results(full_days, train_days, val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict):
    """
    Plot the full timeline with background shading for Training, Validation, and Test segments.
    For Validation and Test segments, overlay DJIA and agent wealth trends.
    For each agent series, use a single label.
    """
    plt.figure(figsize=(14, 7))
    
    # Background shading for segments
    plt.axvspan(train_days[0], train_days[-1], facecolor='gray', alpha=0.1, label='Training Period')
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Test Period')
    
    val_dates_sampled = resample_dates(val_days, TRADE_LEN)
    test_dates_sampled = resample_dates(test_days, TRADE_LEN)
    
    # Resample DJIA wealth for Validation and Test segments and rebase to start at 1
    djia_wealth_val = resample_series(djia_wealth_daily[val_days], TRADE_LEN)
    djia_wealth_val = djia_wealth_val / djia_wealth_val[0]
    djia_wealth_test = resample_series(djia_wealth_daily[test_days], TRADE_LEN)
    djia_wealth_test = djia_wealth_test / djia_wealth_test[0]
    
    # Plot DJIA wealth trends
    plt.plot(val_dates_sampled, djia_wealth_val, color='r', linestyle='-', marker='o', label='DJIA')
    plt.plot(test_dates_sampled, djia_wealth_test, color='r', linestyle='-', marker='o', label=None)
    
    # For agent wealth, plot each series once (use label only for the first segment)
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_dynamic'], color='b', linestyle='-', marker='o', label=None)
    
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho0'], color='darkblue', linestyle='-', marker='o', label=None)
    
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho05'], color='c', linestyle='-.', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho05'], color='c', linestyle='-.', marker='o', label=None)
    
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho1'], color='steelblue', linestyle='-', marker='o', label=None)

    plt.plot(val_dates_sampled, wealth_val_dict['val_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_wo_MSU_rho0'], color='limegreen', linestyle='-', marker='o', label=None)

    plt.plot(val_dates_sampled, wealth_val_dict['val_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_wo_MSU_rho05'], color='g', linestyle='-', marker='o', label=None)

    plt.plot(val_dates_sampled, wealth_val_dict['val_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_wo_MSU_rho1'], color='lawngreen', linestyle='-', marker='o', label=None)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. DJIA", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def plot_yearly_results(val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict):
    """
    Plot yearly wealth trends (using the existing 101 points) for Validation and Test segments.
    For each year, the cumulative wealth is rebased so that it starts at 1.
    The plot includes yearly cumulative wealth for DJIA and for each DeepTrader series.
    """
    # Resample dates to monthly sampling
    val_dates = resample_dates(val_days, TRADE_LEN)
    test_dates = resample_dates(test_days, TRADE_LEN)
    
    # Convert DJIA monthly series to pd.Series and rebase yearly
    djia_monthly_val = pd.Series(resample_series(djia_wealth_daily[val_days], TRADE_LEN), index=val_dates)
    djia_monthly_val = rebase_yearly(djia_monthly_val)
    djia_monthly_test = pd.Series(resample_series(djia_wealth_daily[test_days], TRADE_LEN), index=test_dates)
    djia_monthly_test = rebase_yearly(djia_monthly_test)
    
    # For agent series, convert to pd.Series and rebase yearly
    val_w_MSU_dynamic = pd.Series(wealth_val_dict['val_w_MSU_dynamic'], index=val_dates)
    val_w_MSU_dynamic = rebase_yearly(val_w_MSU_dynamic)
    test_w_MSU_dynamic = pd.Series(wealth_test_dict['test_w_MSU_dynamic'], index=test_dates)
    test_w_MSU_dynamic = rebase_yearly(test_w_MSU_dynamic)
    
    val_w_MSU_rho0 = pd.Series(wealth_val_dict['val_w_MSU_rho0'], index=val_dates)
    val_w_MSU_rho0 = rebase_yearly(val_w_MSU_rho0)
    test_w_MSU_rho0 = pd.Series(wealth_test_dict['test_w_MSU_rho0'], index=test_dates)
    test_w_MSU_rho0 = rebase_yearly(test_w_MSU_rho0)
    
    val_w_MSU_rho05 = pd.Series(wealth_val_dict['val_w_MSU_rho05'], index=val_dates)
    val_w_MSU_rho05 = rebase_yearly(val_w_MSU_rho05)
    test_w_MSU_rho05 = pd.Series(wealth_test_dict['test_w_MSU_rho05'], index=test_dates)
    test_w_MSU_rho05 = rebase_yearly(test_w_MSU_rho05)
    
    val_w_MSU_rho1 = pd.Series(wealth_val_dict['val_w_MSU_rho1'], index=val_dates)
    val_w_MSU_rho1 = rebase_yearly(val_w_MSU_rho1)
    test_w_MSU_rho1 = pd.Series(wealth_test_dict['test_w_MSU_rho1'], index=test_dates)
    test_w_MSU_rho1 = rebase_yearly(test_w_MSU_rho1)


    val_wo_MSU_rho0 = pd.Series(wealth_val_dict['val_wo_MSU_rho0'], index=val_dates)
    val_wo_MSU_rho0 = rebase_yearly(val_wo_MSU_rho0)
    test_wo_MSU_rho0 = pd.Series(wealth_test_dict['test_wo_MSU_rho0'], index=test_dates)
    test_wo_MSU_rho0 = rebase_yearly(test_wo_MSU_rho0)

    val_wo_MSU_rho05 = pd.Series(wealth_val_dict['val_wo_MSU_rho05'], index=val_dates)
    val_wo_MSU_rho05 = rebase_yearly(val_wo_MSU_rho05)
    test_wo_MSU_rho05 = pd.Series(wealth_test_dict['test_wo_MSU_rho05'], index=test_dates)
    test_wo_MSU_rho05 = rebase_yearly(test_wo_MSU_rho05)

    val_wo_MSU_rho1 = pd.Series(wealth_val_dict['val_wo_MSU_rho1'], index=val_dates)
    val_wo_MSU_rho1 = rebase_yearly(val_wo_MSU_rho1)
    test_wo_MSU_rho1 = pd.Series(wealth_test_dict['test_wo_MSU_rho1'], index=test_dates)
    test_wo_MSU_rho1 = rebase_yearly(test_wo_MSU_rho1)

    
    plt.figure(figsize=(12, 6))
    
    # Background shading only for Validation and Test segments.
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Test Period')
    
    # Plot DJIA monthly wealth trends
    plt.plot(val_dates, djia_monthly_val, color='r', linestyle='-', marker='o', label='DJIA')
    plt.plot(test_dates, djia_monthly_test, color='r', linestyle='-', marker='o', label=None)
    
    # Plot agent monthly wealth trends, one label per series.
    plt.plot(val_dates, val_w_MSU_dynamic, color='b', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=Dynamic)')
    plt.plot(test_dates, test_w_MSU_dynamic, color='b', linestyle='-', marker='o', label=None)
    
    plt.plot(val_dates, val_w_MSU_rho0, color='darkblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0)')
    plt.plot(test_dates, test_w_MSU_rho0, color='darkblue', linestyle='-', marker='o', label=None)
    
    plt.plot(val_dates, val_w_MSU_rho05, color='c', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=0.5)')
    plt.plot(test_dates, test_w_MSU_rho05, color='c', linestyle='-', marker='o', label=None)
    
    plt.plot(val_dates, val_w_MSU_rho1, color='steelblue', linestyle='-', marker='o', label='DeepTrader (w/ MSU & ρ=1)')
    plt.plot(test_dates, test_w_MSU_rho1, color='steelblue', linestyle='-', marker='o', label=None)

    plt.plot(val_dates, val_wo_MSU_rho0, color='limegreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0)')
    plt.plot(test_dates, test_wo_MSU_rho0, color='limegreen', linestyle='-', marker='o', label=None)

    plt.plot(val_dates, val_wo_MSU_rho05, color='g', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=0.5)')
    plt.plot(test_dates, test_wo_MSU_rho05, color='g', linestyle='-', marker='o', label=None)

    plt.plot(val_dates, val_wo_MSU_rho1, color='lawngreen', linestyle='-', marker='o', label='DeepTrader (w/o MSU & ρ=1)')
    plt.plot(test_dates, test_wo_MSU_rho1, color='lawngreen', linestyle='-', marker='o', label=None)
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth (Monthly, Yearly Rebased)", fontsize=14)
    plt.title("DeepTrader vs. DJIA (Rebased Each Year)", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper center')
    plt.tight_layout()
    plt.show()

def calculate_periodic_returns(data, period):
    """
    Resample the data to the specified period using the last value of each period,
    then compute period-over-period returns.
    """
    resampled = data.resample(period).last()
    returns = resampled.pct_change().dropna()
    return returns

def calculate_win_rate(periodic_returns, benchmark_column='DowJones'):
    """
    Calculate win rate for each agent column as the ratio of periods 
    where the agent's return is higher than the benchmark's return.
    """
    agent_columns = [col for col in periodic_returns.columns if col != benchmark_column]
    win_rates = {}
    for col in agent_columns:
        wins = (periodic_returns[col] > periodic_returns[benchmark_column]).sum()
        total = len(periodic_returns)
        win_rates[col] = wins / total
    return pd.Series(win_rates)

def main():
    # Load agent wealth arrays and flatten them
    agent_wealth_dict = load_agent_wealth()
    
    # Get full business day range and segments
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Download DJIA data and compute daily cumulative wealth (rebased to 1)
    df_djia = download_djia_data(full_days)
    djia_wealth_daily = compute_cumulative_wealth(df_djia)

    # Resample DJIA wealth for Validation and Test segments and rebase to start at 1
    djia_wealth_val = resample_series(djia_wealth_daily[val_days], TRADE_LEN)
    djia_wealth_val = djia_wealth_val / djia_wealth_val[0]
    djia_wealth_test = resample_series(djia_wealth_daily[test_days], TRADE_LEN)
    djia_wealth_test = djia_wealth_test / djia_wealth_test[0]
    
    # Prepare agent wealth segments (rebased to start at 1)
    val_w_MSU_dynamic = agent_wealth_dict['val_w_MSU_dynamic'] / agent_wealth_dict['val_w_MSU_dynamic'][0]
    val_w_MSU_rho0 = agent_wealth_dict['val_w_MSU_rho0'] / agent_wealth_dict['val_w_MSU_rho0'][0]
    val_w_MSU_rho05 = agent_wealth_dict['val_w_MSU_rho05'] / agent_wealth_dict['val_w_MSU_rho05'][0]
    val_w_MSU_rho1 = agent_wealth_dict['val_w_MSU_rho1'] / agent_wealth_dict['val_w_MSU_rho1'][0]
    val_wo_MSU_rho0 = agent_wealth_dict['val_wo_MSU_rho0'] / agent_wealth_dict['val_wo_MSU_rho0'][0]
    val_wo_MSU_rho05 = agent_wealth_dict['val_wo_MSU_rho05'] / agent_wealth_dict['val_wo_MSU_rho05'][0]
    val_wo_MSU_rho1 = agent_wealth_dict['val_wo_MSU_rho1'] / agent_wealth_dict['val_wo_MSU_rho1'][0]

    test_w_MSU_dynamic = agent_wealth_dict['test_w_MSU_dynamic'] / agent_wealth_dict['test_w_MSU_dynamic'][0]
    test_w_MSU_rho0 = agent_wealth_dict['test_w_MSU_rho0'] / agent_wealth_dict['test_w_MSU_rho0'][0]
    test_w_MSU_rho05 = agent_wealth_dict['test_w_MSU_rho05'] / agent_wealth_dict['test_w_MSU_rho05'][0]
    test_w_MSU_rho1 = agent_wealth_dict['test_w_MSU_rho1'] / agent_wealth_dict['test_w_MSU_rho1'][0]
    test_wo_MSU_rho0 = agent_wealth_dict['test_wo_MSU_rho0'] / agent_wealth_dict['test_wo_MSU_rho0'][0]
    test_wo_MSU_rho05 = agent_wealth_dict['test_wo_MSU_rho05'] / agent_wealth_dict['test_wo_MSU_rho05'][0]
    test_wo_MSU_rho1 = agent_wealth_dict['test_wo_MSU_rho1'] / agent_wealth_dict['test_wo_MSU_rho1'][0]
    
    # --- Individual calculation for metrics on the Dynamic series ---
    metrics_test_w_MSU_dynamic = calculate_metrics(test_w_MSU_dynamic.reshape(1, -1), TRADE_MODE)
    metrics_test_w_MSU_rho0 = calculate_metrics(test_w_MSU_rho0.reshape(1, -1), TRADE_MODE)
    metrics_test_w_MSU_rho05 = calculate_metrics(test_w_MSU_rho05.reshape(1, -1), TRADE_MODE)
    metrics_test_w_MSU_rho1 = calculate_metrics(test_w_MSU_rho1.reshape(1, -1), TRADE_MODE)
    metrics_test_wo_MSU_rho0 = calculate_metrics(test_wo_MSU_rho0.reshape(1, -1), TRADE_MODE)
    metrics_test_wo_MSU_rho05 = calculate_metrics(test_wo_MSU_rho05.reshape(1, -1), TRADE_MODE)
    metrics_test_wo_MSU_rho1 = calculate_metrics(test_wo_MSU_rho1.reshape(1, -1), TRADE_MODE)
    metrics_test_djia_wealth = calculate_metrics(djia_wealth_test.reshape(1,-1), TRADE_MODE)

    print("metrics_test_w_MSU_dynamic:", metrics_test_w_MSU_dynamic)
    print("metrics_test_w_MSU_rho0:", metrics_test_w_MSU_rho0)
    print("metrics_test_w_MSU_rho05:", metrics_test_w_MSU_rho05)
    print("metrics_test_w_MSU_rho1:", metrics_test_w_MSU_rho1)
    print("metrics_test_wo_MSU_rho0:", metrics_test_wo_MSU_rho0)
    print("metrics_test_wo_MSU_rho05:", metrics_test_wo_MSU_rho05)
    print("metrics_test_wo_MSU_rho1:", metrics_test_wo_MSU_rho1)
    print("metrics_test_djia_wealth:", metrics_test_djia_wealth)
    
    
    # Prepare dictionaries for plotting purposes
    wealth_val_dict = {
        'val_w_MSU_dynamic': val_w_MSU_dynamic,
        'val_w_MSU_rho0': val_w_MSU_rho0,
        'val_w_MSU_rho05': val_w_MSU_rho05,
        'val_w_MSU_rho1': val_w_MSU_rho1,
        'val_wo_MSU_rho0': val_wo_MSU_rho0,
        'val_wo_MSU_rho05': val_wo_MSU_rho05,
        'val_wo_MSU_rho1': val_wo_MSU_rho1
    }
    wealth_test_dict = {
        'test_w_MSU_dynamic': test_w_MSU_dynamic,
        'test_w_MSU_rho0': test_w_MSU_rho0,
        'test_w_MSU_rho05': test_w_MSU_rho05,
        'test_w_MSU_rho1': test_w_MSU_rho1,
        'test_wo_MSU_rho0': test_wo_MSU_rho0,
        'test_wo_MSU_rho05': test_wo_MSU_rho05,
        'test_wo_MSU_rho1': test_wo_MSU_rho1
    }
    
    # Plot the results with the full business day x-axis and background shading.
    plot_results(full_days, train_days, val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict)

    # ----- Plot Monthly Results (Directly using the 101 monthly points) -----
    plot_yearly_results(val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict)
    
    # ----- Calculate Win Rates for the Dynamic series without interpolation -----

    # For validation:
    val_dates_sampled = resample_dates(val_days, TRADE_LEN)
    djia_monthly_val = resample_series(djia_wealth_daily[val_days], TRADE_LEN)
    djia_monthly_val = djia_monthly_val / djia_monthly_val[0]
    df_val = pd.DataFrame({
        'val_w_MSU_dynamic': val_w_MSU_dynamic,
        'val_w_MSU_rho1': val_w_MSU_rho1,
        'val_wo_MSU_rho05': val_wo_MSU_rho05,
        'val_wo_MSU_rho1': val_wo_MSU_rho1,
        'DowJones': djia_monthly_val
    }, index=val_dates_sampled)
    
    # For test:
    test_dates_sampled = resample_dates(test_days, TRADE_LEN)
    djia_monthly_test = resample_series(djia_wealth_daily[test_days], TRADE_LEN)
    djia_monthly_test = djia_monthly_test / djia_monthly_test[0]
    df_test = pd.DataFrame({
        'test_w_MSU_dynamic': test_w_MSU_dynamic,
        'val_w_MSU_rho1': val_w_MSU_rho1,
        'val_wo_MSU_rho05': val_wo_MSU_rho05,
        'val_wo_MSU_rho1': val_wo_MSU_rho1,
        'DowJones': djia_monthly_test
    }, index=test_dates_sampled)
    
    # Calculate periodic returns for validation using different period codes:
    monthly_returns_val = calculate_periodic_returns(df_val, 'ME')
    quarterly_returns_val = calculate_periodic_returns(df_val, 'QE')
    semi_annual_returns_val = calculate_periodic_returns(df_val, '6ME')
    annual_returns_val = calculate_periodic_returns(df_val, 'YE')

    print("Validation Periodic Returns for Series:")
    print(f"Annual Returns: {annual_returns_val}")
    
    # Calculate win rates for validation
    monthly_win_rate_val = calculate_win_rate(monthly_returns_val, benchmark_column='DowJones')
    quarterly_win_rate_val = calculate_win_rate(quarterly_returns_val, benchmark_column='DowJones')
    semi_annual_win_rate_val = calculate_win_rate(semi_annual_returns_val, benchmark_column='DowJones')
    annual_win_rate_val = calculate_win_rate(annual_returns_val, benchmark_column='DowJones')
    
    print("Validation Period Win Rates for Series:")
    print(f"Monthly Win Rate: {monthly_win_rate_val}")
    print(f"Quarterly Win Rate: {quarterly_win_rate_val}")
    print(f"Semi-annual Win Rate: {semi_annual_win_rate_val}")
    print(f"Annual Win Rate: {annual_win_rate_val}")
    
    # Calculate periodic returns and win rates for test (using the 101 monthly points)
    monthly_returns_test = calculate_periodic_returns(df_test, 'ME')
    quarterly_returns_test = calculate_periodic_returns(df_test, 'QE')
    semi_annual_returns_test = calculate_periodic_returns(df_test, '6ME')
    annual_returns_test = calculate_periodic_returns(df_test, 'YE')

    print("Test Periodic Returns for Series:")
    print(f"Annual Returns: {annual_returns_test}")
    
    monthly_win_rate_test = calculate_win_rate(monthly_returns_test)
    quarterly_win_rate_test = calculate_win_rate(quarterly_returns_test)
    semi_annual_win_rate_test = calculate_win_rate(semi_annual_returns_test)
    annual_win_rate_test = calculate_win_rate(annual_returns_test)
    
    print("Test Period Win Rates for Series:")
    print(f"Monthly Win Rate: {monthly_win_rate_test}")
    print(f"Quarterly Win Rate: {quarterly_win_rate_test}")
    print(f"Semi-annual Win Rate: {semi_annual_win_rate_test}")
    print(f"Annual Win Rate: {annual_win_rate_test}")

if __name__ == "__main__":
    main()
