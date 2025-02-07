import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from utils.functions import calculate_metrics

logging.basicConfig(level=logging.INFO)

TRADE_MODE = "M"
TRADE_LEN = 21
TARGET_DIR = "."
START_DATE = "2000-01-01"
END_DATE = "2023-12-31"

def load_agent_wealth():
    """
    Load and flatten agent wealth arrays for validation and test.
    Expected shape of each loaded array is (1, 101).
    """
    val_w_MSU_dynamic = np.load('agent_wealth_val_w_MSU_dynamic.npy').flatten()
    val_w_MSU_rho0    = np.load('agent_wealth_val_w_MSU_rho0.npy').flatten()
    val_w_MSU_rho05   = np.load('agent_wealth_val_w_MSU_rho05.npy').flatten()
    val_w_MSU_rho1    = np.load('agent_wealth_val_w_MSU_rho1.npy').flatten()
    
    test_w_MSU_dynamic = np.load('agent_wealth_test_w_MSU_dynamic.npy').flatten()
    test_w_MSU_rho0    = np.load('agent_wealth_test_w_MSU_rho0.npy').flatten()
    test_w_MSU_rho05   = np.load('agent_wealth_test_w_MSU_rho05.npy').flatten()
    test_w_MSU_rho1    = np.load('agent_wealth_test_w_MSU_rho1.npy').flatten()
    
    return {
        'val_w_MSU_dynamic': val_w_MSU_dynamic,
        'val_w_MSU_rho0': val_w_MSU_rho0,
        'val_w_MSU_rho05': val_w_MSU_rho05,
        'val_w_MSU_rho1': val_w_MSU_rho1,
        'test_w_MSU_dynamic': test_w_MSU_dynamic,
        'test_w_MSU_rho0': test_w_MSU_rho0,
        'test_w_MSU_rho05': test_w_MSU_rho05,
        'test_w_MSU_rho1': test_w_MSU_rho1
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
    logging.info(f"Total business days: {total_days}")
    
    train_days = full_days[0:2043]
    val_days   = full_days[2043:4151]
    test_days  = full_days[4151:6260]
    
    logging.info(f"Training days: {len(train_days)}")
    logging.info(f"Validation days: {len(val_days)}")
    logging.info(f"Test days: {len(test_days)}")
    
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
    wealth_rebased = wealth_daily / wealth_daily.iloc[0]  # rebasing: divide by the first value
    return wealth_rebased

def resample_series(series, num_points=101):
    """
    Resample a pandas Series to a fixed number of points.
    This function simply selects num_points evenly spaced indices from the series.
    """
    indices = np.linspace(0, len(series) - 1, num_points, dtype=int)
    return series.iloc[indices].values

def calculate_all_metrics(series_dict, label_suffix=""):
    """
    Calculate metrics for each wealth series using the calculate_metrics function.
    """
    metrics_results = {}
    for key, series in series_dict.items():
        reshaped = series.reshape(1, -1)  # ensure shape is (1, num_points)
        metrics_results[key + label_suffix] = calculate_metrics(reshaped, TRADE_MODE)
    return metrics_results

def plot_results(full_days, train_days, val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict):
    """
    Plot the full timeline with background shading for Training, Validation, and Test segments.
    For Validation and Test segments, overlay DJIA and agent wealth trends.
    """
    plt.figure(figsize=(14, 7))
    
    # Plot background shading for each segment
    plt.axvspan(train_days[0], train_days[-1], facecolor='gray', alpha=0.1, label='Training Period')
    plt.axvspan(val_days[0], val_days[-1], facecolor='gray', alpha=0.3, label='Validation Period')
    plt.axvspan(test_days[0], test_days[-1], facecolor='gray', alpha=0.5, label='Test Period')
    
    def resample_dates(dates, num_points=101):
        indices = np.linspace(0, len(dates) - 1, num_points, dtype=int)
        return dates[indices]
    
    val_dates_sampled = resample_dates(val_days)
    test_dates_sampled = resample_dates(test_days)
    
    # Resample DJIA wealth for Validation and Test segments and rebase to start at 1
    djia_wealth_val = resample_series(djia_wealth_daily[val_days])
    djia_wealth_val = djia_wealth_val / djia_wealth_val[0]
    djia_wealth_test = resample_series(djia_wealth_daily[test_days])
    djia_wealth_test = djia_wealth_test / djia_wealth_test[0]
    
    # Plot DJIA wealth trends (black)
    plt.plot(val_dates_sampled, djia_wealth_val, color='black', linestyle='-', marker='o', label='DJIA (Validation)')
    plt.plot(test_dates_sampled, djia_wealth_test, color='black', linestyle='--', marker='o', label='DJIA (Test)')
    
    # Plot agent wealth curves for Validation
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_dynamic'], color='blue', linestyle='-', marker='o', label='Dynamic (Val)')
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho0'],    color='red', linestyle='--', marker='s', label='ρ=0 (Val)')
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho05'],   color='green', linestyle='-.', marker='^', label='ρ=0.5 (Val)')
    plt.plot(val_dates_sampled, wealth_val_dict['val_w_MSU_rho1'],    color='orange', linestyle=':', marker='D', label='ρ=1 (Val)')
    
    # Plot agent wealth curves for Test
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_dynamic'], color='blue', linestyle='-', marker='o', label='Dynamic (Test)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho0'],    color='red', linestyle='--', marker='s', label='ρ=0 (Test)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho05'],   color='green', linestyle='-.', marker='^', label='ρ=0.5 (Test)')
    plt.plot(test_dates_sampled, wealth_test_dict['test_w_MSU_rho1'],    color='orange', linestyle=':', marker='D', label='ρ=1 (Test)')
    
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.title("DeepTrader vs. DJIA", fontsize=16)
    plt.grid(True)
    plt.legend(fontsize=10, loc='upper left')
    plt.tight_layout()
    plt.show()

def main():
    # Load agent wealth arrays and flatten them
    agent_wealth = load_agent_wealth()
    
    # Get full business day range and segments
    full_days, train_days, val_days, test_days = get_business_day_segments()
    
    # Download DJIA data and compute daily cumulative wealth (rebased to 1)
    df_djia = download_djia_data(full_days)
    djia_wealth_daily = compute_cumulative_wealth(df_djia)
    
    # Prepare agent wealth segments for Validation and Test (rebased to start at 1)
    wealth_val_dict = {
        'val_w_MSU_dynamic': agent_wealth['val_w_MSU_dynamic'] / agent_wealth['val_w_MSU_dynamic'][0],
        'val_w_MSU_rho0':    agent_wealth['val_w_MSU_rho0'] / agent_wealth['val_w_MSU_rho0'][0],
        'val_w_MSU_rho05':   agent_wealth['val_w_MSU_rho05'] / agent_wealth['val_w_MSU_rho05'][0],
        'val_w_MSU_rho1':    agent_wealth['val_w_MSU_rho1'] / agent_wealth['val_w_MSU_rho1'][0]
    }
    wealth_test_dict = {
        'test_w_MSU_dynamic': agent_wealth['test_w_MSU_dynamic'] / agent_wealth['test_w_MSU_dynamic'][0],
        'test_w_MSU_rho0':    agent_wealth['test_w_MSU_rho0'] / agent_wealth['test_w_MSU_rho0'][0],
        'test_w_MSU_rho05':   agent_wealth['test_w_MSU_rho05'] / agent_wealth['test_w_MSU_rho05'][0],
        'test_w_MSU_rho1':    agent_wealth['test_w_MSU_rho1'] / agent_wealth['test_w_MSU_rho1'][0]
    }
    
    # Calculate metrics for each series using calculate_metrics.
    metrics_val = calculate_all_metrics({
        'val_w_MSU_dynamic': wealth_val_dict['val_w_MSU_dynamic'],
        'val_w_MSU_rho0': wealth_val_dict['val_w_MSU_rho0'],
        'val_w_MSU_rho05': wealth_val_dict['val_w_MSU_rho05'],
        'val_w_MSU_rho1': wealth_val_dict['val_w_MSU_rho1']
    }, label_suffix=" (Val)")
    
    metrics_test = calculate_all_metrics({
        'test_w_MSU_dynamic': wealth_test_dict['test_w_MSU_dynamic'],
        'test_w_MSU_rho0': wealth_test_dict['test_w_MSU_rho0'],
        'test_w_MSU_rho05': wealth_test_dict['test_w_MSU_rho05'],
        'test_w_MSU_rho1': wealth_test_dict['test_w_MSU_rho1']
    }, label_suffix=" (Test)")
    
    # Also calculate metrics for DJIA wealth segments.
    djia_wealth_val = resample_series(djia_wealth_daily[val_days])
    djia_wealth_val = djia_wealth_val / djia_wealth_val[0]
    djia_wealth_test = resample_series(djia_wealth_daily[test_days])
    djia_wealth_test = djia_wealth_test / djia_wealth_test[0]
    
    metrics_djia_val = calculate_metrics(djia_wealth_val.reshape(1, -1), TRADE_MODE)
    metrics_djia_test = calculate_metrics(djia_wealth_test.reshape(1, -1), TRADE_MODE)
    
    logging.info("Validation Metrics:")
    logging.info(metrics_val)
    logging.info(f"DJIA (Val): {metrics_djia_val}")
    logging.info("Test Metrics:")
    logging.info(metrics_test)
    logging.info(f"DJIA (Test): {metrics_djia_test}")
    
    # Plot the results with the full business day x-axis and different background colors.
    plot_results(full_days, train_days, val_days, test_days, djia_wealth_daily, wealth_val_dict, wealth_test_dict)

if __name__ == "__main__":
    main()
