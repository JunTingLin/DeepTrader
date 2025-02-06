import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from utils.functions import calculate_metrics


logging.basicConfig(level=logging.INFO)

def main():

    msu_dynamic = np.array([[1.0, 1.01020549, 0.99331695, 1.00888751, 0.91067623, 0.94109122,
                            0.99255158, 0.90993389, 0.92367026, 0.8669656, 0.79741821, 0.7472606,
                            0.76437693, 0.81799155, 0.82616172, 0.80434067, 0.79941158, 0.86251101,
                            0.87476584, 0.88833666, 0.9055071, 0.88670873, 0.97932383, 0.90542295,
                            0.92803245, 0.93608238, 0.87691841, 0.87936126, 0.84496004, 0.87683084,
                            0.90423617, 0.95408367, 0.98435162, 0.9835528, 0.96263212, 0.93534184,
                            0.91410835, 0.92525288, 0.93584364, 1.0360557, 1.02484125, 0.97727914,
                            0.94051327, 0.92342561, 0.88508903]])
    

    # Flatten arrays to shape (48,) if necessary
    if msu_dynamic.ndim == 2 and msu_dynamic.shape[0] == 1:
        msu_dynamic = msu_dynamic[0]
        logging.info(f"msu_dynamic shape: {msu_dynamic.shape}")
    # if msu_rho0.ndim == 2 and msu_rho0.shape[0] == 1:
    #     msu_rho0 = msu_rho0[0]
    # if msu_rho05.ndim == 2 and msu_rho05.shape[0] == 1:
    #     msu_rho05 = msu_rho05[0]
    # if msu_rho1.ndim == 2 and msu_rho1.shape[0] == 1:
    #     msu_rho1 = msu_rho1[0]
    
    # Generate all business days
    all_days = pd.bdate_range(start='2010-01-01', end='2025-01-31')
    # Select 1000 test days
    test_daily_dates = all_days[3000:3936]
    logging.info(f"Test period length = {len(test_daily_dates)} days")
    logging.info(f"Start date = {test_daily_dates[0]}")
    logging.info(f"End date = {test_daily_dates[-1]}")
    
    # Download DJIA data directly using yfinance for the test period
    djia_ticker = yf.Ticker("^DJI")
    df_djia = djia_ticker.history(start=test_daily_dates[0].strftime("%Y-%m-%d"),
                                  end=test_daily_dates[-1].strftime("%Y-%m-%d"))
    # Remove timezone information from the index
    df_djia.index = df_djia.index.tz_localize(None)

    # Reindex to the test daily dates and forward-fill missing values
    df_djia = df_djia.reindex(test_daily_dates)
    df_djia.replace(0, np.nan, inplace=True)
    df_djia.ffill(inplace=True)
    df_djia.bfill(inplace=True)
    
    # Compute cumulative wealth using Close prices (Buy & Hold strategy)
    djia_close = df_djia["Close"].copy()
    daily_return = djia_close.pct_change().fillna(0.0)
    djia_wealth_daily = (1.0 + daily_return).cumprod()
    # Resample DJIA wealth to 21-day intervals to match the DeepTrader sample dates
    djia_wealth = djia_wealth_daily.iloc[::21].reset_index(drop=True)
    logging.info(f"DJIA cumulative return shape: {djia_wealth.shape}")
    
    # x_sub: Dates corresponding to the 48 sample points (every 21 days)
    day_list = list(range(0, len(test_daily_dates), 21))  # e.g., 0, 21, 42, ... (~48 points)
    x_sub = test_daily_dates[day_list]
    
    # Plot multiple curves 
    plt.figure(figsize=(10, 6))
    
    # Plot DJIA (21-day data)
    plt.plot(x_sub, djia_wealth, color='black', linestyle='--', marker='o',
             label='DJIA (Buy & Hold, 21-day)')
    
    # Plot DeepTrader results with different MSU settings
    plt.plot(x_sub, msu_dynamic, color='blue', linestyle='-', marker='o',
             label='DeepTrader (MSU dynamic ρ)')
    # plt.plot(x_sub, msu_rho0, color='red', linestyle='--', marker='s',
    #          label='DeepTrader (MSU, ρ=0)')
    # plt.plot(x_sub, msu_rho05, color='green', linestyle='-.', marker='^',
    #          label='DeepTrader (MSU, ρ=0.5)')
    # plt.plot(x_sub, msu_rho1, color='orange', linestyle=':', marker='D',
    #          label='DeepTrader (MSU, ρ=1)')
    
    plt.title(f"DeepTrader vs. DJIA ({len(test_daily_dates)} Days Test)", fontsize=16)
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Cumulative Wealth", fontsize=14)
    plt.grid(True)
    plt.legend(fontsize=12)
    plt.tight_layout()
    plt.show()

    djia_wealth = djia_wealth.to_frame().T.to_numpy()
    metrics = calculate_metrics(djia_wealth, "M")
    print(metrics)

if __name__ == "__main__":
    main()
