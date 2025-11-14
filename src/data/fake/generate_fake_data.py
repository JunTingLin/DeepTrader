#!/usr/bin/env python3
"""
Generate fake stock data for testing DeepTrader
- Stock A: Bear market (gradual decline over 10 years)
- Stock B: Bull market (gradual rise from low point)
- Market: (A + B) / 2, overall bullish
- Cross point: around middle of test period
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Constants
START_DATE = '2015-01-01'
END_DATE = '2025-03-31'
SEED = 42

# Period definitions
PERIODS = {
    "Training": {"start_idx": 0, "end_idx": 1303, "days": 1304},
    "Validation": {"start_idx": 1304, "end_idx": 2086, "days": 783},
    "Test": {"start_idx": 2087, "end_idx": 2672, "days": 586}
}

def generate_business_days(start_date, end_date):
    """Generate business days using pandas"""
    dates = pd.bdate_range(start=start_date, end=end_date)
    return dates.to_pydatetime(), len(dates)

def generate_bear_market(num_days, start_price=100.0, end_price=60.0, volatility=0.012, seed=42):
    """
    Generate bear market: gradual decline with noise

    Parameters:
    - num_days: total number of days
    - start_price: starting price
    - end_price: ending price
    - volatility: daily volatility
    - seed: random seed
    """
    np.random.seed(seed)

    # Create linear trend from start to end
    linear_trend = np.linspace(start_price, end_price, num_days)

    # Add random walk noise
    noise = np.random.normal(0, volatility * start_price, num_days)
    noise = np.cumsum(noise) * 0.1  # Cumulative noise with dampening

    # Combine trend and noise
    prices = linear_trend + noise

    # Ensure all prices are positive and within reasonable bounds
    prices = np.clip(prices, end_price * 0.7, start_price * 1.3)

    return prices

def generate_bull_market(num_days, start_price=50.0, end_price=140.0, volatility=0.015, seed=43):
    """
    Generate bull market: gradual rise with noise

    Parameters:
    - num_days: total number of days
    - start_price: starting price (low point)
    - end_price: ending price
    - volatility: daily volatility
    - seed: random seed
    """
    np.random.seed(seed)

    # Create linear trend from start to end
    linear_trend = np.linspace(start_price, end_price, num_days)

    # Add random walk noise
    noise = np.random.normal(0, volatility * start_price, num_days)
    noise = np.cumsum(noise) * 0.1  # Cumulative noise with dampening

    # Combine trend and noise
    prices = linear_trend + noise

    # Ensure all prices are positive and within reasonable bounds
    prices = np.clip(prices, start_price * 0.6, end_price * 1.4)

    return prices

def calculate_ror(prices):
    """
    Calculate inter-day return of return: (close_t - close_{t-1}) / close_{t-1}
    First day is 0
    """
    ror = np.zeros(len(prices))
    for i in range(1, len(prices)):
        ror[i] = (prices[i] - prices[i-1]) / prices[i-1]

    # Handle inf and nan
    ror = np.where(np.isinf(ror), 0, ror)
    ror = np.where(np.isnan(ror), 0, ror)

    return ror

def plot_stocks(dates, price_a, price_b, periods, output_file='stocks_AB.png'):
    """Plot stocks A and B with period backgrounds"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert dates to matplotlib format
    dates_mpl = mdates.date2num(dates)

    # Plot background for periods
    colors = {'Training': '#f0f0f0', 'Validation': '#d0d0d0', 'Test': '#b0b0b0'}
    for period_name, period_info in periods.items():
        start_idx = period_info['start_idx']
        end_idx = period_info['end_idx']
        ax.axvspan(dates_mpl[start_idx], dates_mpl[end_idx],
                   alpha=0.3, color=colors[period_name], label=f'{period_name} Period')

    # Plot price lines
    ax.plot(dates, price_a, 'r-', linewidth=2, label='Stock A (Bear)', alpha=0.8)
    ax.plot(dates, price_b, 'b-', linewidth=2, label='Stock B (Bull)', alpha=0.8)

    # Format
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Simulated Stocks: A (Bear) vs B (Bull)', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def plot_market(dates, market_prices, periods, output_file='market_data.png'):
    """Plot market data (A+B)/2 with period backgrounds"""
    fig, ax = plt.subplots(figsize=(14, 7))

    # Convert dates to matplotlib format
    dates_mpl = mdates.date2num(dates)

    # Plot background for periods
    colors = {'Training': '#f0f0f0', 'Validation': '#d0d0d0', 'Test': '#b0b0b0'}
    for period_name, period_info in periods.items():
        start_idx = period_info['start_idx']
        end_idx = period_info['end_idx']
        ax.axvspan(dates_mpl[start_idx], dates_mpl[end_idx],
                   alpha=0.3, color=colors[period_name], label=f'{period_name} Period')

    # Plot market line
    ax.plot(dates, market_prices, 'g-', linewidth=2, label='Market (A+B)/2', alpha=0.8)

    # Format
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.set_title('Simulated Market Data: (A+B)/2', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    plt.xticks(rotation=0)

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

def main():
    print("="*60)
    print("Generating Fake Stock Data for DeepTrader")
    print("="*60)

    # Generate business days
    dates, num_days = generate_business_days(START_DATE, END_DATE)
    print(f"\nTotal business days: {num_days}")
    print(f"Date range: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")

    # Verify periods
    print("\nPeriod breakdown:")
    for period_name, period_info in PERIODS.items():
        print(f"  {period_name}:")
        print(f"    Start Index = {period_info['start_idx']}")
        print(f"    End Index = {period_info['end_idx']}")
        print(f"    Total Business Days = {period_info['days']}")

    # Generate Stock A: Bear market (100 -> 80)
    # Gradual decline, cross point target around test middle (index ~2380)
    # At index 2380 (89% through): A should be ~82.2
    print("\nGenerating Stock A (Bear Market)...")
    price_a = generate_bear_market(
        num_days=num_days,
        start_price=100.0,
        end_price=80.0,
        volatility=0.008,  # Lower volatility for smoother trend
        seed=SEED
    )
    print(f"  Price range: ${price_a.min():.2f} - ${price_a.max():.2f}")
    print(f"  Start: ${price_a[0]:.2f}, End: ${price_a[-1]:.2f}")

    # Generate Stock B: Bull market (65 -> 100)
    # Strong rise to ensure market (A+B)/2 is bullish and cross A around test middle
    # At index 2380 (89% through): B should be ~96.15
    # Market at 2380: (82.2 + 96.15)/2 = 89.17
    # Market overall: (100+65)/2=82.5 -> (80+100)/2=90, up 9%
    print("\nGenerating Stock B (Bull Market)...")
    price_b = generate_bull_market(
        num_days=num_days,
        start_price=65.0,
        end_price=100.0,
        volatility=0.008,  # Matching volatility
        seed=SEED+1
    )
    print(f"  Price range: ${price_b.min():.2f} - ${price_b.max():.2f}")
    print(f"  Start: ${price_b[0]:.2f}, End: ${price_b[-1]:.2f}")

    # Calculate Market: (A + B) / 2
    market_prices = (price_a + price_b) / 2
    market_return = (market_prices[-1] / market_prices[0] - 1) * 100
    print(f"\nMarket Data (A+B)/2:")
    print(f"  Price range: ${market_prices.min():.2f} - ${market_prices.max():.2f}")
    print(f"  Start: ${market_prices[0]:.2f}, End: ${market_prices[-1]:.2f}")
    print(f"  Total return: {market_return:+.2f}%")

    # Check cross point
    cross_points = np.where(np.diff(np.sign(price_a - price_b)))[0]
    if len(cross_points) > 0:
        print(f"\nCross points found at indices: {cross_points}")
        for cp in cross_points:
            period_name = 'Unknown'
            for pname, pinfo in PERIODS.items():
                if pinfo['start_idx'] <= cp <= pinfo['end_idx']:
                    period_name = pname
                    break
            print(f"  Index {cp}: {dates[cp].strftime('%Y-%m-%d')} ({period_name})")

    # Calculate RoR for both stocks
    print("\nCalculating Rate of Return...")
    ror_a = calculate_ror(price_a)
    ror_b = calculate_ror(price_b)

    # Prepare data arrays
    # stocks_data: (num_stocks=2, num_days, num_features=1)
    stocks_data = np.zeros((2, num_days, 1))
    stocks_data[0, :, 0] = price_a  # Stock A
    stocks_data[1, :, 0] = price_b  # Stock B

    # market_data: (num_days, 1)
    market_data = market_prices.reshape(-1, 1)

    # ror: (num_stocks=2, num_days)
    ror = np.vstack([ror_a, ror_b])

    # industry_classification: correlation matrix (2, 2)
    # Use first 1000 days like the original script
    correlation_matrix = np.corrcoef(ror[:, :1000])

    # Save all data
    print("\nSaving data files...")
    np.save('stocks_data.npy', stocks_data)
    print(f"  stocks_data.npy: shape {stocks_data.shape}")

    np.save('market_data.npy', market_data)
    print(f"  market_data.npy: shape {market_data.shape}")

    np.save('ror.npy', ror)
    print(f"  ror.npy: shape {ror.shape}")

    np.save('industry_classification.npy', correlation_matrix)
    print(f"  industry_classification.npy: shape {correlation_matrix.shape}")
    print(f"\nCorrelation matrix:")
    print(correlation_matrix)

    # Generate plots
    print("\nGenerating visualizations...")
    plot_stocks(dates, price_a, price_b, PERIODS, 'stocks_AB.png')
    plot_market(dates, market_prices, PERIODS, 'market_data.png')

    # Summary statistics
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Stock A (Bear): ${price_a[0]:.2f} -> ${price_a[-1]:.2f} ({(price_a[-1]/price_a[0]-1)*100:+.2f}%)")
    print(f"Stock B (Bull): ${price_b[0]:.2f} -> ${price_b[-1]:.2f} ({(price_b[-1]/price_b[0]-1)*100:+.2f}%)")
    print(f"Market (A+B)/2: ${market_prices[0]:.2f} -> ${market_prices[-1]:.2f} ({market_return:+.2f}%)")
    print(f"\nCorrelation between A and B (first 1000 days): {correlation_matrix[0,1]:.4f}")
    print("\nFiles generated:")
    print("  - stocks_data.npy")
    print("  - market_data.npy")
    print("  - ror.npy")
    print("  - industry_classification.npy")
    print("  - stocks_AB.png")
    print("  - market_data.png")
    print("="*60)

if __name__ == '__main__':
    main()
