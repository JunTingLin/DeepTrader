"""
Plot ASU (stocks) and MSU (market) features from specified data files.
Supports both market_data.npy (MSU features) and stocks_data.npy (ASU features).

Usage:
    python plot_features.py --data_dir src/data/DJIA/feature34-Inter-P532
    python plot_features.py --market_file path/to/market_data.npy --stocks_file path/to/stocks_data.npy
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from pathlib import Path

# MSU (Market State Unit) Feature Names
# Based on deeptrader_market.py feature order
MSU_FEATURE_NAMES = [
    'BAMLCC0A4BBBTRIV',      # 1: BBB US Corporate Index
    'BAMLCC0A0CMTRIV',       # 2: US Corp Master Total Return Index
    'BAMLCC0A1AAATRIV',      # 3: AAA US Corporate Index
    'BAMLHYH0A3CMTRIV',      # 4: US High Yield Master II
    'DGS10',                 # 5: 10-Year Treasury Constant Maturity Rate
    'DGS30',                 # 6: 30-Year Treasury Constant Maturity Rate
    'DJI_Open',              # 7: Dow Jones Industrial Average
    'DJI_High',              # 8
    'DJI_Low',               # 9
    'DJI_Close',             # 10
    'DJI_Adj_Close',         # 11
    'DJI_Volume',            # 12
    'xauusd_Open',           # 13: Gold
    'xauusd_High',           # 14
    'xauusd_Low',            # 15
    'xauusd_Close',          # 16
    'VIX_Open',              # 17: Volatility Index
    'VIX_High',              # 18
    'VIX_Low',               # 19
    'VIX_Close',             # 20
    'VIX_Adj_Close',         # 21
    'GSPC_Open',             # 22: S&P 500
    'GSPC_High',             # 23
    'GSPC_Low',              # 24
    'GSPC_Close',            # 25
    'GSPC_Adj_Close',        # 26
    'GSPC_Volume',           # 27
]

# ASU (Agent State Unit) Feature Names
# Based on deeptrader_data_common.py feature order
ASU_FEATURE_NAMES = [
    'Open', 'High', 'Low', 'Close', 'Volume',                                      # 1-5: OHLCV
    'MA20', 'MA60', 'RSI', 'MACD_Signal', 'K', 'D',                               # 6-11: Technical indicators
    'BBands_Upper', 'BBands_Middle', 'BBands_Lower',                              # 12-14: Bollinger Bands
    'Alpha001', 'Alpha002', 'Alpha003', 'Alpha004', 'Alpha006', 'Alpha012',       # 15-20: Alpha factors
    'Alpha019', 'Alpha033', 'Alpha038', 'Alpha040', 'Alpha044', 'Alpha045',       # 21-26
    'Alpha046', 'Alpha051', 'Alpha052', 'Alpha053', 'Alpha054', 'Alpha056',       # 27-32
    'Alpha068', 'Alpha085',                                                        # 33-34
]

# DJIA stock symbols (28 stocks based on feature34-Inter)
DJIA_STOCKS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA"
]


def create_date_range(num_days, start_date='2015-01-01'):
    """Create business day date range"""
    dates = pd.bdate_range(start=start_date, periods=num_days)
    return dates


def plot_msu_features(market_data, output_dir, start_date='2015-01-01', normalize_method='minmax'):
    """
    Plot MSU (market) features

    Args:
        market_data: numpy array of shape (num_days, num_features)
        output_dir: directory to save plots
        start_date: start date for x-axis
        normalize_method: 'minmax' (0-1 normalization), 'none' (raw data), or 'zscore' (z-score normalization)
    """
    num_days, num_features = market_data.shape
    dates = create_date_range(num_days, start_date)

    print(f"\n{'='*60}")
    print(f"Plotting MSU (Market) Features")
    print(f"{'='*60}")
    print(f"Data shape: {market_data.shape}")
    print(f"Number of features: {num_features}")
    print(f"Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"Normalization: {normalize_method}")

    # Group features by category
    feature_groups = {
        'Bond/Treasury Indices': list(range(0, 6)),
        'DJI (Dow Jones)': list(range(6, 12)),
        'Gold (xauusd)': list(range(12, 16)),
        'VIX (Volatility Index)': list(range(16, 21)),
        'S&P 500 (GSPC)': list(range(21, 27)),
    }

    # Determine title suffix based on normalization
    if normalize_method == 'minmax':
        norm_suffix = ' (Min-Max Normalized 0-1)'
    elif normalize_method == 'zscore':
        norm_suffix = ' (Z-Score Normalized)'
    else:
        norm_suffix = ' (Raw Values)'

    # 1. Plot by category
    fig, axes = plt.subplots(len(feature_groups), 1, figsize=(18, 20))
    fig.suptitle(f'MSU Features by Category{norm_suffix}', fontsize=18, y=0.995)

    for idx, (group_name, feature_indices) in enumerate(feature_groups.items()):
        ax = axes[idx]

        for feat_idx in feature_indices:
            if feat_idx < len(MSU_FEATURE_NAMES):
                label = MSU_FEATURE_NAMES[feat_idx]
                feature_data = market_data[:, feat_idx]

                # Apply normalization based on method
                if normalize_method == 'minmax':
                    # Min-Max normalization (0-1)
                    data_range = feature_data.max() - feature_data.min()
                    if data_range > 1e-8:
                        plot_data = (feature_data - feature_data.min()) / data_range
                    else:
                        plot_data = feature_data
                elif normalize_method == 'zscore':
                    # Z-score normalization
                    mean = feature_data.mean()
                    std = feature_data.std()
                    if std > 1e-8:
                        plot_data = (feature_data - mean) / std
                    else:
                        plot_data = feature_data - mean
                else:
                    # No normalization - raw data
                    plot_data = feature_data

                ax.plot(dates, plot_data, label=label, linewidth=1.5, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        if normalize_method == 'minmax':
            ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
        elif normalize_method == 'zscore':
            ax.set_ylabel('Z-Score', fontsize=12)
        else:
            ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{group_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best')

    plt.tight_layout()
    suffix = '_normalized' if normalize_method == 'minmax' else ('_zscore' if normalize_method == 'zscore' else '_raw')
    output_file = output_dir / f'msu_features_by_category{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()

    # 2. Plot all features in grid (original scale)
    n_features = min(num_features, len(MSU_FEATURE_NAMES))
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 3))
    fig.suptitle('MSU Features (Original Scale)', fontsize=18, y=0.995)

    axes_flat = axes.flatten() if n_rows > 1 else [axes] if n_cols == 1 else axes

    for i in range(n_features):
        ax = axes_flat[i]
        label = MSU_FEATURE_NAMES[i] if i < len(MSU_FEATURE_NAMES) else f'Feature {i+1}'

        ax.plot(dates, market_data[:, i], color='steelblue', linewidth=1, alpha=0.7)
        ax.set_title(label, fontsize=11, fontweight='bold')
        ax.set_xlabel('Date', fontsize=9)
        ax.set_ylabel('Value', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='both', which='major', labelsize=8)

    # Hide unused subplots
    for i in range(n_features, len(axes_flat)):
        axes_flat[i].set_visible(False)

    plt.tight_layout()
    output_file = output_dir / 'msu_features_grid.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_asu_features_for_stock(stock_data, stock_name, output_dir, start_date='2015-01-01', normalize_method='none'):
    """
    Plot ASU features for a single stock

    Args:
        stock_data: numpy array of shape (num_days, num_features)
        stock_name: name of the stock
        output_dir: directory to save plots
        start_date: start date for x-axis
        normalize_method: 'minmax' (0-1 normalization), 'none' (raw data), or 'zscore' (z-score normalization)
    """
    num_days, num_features = stock_data.shape
    dates = create_date_range(num_days, start_date)

    # Group features by category
    feature_groups = {
        'OHLCV': list(range(0, 5)),
        'Technical Indicators': list(range(5, 14)),
        'Alpha Factors (1-10)': list(range(14, 24)),
        'Alpha Factors (11-20)': list(range(24, 34)),
    }

    # Determine title suffix based on normalization
    if normalize_method == 'minmax':
        norm_suffix = ' (Min-Max Normalized 0-1)'
    elif normalize_method == 'zscore':
        norm_suffix = ' (Z-Score Normalized)'
    else:
        norm_suffix = ' (Raw Values)'

    # Plot by category
    fig, axes = plt.subplots(len(feature_groups), 1, figsize=(18, 16))
    fig.suptitle(f'ASU Features for {stock_name}{norm_suffix}', fontsize=18, y=0.995)

    for idx, (group_name, feature_indices) in enumerate(feature_groups.items()):
        ax = axes[idx]

        for feat_idx in feature_indices:
            if feat_idx < num_features and feat_idx < len(ASU_FEATURE_NAMES):
                label = ASU_FEATURE_NAMES[feat_idx]
                feature_data = stock_data[:, feat_idx]

                # Apply normalization based on method
                if normalize_method == 'minmax':
                    # Min-Max normalization (0-1)
                    data_range = feature_data.max() - feature_data.min()
                    if data_range > 1e-8:
                        plot_data = (feature_data - feature_data.min()) / data_range
                    else:
                        plot_data = feature_data
                elif normalize_method == 'zscore':
                    # Z-score normalization
                    mean = feature_data.mean()
                    std = feature_data.std()
                    if std > 1e-8:
                        plot_data = (feature_data - mean) / std
                    else:
                        plot_data = feature_data - mean
                else:
                    # No normalization - raw data
                    plot_data = feature_data

                ax.plot(dates, plot_data, label=label, linewidth=1.2, alpha=0.8)

        ax.set_xlabel('Date', fontsize=12)
        if normalize_method == 'minmax':
            ax.set_ylabel('Normalized Value (0-1)', fontsize=12)
        elif normalize_method == 'zscore':
            ax.set_ylabel('Z-Score', fontsize=12)
        else:
            ax.set_ylabel('Value', fontsize=12)
        ax.set_title(f'{group_name}', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9, loc='best', ncol=2)

    plt.tight_layout()
    suffix = '_normalized' if normalize_method == 'minmax' else ('_zscore' if normalize_method == 'zscore' else '_raw')
    output_file = output_dir / f'asu_features_{stock_name}{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def plot_asu_features_comparison(stocks_data, stock_names, feature_idx, feature_name, output_dir, start_date='2015-01-01', normalize_method='none'):
    """
    Plot a specific feature across all stocks

    Args:
        stocks_data: numpy array of shape (num_stocks, num_days, num_features)
        stock_names: list of stock names
        feature_idx: index of feature to plot
        feature_name: name of the feature
        output_dir: directory to save plots
        start_date: start date for x-axis
        normalize_method: 'minmax' (0-1 normalization), 'none' (raw data), or 'zscore' (z-score normalization)
    """
    num_stocks, num_days, num_features = stocks_data.shape
    dates = create_date_range(num_days, start_date)

    # Determine title suffix based on normalization
    if normalize_method == 'minmax':
        norm_suffix = ' (Min-Max Normalized 0-1)'
    elif normalize_method == 'zscore':
        norm_suffix = ' (Z-Score Normalized)'
    else:
        norm_suffix = ' (Raw Values)'

    fig, ax = plt.subplots(1, 1, figsize=(18, 10))

    for stock_idx in range(num_stocks):
        stock_name = stock_names[stock_idx] if stock_idx < len(stock_names) else f'Stock {stock_idx+1}'
        feature_data = stocks_data[stock_idx, :, feature_idx]

        # Apply normalization based on method
        if normalize_method == 'minmax':
            # Min-Max normalization (0-1)
            data_range = feature_data.max() - feature_data.min()
            if data_range > 1e-8:
                plot_data = (feature_data - feature_data.min()) / data_range
            else:
                plot_data = feature_data
        elif normalize_method == 'zscore':
            # Z-score normalization
            mean = feature_data.mean()
            std = feature_data.std()
            if std > 1e-8:
                plot_data = (feature_data - mean) / std
            else:
                plot_data = feature_data - mean
        else:
            # No normalization - raw data
            plot_data = feature_data

        ax.plot(dates, plot_data, label=stock_name, linewidth=1, alpha=0.7)

    ax.set_xlabel('Date', fontsize=14)
    if normalize_method == 'minmax':
        ax.set_ylabel('Normalized Value (0-1)', fontsize=14)
    elif normalize_method == 'zscore':
        ax.set_ylabel('Z-Score', fontsize=14)
    else:
        ax.set_ylabel('Value', fontsize=14)
    ax.set_title(f'Feature Comparison: {feature_name} (All Stocks){norm_suffix}', fontsize=16, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc='best', ncol=4)

    plt.tight_layout()
    suffix = '_normalized' if normalize_method == 'minmax' else ('_zscore' if normalize_method == 'zscore' else '_raw')
    output_file = output_dir / f'asu_comparison_{feature_name}{suffix}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Plot ASU and MSU features')
    parser.add_argument('--data_dir', type=str, help='Directory containing market_data.npy and stocks_data.npy')
    parser.add_argument('--market_file', type=str, help='Path to market_data.npy file')
    parser.add_argument('--stocks_file', type=str, help='Path to stocks_data.npy file')
    parser.add_argument('--output_dir', type=str, default='src/plot/plot_outputs/feature', help='Output directory for plots')
    parser.add_argument('--start_date', type=str, default='2015-01-01', help='Start date for x-axis (YYYY-MM-DD) - Default: 2015-01-01 for feature34-Inter-P532')
    parser.add_argument('--plot_type', type=str, choices=['all', 'msu', 'asu'], default='all',
                       help='Type of plots to generate: all, msu (market only), or asu (stocks only)')
    parser.add_argument('--normalize', type=str, choices=['none', 'minmax', 'zscore'], default='none',
                       help='Normalization method: none (raw data), minmax (0-1), or zscore (z-score). Default: none (shows raw values like DeepTrader uses)')

    args = parser.parse_args()

    # Determine file paths
    if args.data_dir:
        data_dir = Path(args.data_dir)
        market_file = data_dir / 'market_data.npy'
        stocks_file = data_dir / 'stocks_data.npy'
    else:
        market_file = Path(args.market_file) if args.market_file else None
        stocks_file = Path(args.stocks_file) if args.stocks_file else None

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Feature Plotting Tool")
    print(f"{'='*60}")
    print(f"Output directory: {output_dir}")
    print(f"Start date: {args.start_date}")

    # Plot MSU features
    if args.plot_type in ['all', 'msu'] and market_file and market_file.exists():
        print(f"\nLoading market data from: {market_file}")
        market_data = np.load(market_file)
        plot_msu_features(market_data, output_dir, args.start_date, args.normalize)
    elif args.plot_type in ['all', 'msu']:
        print(f"\nWarning: Market data file not found or not specified")

    # Plot ASU features
    if args.plot_type in ['all', 'asu'] and stocks_file and stocks_file.exists():
        print(f"\nLoading stocks data from: {stocks_file}")
        stocks_data = np.load(stocks_file)
        num_stocks, num_days, num_features = stocks_data.shape

        print(f"\n{'='*60}")
        print(f"Plotting ASU (Stocks) Features")
        print(f"{'='*60}")
        print(f"Data shape: {stocks_data.shape}")
        print(f"Number of stocks: {num_stocks}")
        print(f"Number of features: {num_features}")
        print(f"Number of days: {num_days}")

        # Determine stock names
        stock_names = DJIA_STOCKS[:num_stocks] if num_stocks <= len(DJIA_STOCKS) else [f'Stock_{i+1}' for i in range(num_stocks)]

        # Plot each stock individually
        print(f"\nGenerating individual stock plots...")
        for stock_idx in range(num_stocks):
            stock_name = stock_names[stock_idx]
            stock_data = stocks_data[stock_idx, :, :]
            plot_asu_features_for_stock(stock_data, stock_name, output_dir, args.start_date, args.normalize)

        # Plot feature comparisons across stocks (for key features)
        print(f"\nGenerating feature comparison plots...")
        key_features = [
            (3, 'Close'),           # Close price
            (7, 'RSI'),             # RSI
            (14, 'Alpha001'),       # First alpha factor
        ]

        for feat_idx, feat_name in key_features:
            if feat_idx < num_features:
                plot_asu_features_comparison(stocks_data, stock_names, feat_idx, feat_name, output_dir, args.start_date, args.normalize)

    elif args.plot_type in ['all', 'asu']:
        print(f"\nWarning: Stocks data file not found or not specified")

    print(f"\n{'='*60}")
    print(f"Plotting complete!")
    print(f"All plots saved to: {output_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
