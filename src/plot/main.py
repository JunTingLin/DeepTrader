# -------------------------------
# Main Execution Program
# -------------------------------

from data_processor import process_data
from base_plots import plot_results, plot_yearly_results
from heatmaps import plot_portfolio_heatmap, plot_profit_heatmap
from stock_trends import plot_stock_price_trends
from analysis import calculate_periodic_returns_df, calculate_win_rate_df, compute_metrics_df
from config import (
    config, get_stock_symbols, START_DATE, END_DATE, 
    EXPERIMENT_IDS, OUTPUTS_BASE_PATH
)

def main():
    print(f"Using {config['name']} market configuration")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Benchmark: {config['benchmark_label']}")
    
    # Load stock symbols
    symbols = get_stock_symbols()
    # symbols = get_stock_symbols()[:2]  # for quick testing
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
        
        # Plot and save portfolio heatmaps
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plot=True)
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plot=True)
        
        # Plot and save profit heatmaps
        print(f"Plotting profit heatmaps for {exp_id}...")
        plot_profit_heatmap(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plot=True)
        plot_profit_heatmap(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plot=True)
        
        # Plot and save stock price trends with trading positions
        print(f"Generating stock price trend plots for {exp_id}...")
        print(f"This will create {len(symbols) * 2} PNG files (val + test for each stock)")
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'val', save_plots=True)
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'test', save_plots=True)
    
    # Compute periodic returns and win rates for validation period
    period_codes = ['ME', 'QE', '6ME', 'YE']
    print("\nPeriodic Returns and Win Rates (Validation):")
    for period in period_codes:
        returns_val = calculate_periodic_returns_df(df_val, period)
        win_rate_val = calculate_win_rate_df(returns_val)
        print(f"\nValidation Period: {period}")
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