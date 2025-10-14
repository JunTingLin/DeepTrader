# -------------------------------
# Main Execution Program
# -------------------------------

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfaults
import gc  # For garbage collection

from data_processor import process_data
from base_plots import plot_results, plot_yearly_results
from heatmaps import plot_portfolio_heatmap, plot_profit_heatmap, plot_rho_heatmap
from stock_trends import plot_stock_price_trends, plot_step_analysis, plot_msu_step_analysis, print_step_score_ranking, plot_step_score_scatter, plot_all_steps_score_scatter
from analysis import calculate_periodic_returns_df, calculate_win_rate_df, compute_metrics_df, compute_correlation_metrics, compute_prediction_accuracy
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
        
        # Plot and save rho (MSU allocation) heatmaps
        print(f"Plotting rho (MSU allocation) heatmaps for {exp_id}...")
        plot_rho_heatmap(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plot=True)
        plot_rho_heatmap(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plot=True)
        
        # Plot and save stock price trends with trading positions
        print(f"Generating stock price trend plots for {exp_id}...")
        print(f"This will create {len(symbols) * 2} PNG files (val + test for each stock)")
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'val', save_plots=True)
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'test', save_plots=True)
        
        # Print step score ranking (only prints, no plotting)
        # print(f"Printing step score rankings for {exp_id}...")
        # print_step_score_ranking(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val')
        # print_step_score_ranking(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test')
        
        # Plot score vs return scatter plots for all trading steps
        print(f"Generating score vs return scatter plots for {exp_id}...")
        print(f"This will create {len(df_val.index)} val + {len(df_test.index)} test scatter plots")
        plot_step_score_scatter(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plots=True)
        plot_step_score_scatter(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plots=True)
        
        # Plot combined scatter plots (all steps in one chart)
        print(f"Generating combined scatter plots for {exp_id}...")
        plot_all_steps_score_scatter(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val')
        plot_all_steps_score_scatter(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test')
        
        # Plot and save step analysis with all stocks per step
        print(f"Generating step analysis plots for {exp_id}...")
        print(f"This will create {len(df_val.index)} val + {len(df_test.index)} test step analysis files")
        plot_step_analysis(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plots=True)
        plot_step_analysis(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plots=True)
        
        # Plot and save MSU step analysis with DJIA trends and rho values
        print(f"Generating MSU step analysis plots for {exp_id}...")
        print(f"This will create {len(df_val.index)} val + {len(df_test.index)} test MSU analysis files")
        plot_msu_step_analysis(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plots=True)
        plot_msu_step_analysis(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plots=True)

        # Force garbage collection to free memory
        gc.collect()
    
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

    # Compute correlation metrics for different strategies
    print("\n" + "="*60)
    print("=== Correlation Metrics (Score vs Return) ===")
    print("="*60)

    for exp_id in EXPERIMENT_IDS:
        print(f"\nExperiment: {exp_id}")
        print("-"*50)

        # Correlation analysis
        print("\n[Correlation analysis - All 30 stocks]")

        # Validation period
        val_corr = compute_correlation_metrics(exp_id, OUTPUTS_BASE_PATH, 'val')
        if val_corr:
            print(f"  Validation Period:")
            print(f"    Overall Pearson:  {val_corr['overall_pearson']:7.4f}")
            print(f"    Overall Spearman: {val_corr['overall_spearman']:7.4f}")
            print(f"    Mean Step Pearson:  {val_corr['mean_step_pearson']:7.4f}")
            print(f"    Mean Step Spearman: {val_corr['mean_step_spearman']:7.4f}")
            print(f"    Valid Steps: {val_corr['valid_steps']}/{val_corr['total_steps']}")

        # Testing period
        test_corr = compute_correlation_metrics(exp_id, OUTPUTS_BASE_PATH, 'test')
        if test_corr:
            print(f"  Testing Period:")
            print(f"    Overall Pearson:  {test_corr['overall_pearson']:7.4f}")
            print(f"    Overall Spearman: {test_corr['overall_spearman']:7.4f}")
            print(f"    Mean Step Pearson:  {test_corr['mean_step_pearson']:7.4f}")
            print(f"    Mean Step Spearman: {test_corr['mean_step_spearman']:7.4f}")
            print(f"    Valid Steps: {test_corr['valid_steps']}/{test_corr['total_steps']}")

    # Compute prediction precision and recall metrics
    print("\n" + "="*60)
    print("=== Precision and Recall (Long should rise, Short should fall) ===")
    print("="*60)

    for exp_id in EXPERIMENT_IDS:
        print(f"\nExperiment: {exp_id}")
        print("-"*50)

        # Precision & Recall Analysis (Mean Step only)
        val_acc = compute_prediction_accuracy(exp_id, OUTPUTS_BASE_PATH, 'val')
        if val_acc and val_acc.get('total_predicted', 0) > 0:
            val_step_k = int(val_acc.get('avg_long_positions_per_step', 4))  # K per step
            print(f"\n[Precision & Recall Analysis - Mean Step]")

            print(f"  Validation Period:")
            print(f"    Long Precision@{val_step_k}:   {val_acc['mean_step_long_precision']:.1%}")
            print(f"    Short Precision@{val_step_k}:  {val_acc['mean_step_short_precision']:.1%}")
            print(f"    Overall Precision@{int(val_acc.get('avg_positions_per_step', 8))}: {val_acc['mean_step_overall_precision']:.1%}")
            print(f"    Long Recall@{val_step_k}:      {val_acc['mean_step_long_recall']:.1%}")
            print(f"    Short Recall@{val_step_k}:     {val_acc['mean_step_short_recall']:.1%}")
            print(f"    Overall Recall@{int(val_acc.get('avg_positions_per_step', 8))}: {val_acc['mean_step_overall_recall']:.1%}")

        # Testing period
        test_acc = compute_prediction_accuracy(exp_id, OUTPUTS_BASE_PATH, 'test')
        if test_acc and test_acc.get('total_predicted', 0) > 0:
            test_step_k = int(test_acc.get('avg_long_positions_per_step', 4))  # K per step
            print(f"  Testing Period:")
            print(f"    Long Precision@{test_step_k}:   {test_acc['mean_step_long_precision']:.1%}")
            print(f"    Short Precision@{test_step_k}:  {test_acc['mean_step_short_precision']:.1%}")
            print(f"    Overall Precision@{int(test_acc.get('avg_positions_per_step', 8))}: {test_acc['mean_step_overall_precision']:.1%}")
            print(f"    Long Recall@{test_step_k}:      {test_acc['mean_step_long_recall']:.1%}")
            print(f"    Short Recall@{test_step_k}:     {test_acc['mean_step_short_recall']:.1%}")
            print(f"    Overall Recall@{int(test_acc.get('avg_positions_per_step', 8))}: {test_acc['mean_step_overall_recall']:.1%}")

if __name__ == "__main__":
    main()