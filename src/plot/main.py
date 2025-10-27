# -------------------------------
# Main Execution Program
# -------------------------------

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to avoid segfaults
import gc  # For garbage collection
import logging
import sys
from datetime import datetime

from data_processor import process_data
from base_plots import plot_results, plot_yearly_results
from heatmaps import plot_portfolio_heatmap, plot_future_return_heatmap, plot_profit_heatmap, plot_market_profit_heatmap, plot_precision_analysis_heatmap, plot_individual_stock_returns_heatmap
from stock_trends import plot_stock_price_trends, plot_step_analysis, plot_msu_step_analysis, plot_step_score_scatter, plot_all_steps_score_scatter, plot_rho_with_market_trend
from analysis import calculate_periodic_returns_df, calculate_win_rate_df, compute_metrics_df, compute_correlation_metrics, compute_prediction_accuracy, calculate_msu_market_accuracy
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
    
    # Plot market profit heatmaps
    print("\n=== Market Profit Heatmaps ===")
    print("Plotting market monthly return heatmaps...")
    plot_market_profit_heatmap(df_val, 'val', save_plot=True)
    plot_market_profit_heatmap(df_test, 'test', save_plot=True)

    # Plot portfolio visualizations for each experiment
    print("\n=== Portfolio Visualizations ===")
    for exp_id in EXPERIMENT_IDS:
        print(f"\nVisualizing portfolio for {exp_id}...")
        
        # Plot and save portfolio heatmaps
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plot=True, show_ranks=True, show_scores=True)
        plot_portfolio_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plot=True, show_ranks=True, show_scores=True)

        # Plot and save future return heatmaps (for overlay comparison)
        plot_future_return_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plot=True)
        plot_future_return_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plot=True)
        
        # Plot and save profit heatmaps
        print(f"Plotting profit heatmaps for {exp_id}...")
        plot_profit_heatmap(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plot=True)
        plot_profit_heatmap(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plot=True)

        # Plot and save precision analysis heatmaps
        print(f"Plotting precision analysis heatmaps for {exp_id}...")
        plot_precision_analysis_heatmap(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plot=True)
        plot_precision_analysis_heatmap(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plot=True)

        # Plot and save rho with market trend
        print(f"Plotting rho with market trend for {exp_id}...")
        plot_rho_with_market_trend(exp_id, OUTPUTS_BASE_PATH, df_val.index, 'val', save_plot=True)
        plot_rho_with_market_trend(exp_id, OUTPUTS_BASE_PATH, df_test.index, 'test', save_plot=True)

        # Plot and save individual stock returns heatmaps
        print(f"Plotting individual stock returns heatmaps for {exp_id}...")
        plot_individual_stock_returns_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_val.index, 'val', save_plot=True)
        plot_individual_stock_returns_heatmap(exp_id, OUTPUTS_BASE_PATH, symbols, df_test.index, 'test', save_plot=True)

        # Plot and save stock price trends with trading positions
        print(f"Generating stock price trend plots for {exp_id}...")
        print(f"This will create {len(symbols) * 2} PNG files (val + test for each stock)")
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'val', save_plots=True)
        plot_stock_price_trends(exp_id, OUTPUTS_BASE_PATH, symbols, 'test', save_plots=True)

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
            print(f"    Long Precision@{val_step_k}:   {val_acc['mean_step_long_precision']:.1%} (of {val_step_k} stocks predicted long, % that went up)")
            print(f"    Short Precision@{val_step_k}:  {val_acc['mean_step_short_precision']:.1%} (of {val_step_k} stocks predicted short, % that went down)")
            print(f"    Overall Precision@{int(val_acc.get('avg_positions_per_step', 8))}: {val_acc['mean_step_overall_precision']:.1%} (of all {int(val_acc.get('avg_positions_per_step', 8))} predictions, % correct direction)")
            print(f"    Long Recall@{val_step_k}:      {val_acc['mean_step_long_recall']:.1%} (of top {val_step_k} actual gainers, % we caught by going long)")
            print(f"    Short Recall@{val_step_k}:     {val_acc['mean_step_short_recall']:.1%} (of bottom {val_step_k} actual losers, % we caught by going short)")
            print(f"    Overall Recall@{int(val_acc.get('avg_positions_per_step', 8))}: {val_acc['mean_step_overall_recall']:.1%} (of top/bottom performers, % we caught)")

        # Testing period
        test_acc = compute_prediction_accuracy(exp_id, OUTPUTS_BASE_PATH, 'test')
        if test_acc and test_acc.get('total_predicted', 0) > 0:
            test_step_k = int(test_acc.get('avg_long_positions_per_step', 4))  # K per step
            print(f"  Testing Period:")
            print(f"    Long Precision@{test_step_k}:   {test_acc['mean_step_long_precision']:.1%} (of {test_step_k} stocks predicted long, % that went up)")
            print(f"    Short Precision@{test_step_k}:  {test_acc['mean_step_short_precision']:.1%} (of {test_step_k} stocks predicted short, % that went down)")
            print(f"    Overall Precision@{int(test_acc.get('avg_positions_per_step', 8))}: {test_acc['mean_step_overall_precision']:.1%} (of all {int(test_acc.get('avg_positions_per_step', 8))} predictions, % correct direction)")
            print(f"    Long Recall@{test_step_k}:      {test_acc['mean_step_long_recall']:.1%} (of top {test_step_k} actual gainers, % we caught by going long)")
            print(f"    Short Recall@{test_step_k}:     {test_acc['mean_step_short_recall']:.1%} (of bottom {test_step_k} actual losers, % we caught by going short)")
            print(f"    Overall Recall@{int(test_acc.get('avg_positions_per_step', 8))}: {test_acc['mean_step_overall_recall']:.1%} (of top/bottom performers, % we caught)")

        # MSU Market Direction Analysis
        print(f"\n[MSU Market Direction Analysis for {exp_id}]")
        # Validation period
        val_msu = calculate_msu_market_accuracy(f"{OUTPUTS_BASE_PATH}/{exp_id}", 'val')
        if val_msu and val_msu.get('total_predictions', 0) > 0:
            print(f"  Validation Period:")
            print(f"    Overall Accuracy: {val_msu['overall_accuracy']:.1%} ({val_msu['correct_predictions']}/{val_msu['total_predictions']} correct)")
            print("")

            # Bullish metrics
            if val_msu['predicted_bullish'] > 0:
                print(f"    Bullish Precision: {val_msu['bullish_precision']:.1%} ({val_msu['bullish_tp']}/{val_msu['predicted_bullish']} when predicted bullish, market went up)")
            if val_msu['actual_up'] > 0:
                print(f"    Bullish Recall:    {val_msu['bullish_recall']:.1%} ({val_msu['bullish_tp']}/{val_msu['actual_up']} when market went up, predicted bullish)")

            # Bearish metrics
            if val_msu['predicted_bearish'] > 0:
                print(f"    Bearish Precision: {val_msu['bearish_precision']:.1%} ({val_msu['bearish_tp']}/{val_msu['predicted_bearish']} when predicted bearish, market went down)")
            if val_msu['actual_down'] > 0:
                print(f"    Bearish Recall:    {val_msu['bearish_recall']:.1%} ({val_msu['bearish_tp']}/{val_msu['actual_down']} when market went down, predicted bearish)")

            # Neutral metrics (if any)
            if val_msu['predicted_neutral'] > 0:
                print(f"    Neutral Precision: {val_msu['neutral_precision']:.1%} ({val_msu['neutral_tp']}/{val_msu['predicted_neutral']} when predicted neutral, market flat)")
            if val_msu['actual_flat'] > 0:
                print(f"    Neutral Recall:    {val_msu['neutral_recall']:.1%} ({val_msu['neutral_tp']}/{val_msu['actual_flat']} when market flat, predicted neutral)")

        # Testing period
        test_msu = calculate_msu_market_accuracy(f"{OUTPUTS_BASE_PATH}/{exp_id}", 'test')
        if test_msu and test_msu.get('total_predictions', 0) > 0:
            print(f"  Testing Period:")
            print(f"    Overall Accuracy: {test_msu['overall_accuracy']:.1%} ({test_msu['correct_predictions']}/{test_msu['total_predictions']} correct)")
            print("")

            # Bullish metrics
            if test_msu['predicted_bullish'] > 0:
                print(f"    Bullish Precision: {test_msu['bullish_precision']:.1%} ({test_msu['bullish_tp']}/{test_msu['predicted_bullish']} when predicted bullish, market went up)")
            if test_msu['actual_up'] > 0:
                print(f"    Bullish Recall:    {test_msu['bullish_recall']:.1%} ({test_msu['bullish_tp']}/{test_msu['actual_up']} when market went up, predicted bullish)")

            # Bearish metrics
            if test_msu['predicted_bearish'] > 0:
                print(f"    Bearish Precision: {test_msu['bearish_precision']:.1%} ({test_msu['bearish_tp']}/{test_msu['predicted_bearish']} when predicted bearish, market went down)")
            if test_msu['actual_down'] > 0:
                print(f"    Bearish Recall:    {test_msu['bearish_recall']:.1%} ({test_msu['bearish_tp']}/{test_msu['actual_down']} when market went down, predicted bearish)")

            # Neutral metrics (if any)
            if test_msu['predicted_neutral'] > 0:
                print(f"    Neutral Precision: {test_msu['neutral_precision']:.1%} ({test_msu['neutral_tp']}/{test_msu['predicted_neutral']} when predicted neutral, market flat)")
            if test_msu['actual_flat'] > 0:
                print(f"    Neutral Recall:    {test_msu['neutral_recall']:.1%} ({test_msu['neutral_tp']}/{test_msu['actual_flat']} when market flat, predicted neutral)")

if __name__ == "__main__":
    main()