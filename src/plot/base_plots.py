# -------------------------------
# Base Plotting Functions - Wealth Charts
# -------------------------------

import matplotlib.pyplot as plt
from config import config, AGENT_COLORS, AGENT_LINESTYLES, AGENT_MARKERS, AGENT_LABELS

def plot_results(df_val, df_test, train_days, val_days, test_days, save_plot=True):
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

    if save_plot:
        import os
        output_path = 'plot_outputs'
        os.makedirs(output_path, exist_ok=True)
        filename = f'{output_path}/cumulative_wealth.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()

def plot_yearly_results(df_val, df_test, val_days, test_days, save_plot=True):
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

    if save_plot:
        import os
        output_path = 'plot_outputs'
        os.makedirs(output_path, exist_ok=True)
        filename = f'{output_path}/cumulative_wealth_yearly.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
        plt.close()