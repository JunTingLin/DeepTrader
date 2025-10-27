# -------------------------------
# Market Configurations and Constants
# -------------------------------

import os

# Stock symbols for different markets

# 2025-05-29 wiki updated
# DJIA_STOCKS = [
#     "AAPL", "AMGN", "AMZN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS",
#     "GS", "HD", "HON", "IBM", "JNJ", "JPM", "KO", "MCD", "MMM", "MRK",
#     "MSFT", "NKE", "NVDA", "PG", "SHW", "TRV", "UNH", "V", "VZ", "WMT"
# ]

# 之前學姐的版本
DJIA_STOCKS = [
    "AAPL", "AMGN", "AXP", "BA", "CAT", "CRM", "CSCO", "CVX", "DIS", "GS",
    "HD", "HON", "HPQ", "IBM", "INTC", "JNJ", "JPM", "KO", "MCD", "MMM",
    "MRK", "MSFT", "NKE", "PFE", "PG", "TRV", "UNH", "V", "VZ", "WBA"
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
        'end_date': "2025-03-31",
        'market_file': "^DJI.csv",
        'stock_symbols': DJIA_STOCKS,
        'benchmark_column': 'DowJones',
        'benchmark_label': 'DJIA',
        'title': 'DeepTrader vs. DJIA',
        'train_end': 1304,
        'val_end': 2087,
        'test_end': 2673,
        'experiment_ids': [
            '0718/181011',
            '1026/030156',
            '1026/135333'
        ],
        'plot_ylim': None,
        'json_files': {
            'test_results': 'test_results.json',
            'val_results': 'val_results.json'
        }
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
        'plot_ylim': None,
        'json_files': {
            'test_results': 'test_results.json',
            'val_results': 'val_results.json'
        }
    }
}

# -------------------------------
# Global Configuration
# -------------------------------
# Change this to 'TW' or 'US' to switch markets
CURRENT_MARKET = 'US'
config = MARKET_CONFIGS[CURRENT_MARKET]
JSON_FILES = config['json_files'].copy()

# -------------------------------
# Constants
# -------------------------------
TRADE_MODE = "M"    # "M": Monthly mode (12 trading periods per year)
TRADE_LEN = 21      # Sampling interval: 21 business days per sample
START_DATE = config['start_date']
END_DATE = config['end_date']
WEALTH_MODE = 'inter'  # 'inter' or 'intra' for daily returns

# Portfolio selection parameters (will be auto-detected from JSON files)

# -------------------------------
# Paths
# -------------------------------
OUTPUTS_BASE_PATH = '../outputs'
EXPERIMENT_IDS = config['experiment_ids']
STOCK_DATA_PATH = '../data/DJIA/feature34-Inter-P532/stocks_data.npy'
MARKET_DATA_PATH = '../data/DJIA/feature34-Inter-P532/market_data.npy'
STOCK_PRICE_INDEX = 0     # Stock open price index in the 34 stock features
MARKET_PRICE_INDEX = 9    # Market close price index in the 27 market features

# -------------------------------
# Plotting Style Configuration
# -------------------------------
AGENT_COLORS = ['b', 'darkblue', 'c', 'steelblue', 'limegreen', 'g', 'lawngreen', 'purple', 'orange', 'brown']
AGENT_LINESTYLES = ['-', '-', '-.', '-', '-', '-', '-', '--', '--', ':']
AGENT_MARKERS = ['o'] * 10  # Use 'o' marker for all agents
AGENT_LABELS = ['Agent 1', 'Agent 2', 'Agent 3', 'Agent 4', 'Agent 5', 'Agent 6', 'Agent 7', 'Agent 8', 'Agent 9', 'Agent 10']

# -------------------------------
# Utility Functions
# -------------------------------
def get_stock_symbols():
    """Get stock symbols for the current market."""
    return config['stock_symbols']

def set_market(market_code):
    """
    Switch to a different market configuration.
    
    Args:
        market_code (str): 'TW' for Taiwan market or 'US' for US market
    """
    global CURRENT_MARKET, config, START_DATE, END_DATE, EXPERIMENT_IDS, JSON_FILES

    if market_code not in MARKET_CONFIGS:
        raise ValueError(f"Invalid market code. Use one of: {list(MARKET_CONFIGS.keys())}")

    CURRENT_MARKET = market_code
    config = MARKET_CONFIGS[CURRENT_MARKET]
    START_DATE = config['start_date']
    END_DATE = config['end_date']
    EXPERIMENT_IDS = config['experiment_ids']
    JSON_FILES = config['json_files'].copy()
    
    print(f"Switched to {config['name']} market")

def set_json_files(test_results=None, val_results=None):
    """
    Set custom JSON file names for results.

    Args:
        test_results (str, optional): Custom name for test results JSON file
        val_results (str, optional): Custom name for validation results JSON file

    Example:
        set_json_files(test_results='test_results_rho_0.5.json',
                      val_results='val_results_rho_0.5.json')
    """
    global JSON_FILES

    if test_results is not None:
        JSON_FILES['test_results'] = test_results
    if val_results is not None:
        JSON_FILES['val_results'] = val_results

    print(f"Updated JSON files configuration: {JSON_FILES}")

def get_results_paths(experiment_id):
    """
    Get full paths to the results JSON files for a given experiment.

    Args:
        experiment_id (str): The experiment ID (e.g., '0718/181011')

    Returns:
        dict: Dictionary containing full paths to test and validation results
    """
    base_path = os.path.join(OUTPUTS_BASE_PATH, experiment_id)
    return {
        'test_results': os.path.join(base_path, JSON_FILES['test_results']),
        'val_results': os.path.join(base_path, JSON_FILES['val_results'])
    }