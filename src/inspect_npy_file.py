import numpy as np

def verify_data(file_path):
    """Basic data verification function, displays overall statistics"""
    data = np.load(file_path, allow_pickle=True)
    
    print(f"File Name: {file_path}")
    print(f"Data Shape: {data.shape}")
    print(f"Data Type: {data.dtype}")

    print(f"Checking {file_path}")
    print("NaN count:", np.isnan(data).sum() if np.issubdtype(data.dtype, np.floating) else "Not Applicable")
    print("Inf count:", np.isinf(data).sum() if np.issubdtype(data.dtype, np.floating) else "Not Applicable")
    print("0 count:", (data == 0).sum() if np.issubdtype(data.dtype, np.floating) else "Not Applicable")
    print("\n")

def detailed_stats_stocks_data(file_path):
    """Provides detailed statistics for stocks_data.npy by stock and feature"""
    data = np.load(file_path, allow_pickle=True)
    # Assuming shape is (stocks, time_points, features)
    num_stocks, time_points, num_features = data.shape
    
    print(f"Detailed Statistics - {file_path}:")
    print(f"Shape: {data.shape} - (num_stocks, time_points, num_features)")
    
    # Calculate statistics for each feature of each stock
    print("\nDetailed statistics for each stock's features (showing only problematic items):")
    problem_found = False
    for stock_idx in range(num_stocks):
        stock_has_problem = False
        stock_problems = []
        
        for feature_idx in range(num_features):
            feature_data = data[stock_idx, :, feature_idx]
            nan_count = np.isnan(feature_data).sum()
            inf_count = np.isinf(feature_data).sum()
            zero_count = (feature_data == 0).sum()
            
            # Only track problematic features (non-zero counts)
            if nan_count > 0 or inf_count > 0 or zero_count > 0:
                stock_has_problem = True
                stock_problems.append(f"  Feature {feature_idx+1}: NaN={nan_count}, Inf={inf_count}, Zero={zero_count}")
        
        # Only print stocks with problems
        if stock_has_problem:
            problem_found = True
            print(f"\nStock {stock_idx+1}:")
            for problem in stock_problems:
                print(problem)
    
    if not problem_found:
        print("  No problematic data found in any stock's features.")
    
    # Summarize by feature across all stocks
    print("\nSummary by feature across all stocks (showing only problematic features):")
    problem_found = False
    for feature_idx in range(num_features):
        feature_data = data[:, :, feature_idx]
        nan_count = np.isnan(feature_data).sum()
        inf_count = np.isinf(feature_data).sum()
        zero_count = (feature_data == 0).sum()
        
        # Only print problematic features
        if nan_count > 0 or inf_count > 0 or zero_count > 0:
            problem_found = True
            print(f"  Feature {feature_idx+1}: NaN={nan_count}, Inf={inf_count}, Zero={zero_count}")
    
    if not problem_found:
        print("  No problematic data found in any feature.")

def detailed_stats_market_data(file_path):
    """Provides detailed statistics for market_data.npy by feature"""
    data = np.load(file_path, allow_pickle=True)
    # Assuming shape is (time_points, features)
    time_points, num_features = data.shape
    
    print(f"Detailed Statistics - {file_path}:")
    print(f"Shape: {data.shape} - (time_points, num_features)")
    
    # Statistics for each feature
    print("\nFeature statistics (showing only problematic features):")
    problem_found = False
    for feature_idx in range(num_features):
        feature_data = data[:, feature_idx]
        nan_count = np.isnan(feature_data).sum()
        inf_count = np.isinf(feature_data).sum()
        zero_count = (feature_data == 0).sum()
        
        # Only print problematic features
        if nan_count > 0 or inf_count > 0 or zero_count > 0:
            problem_found = True
            print(f"  Feature {feature_idx+1}: NaN={nan_count}, Inf={inf_count}, Zero={zero_count}")
    
    if not problem_found:
        print("  No problematic data found in any feature.")

def detailed_stats_ror(file_path):
    """Provides detailed statistics for ror.npy by stock"""
    data = np.load(file_path, allow_pickle=True)
    # Assuming shape is (stocks, time_points)
    num_stocks, time_points = data.shape
    
    print(f"Detailed Statistics - {file_path}:")
    print(f"Shape: {data.shape} - (num_stocks, time_points)")
    
    # Statistics for each stock
    print("\nStock statistics (showing only problematic stocks):")
    problem_found = False
    for stock_idx in range(num_stocks):
        stock_data = data[stock_idx, :]
        nan_count = np.isnan(stock_data).sum()
        inf_count = np.isinf(stock_data).sum()
        zero_count = (stock_data == 0).sum()
        
        # Only print problematic stocks
        if nan_count > 0 or inf_count > 0 or zero_count > 0:
            problem_found = True
            print(f"  Stock {stock_idx+1}: NaN={nan_count}, Inf={inf_count}, Zero={zero_count}")
    
    if not problem_found:
        print("  No problematic data found in any stock.")

# File paths
stocks_data_file_path = r"data\DJIA\stocks_data.npy"
market_data_file_path = r"data\DJIA\feature33\market_data.npy"
ror_file_path = r"data\DJIA\ror.npy"
industry_classification_file_path = r"data\DJIA\industry_classification.npy"

# Run detailed statistics
print("=" * 50)
verify_data(stocks_data_file_path)
detailed_stats_stocks_data(stocks_data_file_path)
print("=" * 50)
verify_data(market_data_file_path)
detailed_stats_market_data(market_data_file_path)
print("=" * 50)
verify_data(ror_file_path)
detailed_stats_ror(ror_file_path)
print("=" * 50)
verify_data(industry_classification_file_path)