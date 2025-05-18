import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directory for plots
output_dir = r'outputs_4/shap_plots_1'
os.makedirs(output_dir, exist_ok=True)

# Load data
stocks_data = np.load(r"data\DJIA\feature33-fill\stocks_data.npy")
market_data = np.load(r"data\DJIA\feature33-fill\market_data.npy")

print(f"Loaded data shapes:")
print(f"  stocks_data: {stocks_data.shape}")
print(f"  market_data: {market_data.shape}")

num_stocks = stocks_data.shape[0]
num_days = stocks_data.shape[1]
stock_ror = np.zeros((num_stocks, num_days))

# Calculate the return for each stock
for stock_idx in range(num_stocks):
    for day in range(1, num_days):  # Starting from the second day
        stock_ror[stock_idx, day] = (stocks_data[stock_idx, day, 0] - stocks_data[stock_idx, day-1, 0]) / stocks_data[stock_idx, day-1, 0]
stock_ror = np.nan_to_num(stock_ror, nan=0, posinf=0, neginf=0)
print(f"Calculated stock_ror with shape: {stock_ror.shape}")


# Calculate DJI market returns (using index 9 which is DJI_Close)
dji_close = market_data[:, 9]
market_ror = np.zeros(len(dji_close))
for i in range(1, len(dji_close)):
    market_ror[i] = (dji_close[i] - dji_close[i-1]) / dji_close[i-1]
market_ror = np.nan_to_num(market_ror, nan=0, posinf=0, neginf=0)

print(f"Calculated market_ror with shape: {market_ror.shape}")

# Generate feature names for better interpretability
stock_feature_names = [f"Stock_Feature_{i+1}" for i in range(stocks_data.shape[2])]
market_feature_names = [f"Market_Feature_{i+1}" for i in range(market_data.shape[1])]
all_feature_names = stock_feature_names + market_feature_names

# Process each stock
for stock_idx in range(stocks_data.shape[0]):
    print(f"Processing stock {stock_idx+1}/{stocks_data.shape[0]}")
    
    try:
        # Prepare features and target
        # For each day, concatenate stock-specific features with market features
        X = np.zeros((stocks_data.shape[1], stocks_data.shape[2] + market_data.shape[1]))
        
        for day in range(stocks_data.shape[1]):
            X[day] = np.concatenate([stocks_data[stock_idx, day], market_data[day]])
        
        y = stock_ror[stock_idx]
        
        # Skip day 0 (as return is 0)
        X = X[1:]
        y = y[1:]
        
        # Convert to float64 to avoid potential precision issues
        X = X.astype(np.float64)
        y = y.astype(np.float64)
        
        # Check for NaN or infinite values after preprocessing
        if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any():
            print(f"  WARNING: Stock {stock_idx+1} still has NaN or infinite values. Skipping...")
            continue
        
        # Train model (use a reasonable number of estimators for demonstration)
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=all_feature_names, show=False)
        plt.title(f"Stock {stock_idx+1} SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stock_{stock_idx+1}_shap_summary.png")
        plt.close()
        
        # Create partial dependence plots for top 5 features
        # First, identify top 5 features by mean absolute SHAP value
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = feature_importance.argsort()[-5:]  # Get indices of top 5 features
        
        for i, feature_idx in enumerate(top_features_idx):
            try:
                plt.figure(figsize=(10, 6))
                feature_name = all_feature_names[feature_idx]
                
                # Create partial dependence plot
                shap.plots.partial_dependence(
                    feature_idx, 
                    model.predict, 
                    X,
                    feature_names=all_feature_names,
                    ice=False,
                    model_expected_value=True,
                    feature_expected_value=True,
                    show=False
                )
                plt.title(f"Stock {stock_idx+1}: Partial Dependence for {feature_name}")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/stock_{stock_idx+1}_feature_{feature_name}_pdp.png")
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not create PDP for stock {stock_idx+1}, feature {feature_name}: {str(e)}")
    
    except Exception as e:
        print(f"  Error processing stock {stock_idx+1}: {str(e)}")

# Process market index (DJI)
print("Processing market index (DJI)")

try:
    # For market analysis, we'll use only market features to predict market returns
    X = market_data[1:]  # Skip day 0
    y = market_ror[1:]   # Skip day 0
    
    # Convert to float64
    X = X.astype(np.float64)
    y = y.astype(np.float64)
    
    # Check for remaining issues
    if np.isnan(X).any() or np.isinf(X).any() or np.isnan(y).any() or np.isinf(y).any():
        print(f"  WARNING: Market data still has NaN or infinite values. Skipping...")
    else:
        # Train model
        model = GradientBoostingRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        
        # Create SHAP explainer
        explainer = shap.Explainer(model)
        shap_values = explainer(X)
        
        # Create SHAP summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X, feature_names=market_feature_names, show=False)
        plt.title("DJI Market SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dji_market_shap_summary.png")
        plt.close()
        
        # Create partial dependence plots for top 5 features
        feature_importance = np.abs(shap_values.values).mean(0)
        top_features_idx = feature_importance.argsort()[-5:]  # Get indices of top 5 features
        
        for i, feature_idx in enumerate(top_features_idx):
            try:
                plt.figure(figsize=(10, 6))
                feature_name = market_feature_names[feature_idx]
                
                # Create partial dependence plot
                shap.plots.partial_dependence(
                    feature_idx, 
                    model.predict, 
                    X,
                    feature_names=market_feature_names,
                    ice=False,
                    model_expected_value=True,
                    feature_expected_value=True,
                    show=False
                )
                plt.title(f"DJI Market: Partial Dependence for {feature_name}")
                plt.tight_layout()
                plt.savefig(f"{output_dir}/dji_market_feature_{feature_name}_pdp.png")
                plt.close()
            except Exception as e:
                print(f"  Warning: Could not create PDP for market feature {feature_name}: {str(e)}")
except Exception as e:
    print(f"  Error processing market data: {str(e)}")

print(f"SHAP analysis completed. Results saved in '{output_dir}' directory.")