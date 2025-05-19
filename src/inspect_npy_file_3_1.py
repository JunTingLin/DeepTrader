import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
import os
import warnings

# Suppress warnings
warnings.filterwarnings('ignore')

# Output directory for SHAP plots
output_dir = 'outputs_4/shap_plots_1'
os.makedirs(output_dir, exist_ok=True)

# Load data
stocks_data = np.load(r"data/DJIA/feature33-fill/stocks_data.npy")
market_data = np.load(r"data/DJIA/feature33-fill/market_data.npy")

num_stocks, num_days, num_stock_feats = stocks_data.shape
num_market_feats = market_data.shape[1]

# Feature names
stock_feature_names = [f"Stock_Feature_{i+1}" for i in range(num_stock_feats)]
market_feature_names = [f"Market_Feature_{i+1}" for i in range(num_market_feats)]

# Rolling window for future-average labeling
window = 5

# Helper: compute binary label array for "future window-day average close > today's close"
def compute_future_label(close_prices, window):
    labels = np.zeros_like(close_prices, dtype=int)
    # only up to num_days - window
    for t in range(len(close_prices) - window):
        future_avg = close_prices[t+1:t+1+window].mean()
        labels[t] = int(future_avg > close_prices[t])
    return labels

# Process each stock
for stock_idx in range(num_stocks):
    print(f"Processing stock {stock_idx+1}/{num_stocks}")
    try:
        # Extract close prices for labeling
        close_prices = stocks_data[stock_idx, :, 3]
        labels_full = compute_future_label(close_prices, window)
        # Build feature matrix X_full (day x stock-only features)
        X_full = stocks_data[stock_idx, :, :]
        # Truncate last 'window' days (no label)
        X = X_full[:-window]
        y = labels_full[:-window]

        # convert to float
        X = X.astype(np.float64)
        y = y.astype(int)

        # Skip if invalid
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"  WARNING: Stock {stock_idx+1} has NaN/inf in features. Skipping.")
            continue

        # Train binary classifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        # SHAP explainer (TreeExplainer for classifier)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)
        # Determine SHAP values for positive class
        if isinstance(shap_values, list):
            # shap_values[1] for positive class when list
            shap_pos = shap_values[1]
        else:
            # shap_values is already a 2D matrix
            shap_pos = shap_values

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_pos, X, feature_names=stock_feature_names, show=False)
        plt.title(f"Stock {stock_idx+1} SHAP (5-day avg up/down)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/stock_{stock_idx+1}_shap_summary.png")
        plt.close()

        # Partial dependence for top 5 features
        importance = np.abs(shap_pos).mean(axis=0)
        top_idx = importance.argsort()[-5:]
        for feature_idx in top_idx:
            fname = stock_feature_names[feature_idx]
            plt.figure(figsize=(10, 6))
            shap.plots.partial_dependence(
                feature_idx,
                lambda z: model.predict_proba(z)[:,1],
                X,
                feature_names=stock_feature_names,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            plt.title(f"Stock {stock_idx+1}: PDP for {fname}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/stock_{stock_idx+1}_pdp_{fname}.png")
            plt.close()

    except Exception as e:
        print(f"  Error processing stock {stock_idx+1}: {e}")

# Process market index (DJI)
print("Processing market index (DJI)")
try:
    # Extract DJI close prices (assume close at index 9)
    dji_close = market_data[:, 9]
    market_labels = compute_future_label(dji_close, window)
    # Truncate
    Xm = market_data[:-window]
    ym = market_labels[:-window]
    Xm = Xm.astype(np.float64)
    ym = ym.astype(int)

    if not (np.isnan(Xm).any() or np.isinf(Xm).any()):
        model_m = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model_m.fit(Xm, ym)

        expl_m = shap.TreeExplainer(model_m)
        # Determine SHAP values for positive class for market index
        shap_values_m = expl_m.shap_values(Xm)
        if isinstance(shap_values_m, list):
            shap_vals_m = shap_values_m[1]
        else:
            shap_vals_m = shap_values_m

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_vals_m, Xm, feature_names=market_feature_names, show=False)
        plt.title("DJI Market SHAP (5-day avg up/down)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/dji_market_shap_summary.png")
        plt.close()

        imp_m = np.abs(shap_vals_m).mean(axis=0)
        top_m = imp_m.argsort()[-5:]
        for feature_idx in top_m:
            fname = market_feature_names[feature_idx]
            plt.figure(figsize=(10, 6))
            shap.plots.partial_dependence(
                feature_idx,
                lambda z: model_m.predict_proba(z)[:,1],
                Xm,
                feature_names=market_feature_names,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            plt.title(f"DJI: PDP for {fname}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/dji_pdp_{fname}.png")
            plt.close()
    else:
        print("  WARNING: Market data has NaN/inf. Skipping DJI SHAP.")
except Exception as e:
    print(f"  Error processing DJI: {e}")

print("SHAP classification analysis completed.")
