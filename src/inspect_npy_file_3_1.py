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
stock_feature_names = [
    "Open", "High", "Low", "Close", "Volume",
    "MA20", "MA60", "RSI", "MACD_Signal", "K",
    "D", "BBands_Upper", "BBands_Middle", "BBands_Lower", "Alpha001",
    "Alpha002", "Alpha003", "Alpha004", "Alpha006", "Alpha012",
    "Alpha019", "Alpha033", "Alpha038", "Alpha040", "Alpha044",
    "Alpha045", "Alpha046", "Alpha051", "Alpha052", "Alpha053",
    "Alpha054", "Alpha056", "Alpha068", "Alpha085"
]
market_feature_names = [
    "BAMLCC0A4BBBTRIV", "BAMLCC0A0CMTRIV", "BAMLCC0A1AAATRIV", "BAMLHYH0A3CMTRIV", "DGS10",
    "DGS30", "DJI_Open", "DJI_High", "DJI_Low", "DJI_Close",
    "DJI_Adj Close", "DJI_Volume", "XAUUSD_Open", "XAUUSD_High", "XAUUSD_Low",
    "XAUUSD_Close", "VIX_Open", "VIX_High", "VIX_Low", "VIX_Close",
    "VIX_Adj Close", "GSPC_Open", "GSPC_High", "GSPC_Low", "GSPC_Close",
    "GSPC_Adj Close", "GSPC_Volume"
]
stock_names = [
    "MMM", "AXP", "AMGN", "AAPL", "BA", "CAT", "CVX", "CSCO", "KO", "HPQ",
    "GS", "HD", "HON", "IBM", "INTC", "JNJ", "JPM", "MCD", "MRK", "MSFT",
    "NKE", "PFE", "PG", "TRV", "UNH", "VZ", "WBA", "DIS"
]

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
    stock_name = stock_names[stock_idx] if stock_idx < len(stock_names) else f"Stock_{stock_idx+1}"
    print(f"Processing stock {stock_idx+1}/{num_stocks}: {stock_name}")
    try:
        # Extract close prices for labeling
        close_prices = stocks_data[stock_idx, :, 3]
        labels_full = compute_future_label(close_prices, window)
        # Build feature matrix X_full (day x stock-only features)
        X_full = stocks_data[stock_idx, :, :]
        # Truncate last 'window' days (no label)
        X = X_full[:-window]
        y = labels_full[:-window]

        # Skip if invalid
        if np.isnan(X).any() or np.isinf(X).any():
            print(f"WARNING: {stock_name} has NaN/inf in features. Skipping.")
            continue

        X_train = X[:2086]
        y_train = y[:2086]
        X_test  = X[2086:len(X)]
        y_test  = y[2086:len(y)]

        # Train binary classifier
        model = GradientBoostingClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # SHAP explainer (TreeExplainer for classifier)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        # Determine SHAP values for positive class
        if isinstance(shap_values, list):
            # shap_values[1] for positive class when list
            shap_pos = shap_values[1]
        else:
            # shap_values is already a 2D matrix
            shap_pos = shap_values

        # Summary plot
        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_pos, X_test, feature_names=stock_feature_names, show=False)
        plt.title(f"{stock_name} SHAP (5-day avg up/down)")
        plt.tight_layout()
        plt.savefig(f"{output_dir}/{stock_name}_shap_summary.png")
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
                X_test,
                feature_names=stock_feature_names,
                ice=False,
                model_expected_value=True,
                feature_expected_value=True,
                show=False
            )
            plt.title(f"{stock_name}: PDP for {fname}")
            plt.tight_layout()
            plt.savefig(f"{output_dir}/{stock_name}_pdp_{fname}.png")
            plt.close()

    except Exception as e:
        print(f"  Error processing {stock_name}: {e}")


# Process market index (DJI)
print("Processing market index (DJI)")
try:
    # Extract DJI close prices (assume close at index 9)
    dji_close = market_data[:, 9]
    market_labels = compute_future_label(dji_close, window)
    # Truncate
    Xm = market_data[:-window]
    ym = market_labels[:-window]


    if np.isnan(Xm).any() or np.isinf(Xm).any():
        print("WARNING: Market data has NaN/inf. Skipping DJI SHAP.")
        exit()

    Xm_train = Xm[:2086]
    ym_train = ym[:2086]
    Xm_test = Xm[2086 : len(Xm)]
    ym_test = ym[2086 : len(ym)]

    model_m = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model_m.fit(Xm_train, ym_train)

    expl_m = shap.TreeExplainer(model_m)
    # Determine SHAP values for positive class for market index
    shap_values_m = expl_m.shap_values(Xm_test)
    if isinstance(shap_values_m, list):
        shap_vals_m = shap_values_m[1]
    else:
        shap_vals_m = shap_values_m

    plt.figure(figsize=(12, 8))
    shap.summary_plot(shap_vals_m, Xm_test, feature_names=market_feature_names, show=False)
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
            Xm_test,
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
    
except Exception as e:
    print(f"  Error processing DJI: {e}")

print("SHAP classification analysis completed.")
