import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------
# Step 1: Load data and fill NaN/Inf
# ------------------------------
# Load data from .npy file
data = np.load(r"data\DJIA\feature33\stocks_data.npy")  # data shape: [num_stocks, num_days, num_ASU_features]

# Convert Inf to NaN
data[np.isinf(data)] = np.nan

# Fill initial NaN as 0 (safety fallback)
data = np.where(np.isnan(data), 0, data)

# Apply bfill (along time axis) for each stock and each feature
for stock_idx in range(data.shape[0]):
    for feature_idx in range(data.shape[2]):
        ts = data[stock_idx, :, feature_idx]
        ts_df = pd.Series(ts)
        ts_bfilled = ts_df.fillna(method='bfill').values
        data[stock_idx, :, feature_idx] = ts_bfilled

assert not np.isnan(data).any(), "Still have NaNs after bfill"

# ------------------------------
# Step 2: Define training and validation segments
# ------------------------------
# Define the two 8-year segments
training_segment = ("Training (4 years)", 2086, 3130)
validation_segment = ("Validation (4 years)", 3130, 4174)

# Use 2086 days (shorter length) for both segments
target_days = 2086

# ------------------------------
# Step 3: Compute averaged time series for each segment
# ------------------------------
def get_segment_avg(data, start, end, target_days):
    """
    Slice the data from start to end along the time axis,
    take only the first target_days, then average over stocks.
    """
    # Slice along num_days axis; data has shape [num_stocks, num_days, num_features]
    segment = data[:, start:end, :]  # shape: [num_stocks, segment_length, num_features]
    # Ensure we only take target_days
    segment = segment[:, :target_days, :]  # shape: [num_stocks, target_days, num_features]
    # Average over stocks (axis=0) -> result shape: [target_days, num_features]
    segment_avg = np.mean(segment, axis=0)
    return segment_avg

def normalize_timeseries(ts):
    """
    Normalize the time series (z-score normalization) along the time axis for each feature.
    ts: shape (target_days, num_features)
    Returns: normalized time series with mean=0 and std=1 for each feature.
    """
    # Compute mean and std for each feature (axis=0: over time)
    mean = np.mean(ts, axis=0)
    std = np.std(ts, axis=0)
    # Avoid division by zero: if std==0, replace it with 1
    std_fixed = np.where(std == 0, 1, std)
    normalized_ts = (ts - mean) / std_fixed
    return normalized_ts

# Get training averaged time series
train_name, train_start, train_end = training_segment
training_avg = get_segment_avg(data, train_start, train_end, target_days)
# Normalize training time series
training_avg_norm = normalize_timeseries(training_avg)

# Get validation averaged time series
val_name, val_start, val_end = validation_segment
validation_avg = get_segment_avg(data, val_start, val_end, target_days)
# Normalize validation time series
validation_avg_norm = normalize_timeseries(validation_avg)

# ------------------------------
# Step 4: Compute correlation matrix and plot trend comparisons
# ------------------------------
num_features = data.shape[2]

# Initialize correlation matrix of shape (num_features, num_features)
corr_matrix = np.zeros((num_features, num_features))

# Compute Pearson correlation for each pair: train feature i vs validation feature j
for i in range(num_features):
    for j in range(num_features):
        # Compute Pearson correlation coefficient
        corr, _ = pearsonr(training_avg_norm[:, i], validation_avg_norm[:, j])
        corr_matrix[i, j] = corr
        
        # Plot the trend comparison
        if i == j and (i==0 or i==4 or i==26):
            plt.figure(figsize=(10, 6))
            plt.plot(training_avg_norm[:, i], label=f"Training Feature {i}", color='blue')
            plt.plot(validation_avg_norm[:, j], label=f"Validation Feature {j}", color='orange')
            plt.title(f"Trend Comparison: Training Feature {i} vs Validation Feature {j}\n"
                    f"Correlation: {corr:.2f}")
            plt.xlabel("Time (Days)")
            plt.ylabel("Normalized Value")
            plt.legend()
            plt.tight_layout()
            plt.show()

# ------------------------------
# Step 5: Plot correlation heatmap
# ------------------------------
plt.figure(figsize=(10, 8))
# Use imshow to plot the correlation matrix with consistent color scale
plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
plt.colorbar(label='Pearson Correlation')

# Set x and y ticks as feature indices
plt.xticks(ticks=np.arange(num_features), labels=np.arange(num_features))
plt.yticks(ticks=np.arange(num_features), labels=np.arange(num_features))

plt.title(f"Correlation Heatmap: {train_name} vs {val_name}")
plt.xlabel("Validation Feature Index")
plt.ylabel("Training Feature Index")
plt.tight_layout()
plt.show()