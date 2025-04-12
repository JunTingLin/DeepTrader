import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# ------------------------------
# Step 1: Load data and fill NaN/Inf
# ------------------------------
# Load data from .npy file; adjust the path as needed
data = np.load("data\DJIA\stocks_data.npy")  # data shape: [num_stocks, num_days, num_ASU_features]

# Convert Inf to NaN
data[np.isinf(data)] = np.nan

# Compute global mean ignoring NaN
global_mean = np.nanmean(data)

# Replace NaN with global mean
data = np.where(np.isnan(data), global_mean, data)

# ------------------------------
# Step 2: Define training and validation segments
# ------------------------------
# Each segment is defined by (segment_name, start_index, end_index, total_business_days)
segments = [
    ("train-1", 0, 1043, 1043),
    ("train-2", 260, 1305, 1045),
    ("train-3", 521, 1565, 1044),
    ("train-4", 782, 1825, 1043),
    ("train-5", 1043, 2086, 1043),
    ("train-6", 1305, 2348, 1043),
    ("train-7", 1565, 2609, 1044),
    ("train-8", 1825, 2870, 1045),
    ("train-9", 2086, 3130, 1044),
    ("train-10", 2348, 3391, 1043),
    ("train-11", 2609, 3652, 1043),
    ("train-12", 2870, 3913, 1043)
]

# Validation segment definition
validation_segment = ("Validation", 3130, 4174, 1044)

# For consistent comparison, we only take the first 1043 days from each segment
target_days = 1043

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
    # Ensure we only take target_days (assume segment_length >= target_days)
    segment = segment[:, :target_days, :]  # shape: [num_stocks, target_days, num_features]
    # Average over stocks (axis=0) -> result shape: [target_days, num_features]
    segment_avg = np.mean(segment, axis=0)
    return segment_avg

# Get validation averaged time series (only take first target_days)
val_name, val_start, val_end, _ = validation_segment
validation_avg = get_segment_avg(data, val_start, val_end, target_days)

# ------------------------------
# Step 4: Compute correlation matrix and plot heatmap for each training segment
# ------------------------------
num_features = data.shape[2]

for seg in segments:
    seg_name, start_idx, end_idx, _ = seg
    # Get averaged time series for the train segment (shape: [target_days, num_features])
    train_avg = get_segment_avg(data, start_idx, end_idx, target_days) #(1043, 5)
    
    # Initialize correlation matrix of shape (num_features, num_features)
    corr_matrix = np.zeros((num_features, num_features))
    
    # Compute Pearson correlation for each pair: train feature i vs validation feature j
    for i in range(num_features):
        for j in range(num_features):
            # Compute Pearson correlation coefficient between two time series arrays of length target_days
            # pearsonr returns a tuple: (correlation, p-value); we use the correlation coefficient.
            corr, _ = pearsonr(train_avg[:, i], validation_avg[:, j])
            corr_matrix[i, j] = corr

    # ------------------------------
    # Step 5: Plot heatmap using matplotlib
    # ------------------------------
    plt.figure(figsize=(8, 6))
    # Use imshow to plot the correlation matrix; set vmin and vmax for a consistent color scale
    plt.imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation')
    
    # Set x and y ticks as feature indices
    plt.xticks(ticks=np.arange(num_features), labels=np.arange(num_features))
    plt.yticks(ticks=np.arange(num_features), labels=np.arange(num_features))
    
    plt.title(f"Correlation Heatmap: {seg_name} vs {val_name}")
    plt.xlabel("Validation Feature Index")
    plt.ylabel("Training Feature Index")
    plt.tight_layout()
    plt.show()
