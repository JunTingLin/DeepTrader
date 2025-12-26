"""
Dataset loader for MSU_LSTM training

Loads market data and ground truth rho labels for training/validation/testing.
"""

import numpy as np
import json
import torch
from torch.utils.data import Dataset


class MSUDataset(Dataset):
    """
    Dataset for MSU training

    Args:
        market_data_path (str): Path to market_data.npy [T, num_features]
        ground_truth_path (str): Path to ground truth JSON file
        window_len (int): Window length in weeks (default: 13)
        feature_idx (int): Which feature to use for input (default: None = use all)

    Returns:
        X: [window_len, num_features] - Input window (weekly sampled)
        rho: scalar - Target rho value
        idx: int - Sample index
    """

    def __init__(self, market_data_path, ground_truth_path, window_len=13, feature_idx=None):
        super(MSUDataset, self).__init__()

        # Load market data
        self.market_data = np.load(market_data_path)  # [T, num_features]
        self.window_len = window_len
        self.feature_idx = feature_idx

        # Load ground truth
        with open(ground_truth_path, 'r') as f:
            gt_data = json.load(f)

        self.ground_truths = gt_data['ground_truth_records']
        self.metadata = gt_data['metadata']

        print(f"Loaded dataset:")
        print(f"  Market data shape: {self.market_data.shape}")
        print(f"  Number of samples: {len(self.ground_truths)}")
        print(f"  Window length: {window_len} weeks")
        print(f"  Feature index: {feature_idx if feature_idx is not None else 'All features'}")

    def __len__(self):
        return len(self.ground_truths)

    def __getitem__(self, idx):
        """
        Get a single sample

        Returns:
            X: [window_len, num_features] - Weekly sampled input
            rho: scalar - Target rho value
            sample_idx: int - Data index from ground truth
        """
        gt = self.ground_truths[idx]

        # Get input window indices
        input_start = gt['input_start']
        input_end = gt['input_end']

        # Extract input window
        input_daily = self.market_data[input_start:input_end]  # [window_days, num_features]

        # Extract weekly data (every 5th day, starting from day 4)
        input_weekly = self.extract_weekly_data(input_daily)  # [window_len, num_features]

        # Normalize using z-score
        input_normed, mean, std = self.normalize_weekly_data(input_weekly)

        # Get target rho
        rho = gt['rho']

        # Convert to tensors
        X = torch.FloatTensor(input_normed)  # [window_len, num_features]
        rho = torch.FloatTensor([rho]).squeeze()  # scalar

        return X, rho, gt['idx']

    def extract_weekly_data(self, daily_data):
        """
        Extract weekly data from daily data (every 5 days)

        Args:
            daily_data: [window_days, num_features]

        Returns:
            weekly_data: [window_len, num_features]
        """
        # Take every 5th day starting from day 4 (0-indexed: day 4, 9, 14, ...)
        if self.feature_idx is not None:
            # Use only specific feature
            weekly_data = daily_data[4::5, self.feature_idx:self.feature_idx+1][:self.window_len]
        else:
            # Use all features
            weekly_data = daily_data[4::5][:self.window_len]

        return weekly_data

    def normalize_weekly_data(self, weekly_data):
        """
        Z-score normalization

        Args:
            weekly_data: [window_len, num_features]

        Returns:
            normed: [window_len, num_features]
            mean: [num_features]
            std: [num_features]
        """
        mean = np.mean(weekly_data, axis=0)
        std = np.std(weekly_data, axis=0)

        # Avoid division by zero
        std = np.where(std == 0, 1.0, std)

        normed = (weekly_data - mean) / std

        return normed, mean, std


def get_dataloaders(data_dir, batch_size=32, num_workers=4, feature_idx=None):
    """
    Create train/val/test dataloaders

    Args:
        data_dir (str): Directory containing market_data.npy and ground truth files
        batch_size (int): Batch size for training
        num_workers (int): Number of workers for data loading
        feature_idx (int): Which feature to use (None = all features)

    Returns:
        train_loader, val_loader, test_loader
    """
    import os
    from torch.utils.data import DataLoader

    market_data_path = os.path.join(data_dir, 'market_data.npy')

    # Create datasets
    train_dataset = MSUDataset(
        market_data_path=market_data_path,
        ground_truth_path=os.path.join(data_dir, 'MSU_train_ground_truth_step1.json'),
        feature_idx=feature_idx
    )

    val_dataset = MSUDataset(
        market_data_path=market_data_path,
        ground_truth_path=os.path.join(data_dir, 'MSU_val_ground_truth_step21.json'),
        feature_idx=feature_idx
    )

    test_dataset = MSUDataset(
        market_data_path=market_data_path,
        ground_truth_path=os.path.join(data_dir, 'MSU_test_ground_truth_step21.json'),
        feature_idx=feature_idx
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    # Test dataset
    print("Testing MSU Dataset...")

    data_dir = 'src/data/DJIA/feature34-Inter-P532'
    market_data_path = f'{data_dir}/market_data.npy'
    ground_truth_path = f'{data_dir}/MSU_train_ground_truth_step1.json'

    # Test with all features
    print("\n=== Test with all features ===")
    dataset = MSUDataset(market_data_path, ground_truth_path, feature_idx=None)
    X, rho, idx = dataset[0]
    print(f"Sample 0:")
    print(f"  X shape: {X.shape}")
    print(f"  rho: {rho.item():.6f}")
    print(f"  idx: {idx}")

    # Test with single feature
    print("\n=== Test with feature 0 only ===")
    dataset_single = MSUDataset(market_data_path, ground_truth_path, feature_idx=0)
    X, rho, idx = dataset_single[0]
    print(f"Sample 0:")
    print(f"  X shape: {X.shape}")
    print(f"  rho: {rho.item():.6f}")
    print(f"  idx: {idx}")

    # Test dataloaders
    print("\n=== Test dataloaders ===")
    train_loader, val_loader, test_loader = get_dataloaders(data_dir, batch_size=16, num_workers=0)
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")

    # Test one batch
    for X_batch, rho_batch, idx_batch in train_loader:
        print(f"\nFirst batch:")
        print(f"  X shape: {X_batch.shape}")
        print(f"  rho shape: {rho_batch.shape}")
        print(f"  rho values: {rho_batch[:5]}")
        break
