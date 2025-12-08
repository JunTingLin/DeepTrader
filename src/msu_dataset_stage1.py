"""
MSU Dataset for Stage 1 Pretraining

Self-supervised learning: Masked market feature reconstruction

Input:  market_data (13 weeks, 27 features) with some weeks masked
Output: original market_data (for reconstruction)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class MSUDataset(Dataset):
    """
    MSU Stage 1 Pretraining Dataset (Self-Supervised)

    Loads market data for masked reconstruction task.

    Inputs:
        - market_data: [window_len, market_features] normalized weekly data

    Outputs:
        - market_data: [window_len, market_features] original data (for reconstruction target)

    Note: Masking is done in the training loop, not in the dataset
    """

    def __init__(self,
                 data_dir,
                 split='train',
                 window_len=13):
        """
        Args:
            data_dir: Directory containing market_data.npy
            split: 'train', 'val', or 'test'
            window_len: Input window length in weeks (13)
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.window_len = window_len
        self.window_days = window_len * 5  # 13 weeks = 65 days

        # Load market data
        market_data_path = os.path.join(data_dir, 'market_data.npy')
        self.market_data = np.load(market_data_path)  # [T, market_features]

        # Define split ranges (same as before from hyper.json)
        if split == 'train':
            start_idx = 0
            end_idx = 1304
        elif split == 'val':
            start_idx = 1304
            end_idx = 2087
        else:  # test
            start_idx = 2087
            end_idx = 2673

        # Generate valid indices (need window_days history)
        min_idx = self.window_days - 1  # 64
        max_idx = end_idx - 1

        # Use sliding window with step=1 for training (more data)
        self.indices = list(range(max(start_idx, min_idx), max_idx))

        print(f"MSUDataset [{split}]: {len(self.indices)} samples")
        print(f"  Market data shape: {self.market_data.shape}")
        print(f"  Index range: [{min(self.indices)}, {max(self.indices)}]")
        print(f"  Self-supervised: No labels needed!")

    def _extract_weekly_data(self, daily_data):
        """
        Extract weekly data from daily data (every 5 days)

        Args:
            daily_data: [window_days, market_features]

        Returns:
            weekly_data: [window_len, market_features]
        """
        # Take every 5th day starting from day 4 (0-indexed: day 4, 9, 14, ...)
        weekly_data = daily_data[4::5]
        return weekly_data

    def _normalize_weekly_data(self, weekly_data):
        """
        Z-score normalization across time dimension

        Args:
            weekly_data: [window_len, market_features]

        Returns:
            normed: [window_len, market_features]
        """
        mean = np.mean(weekly_data, axis=0, keepdims=True)
        std = np.std(weekly_data, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normed = (weekly_data - mean) / std
        return normed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, i):
        """
        Returns:
            market_data: [window_len, market_features] torch.FloatTensor

        Note: Returns the same data as both input and target.
              Masking will be applied in the training loop.
        """
        idx = self.indices[i]

        # Extract input window (past window_days = 65 days)
        input_start = idx - self.window_days + 1
        input_end = idx + 1
        daily_data = self.market_data[input_start:input_end]  # [65, market_features]

        # Extract weekly data
        weekly_data = self._extract_weekly_data(daily_data)  # [13, market_features]

        # Normalize
        market_data_normalized = self._normalize_weekly_data(weekly_data)  # [13, market_features]

        # Convert to tensor
        market_data_tensor = torch.from_numpy(market_data_normalized).float()  # [13, market_features]

        return market_data_tensor


def test_dataset():
    """Test the dataset"""
    import os

    # Get the directory of this script
    data_dir = os.path.dirname(os.path.abspath(__file__))

    print("="*80)
    print("Testing MSUDataset")
    print("="*80)

    # Test training set
    print("\n[TRAIN SET]")
    train_dataset = MSUDataset(data_dir, split='train')
    print(f"Dataset length: {len(train_dataset)}")

    # Test a sample
    market_input, targets = train_dataset[0]
    print(f"Sample 0:")
    print(f"  market_input shape: {market_input.shape}")
    print(f"  market_input dtype: {market_input.dtype}")
    print(f"  market_input range: [{market_input.min():.4f}, {market_input.max():.4f}]")
    print(f"  targets: {targets.tolist()}")
    print(f"  targets shape: {targets.shape}")
    print(f"  targets dtype: {targets.dtype}")
    print(f"    raw_total_return: {targets[0].item():.6f}")
    print(f"    raw_total_sigma: {targets[1].item():.6f}")

    # Test validation set
    print("\n[VAL SET]")
    val_dataset = MSUDataset(data_dir, split='val')
    print(f"Dataset length: {len(val_dataset)}")

    # Test test set
    print("\n[TEST SET]")
    test_dataset = MSUDataset(data_dir, split='test')
    print(f"Dataset length: {len(test_dataset)}")

    # Test DataLoader
    print("\n[DATALOADER TEST]")
    from torch.utils.data import DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    for batch_idx, (market_input, targets) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  market_input shape: {market_input.shape}")
        print(f"  targets shape: {targets.shape}")
        print(f"  targets[:5, 0] (returns): {targets[:5, 0].tolist()}")
        print(f"  targets[:5, 1] (sigmas): {targets[:5, 1].tolist()}")
        if batch_idx >= 2:
            break

    print("\n✅ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()
