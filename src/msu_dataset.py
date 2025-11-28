"""
MSU Dataset for Stage 1 Pretraining

This dataset loads market data and ground truth for MSU Stage 1 binary classification.

Input:  market_data (13 weeks)
Output: trend_label (0.0 or 1.0)
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os


class MSUDataset(Dataset):
    """
    MSU Stage 1 Pretraining Dataset

    Loads market data and corresponding trend labels for MSU Stage 1 pretraining.

    Inputs:
        - market_data: [window_len, market_features] normalized weekly data

    Outputs:
        - trend_label: float in {0.0, 1.0}
            1.0 if raw_mu > 0 (uptrend)
            0.0 if raw_mu <= 0 (downtrend/flat)
    """

    def __init__(self,
                 data_dir,
                 split='train',
                 window_len=13,
                 trade_len=21):
        """
        Args:
            data_dir: Directory containing market_data.npy and *_ground_truth.json files
            split: 'train', 'val', or 'test'
            window_len: Input window length in weeks (13)
            trade_len: Prediction horizon in days (21)
        """
        super().__init__()

        self.data_dir = data_dir
        self.split = split
        self.window_len = window_len
        self.trade_len = trade_len
        self.window_days = window_len * 5  # 13 weeks = 65 days

        # Load market data
        market_data_path = os.path.join(data_dir, 'market_data.npy')
        self.market_data = np.load(market_data_path)  # [T, market_features]

        # Load ground truth (MSU-specific naming)
        gt_path = os.path.join(data_dir, f'MSU_{split}_ground_truth.json')
        with open(gt_path, 'r') as f:
            gt_data = json.load(f)

        # Extract records
        self.records = gt_data['ground_truth_records']

        # Build index mapping: idx -> record
        self.idx_to_record = {}
        for record in self.records:
            # Use input_end as the index (the last day of input window)
            idx = record['input_end'] - 1  # input_end is exclusive, so -1 to get the actual last day
            self.idx_to_record[idx] = record

        self.indices = sorted(list(self.idx_to_record.keys()))

        print(f"MSUDataset [{split}]: {len(self.indices)} samples")
        print(f"  Market data shape: {self.market_data.shape}")
        print(f"  Index range: [{min(self.indices)}, {max(self.indices)}]")

        # Statistics
        trend_labels = [self.idx_to_record[idx]['trend_label'] for idx in self.indices]
        uptrend_count = sum(trend_labels)
        print(f"  Uptrend samples: {uptrend_count}/{len(trend_labels)} ({100*uptrend_count/len(trend_labels):.1f}%)")

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
            market_input: [window_len, market_features] torch.FloatTensor
            trend_label: scalar torch.FloatTensor in {0.0, 1.0}
        """
        idx = self.indices[i]
        record = self.idx_to_record[idx]

        # Extract input window (past window_days = 65 days)
        input_start = record['input_start']
        input_end = record['input_end']
        daily_data = self.market_data[input_start:input_end]  # [65, market_features]

        # Extract weekly data
        weekly_data = self._extract_weekly_data(daily_data)  # [13, market_features]

        # Normalize
        market_input = self._normalize_weekly_data(weekly_data)  # [13, market_features]

        # Get trend label
        trend_label = record['trend_label']  # 0.0 or 1.0

        # Convert to tensors
        market_input = torch.from_numpy(market_input).float()  # [13, market_features]
        trend_label = torch.tensor(trend_label, dtype=torch.float32)  # scalar

        return market_input, trend_label


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
    market_input, trend_label = train_dataset[0]
    print(f"Sample 0:")
    print(f"  market_input shape: {market_input.shape}")
    print(f"  market_input dtype: {market_input.dtype}")
    print(f"  market_input range: [{market_input.min():.4f}, {market_input.max():.4f}]")
    print(f"  trend_label: {trend_label.item()}")
    print(f"  trend_label dtype: {trend_label.dtype}")

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

    for batch_idx, (market_input, trend_label) in enumerate(train_loader):
        print(f"Batch {batch_idx}:")
        print(f"  market_input shape: {market_input.shape}")
        print(f"  trend_label shape: {trend_label.shape}")
        print(f"  trend_label values: {trend_label[:5].tolist()}")
        if batch_idx >= 2:
            break

    print("\n✅ Dataset test passed!")


if __name__ == '__main__':
    test_dataset()
