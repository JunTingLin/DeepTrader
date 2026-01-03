"""
Dataset loader for ASU Supervised Pre-training

Loads stock data and computes future returns as ground truth for ranking/regression tasks.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class ASUDataset(Dataset):
    """
    Dataset for ASU supervised pre-training

    Data format:
        stocks_data: [num_stocks, num_days, num_features]
        ror: [num_stocks, num_days] - daily return on return

    Args:
        stocks_data_path (str): Path to stocks_data.npy
        ror_path (str): Path to ror.npy
        start_idx (int): Start index in time dimension
        end_idx (int): End index in time dimension
        window_len (int): Input window length (default: 13)
        horizon (int): Future return horizon in days (default: 21)
        stride (int): Stride for sliding window (default: 1)

    Returns:
        stocks_window: [num_stocks, window_len, num_features] - Input window
        future_ror: [num_stocks] - Future return for each stock
        mask: [num_stocks] - True if stock has NaN/missing data
        time_idx: int - Time index of this sample
    """

    def __init__(self, stocks_data_path, ror_path, start_idx, end_idx,
                 window_len=13, horizon=21, stride=1):
        super(ASUDataset, self).__init__()

        # Load data
        self.stocks_data = np.load(stocks_data_path)  # [num_stocks, num_days, num_features]
        self.ror = np.load(ror_path)  # [num_stocks, num_days]

        self.window_len = window_len
        self.horizon = horizon
        self.stride = stride
        self.start_idx = start_idx
        self.end_idx = end_idx

        # Data shape
        self.num_stocks, self.num_days, self.num_features = self.stocks_data.shape

        # Compute valid sample indices
        # Match DeepTrader's num_steps calculation: (end_idx - start_idx) // stride
        min_history = (self.window_len + 1) * 5 - 1
        self.valid_indices = []

        # Calculate expected number of steps (matching DeepTrader)
        num_steps = (end_idx - start_idx) // stride

        for step in range(num_steps):
            t = start_idx + step * stride
            # Check if we have enough history and future
            if t >= min_history and t + self.horizon < self.num_days:
                self.valid_indices.append(t)

        print(f"Loaded ASU dataset:")
        print(f"  Stocks data shape: {self.stocks_data.shape}")
        print(f"  RoR shape: {self.ror.shape}")
        print(f"  Time range: [{start_idx}, {end_idx})")
        print(f"  Window length: {window_len} days")
        print(f"  Future horizon: {horizon} days")
        print(f"  Stride: {stride}")
        print(f"  Number of valid samples: {len(self.valid_indices)}")

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a single sample (模仿 DataGenerator 的周採樣方式)

        Returns:
            stocks_window: [num_stocks, window_len, num_features] - 13 週的數據（每週取週五）
            future_ror: [num_stocks] - cumulative return over horizon
            mask: [num_stocks] - True if stock has NaN/missing data
            time_idx: int
        """
        t = self.valid_indices[idx]

        # 模仿 DataGenerator._get_data() 的周採樣方式
        # 1. 取 (window_len + 1) * 5 = 70 天的原始數據
        raw_window = self.stocks_data[:, t - (self.window_len + 1) * 5 + 1:t + 1, :]  # [num_stocks, 70, num_features]

        # 2. Reshape 成 (window_len + 1) 週，每週 5 天
        # raw_window shape: [num_stocks, 14 weeks, 5 days, num_features]
        raw_window = raw_window.reshape(self.num_stocks, self.window_len + 1, 5, self.num_features)

        # 3. 取後 13 週，每週取最後一天（週五）
        stocks_window = raw_window[:, 1:, -1, :]  # [num_stocks, 13, num_features]

        # Compute future return: cumulative return from t to t+horizon
        future_returns = self.ror[:, t + 1:t + 1 + self.horizon]  # [num_stocks, horizon]

        # Cumulative return: prod(1 + r_i) - 1
        future_ror = np.prod(1 + future_returns, axis=1) - 1  # [num_stocks]

        # Create mask for stocks with NaN/Inf values
        mask = np.isnan(stocks_window).any(axis=(1, 2)) | np.isnan(future_ror) | np.isinf(future_ror)

        # Replace NaN/Inf with 0 for numerical stability
        stocks_window = np.where(np.isnan(stocks_window) | np.isinf(stocks_window), 0, stocks_window)
        future_ror = np.where(np.isnan(future_ror) | np.isinf(future_ror), 0, future_ror)

        # Convert to tensors
        stocks_window = torch.FloatTensor(stocks_window)  # [num_stocks, window_len=13, num_features]
        future_ror = torch.FloatTensor(future_ror)  # [num_stocks]
        mask = torch.BoolTensor(mask)  # [num_stocks]

        return stocks_window, future_ror, mask, t

    def get_num_stocks(self):
        """Return number of stocks"""
        return self.num_stocks

    def get_num_features(self):
        """Return number of features"""
        return self.num_features


def get_dataloaders(data_dir, train_idx, train_idx_end, val_idx, test_idx, test_idx_end,
                    window_len=13, horizon=21, batch_size=32, num_workers=4,
                    train_stride=1, val_stride=5, test_stride=5):
    """
    Create train/val/test dataloaders

    Args:
        data_dir (str): Directory containing stocks_data.npy and ror.npy
        train_idx (int): Training start index
        train_idx_end (int): Training end index
        val_idx (int): Validation start index
        test_idx (int): Test start index
        test_idx_end (int): Test end index
        window_len (int): Input window length
        horizon (int): Future return horizon
        batch_size (int): Batch size
        num_workers (int): Number of workers for data loading
        train_stride (int): Stride for training samples (default: 1 for dense sampling)
        val_stride (int): Stride for validation samples (default: 5 for faster validation)
        test_stride (int): Stride for test samples (default: 5)

    Returns:
        train_loader, val_loader, test_loader, num_stocks, num_features
    """
    import os

    stocks_data_path = os.path.join(data_dir, 'stocks_data.npy')
    ror_path = os.path.join(data_dir, 'ror.npy')

    # Create datasets
    train_dataset = ASUDataset(
        stocks_data_path, ror_path,
        start_idx=train_idx,
        end_idx=train_idx_end,
        window_len=window_len,
        horizon=horizon,
        stride=train_stride
    )

    val_dataset = ASUDataset(
        stocks_data_path, ror_path,
        start_idx=val_idx,
        end_idx=test_idx,
        window_len=window_len,
        horizon=horizon,
        stride=val_stride
    )

    test_dataset = ASUDataset(
        stocks_data_path, ror_path,
        start_idx=test_idx,
        end_idx=test_idx_end,
        window_len=window_len,
        horizon=horizon,
        stride=test_stride
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

    num_stocks = train_dataset.get_num_stocks()
    num_features = train_dataset.get_num_features()

    return train_loader, val_loader, test_loader, num_stocks, num_features


if __name__ == '__main__':
    """Test the dataset loader"""
    import sys
    import os

    # Add parent directory to path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print("="*80)
    print("Testing ASU Dataset Loader")
    print("="*80)

    # Test with DJIA data
    data_dir = './src/data/DJIA/feature34-Inter-P532/'

    if not os.path.exists(data_dir):
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Create dataloaders
    train_loader, val_loader, test_loader, num_stocks, num_features = get_dataloaders(
        data_dir=data_dir,
        train_idx=0,
        train_idx_end=1304,
        val_idx=1304,
        test_idx=2087,
        test_idx_end=2673,
        window_len=13,
        horizon=21,
        batch_size=4,
        num_workers=0,
        train_stride=10,  # Use stride=10 for faster testing
        val_stride=10,
        test_stride=10
    )

    print(f"\nDataset info:")
    print(f"  Number of stocks: {num_stocks}")
    print(f"  Number of features: {num_features}")

    # Test train loader
    print(f"\nTesting train loader:")
    for batch_idx, (stocks_window, future_ror, mask, time_idx) in enumerate(train_loader):
        print(f"  Batch {batch_idx}:")
        print(f"    stocks_window shape: {stocks_window.shape}")
        print(f"    future_ror shape: {future_ror.shape}")
        print(f"    mask shape: {mask.shape}")
        print(f"    time_idx: {time_idx.tolist()}")
        print(f"    future_ror stats: min={future_ror.min():.4f}, max={future_ror.max():.4f}, mean={future_ror.mean():.4f}")
        print(f"    masked stocks per sample: {mask.sum(dim=1).tolist()}")

        if batch_idx >= 2:
            break

    print("\n" + "="*80)
    print("Dataset loader test completed!")
    print("="*80)
