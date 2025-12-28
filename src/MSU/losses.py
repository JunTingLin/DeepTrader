"""
Custom Loss Functions for MSU Training

包含各種適合金融時序預測的損失函數：
- Sharpe Ratio Loss: 直接優化風險調整後收益
- IC Loss (Information Coefficient): 基於排序的相關性
- Combined Loss: MSE + Correlation/IC 混合
"""

import torch
import torch.nn as nn


class SharpeRatioLoss(nn.Module):
    """
    Sharpe Ratio Loss - 直接優化投資組合的 Sharpe Ratio

    核心概念：
    - 如果預測方向正確（pred 和 true 同號），就能賺錢
    - Sharpe Ratio = (平均收益 - 無風險利率) / 收益標準差
    - 最大化 Sharpe Ratio = 最小化 -Sharpe Ratio

    適用場景：
    - 預測結果直接用於交易決策
    - 在乎風險調整後的收益，而非絕對預測準確度

    Args:
        risk_free_rate (float): 無風險利率 (default: 0.0)
        eps (float): 數值穩定性參數 (default: 1e-8)
        clip_sharpe (float): Sharpe Ratio 裁剪上限，避免極端值 (default: 10.0)

    Returns:
        loss: -Sharpe Ratio (越小越好)
    """

    def __init__(self, risk_free_rate=0.0, eps=1e-8, clip_sharpe=10.0):
        super().__init__()
        self.rf = risk_free_rate
        self.eps = eps
        self.clip_sharpe = clip_sharpe

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [batch_size] - 預測的 rho (市場趨勢參數)
            y_true: [batch_size] - 真實的 rho

        Returns:
            loss: -Sharpe Ratio
        """
        # 策略收益：如果預測方向和真實方向一致，就能獲得收益
        # 簡化模型：假設按照預測的 rho 比例做多/空，真實收益按 true rho
        # returns = pred * true
        # 如果 pred > 0.5 且 true > 0.5 (都看漲) → returns > 0 (賺錢)
        # 如果 pred < 0.5 且 true < 0.5 (都看跌) → returns > 0 (賺錢)

        # 將 rho [0, 1] 轉換為方向 [-1, 1]
        # rho > 0.5 → 看漲 (正)
        # rho < 0.5 → 看跌 (負)
        pred_direction = (y_pred - 0.5) * 2  # [-1, 1]
        true_direction = (y_true - 0.5) * 2  # [-1, 1]

        # 計算策略收益：方向一致時為正，方向相反時為負
        returns = pred_direction * true_direction

        # 計算 Sharpe Ratio
        mean_return = returns.mean()
        std_return = returns.std()

        # Sharpe Ratio = (mean - rf) / std
        sharpe = (mean_return - self.rf) / (std_return + self.eps)

        # Clip 避免極端值
        sharpe = torch.clamp(sharpe, -self.clip_sharpe, self.clip_sharpe)

        # Loss = -Sharpe (最小化 loss = 最大化 Sharpe)
        loss = -sharpe

        return loss


class ICLoss(nn.Module):
    """
    IC Loss (Information Coefficient) - 基於排序的相關性

    IC = Spearman Rank Correlation
    - 對離群值不敏感
    - 關注相對排序而非絕對數值
    - 量化金融的標準指標

    Args:
        eps (float): 數值穩定性參數 (default: 1e-8)

    Returns:
        loss: 1 - IC (越小越好)
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [batch_size]
            y_true: [batch_size]

        Returns:
            loss: 1 - Spearman correlation
        """
        # 計算排序 (rank)
        # argsort 兩次得到 rank
        rank_pred = torch.argsort(torch.argsort(y_pred)).float()
        rank_true = torch.argsort(torch.argsort(y_true)).float()

        # 在 rank 上計算 Pearson correlation
        rank_pred_centered = rank_pred - rank_pred.mean()
        rank_true_centered = rank_true - rank_true.mean()

        numerator = (rank_pred_centered * rank_true_centered).sum()
        denominator = torch.sqrt(
            (rank_pred_centered ** 2).sum() * (rank_true_centered ** 2).sum()
        )

        ic = numerator / (denominator + self.eps)

        # Loss = 1 - IC
        loss = 1.0 - ic

        return loss


class CorrelationLoss(nn.Module):
    """
    Correlation Loss - 最大化 Pearson Correlation

    直接優化預測值和真實值的線性相關性

    Args:
        eps (float): 數值穩定性參數 (default: 1e-8)

    Returns:
        loss: 1 - correlation (越小越好)
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [batch_size]
            y_true: [batch_size]

        Returns:
            loss: 1 - Pearson correlation
        """
        # Center the data
        y_pred_centered = y_pred - y_pred.mean()
        y_true_centered = y_true - y_true.mean()

        # Pearson correlation
        numerator = (y_pred_centered * y_true_centered).sum()
        denominator = torch.sqrt(
            (y_pred_centered ** 2).sum() * (y_true_centered ** 2).sum()
        )

        correlation = numerator / (denominator + self.eps)

        # Loss = 1 - correlation
        # correlation = 1  → loss = 0 (完美)
        # correlation = 0  → loss = 1
        # correlation = -1 → loss = 2 (最差)
        loss = 1.0 - correlation

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss - MSE + Sharpe/IC/Correlation 混合

    兼顧數值準確性和趨勢一致性

    Args:
        alpha (float): MSE 權重 (default: 0.5)
        beta (float): 趨勢 loss 權重 (default: 0.5)
        trend_loss_type (str): 趨勢損失類型 ('sharpe', 'ic', 'corr')
        eps (float): 數值穩定性參數

    Returns:
        loss: alpha * MSE + beta * trend_loss
    """

    def __init__(self, alpha=0.5, beta=0.5, trend_loss_type='sharpe', eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        self.mse_loss = nn.MSELoss()

        if trend_loss_type == 'sharpe':
            self.trend_loss = SharpeRatioLoss(eps=eps)
        elif trend_loss_type == 'ic':
            self.trend_loss = ICLoss(eps=eps)
        elif trend_loss_type == 'corr':
            self.trend_loss = CorrelationLoss(eps=eps)
        else:
            raise ValueError(f"Unknown trend_loss_type: {trend_loss_type}")

        self.trend_loss_type = trend_loss_type

    def forward(self, y_pred, y_true):
        """
        Args:
            y_pred: [batch_size]
            y_true: [batch_size]

        Returns:
            loss: combined loss
        """
        mse = self.mse_loss(y_pred, y_true)
        trend = self.trend_loss(y_pred, y_true)

        return self.alpha * mse + self.beta * trend


def get_loss_function(loss_type='MSE', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type (str): 'MSE', 'MAE', 'Sharpe', 'IC', 'Corr', 'Combined'
        **kwargs: Additional arguments for loss function

    Returns:
        loss_fn: Loss function
    """
    loss_type = loss_type.upper()

    if loss_type == 'MSE':
        return nn.MSELoss()

    elif loss_type == 'MAE':
        return nn.L1Loss()

    elif loss_type == 'SHARPE':
        return SharpeRatioLoss(**kwargs)

    elif loss_type == 'IC':
        return ICLoss(**kwargs)

    elif loss_type == 'CORR' or loss_type == 'CORRELATION':
        return CorrelationLoss(**kwargs)

    elif loss_type == 'COMBINED':
        return CombinedLoss(**kwargs)

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


if __name__ == '__main__':
    """測試不同 loss function 的行為"""
    print("="*80)
    print("測試 MSU Loss Functions")
    print("="*80)

    # 創建測試數據
    targets = torch.tensor([0.0, 1.0, 0.2, 0.8, 0.5])

    test_cases = [
        ("完美預測", torch.tensor([0.0, 1.0, 0.2, 0.8, 0.5])),
        ("趨勢正確但偏移", torch.tensor([0.2, 0.9, 0.3, 0.7, 0.5])),
        ("永遠預測平均值", torch.tensor([0.5, 0.5, 0.5, 0.5, 0.5])),
        ("趨勢完全相反", torch.tensor([1.0, 0.0, 0.8, 0.2, 0.5])),
        ("隨機預測", torch.tensor([0.3, 0.7, 0.1, 0.6, 0.9])),
    ]

    # 創建不同的 loss functions
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()
    sharpe_loss = SharpeRatioLoss()
    ic_loss = ICLoss()
    corr_loss = CorrelationLoss()
    combined_loss = CombinedLoss(alpha=0.5, beta=0.5, trend_loss_type='sharpe')

    print(f"\n{'Case':<30} {'MSE':>10} {'MAE':>10} {'Sharpe':>10} {'IC':>10} {'Corr':>10} {'Combined':>10}")
    print("-"*100)

    for name, preds in test_cases:
        mse = mse_loss(preds, targets).item()
        mae = mae_loss(preds, targets).item()
        sharpe = sharpe_loss(preds, targets).item()
        ic = ic_loss(preds, targets).item()
        corr = corr_loss(preds, targets).item()
        comb = combined_loss(preds, targets).item()

        print(f"{name:<30} {mse:>10.4f} {mae:>10.4f} {sharpe:>10.4f} {ic:>10.4f} {corr:>10.4f} {comb:>10.4f}")
