"""
Custom Loss Functions for ASU Supervised Pre-training

Loss functions for stock ranking/scoring tasks:
- Pairwise Ranking Loss: Learn relative ranking between stocks
- Regression Loss: Predict future returns
- Combined Loss: Ranking + Regression 混合
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PairwiseRankingLoss(nn.Module):
    """
    Pairwise Ranking Loss - Learn relative ranking between stocks

    Core concept:
    - If future_ror[i] > future_ror[j], then score[i] should > score[j]
    - Use Hinge Loss: max(0, margin - (score_i - score_j))
    - Only consider pairs with significant RoR difference (avoid noise)

    Use cases:
    - ASU output used for stock ranking/selection
    - Focus on relative performance rather than absolute prediction
    - Aligned with DeepTrader portfolio optimization

    Args:
        margin (float): Ranking margin (default: 0.1)
        min_ror_diff (float): Minimum RoR difference threshold, pairs below this do not contribute to loss (default: 0.01)
        reduction (str): 'mean' or 'sum' (default: 'mean')

    Returns:
        loss: Pairwise ranking loss
    """

    def __init__(self, margin=0.1, min_ror_diff=0.01, reduction='mean'):
        super().__init__()
        self.margin = margin
        self.min_ror_diff = min_ror_diff
        self.reduction = reduction

    def forward(self, scores, future_ror, mask=None):
        """
        Args:
            scores: [batch, num_stocks] - Scores output by ASU
            future_ror: [batch, num_stocks] - Future return of return (ground truth)
            mask: [batch, num_stocks] - Mask for tradable stocks (optional)

        Returns:
            loss: Pairwise ranking loss
        """
        batch_size, num_stocks = scores.shape

        # Handle mask: Set untradable stocks to very small value to exclude their influence
        if mask is not None:
            scores = scores.clone()
            future_ror = future_ror.clone()
            scores[mask] = -1e9
            future_ror[mask] = -1e9

        # Expand to pairwise comparison
        # scores_i: [batch, num_stocks, 1]
        # scores_j: [batch, 1, num_stocks]
        scores_i = scores.unsqueeze(2)
        scores_j = scores.unsqueeze(1)

        ror_i = future_ror.unsqueeze(2)
        ror_j = future_ror.unsqueeze(1)

        # Calculate RoR difference
        ror_diff = ror_i - ror_j  # [batch, num_stocks, num_stocks]

        # Only consider pairs with significant RoR difference (reduce noise)
        significant_pairs = (torch.abs(ror_diff) > self.min_ror_diff).float()

        # When ror_i > ror_j, should have score_i > score_j
        should_rank_higher = (ror_diff > 0).float()

        # Hinge loss: max(0, margin - (score_i - score_j))
        # Only compute when should rank higher and difference is significant
        score_diff = scores_i - scores_j
        loss = torch.relu(self.margin - score_diff) * should_rank_higher * significant_pairs

        # Reduction
        if self.reduction == 'mean':
            # Average only over valid pairs (avoid dividing by too many zeros)
            num_valid_pairs = significant_pairs.sum() + 1e-8
            return loss.sum() / num_valid_pairs
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


class RegressionLoss(nn.Module):
    """
    Regression Loss - Predict future returns

    Directly minimize difference between predicted and actual returns

    Args:
        loss_type (str): 'mse' or 'mae' (default: 'mse')

    Returns:
        loss: Regression loss
    """

    def __init__(self, loss_type='mse'):
        super().__init__()
        self.loss_type = loss_type.lower()

        if self.loss_type == 'mse':
            self.loss_fn = nn.MSELoss(reduction='none')
        elif self.loss_type == 'mae':
            self.loss_fn = nn.L1Loss(reduction='none')
        else:
            raise ValueError(f"Unknown loss_type: {loss_type}")

    def forward(self, predictions, future_ror, mask=None):
        """
        Args:
            predictions: [batch, num_stocks] - 預測的收益率
            future_ror: [batch, num_stocks] - True future returns
            mask: [batch, num_stocks] - Mask for tradable stocks (optional)

        Returns:
            loss: Regression loss (averaged over valid stocks)
        """
        # Compute loss
        loss = self.loss_fn(predictions, future_ror)

        # Apply mask if provided
        if mask is not None:
            # Only compute loss for tradable stocks
            valid_mask = ~mask  # Invert mask: True for valid stocks
            loss = loss * valid_mask.float()
            num_valid = valid_mask.float().sum() + 1e-8
            return loss.sum() / num_valid
        else:
            return loss.mean()


class ICLoss(nn.Module):
    """
    IC Loss (Information Coefficient) - Based on ranking correlation

    IC = Spearman Rank Correlation (跨股票計算)
    - Robust to outliers
    - Focus on relative ranking rather than absolute values
    - Standard metric in quantitative finance

    與 MSU 的 ICLoss 不同：
    - MSU: Compute IC across time (same market at different times)
    - ASU: Compute IC across stocks (different stocks at same time)

    Args:
        eps (float): 數值穩定性參數 (default: 1e-8)

    Returns:
        loss: 1 - IC (越小越好)
    """

    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, scores, future_ror, mask=None):
        """
        Args:
            scores: [batch, num_stocks]
            future_ror: [batch, num_stocks]
            mask: [batch, num_stocks] - Mask for tradable stocks (optional)

        Returns:
            loss: 1 - average IC across batch
        """
        batch_size = scores.shape[0]
        ic_list = []

        for i in range(batch_size):
            score_i = scores[i]
            ror_i = future_ror[i]

            # Apply mask
            if mask is not None:
                valid_mask = ~mask[i]
                score_i = score_i[valid_mask]
                ror_i = ror_i[valid_mask]

            # Skip if not enough stocks
            if len(score_i) < 2:
                continue

            # Compute ranks (argsort twice)
            rank_score = torch.argsort(torch.argsort(score_i)).float()
            rank_ror = torch.argsort(torch.argsort(ror_i)).float()

            # Compute Spearman correlation on ranks
            rank_score_centered = rank_score - rank_score.mean()
            rank_ror_centered = rank_ror - rank_ror.mean()

            numerator = (rank_score_centered * rank_ror_centered).sum()
            denominator = torch.sqrt(
                (rank_score_centered ** 2).sum() * (rank_ror_centered ** 2).sum()
            )

            ic = numerator / (denominator + self.eps)
            ic_list.append(ic)

        # Average IC across batch
        if len(ic_list) == 0:
            return torch.tensor(0.0, device=scores.device)

        avg_ic = torch.stack(ic_list).mean()

        # Loss = 1 - IC
        # IC = 1  → loss = 0 (完美)
        # IC = 0  → loss = 1
        # IC = -1 → loss = 2 (最差)
        loss = 1.0 - avg_ic

        return loss


class CombinedLoss(nn.Module):
    """
    Combined Loss - Ranking + Regression 混合

    Balance relative ranking and numerical prediction

    Recommended configuration:
    - alpha=0.8, beta=0.2: Primarily focus on ranking (推薦用於 ASU)
    - alpha=0.5, beta=0.5: Balance ranking and prediction

    Args:
        alpha (float): Ranking loss 權重 (default: 0.8)
        beta (float): Regression loss 權重 (default: 0.2)
        ranking_loss_type (str): 'pairwise' or 'ic' (default: 'pairwise')
        regression_loss_type (str): 'mse' or 'mae' (default: 'mse')
        margin (float): Pairwise ranking margin (default: 0.1)
        min_ror_diff (float): Minimum RoR difference for pairwise (default: 0.01)

    Returns:
        loss: alpha * ranking_loss + beta * regression_loss
    """

    def __init__(self, alpha=0.8, beta=0.2, ranking_loss_type='pairwise',
                 regression_loss_type='mse', margin=0.1, min_ror_diff=0.01):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

        # Ranking loss
        if ranking_loss_type == 'pairwise':
            self.ranking_loss = PairwiseRankingLoss(margin=margin, min_ror_diff=min_ror_diff)
        elif ranking_loss_type == 'ic':
            self.ranking_loss = ICLoss()
        else:
            raise ValueError(f"Unknown ranking_loss_type: {ranking_loss_type}")

        # Regression loss
        self.regression_loss = RegressionLoss(loss_type=regression_loss_type)

        self.ranking_loss_type = ranking_loss_type
        self.regression_loss_type = regression_loss_type

    def forward(self, scores, predictions, future_ror, mask=None):
        """
        Args:
            scores: [batch, num_stocks] - Scores output by ASU (For ranking)
            predictions: [batch, num_stocks] - 預測的收益率 (For regression)
            future_ror: [batch, num_stocks] - True future returns
            mask: [batch, num_stocks] - Mask for tradable stocks (optional)

        Returns:
            loss: combined loss
            info: dict with individual loss components
        """
        ranking = self.ranking_loss(scores, future_ror, mask)
        regression = self.regression_loss(predictions, future_ror, mask)

        total_loss = self.alpha * ranking + self.beta * regression

        # Return loss and components for logging
        info = {
            'ranking_loss': ranking.item(),
            'regression_loss': regression.item(),
            'total_loss': total_loss.item()
        }

        return total_loss, info


def get_loss_function(loss_type='Ranking', **kwargs):
    """
    Factory function to get loss function

    Args:
        loss_type (str): 'Ranking', 'Pairwise', 'IC', 'Regression', 'MSE', 'MAE', 'Combined'
        **kwargs: Additional arguments for loss function

    Returns:
        loss_fn: Loss function
        need_predictions: Whether this loss needs regression predictions (bool)
    """
    loss_type_upper = loss_type.upper()

    # Ranking-based losses (only need scores)
    if loss_type_upper in ['RANKING', 'PAIRWISE']:
        return PairwiseRankingLoss(**kwargs), False

    elif loss_type_upper == 'IC':
        return ICLoss(**kwargs), False

    # Regression losses (need predictions)
    elif loss_type_upper in ['REGRESSION', 'MSE']:
        return RegressionLoss(loss_type='mse', **kwargs), True

    elif loss_type_upper == 'MAE':
        return RegressionLoss(loss_type='mae', **kwargs), True

    # Combined loss (needs both scores and predictions)
    elif loss_type_upper == 'COMBINED':
        return CombinedLoss(**kwargs), True

    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


if __name__ == '__main__':
    """Test behavior of different loss functions"""
    print("="*80)
    print("Testing ASU Loss Functions")
    print("="*80)

    # Create test data
    # Assume 3 samples, each with 5 stocks
    batch_size = 3
    num_stocks = 5

    # Ground truth: Future returns
    future_ror = torch.tensor([
        [0.05, 0.02, -0.01, 0.03, -0.02],  # Sample 1
        [0.01, -0.03, 0.04, 0.00, 0.02],   # Sample 2
        [-0.01, 0.03, 0.01, -0.02, 0.05],  # Sample 3
    ])

    test_cases = [
        ("完美預測 (scores = future_ror)",
         future_ror.clone(),
         future_ror.clone()),

        ("良好排序但數值偏移",
         future_ror + 0.01,  # scores
         future_ror + 0.01), # predictions

        ("排序正確但收縮",
         future_ror * 0.5,
         future_ror * 0.5),

        ("隨機評分",
         torch.randn(batch_size, num_stocks) * 0.05,
         torch.randn(batch_size, num_stocks) * 0.05),

        ("反向排序 (最差)",
         -future_ror,
         -future_ror),
    ]

    # Create different loss functions
    pairwise_loss = PairwiseRankingLoss(margin=0.05, min_ror_diff=0.01)
    ic_loss = ICLoss()
    mse_loss = RegressionLoss(loss_type='mse')
    mae_loss = RegressionLoss(loss_type='mae')
    combined_loss = CombinedLoss(alpha=0.8, beta=0.2, ranking_loss_type='pairwise')

    print(f"\n{'Case':<35} {'Pairwise':>12} {'IC':>12} {'MSE':>12} {'MAE':>12} {'Combined':>12}")
    print("-"*105)

    for name, scores, predictions in test_cases:
        pw = pairwise_loss(scores, future_ror).item()
        ic = ic_loss(scores, future_ror).item()
        mse = mse_loss(predictions, future_ror).item()
        mae = mae_loss(predictions, future_ror).item()
        comb, _ = combined_loss(scores, predictions, future_ror)
        comb = comb.item()

        print(f"{name:<35} {pw:>12.6f} {ic:>12.6f} {mse:>12.6f} {mae:>12.6f} {comb:>12.6f}")

    print("\n" + "="*80)
    print("Testing complete!")
    print("="*80)
    print("\nExplanation:")
    print("- Pairwise Loss: Closer to 0 means better ranking")
    print("- IC Loss: Closer to 0 means higher ranking correlation (IC ≈ 1)")
    print("- MSE/MAE: Closer to 0 means more accurate numerical prediction")
    print("- Combined: Comprehensive consideration of ranking and numerical prediction")
