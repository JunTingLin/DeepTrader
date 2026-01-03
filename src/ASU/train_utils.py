"""
Training utilities for ASU supervised learning

Shared training utilities used by train_asu_gcn.py and train_asu_te2d.py
"""

import torch
import numpy as np
from scipy.stats import spearmanr


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """Early stopping based on validation metric"""

    def __init__(self, patience=20, mode='max', min_delta=0, verbose=True):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        else:
            if self.mode == 'max':
                improved = score > self.best_score + self.min_delta
            else:
                improved = score < self.best_score - self.min_delta

            if not improved:
                self.counter += 1
                if self.verbose:
                    print(f'EarlyStopping counter: {self.counter}/{self.patience}')
                if self.counter >= self.patience:
                    self.early_stop = True
            else:
                self.best_score = score
                self.best_epoch = epoch
                self.counter = 0


def compute_ic(scores, targets, mask=None):
    """
    Compute Information Coefficient (Spearman correlation) across stocks

    Args:
        scores: [batch, num_stocks]
        targets: [batch, num_stocks]
        mask: [batch, num_stocks] - True for invalid stocks

    Returns:
        avg_ic: Average IC across batch
    """
    batch_size = scores.shape[0]
    ic_list = []

    for i in range(batch_size):
        score_i = scores[i].detach().cpu().numpy()
        target_i = targets[i].detach().cpu().numpy()

        # Apply mask
        if mask is not None:
            valid_mask = ~mask[i].cpu().numpy()
            score_i = score_i[valid_mask]
            target_i = target_i[valid_mask]

        # Skip if not enough stocks
        if len(score_i) < 2:
            continue

        # Compute Spearman correlation
        try:
            ic, _ = spearmanr(score_i, target_i)
            if not np.isnan(ic):
                ic_list.append(ic)
        except:
            pass

    if len(ic_list) == 0:
        return 0.0

    return np.mean(ic_list)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, need_predictions):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (stocks_window, future_ror, mask, _) in enumerate(train_loader):
        stocks_window = stocks_window.to(device)
        future_ror = future_ror.to(device)
        mask = mask.to(device)

        optimizer.zero_grad()
        scores, predictions = model(stocks_window, mask)

        # Compute loss
        if need_predictions and predictions is not None:
            if 'Combined' in criterion.__class__.__name__:
                loss, info = criterion(scores, predictions, future_ror, mask)
            else:
                loss = criterion(predictions, future_ror, mask)
        else:
            loss = criterion(scores, future_ror, mask)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        if batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

    return total_loss / num_batches


def validate(model, val_loader, criterion, device, need_predictions):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_scores = []
    all_targets = []

    with torch.no_grad():
        for stocks_window, future_ror, mask, _ in val_loader:
            stocks_window = stocks_window.to(device)
            future_ror = future_ror.to(device)
            mask = mask.to(device)

            scores, predictions = model(stocks_window, mask)

            if need_predictions and predictions is not None:
                if 'Combined' in criterion.__class__.__name__:
                    loss, info = criterion(scores, predictions, future_ror, mask)
                else:
                    loss = criterion(predictions, future_ror, mask)
            else:
                loss = criterion(scores, future_ror, mask)

            total_loss += loss.item()
            num_batches += 1

            all_scores.append(scores)
            all_targets.append(future_ror)

    avg_loss = total_loss / num_batches
    all_scores = torch.cat(all_scores, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    avg_ic = compute_ic(all_scores, all_targets)

    return avg_loss, avg_ic
