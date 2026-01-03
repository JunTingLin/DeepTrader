"""
Training script for ASU_GCN Supervised Pre-training

使用 TCN + SAGCN (GraphConvNet) + Spatial Attention 的監督學習預訓練
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.ASU import ASU
from ASU.asu_dataset import get_dataloaders
from ASU.losses_asu import get_loss_function
from ASU.train_utils import set_seed, EarlyStopping, train_one_epoch, validate


class ASU_GCN_Wrapper(nn.Module):
    """
    簡單包裝 ASU（GCN 版本），支持可選的 regression head
    """
    def __init__(self, asu_model, add_regression_head=False):
        super().__init__()
        self.asu = asu_model
        self.add_regression_head = add_regression_head

        if add_regression_head:
            self.regression_head = nn.Linear(1, 1)

    def forward(self, inputs, mask):
        scores = self.asu(inputs, mask)  # [batch, num_stocks]

        if self.add_regression_head:
            predictions = self.regression_head(scores.unsqueeze(-1)).squeeze(-1)
            return scores, predictions
        else:
            return scores, None

    def get_asu_state_dict(self):
        """獲取 ASU 的權重（用於遷移學習）"""
        return self.asu.state_dict()

    def load_asu_state_dict(self, state_dict):
        """載入 ASU 的權重"""
        self.asu.load_state_dict(state_dict)


def main(args):
    set_seed(args.seed)

    # Create output directory with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"ASU_GCN_{args.loss_function}_w{args.window_len}_h{args.horizon}_{timestamp}"
    output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save args and timestamp
    args_dict = vars(args)
    args_dict['timestamp'] = timestamp
    args_dict['run_name'] = run_name
    args_dict['model_type'] = 'ASU_GCN'
    with open(output_dir / 'args.json', 'w') as f:
        json.dump(args_dict, f, indent=2)

    # Setup tensorboard
    writer = SummaryWriter(output_dir / 'tensorboard')

    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Load data
    print("\nLoading data...")
    train_loader, val_loader, test_loader, num_stocks, num_features = get_dataloaders(
        data_dir=args.data_dir,
        train_idx=args.train_idx,
        train_idx_end=args.train_idx_end,
        val_idx=args.val_idx,
        test_idx=args.test_idx,
        test_idx_end=args.test_idx_end,
        window_len=args.window_len,
        horizon=args.horizon,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_stride=args.train_stride,
        val_stride=args.val_stride,
        test_stride=args.test_stride
    )

    # Get loss function
    print(f"\nCreating ASU_GCN model...")
    criterion, need_predictions = get_loss_function(
        loss_type=args.loss_function,
        margin=args.margin,
        min_ror_diff=args.min_ror_diff,
        alpha=args.alpha,
        beta=args.beta,
        ranking_loss_type=args.ranking_loss_type,
        regression_loss_type=args.regression_loss_type
    )

    # Create ASU model (GCN version: transformer_asu_bool=False)
    asu_base = ASU(
        num_nodes=num_stocks,
        in_features=num_features,
        hidden_dim=args.hidden_dim,
        window_len=args.window_len,
        dropout=args.dropout,
        kernel_size=args.kernel_size,
        layers=args.num_blocks,
        supports=None,
        spatial_bool=args.spatial_bool,
        addaptiveadj=args.addaptiveadj,
        aptinit=None,
        transformer_asu_bool=False,  # GCN 版本
        num_assets=num_stocks
    ).to(device)

    # Wrap with regression head if needed
    model = ASU_GCN_Wrapper(asu_base, add_regression_head=need_predictions).to(device)

    print(f"Model created with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"Loss function: {args.loss_function}")
    print(f"Need regression predictions: {need_predictions}")

    # Optimizer
    if args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=args.patience, mode='max', verbose=True)

    # Training loop
    print("\n" + "="*80)
    print("Starting training...")
    print("="*80)

    best_val_ic = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 40)

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer, need_predictions)
        val_loss, val_ic = validate(model, val_loader, criterion, device, need_predictions)

        scheduler.step(val_ic)

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Metrics/val_ic', val_ic, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss: {val_loss:.6f}")
        print(f"Val IC: {val_ic:.6f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'asu_state_dict': model.get_asu_state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ic': val_ic,
                'val_loss': val_loss,
                'args': args_dict,
                'timestamp': timestamp
            }, output_dir / 'best_asu.pth')
            print(f"✓ Saved best model (IC: {val_ic:.6f})")

        if (epoch + 1) % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'asu_state_dict': model.get_asu_state_dict(),
                'full_model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_ic': val_ic,
            }, output_dir / f'checkpoint_epoch{epoch+1}.pth')

        early_stopping(val_ic, epoch)
        if early_stopping.early_stop:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break

    # Test
    print("\n" + "="*80)
    print("Testing best model...")
    print("="*80)

    checkpoint = torch.load(output_dir / 'best_asu.pth')
    model.load_asu_state_dict(checkpoint['asu_state_dict'])

    test_loss, test_ic = validate(model, test_loader, criterion, device, need_predictions)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"Test IC: {test_ic:.6f}")

    results = {
        'model_type': 'ASU_GCN',
        'best_epoch': best_epoch,
        'best_val_ic': best_val_ic,
        'test_loss': test_loss,
        'test_ic': test_ic,
        'timestamp': timestamp,
        'run_name': run_name
    }

    with open(output_dir / 'test_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Training complete!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Best ASU weights: {output_dir / 'best_asu.pth'}")
    print(f"✓ Timestamp: {timestamp}")

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train ASU_GCN with supervised learning')

    parser.add_argument('--data_dir', type=str, default='./src/data/DJIA/feature34-Inter-P532/')
    parser.add_argument('--output_dir', type=str, default='./src/ASU/checkpoints')
    parser.add_argument('--train_idx', type=int, default=0)
    parser.add_argument('--train_idx_end', type=int, default=1304)
    parser.add_argument('--val_idx', type=int, default=1304)
    parser.add_argument('--test_idx', type=int, default=2087)
    parser.add_argument('--test_idx_end', type=int, default=2673)
    parser.add_argument('--window_len', type=int, default=13)
    parser.add_argument('--horizon', type=int, default=21)
    parser.add_argument('--train_stride', type=int, default=1)
    parser.add_argument('--val_stride', type=int, default=5)
    parser.add_argument('--test_stride', type=int, default=5)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--kernel_size', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--spatial_bool', action='store_true')
    parser.add_argument('--addaptiveadj', action='store_true', default=True)
    parser.add_argument('--loss_function', type=str, default='Combined',
                        choices=['Ranking', 'Pairwise', 'IC', 'Regression', 'MSE', 'MAE', 'Combined'])
    parser.add_argument('--margin', type=float, default=0.1)
    parser.add_argument('--min_ror_diff', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=0.8)
    parser.add_argument('--beta', type=float, default=0.2)
    parser.add_argument('--ranking_loss_type', type=str, default='pairwise')
    parser.add_argument('--regression_loss_type', type=str, default='mse')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='adam')
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=0.0001)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--save_interval', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', action='store_true', default=True)

    args = parser.parse_args()
    main(args)
