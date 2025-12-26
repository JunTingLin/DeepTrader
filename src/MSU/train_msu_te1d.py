"""
Training script for MSU_TE1D

Features:
- MSE/MAE loss for rho prediction
- Early stopping based on validation loss
- TensorBoard logging
- Model checkpointing
- Learning rate scheduling
- Optional pre-trained TE_1D encoder loading
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MSU_TE1D import MSU_TE1D
from MSU.msu_dataset import get_dataloaders


def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make cudnn deterministic (may reduce performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class EarlyStopping:
    """
    Early stopping to stop training when validation loss doesn't improve
    """

    def __init__(self, patience=20, min_delta=0, verbose=True):
        """
        Args:
            patience (int): How many epochs to wait after last improvement
            min_delta (float): Minimum change to qualify as improvement
            verbose (bool): Print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_epoch = 0

    def __call__(self, val_loss, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_epoch = epoch
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter}/{self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.best_epoch = epoch
            self.counter = 0


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer):
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for batch_idx, (X, rho_target, _) in enumerate(train_loader):
        X = X.to(device)  # [batch, window_len, in_features]
        rho_target = rho_target.to(device)  # [batch]

        # Forward pass
        optimizer.zero_grad()
        rho_pred = model(X)  # [batch]

        # Compute loss
        loss = criterion(rho_pred, rho_target)

        # Backward pass
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        # Log to tensorboard every 10 batches
        if batch_idx % 10 == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Loss/train_batch', loss.item(), global_step)

    avg_loss = total_loss / num_batches
    return avg_loss


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for X, rho_target, _ in val_loader:
            X = X.to(device)
            rho_target = rho_target.to(device)

            # Forward pass
            rho_pred = model(X)

            # Compute loss
            loss = criterion(rho_pred, rho_target)

            total_loss += loss.item()
            num_batches += 1

            all_preds.extend(rho_pred.cpu().numpy().tolist())
            all_targets.extend(rho_target.cpu().numpy().tolist())

    # Compute metrics
    avg_loss = total_loss / num_batches
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # Always compute both MSE and MAE for consistency, regardless of training loss
    mse = np.mean((all_preds - all_targets) ** 2)
    mae = np.mean(np.abs(all_preds - all_targets))
    rmse = np.sqrt(mse)

    # Compute correlation (handle edge cases)
    if len(all_preds) > 1:
        corr = np.corrcoef(all_preds, all_targets)[0, 1]
        if np.isnan(corr):
            corr = 0.0
    else:
        corr = 0.0

    return {
        'loss': avg_loss,  # This is the actual training loss (MSE or MAE depending on criterion)
        'mse': mse,
        'mae': mae,
        'rmse': rmse,
        'correlation': corr,
        'predictions': all_preds.tolist(),
        'targets': all_targets.tolist()
    }


def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, is_best=False):
    """Save model checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }

    # Save latest checkpoint
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pth')
    torch.save(checkpoint, latest_path)

    # Save best checkpoint
    if is_best:
        best_path = os.path.join(checkpoint_dir, 'best_checkpoint.pth')
        torch.save(checkpoint, best_path)
        print(f"  💾 Saved best checkpoint (val_loss: {val_loss:.6f})")


def train(args):
    """Main training function"""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create output directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f"MSU_TE1D_{timestamp}"
    checkpoint_dir = os.path.join(args.checkpoint_dir, run_name)
    os.makedirs(checkpoint_dir, exist_ok=True)

    print(f"\n📁 Output directory:")
    print(f"  {checkpoint_dir}")

    # Save config
    config = vars(args)
    config['device'] = str(device)
    config['run_name'] = run_name
    config['loss_function'] = 'MAE' if args.use_mae else 'MSE'
    with open(os.path.join(checkpoint_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Create dataloaders
    print(f"\n📊 Loading data from: {args.data_dir}")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        feature_idx=args.feature_idx
    )

    print(f"\n📈 Dataset statistics:")
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Train batches: {len(train_loader)}")

    # Get input features from dataset
    sample_X, _, _ = train_loader.dataset[0]
    in_features = sample_X.shape[1]  # [window_len, in_features]
    print(f"\n🔢 Input features: {in_features}")

    # Create model
    model = MSU_TE1D(
        in_features=in_features,
        window_len=args.window_len,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        heads=args.heads,
        mlp_dim=args.mlp_dim,
        dim_head=args.dim_head,
        dropout=args.dropout,
        emb_dropout=args.emb_dropout,
        mlp_hidden_dim=args.mlp_hidden_dim
    ).to(device)

    # Load pre-trained encoder if specified
    if args.pretrained_encoder:
        print(f"\n🔄 Loading pre-trained TE_1D encoder from: {args.pretrained_encoder}")
        model.load_pretrained_encoder(args.pretrained_encoder)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n🧠 Model: MSU_TE1D")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Loss and optimizer
    use_mae = args.use_mae if hasattr(args, 'use_mae') else False
    if use_mae:
        criterion = nn.L1Loss()  # MAE
        print("  Using MAE Loss (more tolerant to extreme predictions)")
    else:
        criterion = nn.MSELoss()  # Default
        print("  Using MSE Loss (conservative, predicts middle values)")

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # TensorBoard writer (logs directly to checkpoint_dir)
    writer = SummaryWriter(checkpoint_dir)

    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)

    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    print("="*80)

    best_val_loss = float('inf')

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 80)

        # Train
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch, writer)

        # Validate
        val_results = validate(model, val_loader, criterion, device)
        val_loss = val_results['loss']
        val_mae = val_results['mae']
        val_rmse = val_results['rmse']
        val_corr = val_results['correlation']

        # Update learning rate
        scheduler.step(val_loss)

        # Log to tensorboard
        writer.add_scalar('train/loss', train_loss, epoch)
        writer.add_scalar('val/loss', val_loss, epoch)
        writer.add_scalar('val/mae', val_mae, epoch)
        writer.add_scalar('val/rmse', val_rmse, epoch)
        writer.add_scalar('val/correlation', val_corr, epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)

        # Print epoch summary
        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f} | MAE: {val_mae:.6f} | RMSE: {val_rmse:.6f} | Corr: {val_corr:.4f}")
        print(f"LR:         {optimizer.param_groups[0]['lr']:.6e}")

        # Save checkpoint
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        save_checkpoint(model, optimizer, epoch, train_loss, val_loss, checkpoint_dir, is_best)

        # Early stopping
        early_stopping(val_loss, epoch)
        if early_stopping.early_stop:
            print(f"\n⚠️  Early stopping triggered at epoch {epoch}")
            print(f"   Best epoch was {early_stopping.best_epoch} with val_loss={early_stopping.best_loss:.6f}")
            break

    # Training complete
    print("\n" + "="*80)
    print("✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    print("\nTo view TensorBoard, run:")
    print(f"  tensorboard --logdir {checkpoint_dir}")

    # Training complete
    print("\n" + "="*80)
    print("✅ Training complete!")
    print("="*80)
    print(f"\n📊 To evaluate the model, run:")
    print(f"  python src/MSU/evaluate_msu_te1d.py --checkpoint_dir {checkpoint_dir}")
    print(f"  python src/MSU/evaluate_msu_te1d.py --checkpoint_dir {checkpoint_dir} --split val")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train MSU_TE1D for market trend prediction')

    # Data
    parser.add_argument('--data_dir', type=str, default='src/data/DJIA/feature34-Inter-P532',
                        help='Directory containing market data and ground truth')
    parser.add_argument('--feature_idx', type=int, default=None,
                        help='Feature index to use (None = all features)')

    # Model architecture
    parser.add_argument('--window_len', type=int, default=13,
                        help='Window length in weeks')
    parser.add_argument('--hidden_dim', type=int, default=128,
                        help='Transformer hidden dimension')
    parser.add_argument('--depth', type=int, default=2,
                        help='Number of transformer layers')
    parser.add_argument('--heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--mlp_dim', type=int, default=32,
                        help='MLP dimension in transformer')
    parser.add_argument('--dim_head', type=int, default=4,
                        help='Dimension per attention head')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--emb_dropout', type=float, default=0.1,
                        help='Embedding dropout rate')
    parser.add_argument('--mlp_hidden_dim', type=int, default=128,
                        help='Hidden dimension for prediction MLP head')

    # Pre-trained encoder
    parser.add_argument('--pretrained_encoder', type=str, default=None,
                        help='Path to pre-trained Stage 1 PMSU checkpoint (optional)')

    # Training
    parser.add_argument('--epochs', type=int, default=200,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (L2 regularization)')
    parser.add_argument('--patience', type=int, default=30,
                        help='Early stopping patience')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--checkpoint_dir', type=str, default='src/MSU/checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    # Loss function
    parser.add_argument('--use_mae', action='store_true',
                        help='Use MAE loss instead of MSE (better for extreme predictions)')

    args = parser.parse_args()

    # Set random seed for reproducibility
    set_seed(args.seed)
    print(f"🎲 Random seed set to: {args.seed}")
    print("="*80)

    # Print configuration
    print("="*80)
    print("MSU_TE1D Training Configuration")
    print("="*80)
    for key, value in vars(args).items():
        print(f"{key:20s}: {value}")
    print("="*80)

    train(args)


if __name__ == '__main__':
    main()
