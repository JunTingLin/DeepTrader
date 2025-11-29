"""
MSU Stage 1 Pretraining Script

Train MSU with trend prediction task (binary classification).

Usage:
    python pretrain_msu_stage1.py \
        --data_dir ./data/fake \
        --save_dir ./checkpoints/msu_stage1 \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3
"""

import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import PMSU model and dataset
from msu_dataset_stage1 import MSUDataset
from model.PMSU import PMSU


def train_one_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc='Training')
    for market_input, trend_label in pbar:
        market_input = market_input.to(device)  # [batch, 13, market_features]
        trend_label = trend_label.to(device)    # [batch]

        # Forward
        logits = model(market_input).squeeze(-1)  # [batch]

        # Loss (BCEWithLogitsLoss expects logits, not probs)
        loss = criterion(logits, trend_label)

        # Get predictions (apply sigmoid for accuracy calculation)
        probs = torch.sigmoid(logits)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        predicted = (probs > 0.5).float()
        correct += (predicted == trend_label).sum().item()
        total += trend_label.size(0)

        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})

    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for market_input, trend_label in val_loader:
            market_input = market_input.to(device)
            trend_label = trend_label.to(device)

            # Forward
            logits = model(market_input).squeeze(-1)  # [batch]

            # Loss (BCEWithLogitsLoss expects logits, not probs)
            loss = criterion(logits, trend_label)

            # Get predictions (apply sigmoid for accuracy calculation)
            probs = torch.sigmoid(logits)

            # Statistics
            total_loss += loss.item()
            predicted = (probs > 0.5).float()
            correct += (predicted == trend_label).sum().item()
            total += trend_label.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu')
    print(f"Using device: {device}")

    # Create save directory with timestamp
    timestamp = datetime.now().strftime('%m%d/%H%M%S')
    save_dir = os.path.join(args.save_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)

    print(f"Save directory: {save_dir}")

    # TensorBoard writer
    writer = SummaryWriter(save_dir)

    # Save configuration
    config = vars(args)
    config['save_dir'] = save_dir  # Update with timestamped directory
    with open(os.path.join(save_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # Load datasets
    print("Loading datasets...")
    train_dataset = MSUDataset(args.data_dir, split='train')
    val_dataset = MSUDataset(args.data_dir, split='val')

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True if device.type == 'cuda' else False
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Compute class distribution for weighting
    print("\nComputing class distribution...")
    train_labels = []
    for _, label in train_dataset:
        train_labels.append(label.item())
    train_labels = torch.tensor(train_labels)

    n_positive = train_labels.sum().item()
    n_negative = len(train_labels) - n_positive
    n_total = len(train_labels)

    print(f"Class distribution (train):")
    print(f"  Positive (1): {n_positive} ({n_positive/n_total*100:.2f}%)")
    print(f"  Negative (0): {n_negative} ({n_negative/n_total*100:.2f}%)")
    print(f"  Imbalance ratio: {n_positive/n_negative:.2f}:1")

    # Initialize model
    print("\nInitializing PMSU model (Stage 1: trend prediction)...")

    # Get market features from dataset
    market_features = train_dataset.market_data.shape[1]  # Should be 1 for fake data
    print(f"Market features: {market_features}")

    # Initialize PMSU model (all hyperparameters are fixed inside PMSU)
    model = PMSU(in_features=market_features).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer with class weighting
    # pos_weight: weight for positive class (how much more to weight positive samples)
    # For imbalanced data where positive is majority: pos_weight = n_positive / n_negative
    # This makes the model pay more attention to the minority class (negative)
    if args.pos_weight is not None:
        pos_weight = torch.tensor([args.pos_weight]).to(device)
        print(f"\nClass weighting (manual):")
        print(f"  pos_weight = {pos_weight.item():.4f} (user-specified)")
        print(f"  This increases the penalty for positive class errors by {pos_weight.item():.2f}x")
    else:
        pos_weight = torch.tensor([n_positive / n_negative]).to(device)
        print(f"\nClass weighting (auto-calculated):")
        print(f"  pos_weight = {pos_weight.item():.4f} (positive/negative ratio)")
        print(f"  This increases the penalty for positive class errors by {pos_weight.item():.2f}x")
    print(f"  Goal: Make model pay more attention to minority class (negative)")

    # Use BCEWithLogitsLoss which combines Sigmoid + BCE and accepts pos_weight
    # Note: We need to modify forward() to output logits instead of probs
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    best_val_acc = 0
    early_stopping_counter = 0
    early_stopping_patience = 20

    print("\nStarting training...")
    print(f"Early stopping: patience = {early_stopping_patience} epochs")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*80)

        # Train
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/Accuracy', train_acc, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/Accuracy', val_acc, epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model and check early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'best_val_acc': best_val_acc,
                'market_features': market_features,
                'args': vars(args),
            }

            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model (acc: {val_acc:.2f}%)")
        else:
            early_stopping_counter += 1
            print(f"⏳ No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs")
                print(f"Best Val Acc: {best_val_acc:.2f}% at epoch {epoch - early_stopping_patience}")
                break

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
            }

            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint at epoch {epoch}")

    # Close writer
    writer.close()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.4f}")
    print(f"Best Val Acc: {best_val_acc:.2f}%")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSU Stage 1 Pretraining')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing MSU_*_ground_truth.json files')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--pos_weight', type=float, default=None,
                        help='Positive class weight for BCEWithLogitsLoss (default: n_pos/n_neg, auto-calculated)')

    # Device
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./checkpoints/msu_stage1',
                        help='Directory to save checkpoints (default: ./checkpoints/msu_stage1)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    args = parser.parse_args()

    main(args)
