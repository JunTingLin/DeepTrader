"""
MSU Stage 1 Pretraining Script (Masked Reconstruction)

Train MSU encoder with self-supervised masked reconstruction task.

Task: Given masked market data, reconstruct the original values at masked positions.
Strategy: Mask entire weeks (30% of 13 weeks = ~4 weeks)

Usage:
    python MSU/pretrain_msu_stage1.py \
        --data_dir ./data/DJIA/feature34-Inter-P532 \
        --save_dir ./MSU/checkpoints/msu_stage1_masked \
        --epochs 100 \
        --batch_size 32 \
        --lr 1e-3 \
        --mask_ratio 0.3
"""

import os
import argparse
import json
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

# Import PMSU model and dataset
from MSU.msu_dataset_stage1 import MSUDataset
from model.PMSU import PMSU


def create_mask(batch_size, seq_len, mask_ratio=0.3):
    """
    Create random temporal mask (mask entire weeks).

    Args:
        batch_size: Batch size
        seq_len: Sequence length (13 weeks)
        mask_ratio: Ratio of weeks to mask (default: 0.3)

    Returns:
        mask: Boolean tensor [batch, seq_len] where True = masked position
    """
    num_masked = int(seq_len * mask_ratio)  # e.g., 13 * 0.3 = 3.9 ≈ 4 weeks

    # For each sample, randomly select weeks to mask
    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    for i in range(batch_size):
        # Randomly select indices to mask
        masked_indices = torch.randperm(seq_len)[:num_masked]
        mask[i, masked_indices] = True

    return mask


def train_one_epoch(model, train_loader, criterion, optimizer, device, mask_ratio=0.3):
    """
    Train for one epoch with masked reconstruction.

    Args:
        model: PMSU model
        train_loader: Training data loader
        criterion: Loss function (MSE)
        optimizer: Optimizer
        device: Device
        mask_ratio: Ratio of weeks to mask (default: 0.3)

    Returns:
        avg_loss: Average reconstruction loss
        avg_mse_masked: Average MSE on masked positions only
    """
    model.train()
    total_loss = 0
    total_mse_masked = 0

    pbar = tqdm(train_loader, desc='Training')
    for market_input in pbar:
        market_input = market_input.to(device)  # [batch, 13, 27]
        batch_size, seq_len, num_features = market_input.shape

        # Create original data (target for reconstruction)
        original_data = market_input.clone()  # [batch, 13, 27]

        # Create mask: [batch, 13] where True = masked position
        mask = create_mask(batch_size, seq_len, mask_ratio).to(device)

        # Apply mask: set masked positions to 0
        # Expand mask to [batch, 13, 27]
        mask_expanded = mask.unsqueeze(-1).expand_as(market_input)  # [batch, 13, 27]
        masked_input = market_input.clone()
        masked_input[mask_expanded] = 0.0  # Mask entire weeks (all 27 features)

        # Forward: reconstruct from masked input
        reconstructed = model(masked_input)  # [batch, 13, 27]

        # Loss: MSE on ALL positions (both masked and unmasked)
        # - Masked positions: Learn to reconstruct from context
        # - Unmasked positions: Learn identity mapping (preserve input)
        loss = criterion(reconstructed, original_data)

        # Also compute MSE on masked positions only for monitoring
        mse_masked = F.mse_loss(reconstructed[mask_expanded], original_data[mask_expanded])

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        total_mse_masked += mse_masked.item()

        pbar.set_postfix({'loss': f'{loss.item():.5f}', 'mse_masked': f'{mse_masked.item():.5f}'})

    avg_loss = total_loss / len(train_loader)
    avg_mse_masked = total_mse_masked / len(train_loader)

    return avg_loss, avg_mse_masked


def validate(model, val_loader, criterion, device, mask_ratio=0.3):
    """
    Validate the model with masked reconstruction.

    Args:
        model: PMSU model
        val_loader: Validation data loader
        criterion: Loss function (MSE)
        device: Device
        mask_ratio: Ratio of weeks to mask (default: 0.3)

    Returns:
        avg_loss: Average reconstruction loss
        avg_mse_masked: Average MSE on masked positions only
    """
    model.eval()
    total_loss = 0
    total_mse_masked = 0

    with torch.no_grad():
        for market_input in val_loader:
            market_input = market_input.to(device)  # [batch, 13, 27]
            batch_size, seq_len, num_features = market_input.shape

            # Create original data
            original_data = market_input.clone()

            # Create mask
            mask = create_mask(batch_size, seq_len, mask_ratio).to(device)

            # Apply mask
            mask_expanded = mask.unsqueeze(-1).expand_as(market_input)
            masked_input = market_input.clone()
            masked_input[mask_expanded] = 0.0

            # Forward
            reconstructed = model(masked_input)

            # Loss on ALL positions (not just masked)
            loss = criterion(reconstructed, original_data)
            mse_masked = F.mse_loss(reconstructed[mask_expanded], original_data[mask_expanded])

            # Statistics
            total_loss += loss.item()
            total_mse_masked += mse_masked.item()

    avg_loss = total_loss / len(val_loader)
    avg_mse_masked = total_mse_masked / len(val_loader)

    return avg_loss, avg_mse_masked


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

    # Initialize model
    print("\nInitializing PMSU model (Stage 1: masked reconstruction)...")

    # Get market features from dataset
    market_features = train_dataset.market_data.shape[1]  # Should be 27 for DJIA
    print(f"Market features: {market_features}")

    # Initialize PMSU model (all hyperparameters are fixed inside PMSU)
    model = PMSU(in_features=market_features).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Mask ratio: {args.mask_ratio}")

    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10, verbose=True
    )

    # Training loop with early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    early_stopping_patience = 20

    print("\nStarting training...")
    print(f"Early stopping: patience = {early_stopping_patience} epochs")
    print("="*80)

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-"*80)

        # Train
        train_loss, train_mse_masked = train_one_epoch(
            model, train_loader, criterion, optimizer, device, mask_ratio=args.mask_ratio
        )
        print(f"Train Loss: {train_loss:.5f}, MSE (masked): {train_mse_masked:.5f}")

        # Validate
        val_loss, val_mse_masked = validate(
            model, val_loader, criterion, device, mask_ratio=args.mask_ratio
        )
        print(f"Val Loss: {val_loss:.5f}, MSE (masked): {val_mse_masked:.5f}")

        # TensorBoard logging
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Train/MSE_Masked', train_mse_masked, epoch)

        writer.add_scalar('Val/Loss', val_loss, epoch)
        writer.add_scalar('Val/MSE_Masked', val_mse_masked, epoch)

        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        writer.flush()

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model (based on validation loss - lower is better)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0  # Reset counter

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_mse_masked': train_mse_masked,
                'val_loss': val_loss,
                'val_mse_masked': val_mse_masked,
                'best_val_loss': best_val_loss,
                'market_features': market_features,
                'mask_ratio': args.mask_ratio,
                'args': vars(args),
            }

            save_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, save_path)
            print(f"✅ Saved best model (val_loss: {val_loss:.5f})")
        else:
            early_stopping_counter += 1
            print(f"⏳ No improvement for {early_stopping_counter}/{early_stopping_patience} epochs")

            # Early stopping
            if early_stopping_counter >= early_stopping_patience:
                print(f"\n🛑 Early stopping triggered! No improvement for {early_stopping_patience} epochs")
                print(f"Best Val Loss: {best_val_loss:.5f} at epoch {epoch - early_stopping_patience}")
                break

        # Save checkpoint every N epochs
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }

            save_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save(checkpoint, save_path)
            print(f"Saved checkpoint at epoch {epoch}")

    # Close writer
    writer.close()

    print("\n" + "="*80)
    print("Training completed!")
    print(f"Best Val Loss: {best_val_loss:.5f}")
    print(f"Saved to: {save_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MSU Stage 1 Pretraining (Masked Reconstruction)')

    # Data
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Directory containing market_data.npy')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')

    # Masking
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='Ratio of weeks to mask (default: 0.3)')

    # Device
    parser.add_argument('--use_gpu', action='store_true', default=True,
                        help='Use GPU if available (default: True)')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of dataloader workers (default: 4)')

    # Checkpointing
    parser.add_argument('--save_dir', type=str, default='./MSU/checkpoints/msu_stage1_masked',
                        help='Directory to save checkpoints (default: ./MSU/checkpoints/msu_stage1_masked)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')

    args = parser.parse_args()

    main(args)
