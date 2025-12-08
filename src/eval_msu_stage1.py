"""
Evaluation script for MSU Stage 1 pretraining (Masked Reconstruction).
Outputs reconstruction metrics as JSON.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from msu_dataset_stage1 import MSUDataset
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
    num_masked = int(seq_len * mask_ratio)

    mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    for i in range(batch_size):
        masked_indices = torch.randperm(seq_len)[:num_masked]
        mask[i, masked_indices] = True

    return mask


def evaluate_split(model, dataloader, device, mask_ratio=0.3, split_name='test'):
    """
    Evaluate model on a dataset split (masked reconstruction).

    Returns:
        dict: Contains reconstruction evaluation results
    """
    model.eval()
    total_mse_masked = 0
    total_mse_unmasked = 0
    total_mse_all = 0
    total_mae_masked = 0
    n_batches = 0

    # Per-feature MSE (to see which features are easier to reconstruct)
    all_mse_per_feature = []

    with torch.no_grad():
        for market_input in tqdm(dataloader, desc=f'Evaluating {split_name}'):
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

            # MSE on masked positions only
            mse_masked = F.mse_loss(reconstructed[mask_expanded], original_data[mask_expanded])
            mae_masked = torch.mean(torch.abs(reconstructed[mask_expanded] - original_data[mask_expanded]))

            # MSE on unmasked positions only
            unmasked_mask = ~mask_expanded
            mse_unmasked = F.mse_loss(reconstructed[unmasked_mask], original_data[unmasked_mask])

            # MSE on all positions
            mse_all = F.mse_loss(reconstructed, original_data)

            # Per-feature MSE on masked positions
            # Compute MSE for each of the 27 features separately
            mse_per_feat = []
            for feat_idx in range(num_features):
                # Get masked positions for this feature
                feat_mask = mask_expanded[:, :, feat_idx]
                if feat_mask.sum() > 0:  # If there are masked positions
                    feat_mse = F.mse_loss(
                        reconstructed[:, :, feat_idx][feat_mask],
                        original_data[:, :, feat_idx][feat_mask]
                    )
                    mse_per_feat.append(feat_mse.item())
                else:
                    mse_per_feat.append(0.0)

            all_mse_per_feature.append(mse_per_feat)

            # Accumulate
            total_mse_masked += mse_masked.item()
            total_mse_unmasked += mse_unmasked.item()
            total_mse_all += mse_all.item()
            total_mae_masked += mae_masked.item()
            n_batches += 1

    # Average over batches
    avg_mse_masked = total_mse_masked / n_batches
    avg_mse_unmasked = total_mse_unmasked / n_batches
    avg_mse_all = total_mse_all / n_batches
    avg_mae_masked = total_mae_masked / n_batches

    # Average per-feature MSE
    avg_mse_per_feature = np.mean(all_mse_per_feature, axis=0).tolist()

    results = {
        'split': split_name,
        'n_samples': len(dataloader.dataset),
        'mask_ratio': mask_ratio,
        'mse_masked': float(avg_mse_masked),
        'mse_unmasked': float(avg_mse_unmasked),
        'mse_all': float(avg_mse_all),
        'mae_masked': float(avg_mae_masked),
        'mse_per_feature': avg_mse_per_feature,
    }

    return results


def print_summary(results):
    """Print evaluation summary"""
    print(f"\n{'='*80}")
    print(f"Evaluation Results: {results['split'].upper()}")
    print(f"{'='*80}")
    print(f"Total samples: {results['n_samples']}")
    print(f"Mask ratio:    {results['mask_ratio']:.1%}")

    print(f"\nReconstruction Metrics:")
    print(f"  MSE (all positions):      {results['mse_all']:.6f}")
    print(f"  MSE (masked only):        {results['mse_masked']:.6f}")
    print(f"  MSE (unmasked only):      {results['mse_unmasked']:.6f}")
    print(f"  MAE (masked only):        {results['mae_masked']:.6f}")
    print(f"\n  Expected: mse_masked > mse_unmasked (masked is harder)")

    print(f"\nPer-Feature MSE (masked, top 5 easiest to reconstruct):")
    mse_per_feat = results['mse_per_feature']
    sorted_indices = np.argsort(mse_per_feat)
    for idx in sorted_indices[:5]:
        print(f"  Feature {idx:2d}: {mse_per_feat[idx]:.6f}")

    print(f"\nPer-Feature MSE (masked, top 5 hardest to reconstruct):")
    for idx in sorted_indices[-5:][::-1]:
        print(f"  Feature {idx:2d}: {mse_per_feat[idx]:.6f}")

    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSU Stage 1 model (Masked Reconstruction)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (best_model.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to checkpoint directory)')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mask_ratio', type=float, default=0.3,
                        help='Ratio of weeks to mask (default: 0.3)')
    parser.add_argument('--splits', type=str, default='val,test',
                        help='Comma-separated splits to evaluate (e.g., val,test)')

    args = parser.parse_args()

    # Setup output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(args.checkpoint)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading checkpoint from: {args.checkpoint}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Load checkpoint
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Get market features and mask ratio
    market_features = checkpoint.get('market_features', 27)
    mask_ratio = checkpoint.get('mask_ratio', args.mask_ratio)

    print(f"\nModel configuration:")
    print(f"  Market features: {market_features}")
    print(f"  Mask ratio: {mask_ratio:.1%}")
    print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_val_loss' in checkpoint:
        print(f"  Best val loss: {checkpoint['best_val_loss']:.6f}")
    elif 'val_loss' in checkpoint:
        print(f"  Val loss: {checkpoint['val_loss']:.6f}")

    # Initialize PMSU model (all hyperparameters are fixed inside PMSU)
    model = PMSU(in_features=market_features).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"\nModel loaded successfully!")

    # Evaluate on specified splits
    splits = args.splits.split(',')

    for split in splits:
        split = split.strip()
        print(f"\n{'='*80}")
        print(f"Evaluating on {split.upper()} set...")
        print(f"{'='*80}")

        # Load dataset
        try:
            dataset = MSUDataset(data_dir=args.data_dir, split=split)
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

            # Evaluate
            results = evaluate_split(model, dataloader, device, mask_ratio=mask_ratio, split_name=split)

            # Print summary
            print_summary(results)

            # Save results
            result_file = os.path.join(args.output_dir, f'{split}_results.json')
            with open(result_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Saved results to {result_file}")

        except FileNotFoundError as e:
            print(f"Warning: Could not load {split} dataset: {e}")
            continue

    print(f"\n{'='*80}")
    print(f"Evaluation completed!")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
