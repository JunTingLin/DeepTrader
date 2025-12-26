"""
Generic evaluation script for trained MSU models (LSTM or TE1D)

Features:
- Automatically detect model type from config.json
- Load trained checkpoint
- Evaluate on test/val/train set
- Visualize predictions
- Analyze attention weights (if applicable)
- Compute evaluation metrics (MSE, RMSE, MAE, Correlation)
"""

import os
import sys
import json
import argparse

import numpy as np
import matplotlib.pyplot as plt
import torch

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MSU_LSTM import MSU_LSTM
from model.MSU_TE1D import MSU_TE1D
from MSU.msu_dataset import get_dataloaders


def load_checkpoint(checkpoint_path, device):
    """Load model from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    return checkpoint


def create_model_from_config(config, in_features, device):
    """
    Create model instance based on config

    Args:
        config: Configuration dictionary
        in_features: Number of input features
        device: Device to load model on

    Returns:
        model: Model instance (MSU_LSTM or MSU_TE1D)
        model_type: String indicating model type ('lstm' or 'te1d')
    """
    # Detect model type from run_name or explicit model_type field
    run_name = config.get('run_name', '')
    model_type_field = config.get('model_type', '')

    if 'TE1D' in run_name or 'te1d' in model_type_field.lower():
        # MSU_TE1D
        model = MSU_TE1D(
            in_features=in_features,
            window_len=config['window_len'],
            hidden_dim=config['hidden_dim'],
            depth=config.get('depth', 2),
            heads=config.get('heads', 4),
            mlp_dim=config.get('mlp_dim', 32),
            dim_head=config.get('dim_head', 4),
            dropout=config.get('dropout', 0.1),
            emb_dropout=config.get('emb_dropout', 0.1),
            mlp_hidden_dim=config.get('mlp_hidden_dim', 128)
        ).to(device)
        model_type = 'te1d'
        print(f"📦 Created MSU_TE1D model")

    elif 'LSTM' in run_name or 'lstm' in model_type_field.lower():
        # MSU_LSTM
        model = MSU_LSTM(
            in_features=in_features,
            window_len=config['window_len'],
            hidden_dim=config['hidden_dim'],
            dropout=config['dropout']
        ).to(device)
        model_type = 'lstm'
        print(f"📦 Created MSU_LSTM model")

    else:
        raise ValueError(f"Cannot determine model type from config. run_name: {run_name}")

    return model, model_type


def evaluate(model, test_loader, device):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_targets = []
    all_indices = []

    with torch.no_grad():
        for X, rho_target, idx in test_loader:
            X = X.to(device)
            rho_target = rho_target.to(device)

            # Forward pass
            rho_pred = model(X)

            all_preds.extend(rho_pred.cpu().numpy().tolist())
            all_targets.extend(rho_target.cpu().numpy().tolist())
            all_indices.extend(idx.numpy().tolist())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_indices = np.array(all_indices)

    # Compute metrics
    mse = np.mean((all_preds - all_targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(all_preds - all_targets))
    corr = np.corrcoef(all_preds, all_targets)[0, 1]

    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'correlation': corr,
        'predictions': all_preds,
        'targets': all_targets,
        'indices': all_indices
    }


def plot_predictions(results, output_dir, split='test'):
    """Plot predictions vs targets"""
    preds = results['predictions']
    targets = results['targets']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Scatter plot
    axes[0].scatter(targets, preds, alpha=0.6)
    axes[0].plot([targets.min(), targets.max()], [targets.min(), targets.max()], 'r--', lw=2)
    axes[0].set_xlabel('Target Rho')
    axes[0].set_ylabel('Predicted Rho')
    axes[0].set_title(f'Predictions vs Targets ({split.upper()})\nCorrelation: {results["correlation"]:.4f}')
    axes[0].grid(True)

    # Time series plot
    axes[1].plot(targets, label='Target', marker='o', linestyle='-', alpha=0.7)
    axes[1].plot(preds, label='Prediction', marker='x', linestyle='--', alpha=0.7)
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Rho')
    axes[1].set_title(f'Time Series Comparison ({split.upper()})')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'predictions_{split}.png')
    plt.savefig(output_path, dpi=150)
    print(f"  📊 Saved prediction plot to: {output_path}")
    plt.close()


def plot_error_distribution(results, output_dir, split='test'):
    """Plot error distribution"""
    preds = results['predictions']
    targets = results['targets']
    errors = preds - targets

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Error histogram
    axes[0].hist(errors, bins=30, alpha=0.7, edgecolor='black')
    axes[0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_xlabel('Prediction Error (Predicted - Target)')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'Error Distribution ({split.upper()})\nMean: {np.mean(errors):.4f}, Std: {np.std(errors):.4f}')
    axes[0].grid(True, alpha=0.3)

    # Absolute error plot
    abs_errors = np.abs(errors)
    axes[1].plot(abs_errors, marker='o', linestyle='-', alpha=0.6)
    axes[1].axhline(y=np.mean(abs_errors), color='r', linestyle='--', linewidth=2, label=f'MAE: {np.mean(abs_errors):.4f}')
    axes[1].set_xlabel('Sample Index')
    axes[1].set_ylabel('Absolute Error')
    axes[1].set_title(f'Absolute Error per Sample ({split.upper()})')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'error_distribution_{split}.png')
    plt.savefig(output_path, dpi=150)
    print(f"  📊 Saved error distribution plot to: {output_path}")
    plt.close()


def plot_attention_weights(model, test_loader, device, output_dir, split='test', num_samples=5):
    """Plot attention weights for sample inputs"""
    model.eval()

    # Collect samples from first batch
    with torch.no_grad():
        for X, rho_target, idx in test_loader:
            X = X.to(device)
            attn_weights = model.get_attention_weights(X)

            # Limit to num_samples or batch size, whichever is smaller
            actual_num_samples = min(num_samples, X.size(0))

            # Create subplots
            fig, axes = plt.subplots(actual_num_samples, 1, figsize=(10, 3*actual_num_samples))
            if actual_num_samples == 1:
                axes = [axes]

            # Plot each sample
            for i in range(actual_num_samples):
                ax = axes[i]
                attn = attn_weights[i].cpu().numpy()  # [window_len]
                ax.bar(range(len(attn)), attn)
                ax.set_xlabel('Week Index')
                ax.set_ylabel('Attention Weight')
                ax.set_title(f'{split.upper()} Sample {i+1}: Target Rho = {rho_target[i]:.4f}, Index = {idx[i]}')
                ax.grid(True, alpha=0.3)

            break  # Only use first batch

    plt.tight_layout()
    output_path = os.path.join(output_dir, f'attention_weights_{split}.png')
    plt.savefig(output_path, dpi=150)
    print(f"  📊 Saved attention weights plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained MSU model (LSTM or TE1D)')

    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Directory containing checkpoint (e.g., ./src/MSU/checkpoints/MSU_LSTM_20240101_120000)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Directory containing market data (default: read from config.json)')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                        help='Which split to evaluate')

    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load config
    config_path = os.path.join(args.checkpoint_dir, 'config.json')
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"\n📁 Loading from: {args.checkpoint_dir}")

    # Use data_dir from config if not specified
    if args.data_dir is None:
        args.data_dir = config['data_dir']
        print(f"📂 Using data_dir from config: {args.data_dir}")
    else:
        print(f"📂 Using specified data_dir: {args.data_dir}")

    # Load dataloaders
    print(f"\n📊 Loading data...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=config['batch_size'],
        num_workers=0,  # Use 0 for evaluation
        feature_idx=config.get('feature_idx', None)
    )

    # Select split
    if args.split == 'train':
        loader = train_loader
    elif args.split == 'val':
        loader = val_loader
    else:
        loader = test_loader

    # Get input features
    sample_X, _, _ = loader.dataset[0]
    in_features = sample_X.shape[1]

    # Create model based on config
    model, model_type = create_model_from_config(config, in_features, device)

    # Load checkpoint
    checkpoint_path = os.path.join(args.checkpoint_dir, 'best_checkpoint.pth')
    checkpoint = load_checkpoint(checkpoint_path, device)
    model.load_state_dict(checkpoint['model_state_dict'])

    print(f"  ✅ Loaded checkpoint from epoch {checkpoint['epoch']}")
    print(f"  📉 Val loss at checkpoint: {checkpoint['val_loss']:.6f}")

    # Evaluate
    print(f"\n📊 Evaluating on {args.split} set...")
    results = evaluate(model, loader, device)

    # Print metrics
    print("\n" + "="*80)
    print("Evaluation Results")
    print("="*80)
    print(f"Model Type:  {model_type.upper()}")
    print(f"MSE:         {results['mse']:.6f}")
    print(f"RMSE:        {results['rmse']:.6f}")
    print(f"MAE:         {results['mae']:.6f}")
    print(f"Correlation: {results['correlation']:.6f}")
    print("="*80)

    # Save results
    results_path = os.path.join(args.checkpoint_dir, f'{args.split}_evaluation.json')
    results_to_save = {
        'model_type': model_type,
        'mse': float(results['mse']),
        'rmse': float(results['rmse']),
        'mae': float(results['mae']),
        'correlation': float(results['correlation']),
        'predictions': results['predictions'].tolist(),
        'targets': results['targets'].tolist(),
        'indices': results['indices'].tolist()
    }
    with open(results_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)
    print(f"\n💾 Saved results to: {results_path}")

    # Plot predictions and errors
    print(f"\n📊 Generating plots...")
    plot_predictions(results, args.checkpoint_dir, split=args.split)
    plot_error_distribution(results, args.checkpoint_dir, split=args.split)

    # Plot attention weights (both LSTM and TE1D support this)
    plot_attention_weights(model, loader, device, args.checkpoint_dir, split=args.split, num_samples=5)

    print("\n✅ Evaluation complete!")


if __name__ == '__main__':
    main()
