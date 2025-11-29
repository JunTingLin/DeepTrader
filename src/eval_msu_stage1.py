"""
Evaluation script for MSU Stage 1 pretraining.
Outputs basic classification results as JSON.
"""
import argparse
import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from msu_dataset_stage1 import MSUDataset
from model.PMSU import PMSU


def evaluate_split(model, dataloader, device, split_name='val'):
    """
    Evaluate model on a dataset split.

    Returns:
        dict: Contains basic evaluation results
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for X, y in dataloader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)
            all_logits.append(logits.cpu())
            all_labels.append(y.cpu())

    # Concatenate all batches
    logits = torch.cat(all_logits, dim=0).squeeze(-1).numpy()
    labels = torch.cat(all_labels, dim=0).numpy()

    # Get probabilities and predictions
    probs = 1 / (1 + np.exp(-logits))  # Sigmoid
    preds = (probs >= 0.5).astype(int)

    # Compute basic metrics
    correct = (preds == labels).sum()
    accuracy = float(correct / len(labels))
    loss = -np.mean(labels * np.log(probs + 1e-10) + (1 - labels) * np.log(1 - probs + 1e-10))

    # Basic results
    results = {
        'split': split_name,
        'n_samples': int(len(labels)),
        'n_positive': int(labels.sum()),
        'n_negative': int((1 - labels).sum()),
        'loss': float(loss),
        'accuracy': float(accuracy),
        'predictions': {
            'logits': logits.tolist(),
            'probs': probs.tolist(),
            'preds': preds.tolist(),
            'labels': labels.tolist()
        }
    }

    return results


def print_summary(results):
    """Print evaluation summary"""
    print(f"\n{'='*80}")
    print(f"Evaluation Results: {results['split'].upper()}")
    print(f"{'='*80}")
    print(f"Total samples: {results['n_samples']}")
    print(f"  Positive (1): {results['n_positive']} ({results['n_positive']/results['n_samples']*100:.2f}%)")
    print(f"  Negative (0): {results['n_negative']} ({results['n_negative']/results['n_samples']*100:.2f}%)")
    print(f"\nLoss: {results['loss']:.4f}")
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate MSU Stage 1 model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file (best_model.pth)')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory (defaults to checkpoint directory)')
    parser.add_argument('--batch_size', type=int, default=32)
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

    # Get market features
    market_features = checkpoint.get('market_features', 1)  # Default to 1 for fake data

    print(f"\nModel configuration:")
    print(f"  Market features: {market_features}")
    print(f"  Epoch: {checkpoint['epoch']}")
    if 'best_val_acc' in checkpoint:
        print(f"  Best val acc: {checkpoint['best_val_acc']:.4f}")
    else:
        print(f"  Val acc: {checkpoint['val_acc']:.4f}")

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
            results = evaluate_split(model, dataloader, device, split_name=split)

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
