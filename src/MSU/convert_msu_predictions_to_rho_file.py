"""
Convert MSU evaluation results to rho file format compatible with validate.py/test.py

Usage:
    python src/MSU/convert_msu_predictions_to_rho_file.py \
        --checkpoint_dir src/MSU/checkpoints/MSU_LSTM_20251218_200329 \
        --data_dir src/data/DJIA/feature34-Inter-P532 \
        --split test
"""

import argparse
import json
import os


def convert_msu_predictions_to_rho_file(checkpoint_dir, data_dir, split='test'):
    """
    Convert MSU predictions to rho file format

    Args:
        checkpoint_dir: Path to MSU checkpoint (e.g., src/MSU/checkpoints/MSU_LSTM_20251218_200329)
        data_dir: Path to data directory (e.g., src/data/DJIA/feature34-Inter-P532)
        split: 'val' or 'test'
    """

    # Load MSU evaluation results
    eval_file = os.path.join(checkpoint_dir, f'{split}_evaluation.json')
    with open(eval_file, 'r') as f:
        eval_data = json.load(f)

    predictions = eval_data['predictions']
    indices = eval_data['indices']

    print(f"✓ Loaded {len(predictions)} predictions from {eval_file}")

    # Load ground truth file to get the structure
    gt_file = os.path.join(data_dir, f'MSU_{split}_ground_truth_step21.json')
    with open(gt_file, 'r') as f:
        gt_data = json.load(f)

    gt_records = gt_data['ground_truth_records']

    print(f"✓ Loaded {len(gt_records)} ground truth records from {gt_file}")

    # Sanity check
    if len(predictions) != len(gt_records):
        print(f"⚠️  Warning: Number of predictions ({len(predictions)}) != number of GT records ({len(gt_records)})")

    if len(indices) != len(gt_records):
        print(f"⚠️  Warning: Number of indices ({len(indices)}) != number of GT records ({len(gt_records)})")

    # Create output records: use GT structure but replace rho with MSU predictions
    output_records = []
    for i, record in enumerate(gt_records):
        new_record = record.copy()
        new_record['rho'] = predictions[i]  # Replace with MSU prediction
        new_record['original_gt_rho'] = record['rho']  # Keep original GT for reference
        output_records.append(new_record)

    # Create output data
    # NOTE: validate.py/test.py expect 'rho_record' to be a list of float values, not dicts!
    rho_values_only = [record['rho'] for record in output_records]

    output_data = {
        'rho_record': rho_values_only,  # List of float values for validate.py/test.py
        'ground_truth_records': output_records,  # Full records for reference
        'source': f'MSU_LSTM predictions from {os.path.basename(checkpoint_dir)}',
        'split': split,
        'msu_correlation': eval_data['correlation'],
        'msu_mae': eval_data['mae'],
        'msu_rmse': eval_data['rmse']
    }

    # Save output to checkpoint directory
    output_file = os.path.join(checkpoint_dir, f'msu_predicted_rho_{split}.json')
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"✅ Saved to: {output_file}")
    print(f"\n📊 MSU Performance on {split}:")
    print(f"  Correlation: {eval_data['correlation']:.4f}")
    print(f"  MAE: {eval_data['mae']:.4f}")
    print(f"  RMSE: {eval_data['rmse']:.4f}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description='Convert MSU predictions to rho file format')

    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to MSU checkpoint directory')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory')
    parser.add_argument('--split', type=str, default='test', choices=['val', 'test'],
                        help='Which split to convert')

    args = parser.parse_args()

    convert_msu_predictions_to_rho_file(args.checkpoint_dir, args.data_dir, args.split)


if __name__ == '__main__':
    main()
