"""
Generate FinBERT-based Rho for MSU (Market State Unit)
------------------------------------------------------
This script uses FinBERT sentiment analysis on market index news (e.g., 0050/TWII)
to generate rho values without training.

Rho represents the market bullishness:
- rho = 1: Full long (bullish)
- rho = 0: Long-short hedge (bearish/neutral)

Calculation Modes:
1. positive_prob: P(positive) directly as rho [0, 1]
2. sentiment_diff: (P(positive) - P(negative) + 1) / 2, rescaled to [0, 1]
3. weighted: P(positive) * 1 + P(neutral) * 0.5 + P(negative) * 0, [0, 1]

Usage:
    # Using config file (recommended for multiple periods)
    python src/data/generate_finbert_rho.py --config src/data/TWII/finbert_rho_config.json

    # Using command line arguments (single period)
    python src/data/generate_finbert_rho.py \
        --summaries_dir src/data/TWII/summaries_v2/0050 \
        --trading_dates_file src/data/TWII/feature5-sc47-2013-2025-finlab/trading_dates.npy \
        --output_dir src/data/TWII/feature5-sc47-2013-2025-finlab \
        --mode positive_prob \
        --window_days 5 \
        --start_idx 1962 \
        --end_idx 2691
"""

import argparse
import json
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from typing import Tuple, List, Optional, Dict
from transformers import AutoTokenizer, AutoModelForSequenceClassification


class FinBERTRhoGenerator:
    """
    Generate rho values from news summaries using FinBERT.
    """

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone-chinese", device: str = None):
        """
        Initialize FinBERT model.

        Args:
            model_name: HuggingFace model name
            device: "cuda" or "cpu", auto-detect if None
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        print(f"Loading FinBERT model: {model_name}")
        print(f"Device: {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        # Get label mapping
        self.id2label = self.model.config.id2label
        print(f"Label mapping: {self.id2label}")

        # Find indices for each sentiment
        self.positive_idx = None
        self.neutral_idx = None
        self.negative_idx = None

        for idx, label in self.id2label.items():
            label_lower = label.lower()
            if 'positive' in label_lower:
                self.positive_idx = idx
            elif 'neutral' in label_lower:
                self.neutral_idx = idx
            elif 'negative' in label_lower:
                self.negative_idx = idx

        print(f"Positive idx: {self.positive_idx}, Neutral idx: {self.neutral_idx}, Negative idx: {self.negative_idx}")

    @torch.no_grad()
    def get_sentiment_probs(self, text: str) -> Tuple[float, float, float]:
        """
        Get sentiment probabilities for a single text.

        Args:
            text: News summary text

        Returns:
            Tuple of (P(positive), P(neutral), P(negative))
        """
        if not text or text.strip() == "" or text.strip() == "無":
            # Return neutral for empty/no news
            return 0.33, 0.34, 0.33

        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)[0]

        p_positive = probs[self.positive_idx].item() if self.positive_idx is not None else 0.33
        p_neutral = probs[self.neutral_idx].item() if self.neutral_idx is not None else 0.34
        p_negative = probs[self.negative_idx].item() if self.negative_idx is not None else 0.33

        return p_positive, p_neutral, p_negative

    def compute_rho(self, p_positive: float, p_neutral: float, p_negative: float, mode: str) -> float:
        """
        Compute rho based on sentiment probabilities and mode.

        Args:
            p_positive: P(positive)
            p_neutral: P(neutral)
            p_negative: P(negative)
            mode: Calculation mode

        Returns:
            rho value in [0, 1]
        """
        if mode == "positive_prob":
            # Simply use P(positive) as rho
            return p_positive

        elif mode == "sentiment_diff":
            # (P(positive) - P(negative) + 1) / 2
            # Maps [-1, 1] to [0, 1]
            diff = p_positive - p_negative
            return (diff + 1.0) / 2.0

        elif mode == "weighted":
            # P(positive) * 1 + P(neutral) * 0.5 + P(negative) * 0
            return p_positive * 1.0 + p_neutral * 0.5 + p_negative * 0.0

        else:
            raise ValueError(f"Unknown mode: {mode}")


def load_summaries(summaries_dir: str, trading_dates: np.ndarray) -> dict:
    """
    Load news summaries for each trading date.

    Args:
        summaries_dir: Path to summaries directory (e.g., summaries_v2/0050)
        trading_dates: Array of trading dates

    Returns:
        Dictionary mapping date_str -> summary text
    """
    summaries = {}

    for date in tqdm(trading_dates, desc="Loading summaries"):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        json_file = os.path.join(summaries_dir, f"{date_str}.json")

        if os.path.exists(json_file):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    summaries[date_str] = data.get('summary', '無')
            except Exception as e:
                print(f"Error reading {json_file}: {e}")
                summaries[date_str] = '無'
        else:
            summaries[date_str] = '無'

    return summaries


def apply_rolling_window(rho_values: np.ndarray, window_days: int) -> np.ndarray:
    """
    Apply rolling average to rho values.

    Args:
        rho_values: Raw rho values for each day
        window_days: Rolling window size in days

    Returns:
        Smoothed rho values
    """
    if window_days <= 1:
        return rho_values

    # Use pandas rolling mean with min_periods=1 to handle edge cases
    series = pd.Series(rho_values)
    smoothed = series.rolling(window=window_days, min_periods=1).mean()

    return smoothed.values


def generate_finbert_rho_daily(
    summaries_dir: str,
    trading_dates: np.ndarray,
    generator: FinBERTRhoGenerator,
    mode: str = "positive_prob",
    window_days: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate FinBERT-based rho values for all trading dates.

    Args:
        summaries_dir: Path to summaries directory
        trading_dates: Array of trading dates
        generator: FinBERTRhoGenerator instance
        mode: Calculation mode
        window_days: Rolling window size

    Returns:
        Tuple of (rho_values, p_positive_values, p_neutral_values, p_negative_values)
    """
    # Load summaries
    summaries = load_summaries(summaries_dir, trading_dates)

    num_days = len(trading_dates)
    p_positive_values = np.zeros(num_days)
    p_neutral_values = np.zeros(num_days)
    p_negative_values = np.zeros(num_days)

    # Process each day
    print(f"\nProcessing {num_days} trading days with mode='{mode}'...")
    for i, date in enumerate(tqdm(trading_dates, desc="Computing sentiment")):
        date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
        summary = summaries.get(date_str, '無')

        p_pos, p_neu, p_neg = generator.get_sentiment_probs(summary)
        p_positive_values[i] = p_pos
        p_neutral_values[i] = p_neu
        p_negative_values[i] = p_neg

    # Compute raw rho values
    raw_rho_values = np.array([
        generator.compute_rho(p_positive_values[i], p_neutral_values[i], p_negative_values[i], mode)
        for i in range(num_days)
    ])

    # Apply rolling window
    rho_values = apply_rolling_window(raw_rho_values, window_days)

    print(f"\nDaily rho statistics (mode={mode}, window={window_days}):")
    print(f"  Min: {rho_values.min():.4f}")
    print(f"  Max: {rho_values.max():.4f}")
    print(f"  Mean: {rho_values.mean():.4f}")
    print(f"  Std: {rho_values.std():.4f}")

    return rho_values, p_positive_values, p_neutral_values, p_negative_values


def generate_rho_file(
    rho_values: np.ndarray,
    trading_dates: np.ndarray,
    start_idx: int,
    end_idx: int,
    window_len: int,
    trade_len: int,
    mode: str,
    window_days: int,
    output_dir: str
) -> str:
    """
    Generate rho file for a specific period (using index range in filename).

    Args:
        rho_values: Pre-computed rho values for all days
        trading_dates: Array of all trading dates
        start_idx: Start index of the period
        end_idx: End index of the period
        window_len: Window length in weeks (e.g., 13)
        trade_len: Trade length in days (e.g., 21)
        mode: Calculation mode used
        window_days: Rolling window size used
        output_dir: Output directory

    Returns:
        Path to the generated file
    """
    print(f"\n{'='*60}")
    print(f"Generating FinBERT Rho File: {start_idx}-{end_idx}")
    print(f"{'='*60}")

    # Calculate number of steps
    num_steps = (end_idx - start_idx) // trade_len

    print(f"Index range: {start_idx} to {end_idx}")
    print(f"Trade length: {trade_len}")
    print(f"Number of steps: {num_steps}")

    # Create ground truth records
    ground_truth_records = []
    rho_record = []

    for step in range(num_steps):
        cursor = start_idx + step * trade_len

        # Use the rho value at the cursor position (decision day)
        if cursor < len(rho_values):
            rho = float(rho_values[cursor])
        else:
            print(f"Warning: cursor {cursor} out of bounds, using 0.5")
            rho = 0.5

        record = {
            "input_start": cursor - window_len * 5,
            "input_end": cursor + 1,
            "predict_start": cursor,
            "predict_end": cursor + trade_len + 1,
            "step": step + 1,
            "idx": cursor,
            "rho": rho,
            "source": "finbert"
        }

        ground_truth_records.append(record)
        rho_record.append(rho)

    # Statistics for this period
    rho_array = np.array(rho_record)
    print(f"\nPeriod rho statistics:")
    print(f"  Min: {rho_array.min():.4f}")
    print(f"  Max: {rho_array.max():.4f}")
    print(f"  Mean: {rho_array.mean():.4f}")
    print(f"  Std: {rho_array.std():.4f}")

    # Create output JSON
    output_data = {
        "ground_truth_records": ground_truth_records,
        "rho_record": rho_record,
        "metadata": {
            "source": "finbert",
            "mode": mode,
            "window_days": window_days,
            "model": "yiyanghkust/finbert-tone-chinese",
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_steps": num_steps,
            "trade_len": trade_len,
            "window_len": window_len
        }
    }

    # Save to file (using index range in filename)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"finbert_rho_{start_idx}-{end_idx}_mode-{mode}_window-{window_days}.json"
    output_path = os.path.join(output_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved to: {output_path}")

    return output_path


def load_config(config_path: str) -> Dict:
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to config JSON file

    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)

    print(f"Loaded config from: {config_path}")
    return config


def run_with_config(config: Dict):
    """
    Run FinBERT rho generation with config dictionary.

    Args:
        config: Configuration dictionary
    """
    summaries_dir = config['summaries_dir']
    trading_dates_file = config['trading_dates_file']
    output_dir = config['output_dir']
    mode = config.get('mode', 'positive_prob')
    window_days = config.get('window_days', 1)
    window_len = config.get('window_len', 13)
    trade_len = config.get('trade_len', 21)
    periods = config['periods']
    model_name = config.get('model_name', 'yiyanghkust/finbert-tone-chinese')
    save_daily = config.get('save_daily', False)

    print("=" * 80)
    print("FinBERT-based Rho Generator for MSU")
    print("=" * 80)
    print(f"Summaries directory: {summaries_dir}")
    print(f"Trading dates file: {trading_dates_file}")
    print(f"Output directory: {output_dir}")
    print(f"Mode: {mode}")
    print(f"Rolling window: {window_days} days")
    print(f"Number of periods: {len(periods)}")
    print()

    # Load trading dates
    print("Loading trading dates...")
    trading_dates = np.load(trading_dates_file, allow_pickle=True)
    print(f"Loaded {len(trading_dates)} trading dates")
    print(f"Date range: {trading_dates[0]} to {trading_dates[-1]}")

    # Initialize generator
    generator = FinBERTRhoGenerator(model_name=model_name)

    # Generate daily rho values (computed once, reused for all periods)
    rho_values, p_pos, p_neu, p_neg = generate_finbert_rho_daily(
        summaries_dir=summaries_dir,
        trading_dates=trading_dates,
        generator=generator,
        mode=mode,
        window_days=window_days
    )

    # Save daily values if requested
    if save_daily:
        daily_file = os.path.join(output_dir, f"finbert_rho_daily_mode-{mode}_window-{window_days}.npy")
        np.save(daily_file, rho_values)
        print(f"\n✅ Saved daily rho values to: {daily_file}")

        probs_file = os.path.join(output_dir, f"finbert_probs_daily.npz")
        np.savez(probs_file, p_positive=p_pos, p_neutral=p_neu, p_negative=p_neg)
        print(f"✅ Saved daily probabilities to: {probs_file}")

    # Generate rho files for each period
    generated_files = []
    for period in periods:
        start_idx = period['start']
        end_idx = period['end']

        output_path = generate_rho_file(
            rho_values=rho_values,
            trading_dates=trading_dates,
            start_idx=start_idx,
            end_idx=end_idx,
            window_len=window_len,
            trade_len=trade_len,
            mode=mode,
            window_days=window_days,
            output_dir=output_dir
        )
        generated_files.append(output_path)

    print("\n" + "=" * 80)
    print("✅ FinBERT rho generation complete!")
    print(f"Generated {len(generated_files)} files:")
    for f in generated_files:
        print(f"  - {f}")
    print("=" * 80)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Generate FinBERT-based Rho for MSU')

    # Config file (recommended)
    parser.add_argument('--config', type=str,
                        help='Path to config JSON file (recommended for multiple periods)')

    # Command line arguments (for single period)
    parser.add_argument('--summaries_dir', type=str,
                        help='Path to summaries directory (e.g., src/data/TWII/summaries_v2/0050)')
    parser.add_argument('--trading_dates_file', type=str,
                        help='Path to trading_dates.npy file')
    parser.add_argument('--output_dir', type=str,
                        help='Output directory for rho files')
    parser.add_argument('--mode', type=str, default='positive_prob',
                        choices=['positive_prob', 'sentiment_diff', 'weighted'],
                        help='Rho calculation mode (default: positive_prob)')
    parser.add_argument('--window_days', type=int, default=1,
                        help='Rolling window size in days (default: 1, no smoothing)')
    parser.add_argument('--window_len', type=int, default=13,
                        help='Window length in weeks (default: 13)')
    parser.add_argument('--trade_len', type=int, default=21,
                        help='Trade length in days (default: 21)')
    parser.add_argument('--start_idx', type=int,
                        help='Start index of the period')
    parser.add_argument('--end_idx', type=int,
                        help='End index of the period')
    parser.add_argument('--model_name', type=str, default='yiyanghkust/finbert-tone-chinese',
                        help='FinBERT model name')
    parser.add_argument('--save_daily', action='store_true',
                        help='Save daily rho values as .npy file')

    args = parser.parse_args()

    # If config file provided, use it
    if args.config:
        config = load_config(args.config)
        run_with_config(config)
    else:
        # Use command line arguments
        if not all([args.summaries_dir, args.trading_dates_file, args.output_dir, args.start_idx, args.end_idx]):
            parser.error("When not using --config, the following arguments are required: "
                        "--summaries_dir, --trading_dates_file, --output_dir, --start_idx, --end_idx")

        # Build config from command line args
        config = {
            'summaries_dir': args.summaries_dir,
            'trading_dates_file': args.trading_dates_file,
            'output_dir': args.output_dir,
            'mode': args.mode,
            'window_days': args.window_days,
            'window_len': args.window_len,
            'trade_len': args.trade_len,
            'model_name': args.model_name,
            'save_daily': args.save_daily,
            'periods': [
                {'start': args.start_idx, 'end': args.end_idx}
            ]
        }
        run_with_config(config)


if __name__ == '__main__':
    main()
