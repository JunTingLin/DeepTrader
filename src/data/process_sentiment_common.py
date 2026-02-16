"""
Common functions for processing news sentiment using FinBERT.
Supports both Chinese (finbert-tone-chinese) and English (ProsusAI/finbert) models.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Tuple, List, Optional
import json
import os
from tqdm import tqdm


class FinBERTSentimentProcessor:
    """
    Process news text using FinBERT to extract:
    1. Sentiment score: -1 (negative), 0 (neutral), 1 (positive)
    2. CLS embedding: 768-dimensional vector
    """

    # Label mapping for sentiment scores (case-insensitive)
    LABEL_TO_SCORE = {
        # Lowercase
        'negative': -1,
        'neutral': 0,
        'positive': 1,
        # Capitalized (yiyanghkust/finbert-tone-chinese uses this format)
        'Negative': -1,
        'Neutral': 0,
        'Positive': 1,
        # Alternative label formats (some models use LABEL_X)
        'LABEL_0': 0,   # neutral (for ProsusAI/finbert)
        'LABEL_1': 1,   # positive
        'LABEL_2': -1,  # negative
    }

    def __init__(self, model_name: str = "yiyanghkust/finbert-tone-chinese", device: str = None):
        """
        Initialize FinBERT model.

        Args:
            model_name: HuggingFace model name
                - "yiyanghkust/finbert-tone-chinese" for Chinese
                - "ProsusAI/finbert" for English
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

        # Get the hidden size (usually 768 for BERT-based models)
        self.hidden_size = self.model.config.hidden_size
        print(f"Hidden size: {self.hidden_size}")

    @torch.no_grad()
    def process_single(self, text: str) -> Tuple[int, np.ndarray]:
        """
        Process a single text and return sentiment score and CLS embedding.

        Args:
            text: News summary text

        Returns:
            Tuple of (sentiment_score, cls_embedding)
            - sentiment_score: -1, 0, or 1
            - cls_embedding: numpy array of shape (hidden_size,)
        """
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Forward pass with hidden states
        outputs = self.model(**inputs, output_hidden_states=True)

        # Get sentiment prediction
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

        # Map to sentiment score
        # Get label from model config if available
        if hasattr(self.model.config, 'id2label'):
            label = self.model.config.id2label[predicted_class]
            sentiment_score = self.LABEL_TO_SCORE.get(label, 0)
        else:
            # Default mapping: 0=negative, 1=neutral, 2=positive
            sentiment_score = predicted_class - 1  # Maps 0,1,2 to -1,0,1

        # Get CLS embedding from last hidden state
        last_hidden_state = outputs.hidden_states[-1]  # [batch, seq_len, hidden]
        cls_embedding = last_hidden_state[0, 0, :].cpu().numpy()  # [hidden_size]

        return sentiment_score, cls_embedding

    def process_batch(self, texts: List[str], batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process a batch of texts.

        Args:
            texts: List of news summary texts
            batch_size: Batch size for processing

        Returns:
            Tuple of (sentiment_scores, cls_embeddings)
            - sentiment_scores: numpy array of shape (num_texts,)
            - cls_embeddings: numpy array of shape (num_texts, hidden_size)
        """
        sentiment_scores = []
        cls_embeddings = []

        for i in tqdm(range(0, len(texts), batch_size), desc="Processing batches"):
            batch_texts = texts[i:i + batch_size]

            for text in batch_texts:
                # Process every text through FinBERT
                # Handle empty text by using "無" as fallback
                if not text or text.strip() == "":
                    text = "無"
                score, embedding = self.process_single(text)
                sentiment_scores.append(score)
                cls_embeddings.append(embedding)

        return np.array(sentiment_scores), np.array(cls_embeddings)


def load_summaries_from_directory(
    summaries_dir: str,
    stock_ids: List[str],
    trading_dates: np.ndarray
) -> dict:
    """
    Load news summaries from directory structure.

    Args:
        summaries_dir: Path to summaries directory (e.g., src/data/TWII/summaries_v2)
        stock_ids: List of stock IDs (e.g., ['1216', '1301', ...])
        trading_dates: Array of trading dates

    Returns:
        Dictionary mapping (stock_id, date_str) -> summary text
    """
    import pandas as pd

    summaries = {}

    for stock_id in tqdm(stock_ids, desc="Loading summaries"):
        stock_dir = os.path.join(summaries_dir, str(stock_id))

        if not os.path.exists(stock_dir):
            print(f"Warning: Directory not found for stock {stock_id}")
            continue

        for date in trading_dates:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            json_file = os.path.join(stock_dir, f"{date_str}.json")

            if os.path.exists(json_file):
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        summaries[(stock_id, date_str)] = data.get('summary', '無')
                except Exception as e:
                    print(f"Error reading {json_file}: {e}")
                    summaries[(stock_id, date_str)] = '無'
            else:
                summaries[(stock_id, date_str)] = '無'

    return summaries


def generate_sentiment_data(
    summaries_dir: str,
    stock_ids: List[str],
    trading_dates: np.ndarray,
    output_dir: str,
    model_name: str = "yiyanghkust/finbert-tone-chinese",
    batch_size: int = 16
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate sentiment scores and CLS embeddings for all stocks and dates.

    Args:
        summaries_dir: Path to summaries directory
        stock_ids: List of stock IDs
        trading_dates: Array of trading dates
        output_dir: Output directory for .npy files
        model_name: FinBERT model name
        batch_size: Batch size for processing

    Returns:
        Tuple of (sentiment_scores, cls_embeddings)
        - sentiment_scores: shape (num_stocks, num_days)
        - cls_embeddings: shape (num_stocks, num_days, hidden_size)
    """
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    # Initialize processor
    processor = FinBERTSentimentProcessor(model_name=model_name)

    num_stocks = len(stock_ids)
    num_days = len(trading_dates)
    hidden_size = processor.hidden_size

    print(f"\nGenerating sentiment data:")
    print(f"  Stocks: {num_stocks}")
    print(f"  Trading days: {num_days}")
    print(f"  Hidden size: {hidden_size}")

    # Initialize output arrays
    sentiment_scores = np.zeros((num_stocks, num_days), dtype=np.int8)
    cls_embeddings = np.zeros((num_stocks, num_days, hidden_size), dtype=np.float32)

    # Load summaries
    print("\nLoading summaries...")
    summaries = load_summaries_from_directory(summaries_dir, stock_ids, trading_dates)

    # Process each stock
    for stock_idx, stock_id in enumerate(tqdm(stock_ids, desc="Processing stocks")):
        texts = []
        for date in trading_dates:
            date_str = pd.Timestamp(date).strftime('%Y-%m-%d')
            text = summaries.get((stock_id, date_str), '無')
            texts.append(text)

        # Process all texts for this stock
        scores, embeddings = processor.process_batch(texts, batch_size=batch_size)

        sentiment_scores[stock_idx] = scores
        cls_embeddings[stock_idx] = embeddings

    # Save outputs
    sentiment_file = os.path.join(output_dir, 'sentiment_scores.npy')
    embeddings_file = os.path.join(output_dir, 'cls_embeddings.npy')

    np.save(sentiment_file, sentiment_scores)
    np.save(embeddings_file, cls_embeddings)

    print(f"\nSaved outputs:")
    print(f"  {sentiment_file}: shape {sentiment_scores.shape}")
    print(f"  {embeddings_file}: shape {cls_embeddings.shape}")

    return sentiment_scores, cls_embeddings


if __name__ == "__main__":
    # Test with a simple example
    processor = FinBERTSentimentProcessor()

    test_texts = [
        "台積電在法說會後股價飆升至1110元，市值達28.8兆元，推動台股大幅上漲。",
        "公司營收大幅下滑，股價重挫。",
        "無",
    ]

    for text in test_texts:
        score, embedding = processor.process_single(text)
        print(f"Text: {text[:30]}...")
        print(f"  Sentiment: {score}")
        print(f"  Embedding shape: {embedding.shape}")
        print()
