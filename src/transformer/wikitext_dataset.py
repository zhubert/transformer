"""
WikiText-103 Dataset for Language Model Training

What is WikiText-103?
--------------------
WikiText-103 is a curated dataset from Wikipedia articles:
- Size: ~100M tokens (517MB uncompressed)
- Source: High-quality, edited Wikipedia text
- Language: English, formal writing style
- Use case: Benchmarking and educational training

Comparison to FineWeb:
----------------------
| Feature | WikiText-103 | FineWeb |
|---------|-------------|---------|
| Size | 100M tokens | 10B+ tokens |
| Quality | Very high (edited) | High (filtered) |
| Diversity | Limited (encyclopedia) | High (web) |
| Speed | Fast (small) | Slower (large) |
| Perplexity | Lower (easier) | Higher (harder) |

Why use WikiText-103?
---------------------
✓ Fast experiments - download once, fits in memory
✓ Well-known benchmark - compare to published results
✓ Clean text - minimal noise, proper grammar
✓ Educational - see lower perplexity faster (confidence boost!)
✓ Debugging - quick iterations during development

Expected Results:
-----------------
For a small model (6 layers, 256 d_model):
- Random initialization: ~100,000 perplexity
- After 1 epoch: ~200-500 perplexity
- After 5 epochs: ~50-150 perplexity
- After 10 epochs: ~30-80 perplexity

Compare to FineWeb (same model):
- After 5 epochs: ~80-200 perplexity (harder dataset!)

Architecture:
-------------
    HuggingFace WikiText-103
           ↓ (download once)
    ┌─────────────────────┐
    │ WikiTextDataset     │
    │ - Downloads dataset │
    │ - Caches locally    │
    │ - No streaming      │
    └─────────────────────┘
           ↓
    ~/.cache/huggingface/datasets/
"""

import torch
from torch.utils.data import IterableDataset
import tiktoken
from datasets import load_dataset
from datasets.utils import disable_progress_bars
from typing import Iterator, Tuple
import logging
import warnings

# Suppress HuggingFace datasets progress bars and warnings
disable_progress_bars()
logging.getLogger("datasets").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")


class WikiTextDataset(IterableDataset):
    """
    WikiText-103 dataset for language model training and benchmarking.

    This dataset:
    - Downloads WikiText-103 from HuggingFace (once, cached automatically)
    - Tokenizes text using tiktoken (BPE)
    - Yields (input, target) pairs for autoregressive training
    - Supports train/validation/test splits
    """

    def __init__(
        self,
        seq_length: int = 128,
        encoding_name: str = "cl100k_base",
        split: str = "train",
        tokens_per_epoch: int = None,
    ):
        """
        Initialize WikiText-103 dataset.

        Args:
            seq_length: Length of each training sequence (default 128)
            encoding_name: tiktoken encoding to use (cl100k_base ~100k vocab)
            split: Which split to use - "train", "validation", or "test"
                  - "train": ~100M tokens for training
                  - "validation": ~217K tokens for validation during training
                  - "test": ~245K tokens for final evaluation
            tokens_per_epoch: How many tokens to process per epoch (None = unlimited, process entire dataset)
                             This allows you to control epoch length for consistent training
        """
        super().__init__()

        self.seq_length = seq_length
        self.split = split
        self.tokens_per_epoch = tokens_per_epoch

        # Validate split parameter
        if split not in ["train", "validation", "test"]:
            raise ValueError(
                f"split must be 'train', 'validation', or 'test', got '{split}'"
            )

        # Tokenizer setup
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Load dataset (HuggingFace caches automatically)
        # First load downloads to ~/.cache/huggingface/datasets/
        # Subsequent loads use cached version
        self.dataset = load_dataset(
            "wikitext",
            "wikitext-103-raw-v1",
            split=split,
        )

    def _tokenize_and_chunk(
        self, texts: list
    ) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize texts and yield (input, target) pairs.

        Args:
            texts: List of text strings

        Yields:
            (input_tensor, target_tensor) pairs of shape (seq_length,)
        """
        for text in texts:
            # Skip empty lines (WikiText has many section separators)
            if not text.strip():
                continue

            # Tokenize the text
            tokens = self.tokenizer.encode(text)

            # Skip if text is too short
            if len(tokens) < self.seq_length + 1:
                continue

            # Create sequences from tokens
            # Each sequence is seq_length tokens
            # Target is same sequence shifted by 1
            for i in range(0, len(tokens) - self.seq_length - 1, self.seq_length):
                input_seq = tokens[i : i + self.seq_length]
                target_seq = tokens[i + 1 : i + self.seq_length + 1]

                # Only yield if we have full sequences
                if len(input_seq) == self.seq_length and len(target_seq) == self.seq_length:
                    yield (
                        torch.tensor(input_seq, dtype=torch.long),
                        torch.tensor(target_seq, dtype=torch.long),
                    )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset, yielding (input, target) pairs.

        If tokens_per_epoch is set, stops after that many tokens have been processed.
        Otherwise, processes the entire dataset.

        Yields:
            (input_tensor, target_tensor) pairs of shape (seq_length,)
        """
        # WikiText-103 provides 'text' field with raw text
        texts = self.dataset["text"]

        # Tokenize and yield sequences, with optional token limit
        tokens_processed = 0
        for input_seq, target_seq in self._tokenize_and_chunk(texts):
            # Check if yielding this sequence would exceed the limit
            if self.tokens_per_epoch is not None and tokens_processed + self.seq_length > self.tokens_per_epoch:
                break

            yield input_seq, target_seq
            tokens_processed += self.seq_length

    def decode(self, token_ids):
        """
        Convert token IDs back to text.

        Args:
            token_ids: List or tensor of token IDs

        Returns:
            text: Decoded text string
        """
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        # Filter out invalid token IDs
        valid_tokens = [t for t in token_ids if 0 <= t < self.vocab_size]

        try:
            return self.tokenizer.decode(valid_tokens, errors="ignore")
        except Exception:
            return "[decoding error]"

    @property
    def vocab_size(self):
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.n_vocab

    def __len__(self):
        """
        Approximate number of sequences in the dataset.

        Note: This is an estimate based on average tokens per example.
        Actual length will vary based on text length distribution.
        """
        # Rough estimate: average ~500 tokens per example in WikiText-103
        avg_tokens_per_example = 500
        num_examples = len(self.dataset)
        total_tokens = num_examples * avg_tokens_per_example
        num_sequences = total_tokens // self.seq_length
        return num_sequences
