"""
FineWeb Dataset with Shard Caching

This module implements a streaming dataset for HuggingFace's FineWeb dataset
with intelligent shard caching and LRU (Least Recently Used) cleanup.

What is FineWeb?
----------------
FineWeb is a large-scale web crawl dataset from HuggingFace:
- sample-10BT: 10 billion tokens of web text
- High quality, filtered web content
- Perfect for pretraining language models

Problem with Large Datasets:
----------------------------
FineWeb sample-10BT is ~96GB. We can't:
- Download it all at once (too large)
- Keep it all in memory (way too large)
- Re-download every time (too slow)

Our Solution: Streaming + Caching
----------------------------------
1. STREAMING: Download shards on-demand from HuggingFace
2. CACHING: Save downloaded shards to disk for reuse
3. LRU CLEANUP: Automatically delete old shards when cache fills up

Architecture:
-------------
    HuggingFace FineWeb
           ↓ (stream on-demand)
    ┌─────────────────────┐
    │ FineWebDataset      │
    │ - Streams shards    │
    │ - Caches to disk    │
    │ - Tracks usage      │
    └─────────────────────┘
           ↓
    data/fineweb_cache/
    ├── shard_00000.parquet  ← Most recently used
    ├── shard_00001.parquet
    ├── shard_00002.parquet
    ├── ...
    └── cache_metadata.json  ← Tracks usage timestamps

Shard Lifecycle:
----------------
1. Dataset needs shard_00123
2. Check if shard_00123.parquet exists in cache
3a. If YES:
    - Load from disk (fast!)
    - Update "last_accessed" timestamp
    - Return tokenized data
3b. If NO:
    - Stream from HuggingFace
    - Save to data/fineweb_cache/shard_00123.parquet
    - Add to metadata with timestamp
    - Return tokenized data
4. After processing:
    - Check if cache size > max_shards
    - If yes: Delete least recently used shard

LRU Cleanup Example:
--------------------
max_shards = 5

Cache state:
    shard_00010.parquet (accessed 10 min ago)
    shard_00011.parquet (accessed 5 min ago)
    shard_00012.parquet (accessed 2 min ago)
    shard_00013.parquet (accessed 1 min ago)
    shard_00014.parquet (accessed 30 sec ago)  ← 5 shards total

New shard needed: shard_00015
    → Cache full (5 shards)
    → Delete shard_00010 (oldest access)
    → Download shard_00015
    → Cache now has shards 00011-00015

Benefits:
---------
✓ No huge upfront download
✓ Fast re-access of recent shards
✓ Automatic space management
✓ Can train on 10B+ tokens with only 2-5GB cache

Token Sampling Strategy:
------------------------
With 10B total tokens, we can't use all in one epoch. We sample:
- tokens_per_epoch: How many tokens to process per epoch (e.g., 100M)
- This is ~1% of the full dataset
- Each epoch samples different shards for variety
- Prevents overfitting while staying manageable
"""

import torch
from torch.utils.data import IterableDataset
import tiktoken
from pathlib import Path
import json
import time
from datasets import load_dataset
from datasets.utils import disable_progress_bars
from typing import Iterator, Tuple, Optional
import random
import logging
import warnings

# Suppress HuggingFace datasets progress bars and warnings
disable_progress_bars()
logging.getLogger("datasets").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", category=UserWarning, module="datasets")


class FineWebDataset(IterableDataset):
    """
    Streaming dataset for FineWeb with shard caching and LRU cleanup.

    This dataset:
    - Streams shards from HuggingFace on-demand
    - Caches shards locally for fast reuse
    - Automatically manages disk space with LRU cleanup
    - Tokenizes text using tiktoken (BPE)
    - Yields (input, target) pairs for autoregressive training
    """

    def __init__(
        self,
        cache_dir: str = "data/fineweb_cache",
        seq_length: int = 128,
        encoding_name: str = "p50k_base",
        tokens_per_epoch: int = 100_000_000,  # 100M tokens per epoch
        max_cached_shards: int = 5,
        dataset_name: str = "HuggingFaceFW/fineweb",
        dataset_split: str = "sample-10BT",
        seed: Optional[int] = None,
    ):
        """
        Initialize FineWeb dataset with caching.

        Args:
            cache_dir: Directory to cache downloaded shards
            seq_length: Length of each training sequence
            encoding_name: tiktoken encoding to use (p50k_base ~50k vocab)
            tokens_per_epoch: How many tokens to process per epoch
                             100M = ~781K sequences with seq_length=128
                             This allows manageable epochs from huge dataset
            max_cached_shards: Maximum number of shards to keep in cache
                              Each shard is ~300-500MB, so 5 shards = ~2GB
            dataset_name: HuggingFace dataset identifier
            dataset_split: Which split/subset to use
            seed: Random seed for shard selection (None = random each time)
        """
        super().__init__()

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.seq_length = seq_length
        self.tokens_per_epoch = tokens_per_epoch
        self.max_cached_shards = max_cached_shards
        self.dataset_name = dataset_name
        self.dataset_split = dataset_split

        # Tokenizer setup
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Metadata file for tracking shard usage
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()

        # Random seed for reproducible shard selection
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        print(f"FineWeb Dataset initialized:")
        print(f"  Cache directory: {self.cache_dir}")
        print(f"  Tokens per epoch: {self.tokens_per_epoch:,}")
        print(f"  Sequences per epoch: ~{self.tokens_per_epoch // seq_length:,}")
        print(f"  Max cached shards: {max_cached_shards}")
        print(f"  Vocabulary size: {self.tokenizer.n_vocab:,}")

    def _load_metadata(self) -> dict:
        """Load cache metadata from disk."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {"shards": {}}

    def _save_metadata(self):
        """Save cache metadata to disk."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)

    def _update_shard_access(self, shard_id: str):
        """Update last access timestamp for a shard."""
        if shard_id not in self.metadata["shards"]:
            self.metadata["shards"][shard_id] = {
                "first_accessed": time.time(),
                "last_accessed": time.time(),
                "access_count": 1
            }
        else:
            self.metadata["shards"][shard_id]["last_accessed"] = time.time()
            self.metadata["shards"][shard_id]["access_count"] += 1

        self._save_metadata()

    def _cleanup_old_shards(self):
        """Remove least recently used shards if cache exceeds max size."""
        # Get all cached shard files
        cached_files = list(self.cache_dir.glob("shard_*.parquet"))

        if len(cached_files) <= self.max_cached_shards:
            return  # No cleanup needed

        # Sort by last access time (oldest first)
        shards_by_access = sorted(
            self.metadata["shards"].items(),
            key=lambda x: x[1]["last_accessed"]
        )

        # Calculate how many to delete
        num_to_delete = len(cached_files) - self.max_cached_shards

        # Delete oldest shards
        for shard_id, _ in shards_by_access[:num_to_delete]:
            shard_file = self.cache_dir / f"{shard_id}.parquet"
            if shard_file.exists():
                shard_file.unlink()
                # Silently remove old shard

            # Remove from metadata
            del self.metadata["shards"][shard_id]

        self._save_metadata()

    def _get_shard_data(self, shard_idx: int) -> list:
        """
        Get data from a specific shard, using cache if available.

        Args:
            shard_idx: Index of the shard to load

        Returns:
            List of text strings from the shard
        """
        shard_id = f"shard_{shard_idx:05d}"
        shard_file = self.cache_dir / f"{shard_id}.parquet"

        # Try to load from cache
        if shard_file.exists():
            # Silently load from cache
            import pyarrow.parquet as pq
            table = pq.read_table(shard_file)
            texts = table['text'].to_pylist()
            self._update_shard_access(shard_id)
            return texts

        # Cache miss - stream from HuggingFace (silently)

        # Load dataset in streaming mode
        dataset = load_dataset(
            self.dataset_name,
            name=self.dataset_split,
            split="train",
            streaming=True
        )

        # Skip to the desired shard (each shard has ~1000 examples typically)
        # This is approximate - exact shard boundaries depend on dataset structure
        shard_size = 1000
        start_idx = shard_idx * shard_size

        texts = []
        for i, example in enumerate(dataset):
            if i < start_idx:
                continue
            if i >= start_idx + shard_size:
                break
            texts.append(example['text'])

        # Save to cache
        if texts:  # Only save if we got data
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.table({'text': texts})
            pq.write_table(table, shard_file)
            # Silently save to cache

            self._update_shard_access(shard_id)
            self._cleanup_old_shards()

        return texts

    def _tokenize_and_chunk(self, texts: list) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize texts and yield (input, target) pairs.

        Args:
            texts: List of text strings

        Yields:
            (input_tensor, target_tensor) pairs of shape (seq_length,)
        """
        for text in texts:
            # Tokenize the text
            tokens = self.tokenizer.encode(text)

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
                        torch.tensor(target_seq, dtype=torch.long)
                    )

    def __iter__(self) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Iterate through the dataset, yielding (input, target) pairs.

        This streams shards on-demand until we've processed tokens_per_epoch tokens.

        Yields:
            (input_tensor, target_tensor) pairs of shape (seq_length,)
        """
        tokens_processed = 0
        shard_idx = 0

        # We don't know exact shard count, so we'll stream until we hit our token limit
        # Approximate max shards needed (assuming ~500K tokens per shard)
        max_shards_needed = (self.tokens_per_epoch // 500_000) + 1

        # Randomly select which shards to use (for variety across epochs)
        if self.seed is None:
            # Random shards each epoch
            shard_indices = random.sample(range(0, 1000), min(max_shards_needed, 1000))
        else:
            # Deterministic shard selection
            shard_indices = list(range(max_shards_needed))

        for shard_idx in shard_indices:
            if tokens_processed >= self.tokens_per_epoch:
                break

            # Get shard data (from cache or HuggingFace)
            texts = self._get_shard_data(shard_idx)

            # Tokenize and yield sequences
            for input_seq, target_seq in self._tokenize_and_chunk(texts):
                if tokens_processed >= self.tokens_per_epoch:
                    break

                yield input_seq, target_seq
                tokens_processed += self.seq_length

        # Epoch complete (silently)

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
            return self.tokenizer.decode(valid_tokens, errors='ignore')
        except Exception:
            return "[decoding error]"

    @property
    def vocab_size(self):
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.n_vocab
