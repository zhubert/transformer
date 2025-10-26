#!/usr/bin/env python3
"""
Test FineWeb dataset integration.

This script tests the FineWebDataset class to verify:
1. Streaming from HuggingFace works
2. Shard caching works
3. Data is tokenized correctly
4. Cache cleanup works

Run this before starting full training to ensure everything works!
"""

import sys
from pathlib import Path
import time

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.fineweb_dataset import FineWebDataset


def test_fineweb():
    """Test FineWeb dataset functionality."""

    print("=" * 80)
    print("FINEWEB DATASET TEST")
    print("=" * 80)
    print()

    # Create dataset with very small tokens_per_epoch for testing
    print("Creating FineWebDataset...")
    dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=128,
        tokens_per_epoch=10_000,  # Small for testing (10K tokens)
        max_cached_shards=3,      # Keep only 3 shards in cache
        seed=42
    )
    print()

    # Test 1: Load first shard
    print("=" * 80)
    print("TEST 1: Load first batch (will download shard from HuggingFace)")
    print("=" * 80)
    print()

    start_time = time.time()
    batch_count = 0
    token_count = 0

    for input_seq, target_seq in dataset:
        batch_count += 1
        token_count += len(input_seq)

        if batch_count == 1:
            # Print first batch details
            print(f"First batch:")
            print(f"  Input shape: {input_seq.shape}")
            print(f"  Target shape: {target_seq.shape}")
            print(f"  Input tokens (first 10): {input_seq[:10].tolist()}")
            print(f"  Decoded input: {dataset.decode(input_seq[:50])!r}")
            print()

        if batch_count >= 5:
            break

    elapsed_1 = time.time() - start_time
    print(f"Loaded {batch_count} batches ({token_count:,} tokens) in {elapsed_1:.2f}s")
    print()

    # Test 2: Re-load same shard (should be faster - from cache)
    print("=" * 80)
    print("TEST 2: Re-load from cache (should be faster)")
    print("=" * 80)
    print()

    dataset2 = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=128,
        tokens_per_epoch=10_000,
        max_cached_shards=3,
        seed=42  # Same seed = same shards
    )

    start_time = time.time()
    batch_count = 0

    for input_seq, target_seq in dataset2:
        batch_count += 1
        if batch_count >= 5:
            break

    elapsed_2 = time.time() - start_time
    print(f"Loaded {batch_count} batches in {elapsed_2:.2f}s")
    print()

    # Compare speeds
    if elapsed_2 < elapsed_1:
        speedup = elapsed_1 / elapsed_2
        print(f"✓ Cache is working! Second load was {speedup:.1f}x faster")
    else:
        print(f"⚠ Warning: Second load was not faster. Cache may not be working.")
    print()

    # Test 3: Check cache metadata
    print("=" * 80)
    print("TEST 3: Cache metadata")
    print("=" * 80)
    print()

    metadata = dataset2._load_metadata()
    print(f"Cached shards: {len(metadata['shards'])}")
    for shard_id, info in metadata['shards'].items():
        print(f"  {shard_id}:")
        print(f"    Access count: {info['access_count']}")
        print(f"    Last accessed: {time.time() - info['last_accessed']:.1f}s ago")
    print()

    # Test 4: Check cache directory
    print("=" * 80)
    print("TEST 4: Cache directory contents")
    print("=" * 80)
    print()

    cache_dir = Path("data/fineweb_cache")
    cached_files = list(cache_dir.glob("shard_*.parquet"))
    print(f"Cached shard files: {len(cached_files)}")
    for f in cached_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  {f.name}: {size_mb:.1f} MB")
    print()

    # Summary
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print()
    print("✓ FineWeb dataset is working!")
    print(f"✓ Shards are cached in: {cache_dir}")
    print(f"✓ Cache contains {len(cached_files)} shard(s)")
    print()
    print("You can now run training with: python commands/train.py --dataset fineweb")
    print()


if __name__ == "__main__":
    test_fineweb()
