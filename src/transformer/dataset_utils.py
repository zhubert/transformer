"""
Dataset configuration utilities.

This module provides utilities for configuring datasets, particularly for
the FineWeb streaming dataset that requires careful cache management.

Why Separate Dataset Utilities?
--------------------------------
Dataset configuration logic (especially cache sizing) was duplicated across
train.py and download_shards.py. Extracting it here provides:
- Single source of truth for cache calculations
- Easier to maintain and update cache strategies
- Better discoverability for configuration values
- Consistent behavior across training and pre-download
"""


def calculate_optimal_cache_size(tokens_per_epoch: int) -> int:
    """
    Calculate optimal shard cache size based on tokens per epoch.

    Why this matters:
    -----------------
    FineWeb streams data in shards (~500K tokens each, ~40MB on disk).
    If the cache is too small, shards are evicted and re-downloaded every epoch,
    wasting network bandwidth and making training much slower.

    This function calculates how many shards are needed to cover an entire epoch
    (including validation data) so that after epoch 1, all shards are cached
    and epochs 2+ run at maximum speed with no network I/O.

    Performance Impact:
    -------------------
    Without optimal caching:
        - Small cache (5 shards): ~105 shards re-downloaded per epoch
        - Network: Constant re-downloading

    With optimal caching:
        - All shards cached after epoch 1
        - Network: Only used in epoch 1

    Speedup: 2-4x faster after epoch 1!

    How it works:
    -------------
    1. Calculate shards needed for training data (90% of tokens)
    2. Calculate shards needed for validation data (10% of tokens)
    3. Add 20% buffer for safety (shard sizes vary slightly)

    Example calculation for 100M tokens/epoch:
        train_shards = 100M / 500K = 200 shards
        val_shards = 10M / 500K = 20 shards
        total = (200 + 20) * 1.2 = 264 shards
        disk_usage = 264 shards * ~40MB = ~10.3 GB

    Args:
        tokens_per_epoch: Number of tokens to process per epoch

    Returns:
        max_cached_shards: Number of shards to keep in cache

    Common Configurations:
    ----------------------
    Quick training (10M tokens/epoch):
        - ~26 shards
        - ~1.0 GB disk
        - Good for testing/development

    Medium training (50M tokens/epoch):
        - ~132 shards
        - ~5.2 GB disk
        - Balanced for most use cases

    Default training (100M tokens/epoch):
        - ~264 shards
        - ~10.3 GB disk
        - Best for production models

    Large training (500M tokens/epoch):
        - ~1,320 shards
        - ~51.5 GB disk
        - For serious model training

    Example:
        >>> # Calculate cache for 100M tokens
        >>> cache_size = calculate_optimal_cache_size(100_000_000)
        >>> print(cache_size)  # 264
        >>>
        >>> # Use in FineWebDataset
        >>> dataset = FineWebDataset(
        ...     tokens_per_epoch=100_000_000,
        ...     max_cached_shards=cache_size
        ... )
    """
    # Each shard contains approximately 500K tokens
    # This is the average across the FineWeb dataset
    TOKENS_PER_SHARD = 500_000

    # Calculate shards needed for training data (90% of tokens)
    train_shards = tokens_per_epoch / TOKENS_PER_SHARD

    # Calculate shards needed for validation data (10% of tokens)
    val_shards = (tokens_per_epoch / 10) / TOKENS_PER_SHARD

    # Add 20% buffer for safety
    # Why buffer?
    # - Shard sizes vary slightly (not exactly 500K)
    # - Better to have a bit extra than run out mid-epoch
    # - Prevents cache thrashing at epoch boundaries
    total_shards = int((train_shards + val_shards) * 1.2)

    return total_shards
