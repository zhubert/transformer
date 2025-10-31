"""
Test dataset configuration utilities.

Tests for the dataset_utils module, particularly cache size calculation logic.
"""

import pytest
import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.dataset_utils import calculate_optimal_cache_size


class TestCalculateOptimalCacheSize:
    """Test suite for calculate_optimal_cache_size function."""

    def test_quick_mode_10m_tokens(self):
        """Test cache size calculation for quick mode (10M tokens)."""
        tokens = 10_000_000
        cache_size = calculate_optimal_cache_size(tokens)

        # Expected calculation:
        # train_shards = 10M / 500K = 20
        # val_shards = 1M / 500K = 2
        # total = (20 + 2) * 1.2 = 26.4 → 26
        assert cache_size == 26

    def test_medium_mode_50m_tokens(self):
        """Test cache size calculation for medium mode (50M tokens)."""
        tokens = 50_000_000
        cache_size = calculate_optimal_cache_size(tokens)

        # Expected calculation:
        # train_shards = 50M / 500K = 100
        # val_shards = 5M / 500K = 10
        # total = (100 + 10) * 1.2 = 132
        assert cache_size == 132

    def test_default_mode_100m_tokens(self):
        """Test cache size calculation for default mode (100M tokens)."""
        tokens = 100_000_000
        cache_size = calculate_optimal_cache_size(tokens)

        # Expected calculation:
        # train_shards = 100M / 500K = 200
        # val_shards = 10M / 500K = 20
        # total = (200 + 20) * 1.2 = 264
        assert cache_size == 264

    def test_large_mode_500m_tokens(self):
        """Test cache size calculation for large mode (500M tokens)."""
        tokens = 500_000_000
        cache_size = calculate_optimal_cache_size(tokens)

        # Expected calculation:
        # train_shards = 500M / 500K = 1000
        # val_shards = 50M / 500K = 100
        # total = (1000 + 100) * 1.2 = 1320
        assert cache_size == 1320

    def test_small_token_count(self):
        """Test cache size calculation for very small token counts."""
        tokens = 1_000_000  # 1M tokens
        cache_size = calculate_optimal_cache_size(tokens)

        # Expected calculation:
        # train_shards = 1M / 500K = 2
        # val_shards = 100K / 500K = 0.2
        # total = (2 + 0.2) * 1.2 = 2.64 → 2
        assert cache_size == 2

    def test_minimum_token_count(self):
        """Test cache size calculation for minimal token count."""
        tokens = 100_000  # 100K tokens
        cache_size = calculate_optimal_cache_size(tokens)

        # Should return at least 0 (though not practical)
        assert cache_size >= 0
        assert isinstance(cache_size, int)

    def test_returns_integer(self):
        """Test that function always returns an integer."""
        test_values = [10_000_000, 50_000_000, 75_000_000, 100_000_000]
        for tokens in test_values:
            cache_size = calculate_optimal_cache_size(tokens)
            assert isinstance(cache_size, int), f"Expected int, got {type(cache_size)}"

    def test_scales_linearly(self):
        """Test that cache size scales approximately linearly with tokens."""
        tokens_1 = 10_000_000
        tokens_2 = 20_000_000  # Double

        cache_1 = calculate_optimal_cache_size(tokens_1)
        cache_2 = calculate_optimal_cache_size(tokens_2)

        # Cache size should approximately double (within rounding)
        ratio = cache_2 / cache_1
        assert 1.8 <= ratio <= 2.2, f"Expected ~2x scaling, got {ratio}x"

    def test_includes_validation_split(self):
        """Test that calculation includes validation data (10% of tokens)."""
        tokens = 10_000_000

        # Calculate expected shards
        TOKENS_PER_SHARD = 500_000
        train_shards = tokens / TOKENS_PER_SHARD  # 20
        val_shards = (tokens / 10) / TOKENS_PER_SHARD  # 2

        expected = int((train_shards + val_shards) * 1.2)
        actual = calculate_optimal_cache_size(tokens)

        assert actual == expected

    def test_includes_safety_buffer(self):
        """Test that calculation includes 20% safety buffer."""
        tokens = 10_000_000
        TOKENS_PER_SHARD = 500_000

        # Calculate without buffer
        train_shards = tokens / TOKENS_PER_SHARD
        val_shards = (tokens / 10) / TOKENS_PER_SHARD
        without_buffer = int(train_shards + val_shards)  # 22

        # Calculate with buffer
        with_buffer = calculate_optimal_cache_size(tokens)  # 26

        # Buffer should increase the cache size
        assert with_buffer > without_buffer
        # Buffer should be approximately 20%
        buffer_ratio = with_buffer / without_buffer
        assert 1.15 <= buffer_ratio <= 1.25  # Within reasonable range


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
