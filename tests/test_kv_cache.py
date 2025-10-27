"""
Tests for KV-cache functionality.

Tests verify:
1. Correctness: cached generation produces identical results to non-cached
2. Cache structure: proper shapes and growth
3. Prefill/decode modes: correct behavior in each mode
4. Multi-layer caching: all layers handle cache correctly
5. Edge cases: single token, empty cache, etc.

Why Test KV-Cache?
------------------
KV-cache is a performance optimization that MUST NOT change model outputs.
If caching changes outputs, it's a bug! These tests ensure correctness.
"""

import pytest
import torch
from src.transformer.model import DecoderOnlyTransformer


class TestKVCacheCorrectness:
    """Test that KV-cache produces identical outputs to non-cached forward pass."""

    def test_single_forward_with_and_without_cache(self):
        """
        Test that a single forward pass with cache produces same output as without.

        This tests the PREFILL mode - processing a prompt and initializing cache.
        """
        # Small model for fast testing
        vocab_size = 100
        d_model = 64
        num_heads = 4
        num_layers = 2
        seq_len = 10
        batch_size = 2

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=256,
            max_seq_len=100,
            dropout=0.0  # No dropout for deterministic testing
        )
        model.eval()  # Evaluation mode

        # Create input
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            # Without cache
            logits_no_cache, _ = model(x, caches=None)

            # With cache (prefill mode)
            logits_with_cache, caches = model(x, caches=None)

        # Outputs should be identical
        assert torch.allclose(logits_no_cache, logits_with_cache, atol=1e-6), \
            "Cached forward pass should produce identical outputs to non-cached"

        # Cache should be initialized
        assert caches is not None, "Cache should be initialized"
        assert len(caches) == num_layers, f"Should have {num_layers} caches"

    def test_autoregressive_generation_correctness(self):
        """
        Test that autoregressive generation with cache produces same results as without.

        This is the CRITICAL test - cached generation must match non-cached generation!
        """
        # Small model for fast testing
        vocab_size = 100
        d_model = 64
        num_heads = 4
        num_layers = 2
        batch_size = 1
        start_len = 5
        max_len = 15  # Generate 10 new tokens

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=256,
            max_seq_len=100,
            dropout=0.0  # No dropout for deterministic testing
        )
        model.eval()

        # Starting tokens
        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        # Generate WITH cache
        with torch.no_grad():
            generated_with_cache = model.generate(
                start_tokens,
                max_length=max_len,
                sampling_strategy="greedy",  # Deterministic
                use_cache=True
            )

        # Generate WITHOUT cache
        with torch.no_grad():
            generated_without_cache = model.generate(
                start_tokens,
                max_length=max_len,
                sampling_strategy="greedy",  # Deterministic
                use_cache=False
            )

        # Generated sequences should be IDENTICAL
        assert torch.equal(generated_with_cache, generated_without_cache), \
            "Cached generation must produce identical outputs to non-cached generation!"

    def test_generation_with_temperature(self):
        """
        Test that cached generation with temperature/sampling produces valid outputs.

        We can't test for identical outputs with sampling (it's random), but we can
        verify that:
        1. Generation completes without errors
        2. Output shapes are correct
        3. Tokens are within vocab range
        """
        vocab_size = 100
        d_model = 64
        num_heads = 4
        num_layers = 2
        batch_size = 2
        start_len = 5
        max_len = 20

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=256,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        # Generate with cache and temperature
        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                max_length=max_len,
                temperature=0.8,
                sampling_strategy="multinomial",
                use_cache=True
            )

        # Check output shape
        assert generated.shape == (batch_size, max_len), \
            f"Expected shape ({batch_size}, {max_len}), got {generated.shape}"

        # Check all tokens are valid
        assert (generated >= 0).all() and (generated < vocab_size).all(), \
            "Generated tokens should be within vocabulary range"


class TestKVCacheStructure:
    """Test that cache has correct structure and shapes."""

    def test_cache_structure(self):
        """Test that cache has correct dictionary structure."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 3
        seq_len = 8
        batch_size = 1

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            _, caches = model(x, caches=None)

        # Check cache structure
        assert isinstance(caches, list), "Caches should be a list"
        assert len(caches) == num_layers, f"Should have {num_layers} layer caches"

        # Check each layer's cache
        for i, cache in enumerate(caches):
            assert isinstance(cache, dict), f"Layer {i} cache should be a dict"
            assert 'keys' in cache, f"Layer {i} cache should have 'keys'"
            assert 'values' in cache, f"Layer {i} cache should have 'values'"

            keys = cache['keys']
            values = cache['values']

            # Check shapes
            # Shape: (batch, num_heads, seq_len, d_k) where d_k = d_model // num_heads
            d_k = d_model // num_heads
            expected_shape = (batch_size, num_heads, seq_len, d_k)

            assert keys.shape == expected_shape, \
                f"Layer {i} keys shape {keys.shape} != expected {expected_shape}"
            assert values.shape == expected_shape, \
                f"Layer {i} values shape {values.shape} != expected {expected_shape}"

    def test_cache_grows_correctly(self):
        """Test that cache grows by 1 token each decode step."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        initial_len = 5
        batch_size = 1

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        # PREFILL: Initial sequence
        x_initial = torch.randint(0, vocab_size, (batch_size, initial_len))

        with torch.no_grad():
            _, caches = model(x_initial, caches=None)

        # Check initial cache size
        for cache in caches:
            assert cache['keys'].shape[2] == initial_len, \
                f"Initial cache should have seq_len={initial_len}"

        # DECODE: Add 3 new tokens one at a time
        for step in range(1, 4):
            new_token = torch.randint(0, vocab_size, (batch_size, 1))

            with torch.no_grad():
                _, caches = model(new_token, caches=caches)

            # Check cache grew by 1
            expected_len = initial_len + step
            for i, cache in enumerate(caches):
                actual_len = cache['keys'].shape[2]
                assert actual_len == expected_len, \
                    f"After {step} decode steps, layer {i} cache should have " \
                    f"seq_len={expected_len}, got {actual_len}"


class TestPrefillAndDecodeModes:
    """Test prefill and decode modes work correctly."""

    def test_prefill_mode(self):
        """Test that prefill mode processes full sequence."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        seq_len = 10
        batch_size = 2

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, caches = model(x, caches=None)

        # Logits should be for full sequence
        assert logits.shape == (batch_size, seq_len, vocab_size), \
            f"Prefill should return logits for all positions"

        # Cache should cover full sequence
        for cache in caches:
            assert cache['keys'].shape[2] == seq_len, \
                "Prefill cache should have full sequence length"

    def test_decode_mode(self):
        """Test that decode mode only processes new token."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        batch_size = 1

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        # Prefill with initial sequence
        x_initial = torch.randint(0, vocab_size, (batch_size, 5))

        with torch.no_grad():
            _, caches = model(x_initial, caches=None)

        # Decode: process single new token
        new_token = torch.randint(0, vocab_size, (batch_size, 1))

        with torch.no_grad():
            logits, new_caches = model(new_token, caches=caches)

        # Logits should only be for the new token
        assert logits.shape == (batch_size, 1, vocab_size), \
            f"Decode should return logits only for new token, got shape {logits.shape}"

        # Cache should have grown by 1
        for cache in new_caches:
            assert cache['keys'].shape[2] == 6, \
                "Decode cache should have grown from 5 to 6"


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_token_input(self):
        """Test with single token input."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        batch_size = 1

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        # Single token
        x = torch.randint(0, vocab_size, (batch_size, 1))

        with torch.no_grad():
            logits, caches = model(x, caches=None)

        assert logits.shape == (batch_size, 1, vocab_size), \
            "Should handle single token input"

        for cache in caches:
            assert cache['keys'].shape[2] == 1, \
                "Cache should have length 1 for single token"

    def test_batch_size_greater_than_one(self):
        """Test that cache works with batch_size > 1."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        seq_len = 8
        batch_size = 4  # Multiple examples in batch

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        with torch.no_grad():
            logits, caches = model(x, caches=None)

        assert logits.shape == (batch_size, seq_len, vocab_size), \
            "Should handle batch_size > 1"

        for cache in caches:
            assert cache['keys'].shape[0] == batch_size, \
                f"Cache batch dimension should be {batch_size}"

    def test_no_cache_backward_compatibility(self):
        """Test that model still works when not using cache (backward compatibility)."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        seq_len = 10
        batch_size = 2

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Call without specifying caches (should default to None)
        with torch.no_grad():
            logits, returned_caches = model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size), \
            "Should work without cache parameter"

        # Caches are always returned now (even when not explicitly requested)
        # This is for forward compatibility with cached generation
        assert returned_caches is not None, \
            "Caches are always returned"
        assert isinstance(returned_caches, list), \
            "Returned caches should be a list"
        assert len(returned_caches) == num_layers, \
            f"Should have {num_layers} caches"


class TestGenerationModes:
    """Test generation with and without cache."""

    def test_generate_without_cache(self):
        """Test that generation works with use_cache=False."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        batch_size = 1
        start_len = 3
        max_len = 10

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                max_length=max_len,
                sampling_strategy="greedy",
                use_cache=False  # Explicitly disable cache
            )

        assert generated.shape == (batch_size, max_len), \
            f"Generated shape should be ({batch_size}, {max_len})"

    def test_generate_with_cache(self):
        """Test that generation works with use_cache=True."""
        vocab_size = 50
        d_model = 32
        num_heads = 2
        num_layers = 2
        batch_size = 1
        start_len = 3
        max_len = 10

        model = DecoderOnlyTransformer(
            vocab_size=vocab_size,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=128,
            max_seq_len=100,
            dropout=0.0
        )
        model.eval()

        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                max_length=max_len,
                sampling_strategy="greedy",
                use_cache=True  # Use cache
            )

        assert generated.shape == (batch_size, max_len), \
            f"Generated shape should be ({batch_size}, {max_len})"


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v"])
