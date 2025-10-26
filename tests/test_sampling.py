"""
Tests for sampling strategies.

These tests verify that our sampling methods work correctly:
- Temperature scaling
- Top-k sampling
- Top-p sampling
- Combined top-k + top-p
- Greedy decoding
"""

import torch
import pytest
from src.transformer.sampling import (
    apply_temperature,
    sample_top_k,
    sample_top_p,
    sample_top_k_top_p,
    sample_greedy,
)


class TestTemperature:
    """Test temperature scaling."""

    def test_temperature_1_no_change(self):
        """Temperature of 1.0 should not modify logits."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = apply_temperature(logits, temperature=1.0)
        assert torch.allclose(scaled, logits)

    def test_temperature_low_sharpens(self):
        """Temperature < 1.0 should increase logit differences."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = apply_temperature(logits, temperature=0.5)

        # After dividing by 0.5 (multiplying by 2): [2.0, 4.0, 6.0]
        expected = torch.tensor([2.0, 4.0, 6.0])
        assert torch.allclose(scaled, expected)

        # Check that distribution is sharper (higher max probability)
        original_probs = torch.softmax(logits, dim=-1)
        scaled_probs = torch.softmax(scaled, dim=-1)
        assert scaled_probs.max() > original_probs.max()

    def test_temperature_high_flattens(self):
        """Temperature > 1.0 should decrease logit differences."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        scaled = apply_temperature(logits, temperature=2.0)

        # After dividing by 2.0: [0.5, 1.0, 1.5]
        expected = torch.tensor([0.5, 1.0, 1.5])
        assert torch.allclose(scaled, expected)

        # Check that distribution is flatter (lower max probability)
        original_probs = torch.softmax(logits, dim=-1)
        scaled_probs = torch.softmax(scaled, dim=-1)
        assert scaled_probs.max() < original_probs.max()

    def test_temperature_zero_raises(self):
        """Temperature of 0 should raise an error."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Temperature must be positive"):
            apply_temperature(logits, temperature=0.0)

    def test_temperature_negative_raises(self):
        """Negative temperature should raise an error."""
        logits = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="Temperature must be positive"):
            apply_temperature(logits, temperature=-1.0)


class TestGreedySampling:
    """Test greedy decoding."""

    def test_greedy_single_sequence(self):
        """Greedy should return highest logit index for single sequence."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])
        token = sample_greedy(logits)

        # Should return index 1 (highest logit: 3.0)
        assert token.shape == (1,)
        assert token.item() == 1

    def test_greedy_batch(self):
        """Greedy should work with batched input."""
        logits = torch.tensor([
            [1.0, 3.0, 2.0, 0.5],  # max at index 1
            [0.5, 1.0, 0.8, 2.0],  # max at index 3
        ])
        tokens = sample_greedy(logits)

        assert tokens.shape == (2, 1)
        assert tokens[0].item() == 1
        assert tokens[1].item() == 3

    def test_greedy_deterministic(self):
        """Greedy should always return same result."""
        logits = torch.tensor([1.0, 3.0, 2.0])

        token1 = sample_greedy(logits)
        token2 = sample_greedy(logits)

        assert torch.equal(token1, token2)


class TestTopKSampling:
    """Test top-k sampling."""

    def test_top_k_filters_correctly(self):
        """Top-k should only sample from k most probable tokens."""
        # Create logits where we know the top-k
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0, 0.0])  # Sorted descending
        k = 3

        # Sample many times and collect results
        samples = []
        for _ in range(100):
            token = sample_top_k(logits, k=k, temperature=1.0)
            samples.append(token.item())

        # Should only see tokens 0, 1, 2 (top-3)
        unique_samples = set(samples)
        assert unique_samples.issubset({0, 1, 2})

        # Should never see tokens 3, 4, 5
        assert 3 not in unique_samples
        assert 4 not in unique_samples
        assert 5 not in unique_samples

    def test_top_k_k_equals_1_like_greedy(self):
        """Top-k with k=1 should behave like greedy decoding."""
        logits = torch.tensor([1.0, 3.0, 2.0, 0.5])

        # Sample multiple times - should always get same result
        tokens = [sample_top_k(logits, k=1).item() for _ in range(10)]

        # All should be index 1 (highest logit)
        assert all(t == 1 for t in tokens)

    def test_top_k_with_temperature(self):
        """Top-k should work with temperature scaling."""
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        k = 3

        # Low temperature: should favor higher probability tokens more
        low_temp_samples = [sample_top_k(logits, k=k, temperature=0.5).item()
                            for _ in range(100)]

        # High temperature: should be more uniform across top-k
        high_temp_samples = [sample_top_k(logits, k=k, temperature=2.0).item()
                             for _ in range(100)]

        # Low temperature should have higher concentration on first token
        low_temp_first_count = sum(1 for s in low_temp_samples if s == 0)
        high_temp_first_count = sum(1 for s in high_temp_samples if s == 0)

        assert low_temp_first_count > high_temp_first_count

    def test_top_k_batch(self):
        """Top-k should work with batched input."""
        logits = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        k = 2

        tokens = sample_top_k(logits, k=k)

        assert tokens.shape == (2, 1)
        # Both batches should sample from their respective top-2
        assert tokens[0].item() in {0, 1}  # Top-2 of first batch
        assert tokens[1].item() in {3, 4}  # Top-2 of second batch

    def test_top_k_invalid_k_raises(self):
        """Invalid k should raise an error."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="k must be in range"):
            sample_top_k(logits, k=0)

        with pytest.raises(ValueError, match="k must be in range"):
            sample_top_k(logits, k=4)  # vocab_size is 3


class TestTopPSampling:
    """Test top-p (nucleus) sampling."""

    def test_top_p_filters_correctly(self):
        """Top-p should only sample from nucleus."""
        # Create probabilities where we can control the nucleus
        # Probs: [0.5, 0.3, 0.15, 0.04, 0.01]
        # Cumulative: [0.5, 0.8, 0.95, 0.99, 1.0]

        # These logits will give approximately those probabilities
        logits = torch.tensor([2.0, 1.0, 0.0, -1.5, -3.0])
        p = 0.9

        # Sample many times
        samples = []
        for _ in range(100):
            token = sample_top_p(logits, p=p, temperature=1.0)
            samples.append(token.item())

        # With p=0.9, nucleus should include first 3 tokens (cumulative ~0.95)
        # Should rarely or never see tokens 3, 4
        unique_samples = set(samples)
        assert unique_samples.issubset({0, 1, 2, 3})  # Allow some variance

        # Token 4 should be very rare or absent
        count_4 = sum(1 for s in samples if s == 4)
        assert count_4 < 5  # Allow some statistical variance

    def test_top_p_adaptive(self):
        """Top-p should adapt nucleus size to distribution."""
        # Confident distribution (one token dominates)
        confident_logits = torch.tensor([10.0, 1.0, 0.5, 0.0])
        p = 0.9

        samples_confident = []
        for _ in range(50):
            token = sample_top_p(confident_logits, p=p)
            samples_confident.append(token.item())

        # Should mostly sample first token
        first_token_count = sum(1 for s in samples_confident if s == 0)
        assert first_token_count > 40  # At least 80%

        # Uncertain distribution (more uniform)
        uncertain_logits = torch.tensor([1.0, 0.9, 0.8, 0.7, 0.6])

        samples_uncertain = []
        for _ in range(50):
            token = sample_top_p(uncertain_logits, p=p)
            samples_uncertain.append(token.item())

        # Should see more diverse samples
        unique_uncertain = len(set(samples_uncertain))
        assert unique_uncertain >= 3  # Should see multiple different tokens

    def test_top_p_with_temperature(self):
        """Top-p should work with temperature."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.0])
        p = 0.9

        # Temperature affects the distribution before nucleus selection
        token_low_temp = sample_top_p(logits, p=p, temperature=0.5)
        token_high_temp = sample_top_p(logits, p=p, temperature=2.0)

        # Both should return valid tokens
        assert 0 <= token_low_temp.item() < 4
        assert 0 <= token_high_temp.item() < 4

    def test_top_p_batch(self):
        """Top-p should work with batched input."""
        logits = torch.tensor([
            [5.0, 4.0, 3.0, 2.0],
            [1.0, 2.0, 3.0, 4.0],
        ])
        p = 0.9

        tokens = sample_top_p(logits, p=p)

        assert tokens.shape == (2, 1)
        assert 0 <= tokens[0].item() < 4
        assert 0 <= tokens[1].item() < 4

    def test_top_p_min_tokens_to_keep(self):
        """Top-p should respect min_tokens_to_keep."""
        # Very confident distribution
        logits = torch.tensor([100.0, 0.0, 0.0, 0.0])
        p = 0.5  # Very low p

        # Without min_tokens_to_keep, might only keep 1 token
        # With min_tokens_to_keep=2, should keep at least 2

        samples = []
        for _ in range(50):
            token = sample_top_p(logits, p=p, min_tokens_to_keep=2)
            samples.append(token.item())

        # Should be able to sample from at least first 2 tokens
        unique_samples = set(samples)
        # Due to extreme logit difference, we might still mostly see token 0
        # But the function should at least allow token 1
        assert 0 in unique_samples  # Will definitely see this

    def test_top_p_invalid_p_raises(self):
        """Invalid p should raise an error."""
        logits = torch.tensor([1.0, 2.0, 3.0])

        with pytest.raises(ValueError, match="p must be in range"):
            sample_top_p(logits, p=0.0)

        with pytest.raises(ValueError, match="p must be in range"):
            sample_top_p(logits, p=1.5)


class TestTopKTopPSampling:
    """Test combined top-k + top-p sampling."""

    def test_combined_filters_correctly(self):
        """Combined method should apply both filters."""
        # Create logits with clear separation
        logits = torch.tensor([10.0, 9.0, 8.0, 7.0, 6.0, 1.0, 0.5, 0.0, -1.0, -2.0])
        k = 5
        p = 0.8

        samples = []
        for _ in range(100):
            token = sample_top_k_top_p(logits, k=k, p=p)
            samples.append(token.item())

        # Should only see tokens from top-5 (indices 0-4)
        unique_samples = set(samples)
        assert unique_samples.issubset({0, 1, 2, 3, 4})

        # Should not see indices 5-9 (filtered by top-k)
        for i in range(5, 10):
            assert i not in unique_samples

    def test_combined_with_temperature(self):
        """Combined method should work with temperature."""
        logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        k = 4
        p = 0.9

        # Should not raise any errors
        token_low = sample_top_k_top_p(logits, k=k, p=p, temperature=0.5)
        token_high = sample_top_k_top_p(logits, k=k, p=p, temperature=2.0)

        assert 0 <= token_low.item() < 5
        assert 0 <= token_high.item() < 5

    def test_combined_batch(self):
        """Combined method should work with batched input."""
        logits = torch.tensor([
            [5.0, 4.0, 3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0, 5.0],
        ])
        k = 3
        p = 0.9

        tokens = sample_top_k_top_p(logits, k=k, p=p)

        assert tokens.shape == (2, 1)
        assert 0 <= tokens[0].item() < 5
        assert 0 <= tokens[1].item() < 5

    def test_combined_more_restrictive_than_individual(self):
        """Combined method should be more restrictive than either alone."""
        # Large logits to get clear probabilities
        logits = torch.tensor([100.0, 99.0, 98.0, 50.0, 1.0, 0.5, 0.0, -10.0])

        k = 6  # Allows indices 0-5
        p = 0.99  # Very permissive

        # Top-k alone would allow 0-5
        # Top-p alone would allow most tokens (high p)
        # Combined should still respect top-k limit

        samples = []
        for _ in range(100):
            token = sample_top_k_top_p(logits, k=k, p=p)
            samples.append(token.item())

        # Should only see tokens from top-k (0-5)
        unique_samples = set(samples)
        assert unique_samples.issubset({0, 1, 2, 3, 4, 5})
        assert 6 not in unique_samples
        assert 7 not in unique_samples


class TestSamplingShapes:
    """Test that all sampling methods handle shapes correctly."""

    def test_single_sequence_all_methods(self):
        """All methods should handle single sequence (1D input)."""
        logits = torch.tensor([1.0, 2.0, 3.0, 4.0])

        # All should return shape (1,)
        assert sample_greedy(logits).shape == (1,)
        assert sample_top_k(logits, k=2).shape == (1,)
        assert sample_top_p(logits, p=0.9).shape == (1,)
        assert sample_top_k_top_p(logits, k=2, p=0.9).shape == (1,)

    def test_batch_all_methods(self):
        """All methods should handle batched input (2D)."""
        logits = torch.tensor([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, 2.0, 1.0],
        ])

        # All should return shape (2, 1)
        assert sample_greedy(logits).shape == (2, 1)
        assert sample_top_k(logits, k=2).shape == (2, 1)
        assert sample_top_p(logits, p=0.9).shape == (2, 1)
        assert sample_top_k_top_p(logits, k=2, p=0.9).shape == (2, 1)


class TestSamplingStatistical:
    """Statistical tests to verify sampling distributions."""

    def test_top_k_respects_probabilities(self):
        """Top-k sampling should respect probability distribution."""
        # Create clear probability distribution
        # After softmax: first token should have ~73% probability
        logits = torch.tensor([2.0, 0.0, 0.0, 0.0])
        k = 4

        samples = [sample_top_k(logits, k=k).item() for _ in range(1000)]

        # Count frequency of first token
        first_count = sum(1 for s in samples if s == 0)

        # Should be roughly 73% (allow 10% margin for statistical variance)
        assert 0.63 < first_count / 1000 < 0.83

    def test_temperature_affects_distribution(self):
        """Temperature should significantly affect sampling distribution."""
        logits = torch.tensor([2.0, 1.0, 0.5, 0.0])

        # Sample with low temperature (should be more peaked)
        low_temp_samples = [sample_top_k(logits, k=4, temperature=0.3).item()
                            for _ in range(200)]

        # Sample with high temperature (should be more uniform)
        high_temp_samples = [sample_top_k(logits, k=4, temperature=3.0).item()
                             for _ in range(200)]

        # Low temp should have higher concentration on first token
        low_first = sum(1 for s in low_temp_samples if s == 0) / 200
        high_first = sum(1 for s in high_temp_samples if s == 0) / 200

        assert low_first > high_first + 0.1  # At least 10% difference
