"""
Tests for perplexity calculation.

These tests verify that perplexity is calculated correctly and has the
expected mathematical properties.
"""

import torch
import torch.nn as nn
import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.perplexity import (
    calculate_perplexity,
    calculate_perplexity_from_loss,
)


def test_perplexity_from_loss():
    """Test that perplexity = exp(loss) relationship holds."""
    # Test with several loss values
    test_cases = [
        (0.0, 1.0),      # Perfect model: loss=0 → perplexity=1
        (1.0, 2.718),    # loss=1 → perplexity≈e
        (2.0, 7.389),    # loss=2 → perplexity≈e^2
        (3.0, 20.085),   # loss=3 → perplexity≈e^3
    ]

    for loss, expected_ppl in test_cases:
        loss_tensor = torch.tensor(loss)
        perplexity = calculate_perplexity_from_loss(loss_tensor)

        # Check if close to expected value (within 0.01)
        assert abs(perplexity.item() - expected_ppl) < 0.01, \
            f"Expected perplexity {expected_ppl} for loss {loss}, got {perplexity.item()}"


def test_perplexity_perfect_predictions():
    """Test perplexity when model makes perfect predictions."""
    batch_size, seq_len, vocab_size = 2, 5, 100

    # Create targets
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create logits where correct token always has very high score
    logits = torch.zeros(batch_size, seq_len, vocab_size)

    for b in range(batch_size):
        for s in range(seq_len):
            # Give correct token extremely high score (→ probability ≈ 1.0)
            logits[b, s, targets[b, s]] = 100.0
            # All other tokens get 0 (→ probability ≈ 0.0)

    perplexity = calculate_perplexity(logits, targets)

    # Perfect predictions should give perplexity very close to 1.0
    assert perplexity.item() < 1.01, \
        f"Perfect predictions should give perplexity≈1.0, got {perplexity.item()}"


def test_perplexity_random_predictions():
    """Test perplexity when model makes random predictions (uniform distribution)."""
    batch_size, seq_len, vocab_size = 2, 10, 100

    # Create random uniform logits (all tokens equally likely)
    logits = torch.zeros(batch_size, seq_len, vocab_size)

    # Random targets
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    perplexity = calculate_perplexity(logits, targets)

    # Random guessing over vocab_size options should give perplexity ≈ vocab_size
    # Allow some tolerance since we're using random targets
    assert 80 < perplexity.item() < 120, \
        f"Random predictions over {vocab_size} tokens should give perplexity≈{vocab_size}, " \
        f"got {perplexity.item()}"


def test_perplexity_decreases_with_confidence():
    """Test that higher confidence in correct token → lower perplexity."""
    batch_size, seq_len, vocab_size = 2, 5, 100
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Case 1: Low confidence (score = 1.0 for correct token)
    logits_low = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            logits_low[b, s, targets[b, s]] = 1.0

    perplexity_low = calculate_perplexity(logits_low, targets)

    # Case 2: Medium confidence (score = 5.0 for correct token)
    logits_medium = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            logits_medium[b, s, targets[b, s]] = 5.0

    perplexity_medium = calculate_perplexity(logits_medium, targets)

    # Case 3: High confidence (score = 10.0 for correct token)
    logits_high = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            logits_high[b, s, targets[b, s]] = 10.0

    perplexity_high = calculate_perplexity(logits_high, targets)

    # More confidence should mean lower perplexity
    assert perplexity_high < perplexity_medium < perplexity_low, \
        f"Higher confidence should give lower perplexity, but got: " \
        f"low={perplexity_low.item()}, medium={perplexity_medium.item()}, " \
        f"high={perplexity_high.item()}"


def test_perplexity_ignore_index():
    """Test that ignore_index properly excludes tokens from perplexity calculation."""
    batch_size, seq_len, vocab_size = 2, 5, 100
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Create logits that give good predictions
    logits = torch.zeros(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            logits[b, s, targets[b, s]] = 10.0

    # Calculate perplexity without ignoring anything
    perplexity_all = calculate_perplexity(logits, targets)

    # Now set some targets to ignore_index
    targets_with_ignore = targets.clone()
    targets_with_ignore[0, 0] = -100  # Ignore first token of first sequence

    # Calculate perplexity with ignore_index
    perplexity_ignored = calculate_perplexity(logits, targets_with_ignore, ignore_index=-100)

    # Perplexities should be similar (only 1 out of 10 tokens ignored)
    # but not exactly equal
    assert 0.5 < (perplexity_ignored / perplexity_all).item() < 2.0, \
        f"Ignoring one token shouldn't drastically change perplexity, " \
        f"got {perplexity_all.item()} vs {perplexity_ignored.item()}"


def test_perplexity_consistency_with_cross_entropy():
    """Test that calculate_perplexity gives same result as exp(CrossEntropyLoss)."""
    batch_size, seq_len, vocab_size = 4, 10, 50

    # Random logits and targets
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Method 1: Using our calculate_perplexity function
    perplexity = calculate_perplexity(logits, targets)

    # Method 2: Manual calculation with CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(
        logits.view(batch_size * seq_len, vocab_size),
        targets.view(batch_size * seq_len)
    )
    expected_perplexity = torch.exp(loss)

    # Should be very close (within numerical precision)
    assert abs(perplexity.item() - expected_perplexity.item()) < 0.01, \
        f"calculate_perplexity should match exp(CrossEntropyLoss), " \
        f"got {perplexity.item()} vs {expected_perplexity.item()}"


def test_perplexity_batch_consistency():
    """Test that perplexity is consistent across batch sizes."""
    seq_len, vocab_size = 10, 50

    # Create identical sequences
    targets = torch.randint(0, vocab_size, (1, seq_len))
    logits = torch.randn(1, seq_len, vocab_size)

    # Perplexity for single sequence
    perplexity_single = calculate_perplexity(logits, targets)

    # Duplicate to create batch of identical sequences
    targets_batch = targets.repeat(4, 1)  # 4 identical sequences
    logits_batch = logits.repeat(4, 1, 1)

    # Perplexity for batch
    perplexity_batch = calculate_perplexity(logits_batch, targets_batch)

    # Should be the same (within numerical precision)
    assert abs(perplexity_single.item() - perplexity_batch.item()) < 0.01, \
        f"Perplexity should be consistent across batch sizes, " \
        f"got {perplexity_single.item()} for single vs {perplexity_batch.item()} for batch"


if __name__ == "__main__":
    # Run all tests
    print("Running perplexity tests...")
    print()

    test_perplexity_from_loss()
    print("✓ test_perplexity_from_loss passed")

    test_perplexity_perfect_predictions()
    print("✓ test_perplexity_perfect_predictions passed")

    test_perplexity_random_predictions()
    print("✓ test_perplexity_random_predictions passed")

    test_perplexity_decreases_with_confidence()
    print("✓ test_perplexity_decreases_with_confidence passed")

    test_perplexity_ignore_index()
    print("✓ test_perplexity_ignore_index passed")

    test_perplexity_consistency_with_cross_entropy()
    print("✓ test_perplexity_consistency_with_cross_entropy passed")

    test_perplexity_batch_consistency()
    print("✓ test_perplexity_batch_consistency passed")

    print()
    print("All perplexity tests passed! ✓")
