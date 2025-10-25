"""Tests for feed-forward network."""

import pytest
import torch
from src.transformer.feedforward import FeedForward


class TestFeedForward:
    """Tests for feed-forward network (MLP)."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 512
        d_ff = 2048
        ff = FeedForward(d_model, d_ff)

        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = ff(x)

        # Output should have same shape as input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_expansion_in_hidden_layer(self):
        """Test that hidden layer expands to d_ff dimensions."""
        d_model = 256
        d_ff = 1024
        ff = FeedForward(d_model, d_ff)

        # Check that first linear layer has correct dimensions
        assert ff.linear1.in_features == d_model
        assert ff.linear1.out_features == d_ff

        # Check that second linear layer projects back
        assert ff.linear2.in_features == d_ff
        assert ff.linear2.out_features == d_model

    def test_different_expansion_ratios(self):
        """Test with various d_ff to d_model ratios."""
        d_model = 128

        for ratio in [2, 4, 8]:
            d_ff = d_model * ratio
            ff = FeedForward(d_model, d_ff)

            batch_size = 2
            seq_len = 5
            x = torch.randn(batch_size, seq_len, d_model)

            output = ff(x)

            assert output.shape == (batch_size, seq_len, d_model)

    def test_different_batch_and_sequence_sizes(self):
        """Test with various batch and sequence sizes."""
        d_model = 256
        d_ff = 1024
        ff = FeedForward(d_model, d_ff)

        for batch_size in [1, 4, 8]:
            for seq_len in [1, 10, 50]:
                x = torch.randn(batch_size, seq_len, d_model)
                output = ff(x)
                assert output.shape == (batch_size, seq_len, d_model)

    def test_position_wise_independence(self):
        """Test that each position is processed independently."""
        d_model = 64
        d_ff = 256
        ff = FeedForward(d_model, d_ff)

        # Create input with one position
        x_single = torch.randn(1, 1, d_model)
        output_single = ff(x_single)

        # Create input with same value repeated at multiple positions
        x_repeated = x_single.repeat(1, 5, 1)

        # Process all positions together
        ff.eval()  # Disable dropout for deterministic behavior
        with torch.no_grad():
            output_repeated = ff(x_repeated)

        # Due to dropout randomness, use eval mode to ensure
        # same input produces same output
        # All positions should have same output (same input, same network)
        for i in range(5):
            assert torch.allclose(output_repeated[0, i], output_repeated[0, 0], atol=1e-5)

    def test_dropout_training_vs_eval(self):
        """Test that dropout behaves differently in training vs eval mode."""
        d_model = 128
        d_ff = 512
        ff = FeedForward(d_model, d_ff, dropout=0.5)

        x = torch.randn(2, 10, d_model)

        # Training mode: dropout is active
        ff.train()
        output1 = ff(x)
        output2 = ff(x)

        # Outputs should be different due to dropout randomness
        # (very unlikely to be identical with dropout=0.5)
        assert not torch.allclose(output1, output2)

        # Eval mode: dropout is disabled
        ff.eval()
        with torch.no_grad():
            output3 = ff(x)
            output4 = ff(x)

        # Outputs should be identical in eval mode
        assert torch.allclose(output3, output4)

    def test_gradients_flow(self):
        """Test that gradients flow through the network (learnable)."""
        d_model = 128
        d_ff = 512
        ff = FeedForward(d_model, d_ff)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output = ff(x)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that all layers have gradients
        assert ff.linear1.weight.grad is not None
        assert ff.linear1.bias.grad is not None
        assert ff.linear2.weight.grad is not None
        assert ff.linear2.bias.grad is not None

        # Check that input has gradients
        assert x.grad is not None

    def test_no_nans_or_infs(self):
        """Test numerical stability."""
        d_model = 512
        d_ff = 2048
        ff = FeedForward(d_model, d_ff)

        x = torch.randn(2, 10, d_model)
        output = ff(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_zero_dropout(self):
        """Test with zero dropout (deterministic behavior)."""
        d_model = 128
        d_ff = 512
        ff = FeedForward(d_model, d_ff, dropout=0.0)

        x = torch.randn(2, 5, d_model)

        # Even in training mode, should be deterministic with dropout=0
        ff.train()
        output1 = ff(x)
        output2 = ff(x)

        assert torch.allclose(output1, output2)

    def test_gelu_activation_is_used(self):
        """Test that GELU activation is being used."""
        d_model = 64
        d_ff = 256
        ff = FeedForward(d_model, d_ff)

        # Check that activation is GELU
        assert isinstance(ff.activation, torch.nn.GELU)
