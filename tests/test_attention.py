"""Tests for attention mechanisms."""

import pytest
import torch
from src.transformer.attention import ScaledDotProductAttention, MultiHeadAttention


class TestScaledDotProductAttention:
    """Tests for scaled dot-product attention."""

    def test_output_shape(self):
        """Test that output shapes are correct."""
        attention = ScaledDotProductAttention()

        batch_size = 2
        seq_len = 10
        d_k = 64
        d_v = 64

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_v)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_output_shape_different_dimensions(self):
        """Test output shapes with different d_k and d_v."""
        attention = ScaledDotProductAttention()

        batch_size = 3
        seq_len = 5
        d_k = 32
        d_v = 48  # Different from d_k

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_v)
        assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_attention_weights_sum_to_one(self):
        """Test that attention weights sum to 1.0 across sequence dimension."""
        attention = ScaledDotProductAttention()

        batch_size = 2
        seq_len = 8
        d_k = 64
        d_v = 64

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        # Sum across the last dimension (attending to all positions)
        sums = attention_weights.sum(dim=-1)

        # Should sum to 1.0 for each position in each batch
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)

    def test_causal_masking(self):
        """Test that causal masking prevents attending to future positions."""
        attention = ScaledDotProductAttention()

        batch_size = 1
        seq_len = 4
        d_k = 8
        d_v = 8

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        # Create causal mask (upper triangular, excluding diagonal)
        # Position i should not attend to positions j > i
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        mask = mask.unsqueeze(0)  # Add batch dimension

        output, attention_weights = attention(query, key, value, mask=mask)

        # Check that future positions have zero attention weight
        # For position 0, positions 1,2,3 should be 0
        # For position 1, positions 2,3 should be 0, etc.
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert attention_weights[0, i, j].item() == 0.0, \
                    f"Position {i} should not attend to future position {j}"

    def test_simple_known_case(self):
        """Test on a simple case where we can verify the result."""
        attention = ScaledDotProductAttention()

        # Simple case: 1 batch, 2 positions, dimension 2
        # Make query at position 0 very similar to key at position 0
        query = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])  # (1, 2, 2)
        key = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])    # (1, 2, 2)
        value = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])  # (1, 2, 2)

        output, attention_weights = attention(query, key, value)

        # Position 0's query [1,0] is orthogonal to position 1's key [0,1]
        # but identical to position 0's key [1,0]
        # So position 0 should attend mostly to itself
        assert attention_weights[0, 0, 0] > attention_weights[0, 0, 1]

        # Similarly for position 1
        assert attention_weights[0, 1, 1] > attention_weights[0, 1, 0]

    def test_single_token_sequence(self):
        """Test with a single token (edge case)."""
        attention = ScaledDotProductAttention()

        batch_size = 2
        seq_len = 1  # Single token
        d_k = 16
        d_v = 16

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        assert output.shape == (batch_size, seq_len, d_v)
        # With single token, attention weight should be 1.0 to itself
        assert torch.allclose(attention_weights, torch.ones_like(attention_weights))

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        attention = ScaledDotProductAttention()

        for batch_size in [1, 4, 8]:
            seq_len = 6
            d_k = 32
            d_v = 32

            query = torch.randn(batch_size, seq_len, d_k)
            key = torch.randn(batch_size, seq_len, d_k)
            value = torch.randn(batch_size, seq_len, d_v)

            output, attention_weights = attention(query, key, value)

            assert output.shape == (batch_size, seq_len, d_v)
            assert attention_weights.shape == (batch_size, seq_len, seq_len)

    def test_no_nans_or_infs(self):
        """Test numerical stability - no NaNs or Infs in normal operation."""
        attention = ScaledDotProductAttention()

        batch_size = 2
        seq_len = 10
        d_k = 64
        d_v = 64

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()
        assert not torch.isnan(attention_weights).any()
        assert not torch.isinf(attention_weights).any()

    def test_attention_weights_non_negative(self):
        """Test that attention weights are non-negative (result of softmax)."""
        attention = ScaledDotProductAttention()

        batch_size = 2
        seq_len = 7
        d_k = 32
        d_v = 32

        query = torch.randn(batch_size, seq_len, d_k)
        key = torch.randn(batch_size, seq_len, d_k)
        value = torch.randn(batch_size, seq_len, d_v)

        output, attention_weights = attention(query, key, value)

        # All attention weights should be >= 0 (softmax output)
        assert (attention_weights >= 0).all()


class TestMultiHeadAttention:
    """Tests for multi-head attention."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)

        output, _ = mha(x)

        # Output should have same shape as input
        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_num_heads(self):
        """Test with various numbers of heads."""
        d_model = 512

        for num_heads in [1, 2, 4, 8, 16]:
            mha = MultiHeadAttention(d_model, num_heads)

            batch_size = 2
            seq_len = 5
            x = torch.randn(batch_size, seq_len, d_model)

            output, _ = mha(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert mha.d_k == d_model // num_heads

    def test_different_d_model_sizes(self):
        """Test with various d_model sizes."""
        num_heads = 8

        for d_model in [64, 128, 256, 512, 1024]:
            if d_model % num_heads == 0:  # Must be divisible
                mha = MultiHeadAttention(d_model, num_heads)

                batch_size = 2
                seq_len = 5
                x = torch.randn(batch_size, seq_len, d_model)

                output, _ = mha(x)

                assert output.shape == (batch_size, seq_len, d_model)

    def test_d_k_calculation(self):
        """Test that d_k is correctly calculated as d_model / num_heads."""
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        assert mha.d_k == 64  # 512 / 8
        assert mha.d_k * num_heads == d_model

    def test_causal_masking(self):
        """Test that causal masking works through multi-head attention."""
        d_model = 128
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)

        batch_size = 1
        seq_len = 4
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

        # This should not raise an error
        output, _ = mha(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_batch_and_sequence_sizes(self):
        """Test with various batch and sequence sizes."""
        d_model = 256
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        for batch_size in [1, 4, 8]:
            for seq_len in [1, 10, 50]:
                x = torch.randn(batch_size, seq_len, d_model)
                output, _ = mha(x)
                assert output.shape == (batch_size, seq_len, d_model)

    def test_gradients_flow_through_all_projections(self):
        """Test that gradients flow through all linear projections."""
        d_model = 128
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(2, 5, d_model, requires_grad=True)
        output, _ = mha(x)

        # Compute dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Check that all projections have gradients
        assert mha.W_q.weight.grad is not None
        assert mha.W_k.weight.grad is not None
        assert mha.W_v.weight.grad is not None
        assert mha.W_o.weight.grad is not None

        # Check that input has gradients
        assert x.grad is not None

    def test_no_nans_or_infs(self):
        """Test numerical stability."""
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(2, 10, d_model)
        output, _ = mha(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_invalid_d_model_raises_error(self):
        """Test that d_model not divisible by num_heads raises error."""
        d_model = 100  # Not divisible by 8
        num_heads = 8

        with pytest.raises(AssertionError):
            mha = MultiHeadAttention(d_model, num_heads)

    def test_projections_have_correct_dimensions(self):
        """Test that linear projections have correct input/output dimensions."""
        d_model = 512
        num_heads = 8
        mha = MultiHeadAttention(d_model, num_heads)

        # All projections should be d_model → d_model
        assert mha.W_q.in_features == d_model
        assert mha.W_q.out_features == d_model
        assert mha.W_k.in_features == d_model
        assert mha.W_k.out_features == d_model
        assert mha.W_v.in_features == d_model
        assert mha.W_v.out_features == d_model
        assert mha.W_o.in_features == d_model
        assert mha.W_o.out_features == d_model

    def test_single_head_is_valid(self):
        """Test that single-head attention works (edge case)."""
        d_model = 128
        num_heads = 1
        mha = MultiHeadAttention(d_model, num_heads)

        x = torch.randn(2, 5, d_model)
        output, _ = mha(x)

        assert output.shape == (2, 5, d_model)
        assert mha.d_k == d_model  # Single head uses full dimension

    def test_mask_with_batch_dimension(self):
        """Test that masks with batch dimension work correctly."""
        d_model = 128
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)

        batch_size = 2
        seq_len = 4
        x = torch.randn(batch_size, seq_len, d_model)

        # Create causal mask with batch dimension
        mask = torch.triu(torch.ones(batch_size, seq_len, seq_len), diagonal=1).bool()

        output, _ = mha(x, mask=mask)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_uses_scaled_dot_product_attention(self):
        """Test that MultiHeadAttention uses ScaledDotProductAttention."""
        d_model = 128
        num_heads = 4
        mha = MultiHeadAttention(d_model, num_heads)

        # Check that it has a ScaledDotProductAttention instance
        assert isinstance(mha.attention, ScaledDotProductAttention)
