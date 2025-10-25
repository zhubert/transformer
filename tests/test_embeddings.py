"""Tests for embedding layers."""

import pytest
import torch
from src.transformer.embeddings import TokenEmbedding, PositionalEncoding


class TestTokenEmbedding:
    """Tests for token embedding."""

    def test_output_shape(self):
        """Test that output shape is correct."""
        vocab_size = 1000
        d_model = 512
        embedding = TokenEmbedding(vocab_size, d_model)

        batch_size = 4
        seq_len = 10
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len))

        output = embedding(tokens)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        vocab_size = 500
        d_model = 256
        embedding = TokenEmbedding(vocab_size, d_model)

        for batch_size in [1, 2, 8, 16]:
            seq_len = 5
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            output = embedding(tokens)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        vocab_size = 500
        d_model = 256
        embedding = TokenEmbedding(vocab_size, d_model)

        batch_size = 2
        for seq_len in [1, 5, 10, 50]:
            tokens = torch.randint(0, vocab_size, (batch_size, seq_len))
            output = embedding(tokens)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_same_token_same_embedding(self):
        """Test that same token always gets same embedding."""
        vocab_size = 100
        d_model = 64
        embedding = TokenEmbedding(vocab_size, d_model)

        # Create input with repeated token
        token_id = 42
        tokens = torch.tensor([[token_id, 10, token_id, 20]])

        output = embedding(tokens)

        # Position 0 and position 2 should have same embedding (same token)
        assert torch.allclose(output[0, 0], output[0, 2])

        # Position 0 and position 1 should have different embeddings (different tokens)
        assert not torch.allclose(output[0, 0], output[0, 1])

    def test_no_nans_or_infs(self):
        """Test numerical stability."""
        vocab_size = 1000
        d_model = 512
        embedding = TokenEmbedding(vocab_size, d_model)

        tokens = torch.randint(0, vocab_size, (2, 10))
        output = embedding(tokens)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()


class TestPositionalEncoding:
    """Tests for positional encoding."""

    def test_output_shape(self):
        """Test that output shape matches input shape."""
        d_model = 512
        max_seq_len = 1000
        pos_enc = PositionalEncoding(d_model, max_seq_len)

        batch_size = 4
        seq_len = 10
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        assert output.shape == (batch_size, seq_len, d_model)

    def test_positions_are_added_not_concatenated(self):
        """Test that positional encodings are added to input, not concatenated."""
        d_model = 64
        pos_enc = PositionalEncoding(d_model, max_seq_len=100)

        batch_size = 2
        seq_len = 5
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        # Output dimension should stay d_model (not 2*d_model if concatenated)
        assert output.shape[-1] == d_model

    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        d_model = 256
        max_seq_len = 1000
        pos_enc = PositionalEncoding(d_model, max_seq_len)

        batch_size = 2
        for seq_len in [1, 5, 10, 50, 100]:
            x = torch.randn(batch_size, seq_len, d_model)
            output = pos_enc(x)
            assert output.shape == (batch_size, seq_len, d_model)

    def test_same_position_same_embedding_across_batches(self):
        """Test that same position gets same embedding across different batches."""
        d_model = 128
        pos_enc = PositionalEncoding(d_model, max_seq_len=100)

        batch_size = 3
        seq_len = 5
        x = torch.randn(batch_size, seq_len, d_model)

        output = pos_enc(x)

        # The positional contribution should be the same for all batches
        # output[i] = x[i] + pos_emb, so differences in output across batches
        # should equal differences in x
        diff_x_01 = x[0] - x[1]
        diff_output_01 = output[0] - output[1]
        assert torch.allclose(diff_x_01, diff_output_01, atol=1e-6)

    def test_different_positions_different_embeddings(self):
        """Test that different positions get different embeddings."""
        d_model = 128
        pos_enc = PositionalEncoding(d_model, max_seq_len=100)

        # Use zero input to isolate positional embeddings
        batch_size = 1
        seq_len = 10
        x = torch.zeros(batch_size, seq_len, d_model)

        output = pos_enc(x)

        # Since input is zeros, output = positional embeddings
        # Different positions should have different embeddings
        for i in range(seq_len - 1):
            assert not torch.allclose(output[0, i], output[0, i + 1])

    def test_positional_embeddings_are_learned(self):
        """Test that positional embeddings can be learned (have gradients)."""
        d_model = 64
        pos_enc = PositionalEncoding(d_model, max_seq_len=100)

        x = torch.randn(2, 5, d_model)
        output = pos_enc(x)

        # Compute a dummy loss and backpropagate
        loss = output.sum()
        loss.backward()

        # Position embedding layer should have gradients
        assert pos_enc.pos_embedding.weight.grad is not None

    def test_no_nans_or_infs(self):
        """Test numerical stability."""
        d_model = 512
        pos_enc = PositionalEncoding(d_model, max_seq_len=1000)

        x = torch.randn(2, 10, d_model)
        output = pos_enc(x)

        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_exceeding_max_seq_len_raises_error(self):
        """Test that exceeding max_seq_len raises an error."""
        d_model = 64
        max_seq_len = 10
        pos_enc = PositionalEncoding(d_model, max_seq_len)

        # Try to process sequence longer than max_seq_len
        batch_size = 1
        seq_len = 15  # > max_seq_len
        x = torch.randn(batch_size, seq_len, d_model)

        with pytest.raises(IndexError):
            output = pos_enc(x)
