"""Tests for transformer model."""

import pytest
import torch
from src.transformer.model import DecoderOnlyTransformer


class TestDecoderOnlyTransformer:
    """Tests for decoder-only transformer - the complete model!"""

    def test_output_shape(self):
        """Test that output shape is (batch, seq_len, vocab_size)."""
        vocab_size = 1000
        model = DecoderOnlyTransformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, _ = model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_various_model_configurations(self):
        """Test with various model sizes."""
        configs = [
            # (vocab_size, d_model, num_heads, num_layers, d_ff)
            (500, 128, 4, 1, 512),       # Tiny model
            (1000, 256, 8, 6, 1024),     # Small model
            (2000, 512, 8, 6, 2048),     # Medium model (our default)
            (5000, 768, 12, 12, 3072),   # GPT-2 Small size
        ]

        for vocab_size, d_model, num_heads, num_layers, d_ff in configs:
            model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)

            batch_size = 2
            seq_len = 5
            x = torch.randint(0, vocab_size, (batch_size, seq_len))

            logits, _ = model(x)

            assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_different_sequence_lengths(self):
        """Test with various sequence lengths."""
        vocab_size = 1000
        model = DecoderOnlyTransformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

        batch_size = 2
        for seq_len in [1, 5, 10, 50]:
            x = torch.randint(0, vocab_size, (batch_size, seq_len))
            logits, _ = model(x)
            assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_causal_mask_is_created(self):
        """Test that causal mask is correctly generated."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        seq_len = 4
        mask = model.create_causal_mask(seq_len)

        # Check shape
        assert mask.shape == (seq_len, seq_len)

        # Check that it's upper triangular (excluding diagonal)
        expected = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        assert torch.equal(mask, expected)

        # Verify specific positions
        assert mask[0, 0] == False  # Can see itself
        assert mask[0, 1] == True   # Cannot see future
        assert mask[1, 0] == False  # Can see past
        assert mask[2, 3] == True   # Cannot see future

    def test_forward_with_explicit_mask(self):
        """Test forward pass with explicitly provided mask."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        batch_size = 2
        seq_len = 4
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # Create custom mask
        mask = model.create_causal_mask(seq_len)

        logits, _ = model(x, mask=mask)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_forward_without_mask(self):
        """Test that forward pass creates causal mask automatically."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        batch_size = 2
        seq_len = 4
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        # No mask provided - should create one automatically
        logits, _ = model(x)

        assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_stacking_layers(self):
        """Test that different numbers of layers work."""
        vocab_size = 500
        d_model = 128
        num_heads = 4
        d_ff = 512

        for num_layers in [1, 2, 6, 12]:
            model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)

            # Check that correct number of blocks were created
            assert len(model.blocks) == num_layers

            # Test forward pass
            batch_size = 2
            seq_len = 5
            x = torch.randint(0, vocab_size, (batch_size, seq_len))

            logits, _ = model(x)

            assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_all_components_integrated(self):
        """Test that all components are correctly integrated."""
        from src.transformer.embeddings import TokenEmbedding, PositionalEncoding
        from src.transformer.block import TransformerBlock

        vocab_size = 500
        model = DecoderOnlyTransformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

        # Check that all components exist
        assert isinstance(model.token_embedding, TokenEmbedding)
        assert isinstance(model.pos_encoding, PositionalEncoding)
        assert isinstance(model.blocks[0], TransformerBlock)
        assert isinstance(model.ln_f, torch.nn.LayerNorm)
        assert isinstance(model.output_proj, torch.nn.Linear)

        # Check dimensions
        assert model.output_proj.in_features == 128  # d_model
        assert model.output_proj.out_features == 500  # vocab_size

    def test_gradients_flow_through_entire_model(self):
        """Test that gradients flow from output back to embeddings."""
        vocab_size = 500
        model = DecoderOnlyTransformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

        batch_size = 2
        seq_len = 5
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, _ = model(x)

        # Compute dummy loss and backpropagate
        loss = logits.sum()
        loss.backward()

        # Check that gradients exist for all major components
        assert model.token_embedding.embedding.weight.grad is not None
        assert model.pos_encoding.pos_embedding.weight.grad is not None
        assert model.blocks[0].attention.W_q.weight.grad is not None
        assert model.blocks[0].ffn.linear1.weight.grad is not None
        assert model.ln_f.weight.grad is not None
        assert model.output_proj.weight.grad is not None

    def test_no_nans_or_infs(self):
        """Test numerical stability."""
        vocab_size = 1000
        model = DecoderOnlyTransformer(vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024)

        batch_size = 2
        seq_len = 10
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, _ = model(x)

        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_output_is_logits_not_probabilities(self):
        """Test that output is logits (unnormalized), not probabilities."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        batch_size = 1
        seq_len = 3
        x = torch.randint(0, vocab_size, (batch_size, seq_len))

        logits, _ = model(x)

        # Logits can be any value (not restricted to [0, 1])
        # and don't sum to 1 along vocab dimension
        assert not torch.allclose(logits.sum(dim=-1), torch.ones(batch_size, seq_len))

        # Convert to probabilities and verify they sum to 1
        probs = torch.softmax(logits, dim=-1)
        assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)

    def test_different_vocab_sizes(self):
        """Test with various vocabulary sizes."""
        d_model = 128
        num_heads = 4
        num_layers = 2
        d_ff = 512

        for vocab_size in [100, 500, 1000, 5000]:
            model = DecoderOnlyTransformer(vocab_size, d_model, num_heads, num_layers, d_ff)

            batch_size = 2
            seq_len = 5
            x = torch.randint(0, vocab_size, (batch_size, seq_len))

            logits, _ = model(x)

            assert logits.shape == (batch_size, seq_len, vocab_size)

    def test_generate_method(self):
        """Test autoregressive text generation."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        batch_size = 1
        start_len = 3
        max_length = 10

        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        generated = model.generate(start_tokens, max_length)

        # Check output shape
        assert generated.shape == (batch_size, max_length)

        # Check that it starts with start_tokens
        assert torch.equal(generated[:, :start_len], start_tokens)

        # Check that all tokens are valid (within vocabulary)
        assert (generated >= 0).all()
        assert (generated < vocab_size).all()

    def test_generate_with_temperature(self):
        """Test generation with different temperature values."""
        vocab_size = 100
        model = DecoderOnlyTransformer(vocab_size, d_model=64, num_heads=4, num_layers=1, d_ff=256)

        batch_size = 1
        start_len = 2
        max_length = 5

        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        # Test with different temperatures
        for temperature in [0.5, 1.0, 2.0]:
            generated = model.generate(start_tokens, max_length, temperature=temperature)

            assert generated.shape == (batch_size, max_length)
            assert torch.equal(generated[:, :start_len], start_tokens)

    def test_weights_are_initialized(self):
        """Test that weights are properly initialized."""
        vocab_size = 500
        model = DecoderOnlyTransformer(vocab_size, d_model=128, num_heads=4, num_layers=2, d_ff=512)

        # Check that weights are not all zeros (they've been initialized)
        assert not torch.allclose(model.token_embedding.embedding.weight, torch.zeros_like(model.token_embedding.embedding.weight))
        assert not torch.allclose(model.output_proj.weight, torch.zeros_like(model.output_proj.weight))

        # Check that LayerNorm weights are initialized to 1 and biases to 0
        assert torch.allclose(model.ln_f.weight, torch.ones_like(model.ln_f.weight))
        assert torch.allclose(model.ln_f.bias, torch.zeros_like(model.ln_f.bias))
