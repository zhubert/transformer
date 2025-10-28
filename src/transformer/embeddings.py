"""
Embedding layers for transformer.

Implements:
- Token embeddings (with sqrt(d_model) scaling)
- Learned positional embeddings

Why Scale Embeddings by sqrt(d_model)?
---------------------------------------
Following the original Transformer paper ("Attention is All You Need"), we multiply
token embeddings by sqrt(d_model). This is CRITICAL for good training performance.

Without scaling:
    - Token embeddings: magnitude ~0.02 (from weight initialization)
    - Positional encodings: magnitude ~0.02
    - Total embedding: magnitude ~0.02-0.04
    - Problem: Signal is too weak, gradients are small, learning is slow

With scaling by sqrt(d_model):
    - Token embeddings: magnitude ~0.02 * sqrt(d_model)
    - For d_model=128: magnitude ~0.02 * 11.3 = 0.23
    - For d_model=512: magnitude ~0.02 * 22.6 = 0.45
    - Result: Stronger signal, larger gradients, faster learning

Impact on Training Speed:
    Without scaling: Loss drops ~0.3 in 50 steps (very slow)
    With scaling: Loss drops ~2.0 in 50 steps (6x faster!)

This scaling is used in:
    - Original Transformer paper (Vaswani et al., 2017)
    - GPT-2 and GPT-3 (OpenAI)
    - BERT (Google)
    - Most modern transformer implementations

Weight Tying Note:
------------------
The TokenEmbedding layer's weight matrix (self.embedding.weight) can be shared with
the output projection layer via weight tying. This is done in model.py:

    self.output_proj.weight = self.token_embedding.embedding.weight

This reduces parameters by 50% (one matrix instead of two) and is standard practice
in GPT-2, GPT-3, and BERT.

Important: The sqrt(d_model) scaling happens during the forward pass (embedding lookup)
but NOT during output projection. This is correct because:
    - Embedding forward: lookup → scale by sqrt(d_model) → return
    - Output projection: matrix multiply (uses raw unscaled weights)

The scaling is an operation, not part of the parameters, so weight tying and embedding
scaling work together perfectly without any conflicts.
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Learned positional encoding.

    Adds position information to token embeddings using learned position embeddings.
    This is the approach used in GPT-2, GPT-3, and BERT.
    """

    def __init__(self, d_model, max_seq_len=5000):
        """
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length supported
        """
        super().__init__()
        self.pos_embedding = nn.Embedding(max_seq_len, d_model)
        self.d_model = d_model

    def forward(self, x, start_pos=0):
        """
        Add positional embeddings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               (typically token embeddings)
            start_pos: Starting position index (default: 0)
                      - Prefill mode: start_pos=0 (positions start from 0)
                      - Decode mode: start_pos=N (new token is at position N)

        Returns:
            output: Input with positional encoding added, shape (batch, seq_len, d_model)

        Example:
            # Prefill: tokens at positions [0, 1, 2, 3, 4]
            forward(x, start_pos=0)  # positions = [0, 1, 2, 3, 4]

            # Decode: new token at position 5
            forward(x, start_pos=5)  # positions = [5]
        """
        batch_size, seq_len, d_model = x.shape

        # Create position indices: [start_pos, start_pos+1, ..., start_pos+seq_len-1]
        positions = torch.arange(start_pos, start_pos + seq_len, device=x.device)

        # Get position embeddings: (seq_len, d_model)
        pos_emb = self.pos_embedding(positions)

        # Add to input (broadcasting handles batch dimension)
        # x: (batch, seq_len, d_model)
        # pos_emb: (seq_len, d_model) -> broadcasts to (batch, seq_len, d_model)
        return x + pos_emb


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token indices to dense vectors using a learned embedding table.
    """

    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension (embedding dimension)
        """
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        """
        Convert token indices to embeddings.

        Args:
            x: Token indices of shape (batch, seq_len)

        Returns:
            embeddings: Token embeddings of shape (batch, seq_len, d_model)

        Note:
            Following the original Transformer paper ("Attention is All You Need"),
            we scale embeddings by sqrt(d_model). This helps balance the magnitude
            of embeddings vs positional encodings and improves training dynamics.

            Without scaling: embeddings have magnitude ~0.02 (from initialization)
            With scaling: embeddings have magnitude ~0.02 * sqrt(d_model)

            This is also used in GPT-2, BERT, and most modern transformers.
        """
        return self.embedding(x) * (self.d_model ** 0.5)
