"""
Embedding layers for transformer.

Implements:
- Token embeddings
- Learned positional embeddings
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

    def forward(self, x):
        """
        Add positional embeddings to input.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               (typically token embeddings)

        Returns:
            output: Input with positional encoding added, shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=x.device)

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
        """
        return self.embedding(x)
