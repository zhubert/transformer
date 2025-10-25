"""
Embedding layers for transformer.

Implements:
- Token embeddings
- Sinusoidal positional encoding
"""

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding.

    Adds position information to token embeddings using sine and cosine functions.
    """

    def __init__(self, d_model, max_seq_len=5000):
        """
        Args:
            d_model: Model dimension
            max_seq_len: Maximum sequence length to pre-compute
        """
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            output: Input with positional encoding added
        """
        # TODO: Implement
        pass


class TokenEmbedding(nn.Module):
    """
    Token embedding layer.

    Converts token indices to dense vectors.
    """

    def __init__(self, vocab_size, d_model):
        """
        Args:
            vocab_size: Size of vocabulary
            d_model: Model dimension
        """
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, x):
        """
        Args:
            x: Token indices of shape (batch, seq_len)

        Returns:
            embeddings: Token embeddings of shape (batch, seq_len, d_model)
        """
        # TODO: Implement
        pass
