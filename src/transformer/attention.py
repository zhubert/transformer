"""
Attention mechanisms for transformer.

Implements:
- Scaled dot-product attention
- Multi-head attention with causal masking
"""

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    """
    Scaled dot-product attention mechanism.

    Computes attention weights and applies them to values using the formula:
        Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V
    """

    def __init__(self):
        super().__init__()

    def forward(self, query, key, value, mask=None):
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional mask tensor of shape (batch, seq_len, seq_len).
                  Positions with True/1 are masked (set to -inf before softmax).

        Returns:
            output: Attention output of shape (batch, seq_len, d_v)
            attention_weights: Attention weights of shape (batch, seq_len, seq_len)
        """
        # Get the dimension of the key (for scaling)
        d_k = query.size(-1)

        # Compute attention scores: Q·Kᵀ / √d_k
        # query: (batch, seq_len, d_k)
        # key.transpose(-2, -1): (batch, d_k, seq_len)
        # scores: (batch, seq_len, seq_len)
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32))

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # Apply softmax to get attention weights
        # Each row sums to 1.0 (probability distribution over sequence positions)
        attention_weights = torch.softmax(scores, dim=-1)

        # Apply attention weights to values
        # attention_weights: (batch, seq_len, seq_len)
        # value: (batch, seq_len, d_v)
        # output: (batch, seq_len, d_v)
        output = torch.matmul(attention_weights, value)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Runs multiple attention heads in parallel and combines their outputs.
    """

    def __init__(self, d_model, num_heads):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
        """
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask

        Returns:
            output: Multi-head attention output
        """
        # TODO: Implement
        pass
