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

    Computes attention weights and applies them to values.
    """

    def __init__(self):
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional mask tensor

        Returns:
            output: Attention output
            attention_weights: Attention weights for visualization
        """
        # TODO: Implement
        pass


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
