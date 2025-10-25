"""
Feed-forward network for transformer.

Implements position-wise feed-forward network (MLP).
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.

    Two linear transformations with activation in between, applied to each
    position independently and identically.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
        """
        super().__init__()
        # TODO: Implement
        pass

    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            output: Feed-forward output of shape (batch, seq_len, d_model)
        """
        # TODO: Implement
        pass
