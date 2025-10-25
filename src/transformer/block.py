"""
Transformer block.

Implements a single transformer decoder block with:
- Multi-head self-attention
- Feed-forward network
- Layer normalization
- Residual connections
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block.

    Combines multi-head attention and feed-forward network with
    residual connections and layer normalization.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network
            dropout: Dropout probability
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
            output: Block output of shape (batch, seq_len, d_model)
        """
        # TODO: Implement
        pass
