"""
Feed-forward network for transformer.

Implements position-wise feed-forward network (MLP - Multi-Layer Perceptron).

What is an MLP?
---------------
MLP stands for Multi-Layer Perceptron - a simple feedforward neural network
consisting of linear layers with activation functions in between.

In transformers, the feed-forward network is a 2-layer MLP that:
1. Expands the dimension from d_model to d_ff (typically 4x larger)
2. Applies a non-linear activation (GELU)
3. Projects back down to d_model

This is applied independently to each position in the sequence, hence
"position-wise" feed-forward network.

Architecture:
    Input (d_model)
        ↓
    Linear (d_model → d_ff)    # Expand dimension
        ↓
    GELU Activation            # Non-linearity
        ↓
    Dropout                    # Regularization
        ↓
    Linear (d_ff → d_model)    # Project back
        ↓
    Dropout                    # Regularization
        ↓
    Output (d_model)

Example with d_model=512, d_ff=2048:
    (batch, seq_len, 512)  →  Linear  →  (batch, seq_len, 2048)
                           →  GELU    →  (batch, seq_len, 2048)
                           →  Linear  →  (batch, seq_len, 512)
"""

import torch
import torch.nn as nn


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network (MLP).

    A simple 2-layer neural network applied to each position independently.
    The same network (same parameters) is used for all positions in the sequence.

    The typical expansion factor is 4x (d_ff = 4 * d_model), which gives the
    model additional capacity to learn complex transformations.
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        """
        Initialize the feed-forward network.

        Args:
            d_model: Model dimension (input and output size)
            d_ff: Hidden dimension (intermediate layer size)
                  Typically 4 * d_model (e.g., 2048 if d_model is 512)
            dropout: Dropout probability for regularization (default: 0.1)
        """
        super().__init__()

        # First linear layer: expand from d_model to d_ff
        self.linear1 = nn.Linear(d_model, d_ff)

        # Activation function: GELU (Gaussian Error Linear Unit)
        # Used in GPT-2, GPT-3, and BERT
        # Smoother alternative to ReLU
        self.activation = nn.GELU()

        # Dropout for regularization
        self.dropout1 = nn.Dropout(dropout)

        # Second linear layer: project back from d_ff to d_model
        self.linear2 = nn.Linear(d_ff, d_model)

        # Dropout after second linear layer
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Apply position-wise feed-forward network.

        The same transformation is applied to each position independently.
        This means each token at position i gets transformed by the same network,
        regardless of its position in the sequence.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)

        Returns:
            output: Feed-forward output of shape (batch, seq_len, d_model)
                   (same shape as input)
        """
        # Expand: (batch, seq_len, d_model) → (batch, seq_len, d_ff)
        x = self.linear1(x)

        # Apply non-linear activation
        x = self.activation(x)

        # Apply dropout
        x = self.dropout1(x)

        # Project back: (batch, seq_len, d_ff) → (batch, seq_len, d_model)
        x = self.linear2(x)

        # Apply dropout
        x = self.dropout2(x)

        return x
