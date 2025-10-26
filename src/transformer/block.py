"""
Transformer block.

Implements a single transformer decoder block - the fundamental building block
that gets stacked to create the full transformer model.

What is a Transformer Block?
-----------------------------
A transformer block is one repeatable unit that combines all our components:
- Multi-head self-attention (for gathering information from other positions)
- Feed-forward network (for processing that information)
- Layer normalization (for training stability)
- Residual connections (for gradient flow)

The full transformer is just multiple identical blocks stacked together.
GPT-3, for example, has 96 of these blocks stacked.

Architecture (Pre-LN):
----------------------
    Input (batch, seq_len, d_model)
        ↓
    ┌─────────────────────────────────────┐
    │ residual = x                        │
    │ x = layer_norm(x)                   │
    │ x = multi_head_attention(x)         │
    │ x = dropout(x)                      │
    │ x = x + residual  ← Skip connection │
    ├─────────────────────────────────────┤
    │ residual = x                        │
    │ x = layer_norm(x)                   │
    │ x = feed_forward(x)                 │
    │ x = dropout(x)                      │
    │ x = x + residual  ← Skip connection │
    └─────────────────────────────────────┘
        ↓
    Output (batch, seq_len, d_model)

Note: Input shape = Output shape (allows stacking!)

Gradient Flow and Residual Connections:
----------------------------------------
Residual connections (skip connections) are ESSENTIAL for training deep networks.

Without residuals:
    x → f₁(x) → f₂(x) → ... → f₉₆(x)
    Gradients must flow through all 96 transformations
    Result: Gradients vanish (become tiny) - early layers can't learn!

With residuals:
    x → x + f₁(x) → x + f₂(x) → ... → x + f₉₆(x)
    Gradients have a direct "highway" path

    ∂(x + f(x))/∂x = 1 + ∂f(x)/∂x
                     ↑
                  Always at least 1!

    Result: Gradients flow directly from output to input - all layers learn!

The "+ x" creates a gradient highway that makes deep learning actually work.

Pre-LN vs Post-LN:
------------------
Pre-LN (modern - what we use):
    x = x + attention(layer_norm(x))
    x = x + ffn(layer_norm(x))

    Benefits:
    - More stable training
    - Easier to train deep networks
    - Used in GPT-2, GPT-3, and modern LLMs

Post-LN (original paper):
    x = layer_norm(x + attention(x))
    x = layer_norm(x + ffn(x))

    Issues:
    - Less stable for very deep networks
    - Harder to train

Why This Architecture Works:
-----------------------------
1. **Attention** moves information between positions (communication)
2. **FFN** processes each position independently (computation)
3. **Residuals** enable gradient flow (learning deep networks)
4. **LayerNorm** stabilizes training (keeps activations in range)
5. **Dropout** prevents overfitting (regularization)

Division of labor:
- Attention = "What should I pay attention to?"
- FFN = "Now that I have the information, what should I do with it?"
"""

import torch
import torch.nn as nn
from .attention import MultiHeadAttention
from .feedforward import FeedForward


class TransformerBlock(nn.Module):
    """
    Single transformer decoder block with Pre-LN architecture.

    This is the fundamental building block of a transformer. The full model
    is created by stacking multiple identical blocks (with different learned
    parameters).

    Uses Pre-LN (Pre-Layer Normalization) architecture like GPT-2/GPT-3,
    where normalization happens before attention/FFN rather than after.
    """

    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        """
        Initialize transformer block.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            d_ff: Hidden dimension of feed-forward network (typically 4 * d_model)
            dropout: Dropout probability for regularization (default: 0.1)
        """
        super().__init__()

        # Multi-head self-attention
        self.attention = MultiHeadAttention(d_model, num_heads)

        # Position-wise feed-forward network
        self.ffn = FeedForward(d_model, d_ff, dropout)

        # Layer normalization (2 instances - one before attention, one before FFN)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout (2 instances - one after attention, one after FFN)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None, debug=False):
        """
        Apply transformer block with Pre-LN architecture.

        Data flow:
            Input (batch, seq_len, d_model)
                ↓
            First sub-layer (attention with residual):
                residual = x
                x = norm1(x)
                x = attention(x, mask)
                x = dropout(x)
                x = x + residual  ← Gradient highway
                ↓
            Second sub-layer (FFN with residual):
                residual = x
                x = norm2(x)
                x = ffn(x)
                x = dropout(x)
                x = x + residual  ← Gradient highway
                ↓
            Output (batch, seq_len, d_model)

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                  Prevents attending to future positions in decoder
            debug: If True, enable diagnostic prints for NaN detection

        Returns:
            output: Block output of shape (batch, seq_len, d_model)
                   (same shape as input - enables stacking blocks)
        """
        # First sub-layer: Multi-head self-attention with residual connection
        residual = x  # Save input for skip connection
        x = self.norm1(x)  # Pre-LN: normalize before attention
        x = self.attention(x, mask=mask, debug=debug)  # Multi-head attention
        x = self.dropout1(x)  # Dropout for regularization
        x = x + residual  # Residual connection (gradient highway!)

        # Second sub-layer: Feed-forward network with residual connection
        residual = x  # Save for next skip connection
        x = self.norm2(x)  # Pre-LN: normalize before FFN
        x = self.ffn(x)  # Position-wise feed-forward
        x = self.dropout2(x)  # Dropout for regularization
        x = x + residual  # Residual connection (gradient highway!)

        return x
