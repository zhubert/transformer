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

    def forward(self, query, key, value, mask=None, debug=False):
        """
        Compute scaled dot-product attention.

        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional mask tensor of shape (batch, seq_len, seq_len).
                  Positions with True/1 are masked (set to -inf before softmax).
            debug: If True, print diagnostic information for NaN detection

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
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(torch.tensor(d_k, dtype=torch.float32, device=query.device))

        # Debug: Check for NaN in scores
        if debug and (torch.isnan(scores).any() or torch.isinf(scores).any()):
            print(f"  [DEBUG] NaN/Inf in attention scores before mask!")
            print(f"  Query stats: min={query.min():.4f}, max={query.max():.4f}, mean={query.mean():.4f}")
            print(f"  Key stats: min={key.min():.4f}, max={key.max():.4f}, mean={key.mean():.4f}")
            print(f"  Scores stats: min={scores.min():.4f}, max={scores.max():.4f}, mean={scores.mean():.4f}")

        # Apply mask if provided (set masked positions to -inf)
        if mask is not None:
            scores = scores.masked_fill(mask == 1, float('-inf'))

        # Apply softmax to get attention weights
        # Each row sums to 1.0 (probability distribution over sequence positions)
        attention_weights = torch.softmax(scores, dim=-1)

        # Debug: Check for NaN after softmax
        if debug and torch.isnan(attention_weights).any():
            print(f"  [DEBUG] NaN in attention_weights after softmax!")
            print(f"  Scores before softmax stats: min={scores.min():.4f}, max={scores.max():.4f}")

        # Apply attention weights to values
        # attention_weights: (batch, seq_len, seq_len)
        # value: (batch, seq_len, d_v)
        # output: (batch, seq_len, d_v)
        output = torch.matmul(attention_weights, value)

        # Debug: Check for NaN in output
        if debug and torch.isnan(output).any():
            print(f"  [DEBUG] NaN in attention output!")
            print(f"  Value stats: min={value.min():.4f}, max={value.max():.4f}, mean={value.mean():.4f}")

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-head attention mechanism.

    Runs multiple scaled dot-product attention operations in parallel, each called
    a "head". Each head operates on a different learned linear projection of the
    input, allowing the model to attend to information from different representation
    subspaces at different positions.

    Why Multiple Heads?
    -------------------
    Different heads can learn to focus on different aspects of the relationships
    between tokens:
    - Syntactic relationships (grammar, structure)
    - Semantic relationships (meaning, context)
    - Long-range dependencies (discourse, coreference)
    - Local patterns (nearby words)

    How It Works:
    -------------
    1. Project input to Q, K, V using learned linear transformations
    2. Split each into num_heads pieces (each of dimension d_k = d_model / num_heads)
    3. Apply scaled dot-product attention to each head in parallel
    4. Concatenate all head outputs
    5. Apply final linear projection

    Example with d_model=512, num_heads=8:
        Input: (batch, seq_len, 512)
            ↓ Linear projections
        Q, K, V: (batch, seq_len, 512)
            ↓ Split into 8 heads
        Q, K, V: (batch, 8, seq_len, 64)  # 64 = 512/8
            ↓ Attention on each head
        Output per head: (batch, 8, seq_len, 64)
            ↓ Concatenate heads
        Combined: (batch, seq_len, 512)
            ↓ Final linear projection
        Output: (batch, seq_len, 512)

    Why Don't Heads Learn the Same Thing?
    --------------------------------------
    - Different random initialization → different starting points
    - Different learned projections → different "views" of input
    - Optimization rewards diversity → redundancy doesn't help reduce loss
    - Lower-dimensional subspaces → encourages specialization
    - Empirically verified in real models (GPT-2, BERT, etc.)
    """

    def __init__(self, d_model, num_heads):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
        """
        super().__init__()

        # Ensure d_model is divisible by num_heads
        assert d_model % num_heads == 0, \
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads  # Dimension per head

        # Linear projections for Q, K, V
        # Each projects from d_model to d_model, then we split into heads
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Scaled dot-product attention (we already built this!)
        self.attention = ScaledDotProductAttention()

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None, debug=False):
        """
        Apply multi-head attention.

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            mask: Optional causal mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                  Positions with True/1 are masked (can't attend)
            debug: If True, enable diagnostic prints for NaN detection

        Returns:
            output: Multi-head attention output of shape (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape

        # 1. Project input to Q, K, V
        # Each: (batch, seq_len, d_model)
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # 2. Split into multiple heads
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # Then transpose: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 3. Adjust mask dimensions if needed
        if mask is not None:
            # Add head dimension: (seq_len, seq_len) → (1, 1, seq_len, seq_len)
            # or (batch, seq_len, seq_len) → (batch, 1, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 4. Apply scaled dot-product attention to each head in parallel
        # Q, K, V: (batch, num_heads, seq_len, d_k)
        # output: (batch, num_heads, seq_len, d_k)
        # attn_weights: (batch, num_heads, seq_len, seq_len)
        output, attn_weights = self.attention(Q, K, V, mask, debug=debug)

        # 5. Concatenate heads
        # Transpose: (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        # Reshape: (batch, seq_len, num_heads, d_k) → (batch, seq_len, num_heads * d_k)
        #        = (batch, seq_len, d_model)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # 6. Final linear projection
        output = self.W_o(output)

        return output
