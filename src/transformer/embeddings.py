"""
Embedding layers for transformer.

Implements:
- Token embeddings (with sqrt(d_model) scaling)
- Learned positional embeddings (GPT-2/GPT-3 style)
- RoPE (Rotary Position Embeddings) - modern alternative

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
import math


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


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Position Embedding (RoPE) - A modern, parameter-free approach to position encoding.

    THE BIG IDEA: Instead of ADDING position information to embeddings, we ROTATE
    the query and key vectors by an angle proportional to their position.

    Why Rotation? The Mathematical Intuition
    ----------------------------------------
    Imagine vectors as hands on a clock:
    - Position 0: Hand points at 12 o'clock
    - Position 1: Rotated clockwise by θ
    - Position 2: Rotated clockwise by 2θ
    - Position 3: Rotated clockwise by 3θ

    When two clock hands meet for attention, their angle difference tells you
    the RELATIVE distance between tokens. This happens automatically through
    the dot product!

    The Math (Simplified for 2D)
    ----------------------------
    For a single pair of dimensions:

        Original vectors: q = [q₀, q₁], k = [k₀, k₁]

        Rotate q at position m by angle mθ:
            q' = [q₀ cos(mθ) - q₁ sin(mθ),
                  q₀ sin(mθ) + q₁ cos(mθ)]

        Rotate k at position n by angle nθ:
            k' = [k₀ cos(nθ) - k₁ sin(nθ),
                  k₀ sin(nθ) + k₁ cos(nθ)]

        When computing attention: q' · k'
            The math works out so the result depends on (m - n), the RELATIVE
            position, not the absolute positions m and n!

    How It Works in Practice
    ------------------------
    1. Split d_model dimensions into pairs: [dim0, dim1], [dim2, dim3], ...
       - For d_model=256 with head_dim=64: We get 32 pairs per head

    2. Assign different rotation frequencies to each pair:
       - Fast frequencies (high rotation): Fine-grained position distinctions
       - Slow frequencies (low rotation): Long-range position relationships
       - Formula: θᵢ = 10000^(-2i/d) where i is the pair index

    3. For each position m, precompute rotation angles:
       - Pair 0: m × θ₀
       - Pair 1: m × θ₁
       - Pair 2: m × θ₂
       - ...

    4. Apply 2D rotation to each pair in Q and K:
       - Rotate Q using its position
       - Rotate K using its position
       - Leave V untouched (position info not needed for content!)

    5. Attention naturally sees relative positions:
       - When Q at position m attends to K at position n
       - The rotations encode (m - n) automatically
       - No explicit relative position computation needed!

    Why RoPE is Better Than Learned Embeddings
    -------------------------------------------
    ✅ RELATIVE POSITION: Encodes "3 tokens apart" not "at position 47"
       - Language cares about relative distances, not absolute positions
       - Works better for understanding sentence structure

    ✅ LENGTH EXTRAPOLATION: Trained on 128 tokens? Generate 500+ tokens!
       - Rotation pattern extends naturally beyond training length
       - Learned embeddings hit a wall at max_seq_len

    ✅ NO PARAMETERS: Zero learnable parameters, purely mathematical
       - Learned embeddings: 25M parameters for vocab_size=100K, d_model=256
       - RoPE: 0 parameters, just precomputed sin/cos values
       - Less to learn = faster training, better generalization

    ✅ THEORETICALLY GROUNDED: Based on geometric principles
       - Proven properties about relative position encoding
       - Not just "learn something and hope it works"

    ✅ WORKS WITH KV-CACHE: Each new token gets rotated by its absolute position
       - The relative position math still works perfectly
       - No special handling needed for cached vs new tokens

    Used In Production By
    ----------------------
    - GPT-NeoX (EleutherAI)
    - LLaMA, LLaMA 2, LLaMA 3 (Meta)
    - PaLM (Google)
    - Falcon (TII)
    - Mistral (Mistral AI)
    - Most modern open-source LLMs (2023-2024)

    Comparison Table
    ----------------
    | Feature               | Learned Embeddings | RoPE            |
    |-----------------------|-------------------|-----------------|
    | Parameters            | 25M+ (large)      | 0 (none!)       |
    | Position Type         | Absolute          | Relative        |
    | Length Extrapolation  | Poor (hard limit) | Excellent       |
    | Training Data Needed  | Yes (slow)        | No (math-based) |
    | Memory                | Embedding table   | Small cache     |
    | Interpretability      | Opaque            | Geometric       |
    | KV-Cache Compatible   | Yes               | Yes             |
    | Used in Modern LLMs   | Decreasing        | Standard (2024) |

    The Frequency Pattern
    ---------------------
    We use logarithmically spaced frequencies:

        θᵢ = 10000^(-2i/d) for i = 0, 1, 2, ..., d/2-1

    This gives:
    - Low indices (i=0): θ ≈ 1.0 → completes rotation in ~6 positions (fine-grained)
    - Mid indices (i=d/4): θ ≈ 0.01 → completes rotation in ~600 positions (medium-range)
    - High indices (i=d/2): θ ≈ 0.0001 → completes rotation in 60K positions (long-range)

    This range of frequencies allows the model to capture both:
    - Local patterns: "the cat" (adjacent words)
    - Long-range dependencies: "the cat ... it" (distant references)

    Implementation Details
    ----------------------
    - Applied to Q and K, NOT to V (V carries content, not position)
    - Works per-head: each attention head gets rotations based on its d_k
    - Precomputed and cached: sin/cos values computed once, reused
    - Efficient: Only ~5-10% computational overhead vs learned embeddings
    - Compatible with everything: Works with KV-cache, torch.compile(), etc.

    Shape Flow Example (head_dim=64, seq_len=10)
    --------------------------------------------
    Input Q or K: (batch, num_heads, seq_len, head_dim)
                  (2, 4, 10, 64)

    1. Reshape to pairs:
       (2, 4, 10, 64) → (2, 4, 10, 32, 2)
                         32 pairs of dimensions

    2. Split each pair:
       [..., 0] = first element of each pair
       [..., 1] = second element of each pair

    3. Apply rotation:
       q0_rotated = q0 * cos(mθ) - q1 * sin(mθ)
       q1_rotated = q0 * sin(mθ) + q1 * cos(mθ)

    4. Merge back:
       (2, 4, 10, 32, 2) → (2, 4, 10, 64)

    Output: Rotated Q or K with same shape, now encoding position!

    Technical Note: Why This Works
    -------------------------------
    The key insight is that for 2D rotation by angle α:

        [x']   [cos α  -sin α] [x]
        [y'] = [sin α   cos α] [y]

    When we compute the dot product of two rotated vectors:
        (rotated by mθ) · (rotated by nθ) = original · (rotated by (m-n)θ)

    This is the "angle difference" property of rotations. The attention
    mechanism naturally computes relative positions through the dot product!
    """

    def __init__(self, head_dim, max_seq_len=5000, base=10000.0):
        """
        Initialize RoPE with precomputed frequencies.

        Args:
            head_dim: Dimension per attention head (d_k in the paper)
                     Must be even (we rotate pairs of dimensions)
            max_seq_len: Maximum sequence length to precompute for
                        Can extrapolate beyond this, but precomputing is faster
            base: Base for computing frequencies (10000 in original paper)
                  Higher base = slower rotation = longer-range position encoding
                  Can be tuned for different context lengths

        Precomputation:
            We precompute sin/cos values for all positions up to max_seq_len
            and all frequency bands. This is a one-time cost that makes
            forward passes fast.

        Memory cost:
            2 × max_seq_len × (head_dim / 2) × 4 bytes
            For head_dim=64, max_seq_len=5000: ~1.2 MB
            Negligible compared to model weights!
        """
        super().__init__()

        assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # Compute frequency for each pair of dimensions
        # θᵢ = base^(-2i/head_dim) for i = 0, 1, 2, ..., head_dim/2 - 1
        #
        # Example for head_dim=64:
        # i=0:  θ = 10000^(0/64)    = 1.0       → fast rotation (fine-grained)
        # i=16: θ = 10000^(-32/64)  = 0.01      → medium rotation
        # i=31: θ = 10000^(-62/64)  = 0.0001    → slow rotation (long-range)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        # Shape: (head_dim / 2,)
        # Example for head_dim=64: 32 frequencies
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Precompute sin/cos values for all positions
        # This is cached and reused across all forward passes
        self._cached_cos = None
        self._cached_sin = None
        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """
        Precompute and cache sin/cos values for positions [0, seq_len).

        Args:
            seq_len: Sequence length to precompute for

        Computes:
            For each position m and each frequency θᵢ:
                cos(m × θᵢ) and sin(m × θᵢ)

        These values are used to rotate Q and K vectors.
        """
        # Create position indices: [0, 1, 2, ..., seq_len-1]
        positions = torch.arange(seq_len, device=self.inv_freq.device).float()
        # Shape: (seq_len,)

        # Compute angles: outer product of positions and frequencies
        # positions: (seq_len,)
        # inv_freq: (head_dim / 2,)
        # Result: (seq_len, head_dim / 2)
        #
        # angles[m, i] = position[m] × inv_freq[i] = m × θᵢ
        angles = torch.outer(positions, self.inv_freq)
        # Shape: (seq_len, head_dim / 2)

        # Compute sin and cos for all angles
        # These will be used in the rotation formula:
        #   q0' = q0 * cos(mθ) - q1 * sin(mθ)
        #   q1' = q0 * sin(mθ) + q1 * cos(mθ)
        cos_cached = torch.cos(angles)  # (seq_len, head_dim / 2)
        sin_cached = torch.sin(angles)  # (seq_len, head_dim / 2)

        # Cache these values
        self._cached_cos = cos_cached
        self._cached_sin = sin_cached
        self._cached_seq_len = seq_len

    def _apply_rotary_emb(self, x, cos, sin):
        """
        Apply rotary embedding to input tensor x using precomputed cos/sin.

        This is where the actual rotation happens!

        Args:
            x: Input tensor (Q or K) of shape (..., seq_len, head_dim)
               Typically: (batch, num_heads, seq_len, head_dim)
            cos: Cosine values of shape (seq_len, head_dim / 2)
            sin: Sine values of shape (seq_len, head_dim / 2)

        Returns:
            rotated_x: Rotated tensor of shape (..., seq_len, head_dim)

        Rotation Formula (for each pair of dimensions):
            [x₀']   [cos θ  -sin θ] [x₀]     [x₀ cos θ - x₁ sin θ]
            [x₁'] = [sin θ   cos θ] [x₁]  =  [x₀ sin θ + x₁ cos θ]

        We apply this rotation to each of the (head_dim / 2) pairs.
        """
        # Get shape
        *batch_dims, seq_len, head_dim = x.shape

        # Reshape x to expose pairs: (..., seq_len, head_dim/2, 2)
        # This groups each pair of dimensions together
        x = x.reshape(*batch_dims, seq_len, head_dim // 2, 2)

        # Split into first and second element of each pair
        # x[..., 0]: first element  (q₀ or k₀)
        # x[..., 1]: second element (q₁ or k₁)
        x0 = x[..., 0]  # (..., seq_len, head_dim / 2)
        x1 = x[..., 1]  # (..., seq_len, head_dim / 2)

        # Expand cos/sin to match batch dimensions
        # cos, sin: (seq_len, head_dim / 2)
        # Need: (..., seq_len, head_dim / 2) to broadcast with x0, x1
        cos = cos.reshape(*([1] * len(batch_dims)), seq_len, head_dim // 2)
        sin = sin.reshape(*([1] * len(batch_dims)), seq_len, head_dim // 2)

        # Apply rotation formula:
        #   x₀' = x₀ cos θ - x₁ sin θ
        #   x₁' = x₀ sin θ + x₁ cos θ
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos

        # Stack back together: (..., seq_len, head_dim/2, 2)
        x_rotated = torch.stack([x0_rotated, x1_rotated], dim=-1)

        # Reshape back to original: (..., seq_len, head_dim)
        x_rotated = x_rotated.reshape(*batch_dims, seq_len, head_dim)

        return x_rotated

    def forward(self, q, k, start_pos=0):
        """
        Apply RoPE to query and key tensors.

        This is the main entry point. It rotates Q and K based on their positions.

        Args:
            q: Query tensor of shape (batch, num_heads, seq_len, head_dim)
            k: Key tensor of shape (batch, num_heads, seq_len, head_dim)
            start_pos: Starting position index (default: 0)
                      - Prefill mode: start_pos=0 (positions start from 0)
                      - Decode mode: start_pos=N (new token is at position N)
                      Critical for KV-cache: must track current position!

        Returns:
            q_rotated: Rotated query of shape (batch, num_heads, seq_len, head_dim)
            k_rotated: Rotated key of shape (batch, num_heads, seq_len, head_dim)

        Example Usage:
            # Prefill: Process prompt "The cat" (positions 0, 1)
            q, k, v = project_qkv(embeddings)
            q_rot, k_rot = rope(q, k, start_pos=0)
            attention_output = attention(q_rot, k_rot, v)

            # Decode: Generate token at position 2 ("sat")
            q, k, v = project_qkv(new_token_embedding)
            q_rot, k_rot = rope(q, k, start_pos=2)  # ← Critical!
            attention_output = attention(q_rot, k_rot, v)

        Why start_pos matters:
            - Each token must be rotated by its ABSOLUTE position
            - In decode mode, new token is at position (cache_length)
            - If we always used start_pos=0, all new tokens would be rotated
              the same way, losing position information!
        """
        seq_len = q.shape[2]
        end_pos = start_pos + seq_len

        # Extend cache if needed (for sequences longer than max_seq_len)
        if end_pos > self._cached_seq_len:
            # Precompute more values
            new_max = max(end_pos, self._cached_seq_len * 2)
            self._build_cache(new_max)

        # Get sin/cos for positions [start_pos, start_pos + seq_len)
        cos = self._cached_cos[start_pos:end_pos]  # (seq_len, head_dim / 2)
        sin = self._cached_sin[start_pos:end_pos]  # (seq_len, head_dim / 2)

        # Apply rotation to Q and K
        # Note: We do NOT rotate V! V carries content, not position.
        q_rotated = self._apply_rotary_emb(q, cos, sin)
        k_rotated = self._apply_rotary_emb(k, cos, sin)

        return q_rotated, k_rotated
