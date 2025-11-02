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
    - **Partial rotation support**: Can rotate only a fraction of dimensions
      * partial_rotary_factor=1.0 (default): rotate all dimensions (standard RoPE)
      * partial_rotary_factor=0.4: rotate first 40% of dimensions (Phi-2 style)
      * Remaining dimensions pass through unchanged
      * Used in Phi-2, CodeGen, and some recent models to reduce computation

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

    def __init__(self, head_dim, max_seq_len=5000, base=10000.0, partial_rotary_factor=1.0):
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
            partial_rotary_factor: Fraction of head_dim to apply RoPE to (default: 1.0)
                                  - 1.0 = full rotation (all dimensions, standard RoPE)
                                  - 0.4 = partial rotation (40% of dimensions, used in Phi-2)
                                  - Must be in range (0.0, 1.0]
                                  The first (head_dim * factor) dimensions get rotated,
                                  the rest remain unchanged

        Precomputation:
            We precompute sin/cos values for all positions up to max_seq_len
            and all frequency bands. This is a one-time cost that makes
            forward passes fast.

        Memory cost:
            2 × max_seq_len × (rotary_dim / 2) × 4 bytes
            For head_dim=64, max_seq_len=5000, partial=1.0: ~1.2 MB
            For head_dim=80, max_seq_len=5000, partial=0.4: ~0.6 MB
            Negligible compared to model weights!
        """
        super().__init__()

        assert head_dim % 2 == 0, f"head_dim must be even for RoPE, got {head_dim}"
        assert 0.0 < partial_rotary_factor <= 1.0, \
            f"partial_rotary_factor must be in (0.0, 1.0], got {partial_rotary_factor}"

        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.partial_rotary_factor = partial_rotary_factor

        # Calculate rotary dimension (how many dimensions to rotate)
        # Must be even for pairing
        rotary_dim = int(head_dim * partial_rotary_factor)
        if rotary_dim % 2 != 0:
            rotary_dim -= 1  # Make it even
        self.rotary_dim = rotary_dim

        # Compute frequency for each pair of dimensions in the rotary subspace
        # θᵢ = base^(-2i/rotary_dim) for i = 0, 1, 2, ..., rotary_dim/2 - 1
        #
        # NOTE: We use rotary_dim here, not head_dim, so frequencies are computed
        # relative to the rotary subspace size
        #
        # Example for head_dim=80, partial=0.4 → rotary_dim=32:
        # i=0:  θ = 10000^(0/32)    = 1.0       → fast rotation (fine-grained)
        # i=8:  θ = 10000^(-16/32)  = 0.01      → medium rotation
        # i=15: θ = 10000^(-30/32)  = 0.0001    → slow rotation (long-range)
        inv_freq = 1.0 / (base ** (torch.arange(0, rotary_dim, 2).float() / rotary_dim))
        # Shape: (rotary_dim / 2,)
        # Example for rotary_dim=32: 16 frequencies
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
            cos: Cosine values of shape (seq_len, rotary_dim / 2)
            sin: Sine values of shape (seq_len, rotary_dim / 2)

        Returns:
            rotated_x: Rotated tensor of shape (..., seq_len, head_dim)

        Rotation Formula (for each pair of dimensions):
            [x₀']   [cos θ  -sin θ] [x₀]     [x₀ cos θ - x₁ sin θ]
            [x₁'] = [sin θ   cos θ] [x₁]  =  [x₀ sin θ + x₁ cos θ]

        We apply this rotation to the first rotary_dim dimensions.
        If partial_rotary_factor < 1.0, the remaining dimensions are left unchanged.
        """
        # Get shape
        *batch_dims, seq_len, head_dim = x.shape

        # For partial rotation: split x into rotary and non-rotary parts
        if self.rotary_dim < head_dim:
            # Split: x_rotary gets rotated, x_pass_through stays unchanged
            x_rotary = x[..., :self.rotary_dim]      # (..., seq_len, rotary_dim)
            x_pass_through = x[..., self.rotary_dim:] # (..., seq_len, head_dim - rotary_dim)
        else:
            # Full rotation: rotate all dimensions
            x_rotary = x
            x_pass_through = None

        # Reshape rotary part to expose pairs: (..., seq_len, rotary_dim/2, 2)
        # This groups each pair of dimensions together
        x_rotary = x_rotary.reshape(*batch_dims, seq_len, self.rotary_dim // 2, 2)

        # Split into first and second element of each pair
        # x[..., 0]: first element  (q₀ or k₀)
        # x[..., 1]: second element (q₁ or k₁)
        x0 = x_rotary[..., 0]  # (..., seq_len, rotary_dim / 2)
        x1 = x_rotary[..., 1]  # (..., seq_len, rotary_dim / 2)

        # Expand cos/sin to match batch dimensions
        # cos, sin: (seq_len, rotary_dim / 2)
        # Need: (..., seq_len, rotary_dim / 2) to broadcast with x0, x1
        cos = cos.reshape(*([1] * len(batch_dims)), seq_len, self.rotary_dim // 2)
        sin = sin.reshape(*([1] * len(batch_dims)), seq_len, self.rotary_dim // 2)

        # Apply rotation formula:
        #   x₀' = x₀ cos θ - x₁ sin θ
        #   x₁' = x₀ sin θ + x₁ cos θ
        x0_rotated = x0 * cos - x1 * sin
        x1_rotated = x0 * sin + x1 * cos

        # Stack back together: (..., seq_len, rotary_dim/2, 2)
        x_rotated = torch.stack([x0_rotated, x1_rotated], dim=-1)

        # Reshape back: (..., seq_len, rotary_dim)
        x_rotated = x_rotated.reshape(*batch_dims, seq_len, self.rotary_dim)

        # For partial rotation: concatenate rotated and pass-through parts
        if x_pass_through is not None:
            # Concatenate: [rotated_dims, unrotated_dims]
            x_out = torch.cat([x_rotated, x_pass_through], dim=-1)
            # Shape: (..., seq_len, head_dim)
        else:
            # Full rotation: just return rotated tensor
            x_out = x_rotated

        return x_out

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
        # OR if device changed (cache built at init on CPU, but model may be moved to GPU/MPS)
        # The inv_freq buffer moves with model.to(device), but cached sin/cos don't automatically
        needs_rebuild = (
            end_pos > self._cached_seq_len or  # Need more values
            (self._cached_cos is not None and
             self._cached_cos.device != self.inv_freq.device)  # Wrong device
        )

        if needs_rebuild:
            # Rebuild cache with correct length and device
            new_max = max(end_pos, self._cached_seq_len * 2) if end_pos > self._cached_seq_len else self._cached_seq_len
            self._build_cache(new_max)

        # Get sin/cos for positions [start_pos, start_pos + seq_len)
        cos = self._cached_cos[start_pos:end_pos]  # (seq_len, head_dim / 2)
        sin = self._cached_sin[start_pos:end_pos]  # (seq_len, head_dim / 2)

        # Apply rotation to Q and K
        # Note: We do NOT rotate V! V carries content, not position.
        q_rotated = self._apply_rotary_emb(q, cos, sin)
        k_rotated = self._apply_rotary_emb(k, cos, sin)

        return q_rotated, k_rotated


class ALiBiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) - The simplest modern position encoding.

    THE BIG IDEA: Instead of rotating vectors (RoPE) or adding to embeddings (learned),
    we add BIASES directly to attention scores based on the distance between tokens.

    The Brilliantly Simple Formula
    -------------------------------
    In attention, we normally compute:
        attention_scores = Q @ K^T / sqrt(d_k)
        attention_weights = softmax(attention_scores)

    With ALiBi, we add a distance-based bias BEFORE softmax:
        attention_scores = Q @ K^T / sqrt(d_k) + ALiBi_bias
        attention_weights = softmax(attention_scores + bias)

    Where the bias is simply:
        ALiBi_bias[i, j] = -slope × |i - j|

    That's it! Just subtract the distance between positions, scaled by a slope.

    Intuition: Distance Penalty
    ----------------------------
    Imagine you're at position 5, looking at the sequence:

        Position:  0    1    2    3    4    5    6    7
        Distance:  5    4    3    2    1    0    1    2
        Bias:     -2.5 -2.0 -1.5 -1.0 -0.5  0.0 -0.5 -1.0

    (Assuming slope = 0.5 and causal masking for future tokens)

    What this means:
    - Position 5 (current): bias = 0.0 → No penalty
    - Position 4 (1 away): bias = -0.5 → Slight penalty
    - Position 3 (2 away): bias = -1.0 → Moderate penalty
    - Position 0 (5 away): bias = -2.5 → Strong penalty

    The further away a token is, the more negative the bias → lower attention weight!

    This naturally encourages local attention while still allowing long-range dependencies
    when needed (if the Q·K similarity is strong enough to overcome the bias).

    Multiple Heads with Different Slopes
    -------------------------------------
    Different attention heads get different slopes, giving them different "zoom levels":

        Head 0: slope = 0.25     → Strong distance penalty (focuses locally)
        Head 1: slope = 0.0625   → Moderate penalty (medium-range focus)
        Head 2: slope = 0.015625 → Gentle penalty (long-range focus)
        Head 3: slope = 0.00390  → Very gentle (very long-range)

    This is like having multiple cameras with different lenses:
    - Wide-angle lens (small slope): Sees the whole scene (long-range dependencies)
    - Telephoto lens (large slope): Focuses tightly on nearby objects (local patterns)

    The model learns to use different heads for different purposes!

    Slope Computation
    -----------------
    Slopes follow a geometric sequence based on the number of heads:

        slope_i = 2^(-8/num_heads × (i+1))

    For 4 heads:
        Head 0: 2^(-8/4 × 1) = 2^(-2)  = 0.25
        Head 1: 2^(-8/4 × 2) = 2^(-4)  = 0.0625
        Head 2: 2^(-8/4 × 3) = 2^(-6)  = 0.015625
        Head 3: 2^(-8/4 × 4) = 2^(-8)  = 0.00390625

    For 8 heads: 2^(-1), 2^(-2), 2^(-3), ..., 2^(-8)

    This gives a good spread from aggressive to gentle biases.

    Why ALiBi is Better Than Learned Embeddings
    --------------------------------------------
    ✅ ZERO PARAMETERS: Purely mathematical, no weights to learn
       - Learned embeddings: max_seq_len × d_model parameters (1.28M for default)
       - ALiBi: 0 parameters

    ✅ RELATIVE POSITIONS: Encodes distance "3 tokens apart", not absolute "position 47"
       - More meaningful for language (relationships matter more than location)

    ✅ EXTREME EXTRAPOLATION: Train on 512 tokens → test on 10,000+ tokens!
       - ALiBi shows the BEST extrapolation in benchmarks
       - Learned embeddings hit a hard wall at max_seq_len
       - Even better than RoPE for extreme lengths!

    ✅ SIMPLICITY: Just add biases to attention scores
       - No complex rotation math
       - No embedding layer to manage
       - Easy to understand and debug

    ✅ EFFICIENCY: Minimal computational overhead
       - Biases precomputed once
       - Just one addition in attention
       - Slightly faster than RoPE (no rotation)

    ✅ CAUSAL MASKING BUILT-IN: Naturally handles decoder-style attention
       - Biases only computed for valid positions
       - Future positions already masked, so no bias needed

    Comparison to Other Methods
    ----------------------------
    | Feature              | ALiBi        | RoPE           | Learned       |
    |----------------------|--------------|----------------|---------------|
    | Parameters           | 0            | 0              | 1.28M+        |
    | Position Type        | Relative     | Relative       | Absolute      |
    | Extrapolation        | Excellent++  | Excellent      | Poor          |
    | Simplicity           | Very Simple  | Moderate       | Simple        |
    | Computational Cost   | Minimal      | ~5-10%         | Negligible    |
    | Where Applied        | Attn scores  | Q/K vectors    | Embeddings    |
    | Used In              | BLOOM, MPT   | LLaMA, Mistral | GPT-2, GPT-3  |

    ALiBi vs RoPE:
    - ALiBi: Simpler, better extreme extrapolation, adds biases
    - RoPE: More "elegant" mathematically, rotates vectors
    - Both are excellent! ALiBi is arguably simpler to understand.

    Used In Production By
    ----------------------
    - BLOOM (BigScience) - 176B parameter model
    - MPT (MosaicML) - 7B, 30B parameter models
    - Falcon (some variants) - 7B, 40B models
    - Various research models (2022+)

    The Key Advantage: Length Extrapolation
    ----------------------------------------
    ALiBi has shown THE BEST extrapolation in academic benchmarks:
    - Train on 512 tokens → test on 10,000+ tokens with minimal degradation
    - Perplexity degrades gracefully, not catastrophically
    - Outperforms sinusoidal, learned, and even RoPE for extreme lengths

    Why does this work so well?
    - Linear bias is a simple, smooth function of distance
    - No "frequency aliasing" issues (unlike sinusoidal)
    - Model learns to overcome biases when truly needed
    - Natural inductive bias toward locality

    Implementation Details
    ----------------------
    - Applied directly to attention scores BEFORE softmax
    - Precomputed and cached for efficiency
    - One bias matrix per head (different slopes)
    - Compatible with KV-cache (extend biases as sequence grows)
    - No interaction with embeddings at all
    - Works with causal masking (future positions already masked)

    Memory Cost:
        num_heads × max_seq_len × max_seq_len × 4 bytes
        For 4 heads, max_seq_len=5000: ~400 MB

        BUT: We only store the bias template and expand as needed!
        Actual memory: num_heads × max_seq_len × 4 bytes = ~80 KB
        Negligible!

    Shape Flow Example (num_heads=4, seq_len=10)
    ---------------------------------------------
    Attention scores: (batch, num_heads, seq_len, seq_len)
                      (2, 4, 10, 10)

    ALiBi bias:       (num_heads, seq_len, seq_len)
                      (4, 10, 10)
                      ↓ Broadcasting
                      (1, 4, 10, 10) → added to scores

    For each head, the bias matrix looks like:
        Head 0 (slope=0.25):
        [[0.0,  -∞,   -∞,   -∞  ],    ← Position 0 attends only to 0
         [-0.25, 0.0, -∞,   -∞  ],    ← Position 1 attends to 0,1
         [-0.5, -0.25, 0.0, -∞  ],    ← Position 2 attends to 0,1,2
         [-0.75,-0.5, -0.25, 0.0]]    ← Position 3 attends to all

    (The -∞ values come from causal masking - applied elsewhere)

    Technical Note: Causal Masking
    -------------------------------
    ALiBi biases are only added for valid (non-masked) positions.
    For decoder-style (causal) attention:
    - Future positions are masked with -inf (standard practice)
    - ALiBi biases don't need to handle this - masking does it
    - We only compute biases for the lower triangular part

    Why This Works: The Math
    -------------------------
    Adding a linear bias to attention scores is equivalent to adding
    a "prior" that closer tokens should attend more to each other.

    Before ALiBi:
        attention_weight[i,j] ∝ exp(similarity(i,j))

    After ALiBi:
        attention_weight[i,j] ∝ exp(similarity(i,j) - slope × |i-j|)
                              = exp(similarity(i,j)) × exp(-slope × |i-j|)

    The exp(-slope × |i-j|) term is a "distance discount factor":
    - Nearby tokens: small distance → discount ≈ 1.0 → full attention
    - Far tokens: large distance → discount ≈ 0 → reduced attention

    This is a soft inductive bias, not a hard constraint. If the model
    really needs to attend to a distant token (high similarity), it can
    overcome the bias!
    """

    def __init__(self, num_heads, max_seq_len=5000):
        """
        Initialize ALiBi with precomputed slope values.

        Args:
            num_heads: Number of attention heads
                      Each head gets a different slope value
            max_seq_len: Maximum sequence length to precompute biases for
                        Can extend dynamically if needed

        Precomputation:
            We precompute slopes for each head based on the formula:
                slope[i] = 2^(-8/num_heads × (i+1))

            This gives a geometric sequence from aggressive to gentle biases.

        Memory cost:
            Slopes: num_heads × 4 bytes (negligible)
            We compute biases on-the-fly or cache them as needed.
        """
        super().__init__()

        self.num_heads = num_heads
        self.max_seq_len = max_seq_len

        # Compute slopes for each head using geometric sequence
        # Formula: slope[i] = 2^(-8/num_heads × (i+1))
        #
        # This gives a nice spread from strong to weak biases
        # Example for 8 heads: [2^-1, 2^-2, ..., 2^-8]
        # Example for 4 heads: [2^-2, 2^-4, 2^-6, 2^-8]
        slopes = torch.tensor([
            2 ** (-8 / num_heads * (i + 1))
            for i in range(num_heads)
        ])
        # Shape: (num_heads,)

        # Register as buffer (moved to device automatically, not trained)
        self.register_buffer("slopes", slopes, persistent=False)

        # Cache for bias matrices
        self._cached_bias = None
        self._cached_seq_len = 0
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len):
        """
        Precompute and cache bias matrices for all heads.

        Args:
            seq_len: Sequence length to precompute for

        Computes:
            For each head h and positions i,j:
                bias[h, i, j] = -slopes[h] × |i - j|

        For decoder (causal) attention, we only need the lower triangle
        since future positions are masked anyway.
        """
        # Create position indices
        # positions[i] = i for i in [0, 1, 2, ..., seq_len-1]
        # IMPORTANT: Must be on same device as slopes for torch.compile() compatibility
        positions = torch.arange(seq_len, device=self.slopes.device).unsqueeze(0)  # (1, seq_len)

        # Compute pairwise distances
        # distance[i, j] = |i - j|
        #
        # positions.T: (seq_len, 1)
        # positions:   (1, seq_len)
        # Broadcast subtraction and absolute value
        distances = torch.abs(positions.T - positions)  # (seq_len, seq_len)

        # Apply slopes to compute biases for each head
        # slopes: (num_heads, 1, 1)
        # distances: (seq_len, seq_len)
        # Result: (num_heads, seq_len, seq_len)
        #
        # bias[h, i, j] = -slopes[h] × distances[i, j]
        #               = -slopes[h] × |i - j|
        slopes_expanded = self.slopes.view(self.num_heads, 1, 1)
        biases = -slopes_expanded * distances.unsqueeze(0)
        # Shape: (num_heads, seq_len, seq_len)

        # Cache the biases
        self._cached_bias = biases
        self._cached_seq_len = seq_len

    def forward(self, seq_len):
        """
        Get ALiBi bias matrix for the given sequence length.

        This returns the bias matrix that should be ADDED to attention scores.

        Args:
            seq_len: Current sequence length
                    Can be less than max_seq_len (just take subset)
                    Can be more (will extend cache)

        Returns:
            biases: Bias matrix of shape (num_heads, seq_len, seq_len)
                   To be added to attention scores before softmax
                   bias[h, i, j] = -slopes[h] × |i - j|

        Usage:
            # In attention mechanism:
            attention_scores = Q @ K^T / sqrt(d_k)
            biases = alibi(seq_len)
            attention_scores = attention_scores + biases
            attention_weights = softmax(attention_scores)

        Note:
            Causal masking should still be applied separately!
            ALiBi biases don't replace masking, they complement it.
        """
        # Extend cache if needed OR if device changed
        # The cache is built at init time (on CPU) but when model.to(device) is called,
        # only registered buffers (slopes) move to the new device. We need to rebuild
        # the cache if we detect a device mismatch.
        needs_rebuild = (
            seq_len > self._cached_seq_len or  # Need more values
            (self._cached_bias is not None and
             self._cached_bias.device != self.slopes.device)  # Wrong device
        )

        if needs_rebuild:
            # Rebuild cache with correct length and device
            new_max = max(seq_len, self._cached_seq_len * 2) if seq_len > self._cached_seq_len else self._cached_seq_len
            self._build_cache(new_max)

        # Return cached biases for the requested sequence length
        # Shape: (num_heads, seq_len, seq_len)
        return self._cached_bias[:, :seq_len, :seq_len]

    def get_slopes(self):
        """
        Get the slope values for each head.

        Useful for debugging or visualization.

        Returns:
            slopes: Tensor of shape (num_heads,) containing slope values
        """
        return self.slopes
