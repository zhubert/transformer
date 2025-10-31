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

    def forward(self, query, key, value, mask=None, debug=False, alibi_bias=None):
        """
        Compute scaled dot-product attention with optional ALiBi biases.

        Args:
            query: Query tensor of shape (batch, seq_len, d_k)
            key: Key tensor of shape (batch, seq_len, d_k)
            value: Value tensor of shape (batch, seq_len, d_v)
            mask: Optional mask tensor of shape (batch, seq_len, seq_len).
                  Positions with True/1 are masked (set to -inf before softmax).
            debug: If True, print diagnostic information for NaN detection
            alibi_bias: Optional ALiBi bias tensor to add to attention scores
                       Shape: (1, num_heads, query_seq_len, key_seq_len)
                       Added BEFORE applying mask and softmax

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

        # Add ALiBi bias if provided (BEFORE masking!)
        # ALiBi biases encourage attention to nearby tokens
        # bias[i,j] = -slope × |i-j| (negative for all non-zero distances)
        if alibi_bias is not None:
            # ALiBi bias shape: (1, num_heads, query_seq_len, key_seq_len)
            # Scores shape: (batch, num_heads, query_seq_len, key_seq_len)
            # Broadcasting adds bias to all batch elements
            scores = scores + alibi_bias

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

    def __init__(self, d_model, num_heads, rope=None, alibi=None):
        """
        Initialize multi-head attention.

        Args:
            d_model: Model dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            rope: Optional RotaryPositionalEmbedding instance
                 If provided, RoPE will be applied to Q and K
                 Cannot be used together with alibi
            alibi: Optional ALiBiPositionalBias instance
                  If provided, ALiBi biases will be added to attention scores
                  Cannot be used together with rope

        Note: Only one of rope or alibi should be provided. If both are None,
              position info should come from additive embeddings (learned).
        """
        super().__init__()

        # Ensure only one position encoding method is used
        if rope is not None and alibi is not None:
            raise ValueError("Cannot use both RoPE and ALiBi simultaneously. Choose one.")

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

        # Optional position encoding (only one at a time)
        self.rope = rope
        self.alibi = alibi

    def forward(self, x, mask=None, cache=None, debug=False, return_attention_weights=False):
        """
        Apply multi-head attention with optional KV-cache for efficient generation.

        KV-Cache Optimization:
        ----------------------
        During autoregressive generation, we repeatedly compute attention for growing
        sequences. Without caching, this is wasteful:

            Step 1: Input [tok1, tok2]       → Compute K[1,2], V[1,2]
            Step 2: Input [tok1, tok2, tok3] → Compute K[1,2,3], V[1,2,3] ← Redundant!
            Step 3: Input [tok1, tok2, tok3, tok4] → Compute K[1,2,3,4], V[1,2,3,4] ← Redundant!

        KEY INSIGHT: K and V for past tokens never change! We can cache and reuse them.

        With cache:
            Step 1 (PREFILL): Input [tok1, tok2]
                - Compute Q[1,2], K[1,2], V[1,2]
                - Cache K[1,2], V[1,2]
                - Return output + cache

            Step 2 (DECODE): Input [tok3] only!
                - Compute Q[3], K[3], V[3] for new token only
                - Load K_past[1,2], V_past[1,2] from cache
                - Concatenate: K = [K_past[1,2], K[3]], V = [V_past[1,2], V[3]]
                - Attention: Q[3] @ K[1,2,3]^T → only query from new token!
                - Update cache: K[1,2,3], V[1,2,3]
                - Return output + updated cache

        This reduces time complexity from O(n²) to O(n) for generation!

        Two Modes:
        ----------
        1. PREFILL MODE (cache=None):
           - Process full input sequence
           - Compute Q, K, V for all tokens
           - Initialize cache with K, V
           - Used for: initial prompt, training, non-cached generation

        2. DECODE MODE (cache provided):
           - Process only new token(s) (typically 1 token)
           - Compute Q, K, V only for new token(s)
           - Concatenate new K, V with cached K, V
           - Use cached K, V from previous positions
           - Used for: fast autoregressive generation

        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
               - Prefill mode: seq_len = full prompt length
               - Decode mode: seq_len = 1 (single new token)
            mask: Optional causal mask
                  - Prefill mode: (seq_len, seq_len) full causal mask
                  - Decode mode: (1, cached_len + 1) or None
            cache: Optional cache dict with 'keys' and 'values' tensors
                   - None: Prefill mode (initialize cache)
                   - Dict: Decode mode (use and update cache)
                   Structure: {'keys': (batch, num_heads, cached_seq_len, d_k),
                              'values': (batch, num_heads, cached_seq_len, d_k)}
            debug: If True, enable diagnostic prints for NaN detection
            return_attention_weights: If True, return attention weights for interpretability
                                     Default: False (for backward compatibility)

        Returns:
            output: Multi-head attention output of shape (batch, seq_len, d_model)
            new_cache: Updated cache dict with extended keys and values
                      {'keys': (batch, num_heads, total_seq_len, d_k),
                       'values': (batch, num_heads, total_seq_len, d_k)}
                      where total_seq_len = cached_seq_len + seq_len
            attention_weights: (Optional) Attention weights of shape (batch, num_heads, seq_len, total_seq_len)
                             Only returned if return_attention_weights=True
                             Shows which positions each query attends to

        Example Usage:
        --------------
        # Prefill: Process prompt "The cat"
        x_prompt = embeddings([The, cat])  # (1, 2, d_model)
        output, cache = attention(x_prompt, mask=causal_mask, cache=None)
        # cache now contains K[The, cat], V[The, cat]

        # Decode: Generate next token
        x_new = embeddings([sat])  # (1, 1, d_model) - only new token!
        output, cache = attention(x_new, mask=None, cache=cache)
        # cache now contains K[The, cat, sat], V[The, cat, sat]

        # Continue generation
        x_new = embeddings([on])  # (1, 1, d_model)
        output, cache = attention(x_new, mask=None, cache=cache)
        # cache now contains K[The, cat, sat, on], V[The, cat, sat, on]

        Speedup:
        --------
        - Without cache: O(n²) time for generating n tokens
        - With cache: O(n) time for generating n tokens
        - Typical speedup: 10-50x for sequences of 100+ tokens
        - Memory overhead: 2 * num_heads * seq_len * d_k * 4 bytes per layer
        """
        batch_size, seq_len, d_model = x.shape

        # Determine mode based on cache presence
        is_prefill = (cache is None)

        # 1. Project input to Q, K, V
        # PREFILL: Projects all tokens in prompt
        # DECODE: Projects only the new token(s)
        Q = self.W_q(x)  # (batch, seq_len, d_model)
        K = self.W_k(x)  # (batch, seq_len, d_model)
        V = self.W_v(x)  # (batch, seq_len, d_model)

        # 2. Split into multiple heads
        # Reshape: (batch, seq_len, d_model) → (batch, seq_len, num_heads, d_k)
        # Then transpose: (batch, num_heads, seq_len, d_k)
        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        # 2.5. Apply RoPE if configured (BEFORE caching!)
        # RoPE rotates Q and K based on their absolute positions
        # This must happen before concatenating with cached K, V
        if self.rope is not None:
            # Determine starting position for RoPE
            # If using cache, start_pos = length of cached sequence
            # Otherwise, start_pos = 0
            if not is_prefill:
                # DECODE MODE: Get position from cache length
                # All layers should have same cache length, so check cache
                start_pos = cache['keys'].shape[2]  # Shape: (batch, num_heads, seq_len, d_k)
            else:
                # PREFILL MODE: Start from position 0
                start_pos = 0

            # Apply RoPE rotation to Q and K
            # This encodes position information through rotation
            # Q, K: (batch, num_heads, seq_len, d_k)
            # Returns: rotated Q, K with same shape
            Q, K = self.rope(Q, K, start_pos=start_pos)
            # Now Q and K contain position information through their rotations!
            # When we compute Q @ K^T, the relative positions are automatically encoded

        # 3. Handle KV-cache: concatenate with cached K, V
        if not is_prefill:
            # DECODE MODE: We have cached K, V from previous tokens
            # Concatenate cached K, V with new K, V along sequence dimension
            #
            # Example:
            #   Cached: K[tok1, tok2] shape (batch, num_heads, 2, d_k)
            #   New: K[tok3] shape (batch, num_heads, 1, d_k)
            #   Result: K[tok1, tok2, tok3] shape (batch, num_heads, 3, d_k)
            K_cached = cache['keys']    # (batch, num_heads, cached_len, d_k)
            V_cached = cache['values']  # (batch, num_heads, cached_len, d_k)

            K = torch.cat([K_cached, K], dim=2)  # Concat along seq dimension
            V = torch.cat([V_cached, V], dim=2)  # (batch, num_heads, cached_len + seq_len, d_k)

            # Note: Q is NOT cached! We only use the new token's query.
            # The new token queries ALL past tokens (cached + current).

        # 4. Adjust mask dimensions if needed
        if mask is not None:
            # Add head dimension: (seq_len, seq_len) → (1, 1, seq_len, seq_len)
            # or (batch, seq_len, seq_len) → (batch, 1, seq_len, seq_len)
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(0)
            elif mask.dim() == 3:
                mask = mask.unsqueeze(1)

        # 4.5. Apply ALiBi biases if configured (BEFORE attention computation!)
        # ALiBi adds biases directly to attention scores (Q @ K^T)
        # Unlike RoPE which rotates Q/K, ALiBi modifies the similarity scores
        alibi_bias = None
        if self.alibi is not None:
            # Get total sequence length (including cache if present)
            # This is the key_seq_len that attention will use
            key_seq_len = K.shape[2]  # (batch, num_heads, key_seq_len, d_k)

            # Get ALiBi bias matrix for this sequence length
            # Shape: (num_heads, key_seq_len, key_seq_len)
            alibi_bias = self.alibi(key_seq_len)

            # For decode mode with cache, we only need the bias for the new query position
            # attending to all key positions (cached + new)
            query_seq_len = Q.shape[2]  # Usually 1 in decode mode
            if query_seq_len < key_seq_len:
                # DECODE MODE: Q is just the new token, K includes cached + new
                # We need biases for: new query position attending to all keys
                # Take the last query_seq_len rows (corresponding to new queries)
                alibi_bias = alibi_bias[:, -query_seq_len:, :]
                # Shape: (num_heads, query_seq_len, key_seq_len)

            # Add batch dimension for broadcasting
            # (num_heads, query_seq_len, key_seq_len) → (1, num_heads, query_seq_len, key_seq_len)
            alibi_bias = alibi_bias.unsqueeze(0)

            # This will be added to attention scores in the attention function
            # We'll pass it through the mask parameter (combine with causal mask if needed)

        # Combine ALiBi bias with causal mask if both present
        if alibi_bias is not None:
            # ALiBi biases should be ADDED to scores, not masked
            # We need to handle this in the attention mechanism
            # For now, we'll modify the mask parameter to include both
            if mask is not None:
                # Mask has -inf for masked positions
                # ALiBi has distance penalties for valid positions
                # We need to pass both to attention separately
                # Since ScaledDotProductAttention doesn't support this yet,
                # we'll combine them: where mask is True (masked), keep -inf
                # where mask is False (valid), add ALiBi bias
                pass  # Will handle below in attention call

        # 5. Apply scaled dot-product attention to each head in parallel
        # PREFILL: Q, K, V all have same seq_len (full prompt)
        # DECODE: Q has seq_len=1 (new token), K,V have cached_len+1 (all tokens)
        #
        # Attention computes: softmax(Q @ K^T / √d_k + ALiBi_bias) @ V
        # In decode mode:
        #   Q: (batch, num_heads, 1, d_k)        ← New token only
        #   K: (batch, num_heads, cached_len+1, d_k)  ← All tokens
        #   K^T: (batch, num_heads, d_k, cached_len+1)
        #   Q @ K^T: (batch, num_heads, 1, cached_len+1) ← Attend to all!
        #   + ALiBi_bias: (1, num_heads, 1, cached_len+1) ← Distance penalties
        #   output: (batch, num_heads, 1, d_k)   ← Output for new token only

        # Pass ALiBi bias to attention (will be added to scores)
        # We need to update ScaledDotProductAttention to accept bias parameter
        output, attn_weights = self.attention(Q, K, V, mask, debug=debug, alibi_bias=alibi_bias)

        # 6. Concatenate heads
        # Transpose: (batch, num_heads, seq_len, d_k) → (batch, seq_len, num_heads, d_k)
        # Reshape: (batch, seq_len, num_heads, d_k) → (batch, seq_len, d_model)
        current_seq_len = output.size(2)  # Could be 1 (decode) or longer (prefill)
        output = output.transpose(1, 2).contiguous().view(batch_size, current_seq_len, d_model)

        # 7. Final linear projection
        output = self.W_o(output)

        # 8. Create/update cache with current K, V
        # This cache will be passed to the next generation step
        new_cache = {
            'keys': K,      # (batch, num_heads, total_seq_len, d_k)
            'values': V     # (batch, num_heads, total_seq_len, d_k)
        }

        # Optionally return attention weights for interpretability
        if return_attention_weights:
            return output, new_cache, attn_weights
        else:
            return output, new_cache
