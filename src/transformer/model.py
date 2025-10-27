"""
Decoder-only transformer model.

Implements complete GPT-style transformer architecture by combining all components
we've built: embeddings, positional encoding, transformer blocks, and output projection.

What is the Complete Transformer Model?
----------------------------------------
This is where we assemble all our individual components into a working transformer
that can:
- Take token IDs as input
- Process them through multiple transformer blocks
- Produce output logits for next-token prediction

Architecture Overview:
----------------------
    Input: Token IDs (batch, seq_len)
        ↓
    ┌─────────────────────────────────────┐
    │ 1. Token Embedding                  │ ← TokenEmbedding (we built this!)
    │    IDs → dense vectors              │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ 2. Positional Encoding              │ ← PositionalEncoding (we built this!)
    │    Add position information         │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ 3. Transformer Block 1              │ ← TransformerBlock (we built this!)
    │    (attention + FFN + residuals)    │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ 4. Transformer Block 2              │ ← Stack multiple blocks!
    │    (attention + FFN + residuals)    │
    └─────────────────────────────────────┘
        ↓
        ... (stack num_layers blocks total)
        ↓
    ┌─────────────────────────────────────┐
    │ N. Transformer Block N              │
    │    (attention + FFN + residuals)    │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Final Layer Normalization           │ ← Stabilize outputs
    │    (Pre-LN architecture)            │
    └─────────────────────────────────────┘
        ↓
    ┌─────────────────────────────────────┐
    │ Output Projection                   │ ← Linear: d_model → vocab_size
    │    d_model → vocab_size             │
    └─────────────────────────────────────┘
        ↓
    Output: Logits (batch, seq_len, vocab_size)

Shape Flow Example:
-------------------
    Input token IDs:  (batch=2, seq_len=10)
        ↓ Token Embedding
    Embedded:         (2, 10, d_model=512)
        ↓ Positional Encoding
    With positions:   (2, 10, 512)
        ↓ Transformer Block 1
    After block 1:    (2, 10, 512)  ← Same shape!
        ↓ Transformer Block 2
    After block 2:    (2, 10, 512)  ← Same shape!
        ↓ ... (repeat for all blocks)
    After block N:    (2, 10, 512)  ← Same shape!
        ↓ Final LayerNorm
    Normalized:       (2, 10, 512)  ← Same shape!
        ↓ Output Projection
    Logits:           (2, 10, vocab_size=1000)

What are Logits?
----------------
Logits are unnormalized scores (raw outputs) before converting to probabilities.

Why logits instead of probabilities?
1. Numerical stability (avoid tiny numbers like 0.0000001)
2. Better for loss computation (CrossEntropyLoss expects logits)
3. Allow temperature scaling for generation

Example:
    Logits:  [2.3, -1.5, 4.1, 0.8]  ← Raw scores
        ↓ Apply softmax
    Probs:   [0.11, 0.002, 0.66, 0.23]  ← Sum to 1.0

To get predictions:
    probs = torch.softmax(logits, dim=-1)
    next_token = torch.argmax(logits[:, -1, :], dim=-1)

Causal Masking:
---------------
For autoregressive generation (GPT-style), we use causal masking to prevent
attending to future positions:

    Mask (seq_len=4):
    [[0, 1, 1, 1],    Position 0 can only see position 0
     [0, 0, 1, 1],    Position 1 can see 0-1
     [0, 0, 0, 1],    Position 2 can see 0-2
     [0, 0, 0, 0]]    Position 3 can see 0-3

This prevents "cheating" - the model can't look ahead to predict future tokens!

Model Sizes Reference:
----------------------
GPT-2 Small:  num_layers=12, d_model=768,  num_heads=12, d_ff=3072
GPT-2 Medium: num_layers=24, d_model=1024, num_heads=16, d_ff=4096
GPT-2 Large:  num_layers=36, d_model=1280, num_heads=20, d_ff=5120
GPT-3:        num_layers=96, d_model=12288, num_heads=96, d_ff=49152

Our default:  num_layers=6,  d_model=512,  num_heads=8,  d_ff=2048
              (Similar to original Transformer paper)
"""

import torch
import torch.nn as nn
from .embeddings import TokenEmbedding, PositionalEncoding
from .block import TransformerBlock


class DecoderOnlyTransformer(nn.Module):
    """
    Decoder-only transformer model (GPT-style).

    Combines all components we've built:
    - Token embeddings
    - Positional encodings
    - Stack of transformer blocks
    - Output projection to vocabulary

    This is the complete, working transformer that can be trained and used
    for text generation.
    """

    def __init__(
        self,
        vocab_size,
        d_model=512,
        num_heads=8,
        num_layers=6,
        d_ff=2048,
        max_seq_len=5000,
        dropout=0.1,
    ):
        """
        Initialize decoder-only transformer.

        Args:
            vocab_size: Size of vocabulary (number of unique tokens)
            d_model: Model dimension (embedding size)
            num_heads: Number of attention heads (must divide d_model evenly)
            num_layers: Number of transformer blocks to stack
            d_ff: Hidden dimension of feed-forward network (typically 4 * d_model)
            max_seq_len: Maximum sequence length supported
            dropout: Dropout probability for regularization
        """
        super().__init__()

        self.d_model = d_model
        self.num_layers = num_layers

        # 1. Token embedding: converts token IDs to dense vectors
        self.token_embedding = TokenEmbedding(vocab_size, d_model)

        # 2. Positional encoding: adds position information
        self.pos_encoding = PositionalEncoding(d_model, max_seq_len)

        # 3. Stack of transformer blocks
        # Each block has same architecture but different learned parameters
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        # 4. Final layer normalization (Pre-LN architecture)
        self.ln_f = nn.LayerNorm(d_model)

        # 5. Output projection: maps from d_model to vocabulary logits
        self.output_proj = nn.Linear(d_model, vocab_size)

        # Initialize weights for better training stability
        self._init_weights()

    def _init_weights(self):
        """Initialize weights for better training stability."""
        # Initialize linear layers and embeddings
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)

        # Special initialization for output projection (large vocab stability)
        # Use smaller std for the final layer to prevent numerical instability
        torch.nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.01)

    def forward(self, x, mask=None, caches=None, debug=False):
        """
        Forward pass through the transformer with optional KV-cache support.

        KV-Cache in Multi-Layer Transformers:
        -------------------------------------
        Each transformer layer has its own attention mechanism, so each layer needs
        its own separate cache. We manage this with a list of caches:

            caches = [
                cache_layer_0,  # {'keys': tensor, 'values': tensor}
                cache_layer_1,  # {'keys': tensor, 'values': tensor}
                ...
                cache_layer_N
            ]

        Why separate caches per layer?
        - Each layer's attention operates on different representations
        - Layer 0 attends to embeddings
        - Layer 1 attends to layer 0's output
        - Layer N attends to layer N-1's output
        - These are all different, so we can't share K, V across layers

        Two Modes:
        ----------
        PREFILL (caches=None):
            - Process full input sequence (prompt)
            - Each layer initializes its own cache
            - Returns logits + list of initialized caches

        DECODE (caches provided):
            - Process only new token(s)
            - Each layer uses + updates its own cache
            - Returns logits + list of updated caches

        Args:
            x: Input token indices of shape (batch, seq_len)
               - Prefill: seq_len = prompt length
               - Decode: seq_len = 1 (new token)
            mask: Optional causal mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                  If None, a causal mask will be created automatically
            caches: Optional list of caches, one per layer
                    - None: Prefill mode
                    - List[Dict]: Decode mode
                    Length must equal num_layers
            debug: If True, print diagnostic information for NaN detection

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
                   These are unnormalized scores for each token in the vocabulary
            new_caches: List of updated caches (one per layer)
                       Each cache: {'keys': tensor, 'values': tensor}
                       None if caches input was None (for backward compatibility)

        Shape flow (PREFILL):
            Input:  (batch, prompt_len) - token IDs
                ↓ token_embedding
            (batch, prompt_len, d_model)
                ↓ pos_encoding
            (batch, prompt_len, d_model)
                ↓ block_0(x, cache=None)
            (batch, prompt_len, d_model), cache_0  ← Initialize cache
                ↓ block_1(x, cache=None)
            (batch, prompt_len, d_model), cache_1
                ↓ ... all blocks
            (batch, prompt_len, d_model), [cache_0, ..., cache_N]
                ↓ final layer norm
            (batch, prompt_len, d_model)
                ↓ output projection
            (batch, prompt_len, vocab_size) - logits

        Shape flow (DECODE):
            Input:  (batch, 1) - new token ID only!
                ↓ token_embedding
            (batch, 1, d_model)
                ↓ pos_encoding
            (batch, 1, d_model)
                ↓ block_0(x, cache=cache_0)  ← Use cached K, V
            (batch, 1, d_model), updated_cache_0
                ↓ block_1(x, cache=cache_1)
            (batch, 1, d_model), updated_cache_1
                ↓ ... all blocks
            (batch, 1, d_model), [updated_cache_0, ..., updated_cache_N]
                ↓ final layer norm
            (batch, 1, d_model)
                ↓ output projection
            (batch, 1, vocab_size) - logits for new token
        """
        batch_size, seq_len = x.shape

        # Validate caches if provided
        if caches is not None and len(caches) != self.num_layers:
            raise ValueError(
                f"Number of caches ({len(caches)}) must match number of layers ({self.num_layers})"
            )

        # Determine if we're using cache
        use_cache = (caches is not None)

        # Create causal mask if not provided
        # In decode mode with cache, we typically don't need a mask since we're
        # only generating one token at a time, but we keep this for flexibility
        if mask is None and not use_cache:
            # Only create full causal mask in prefill mode
            mask = self.create_causal_mask(seq_len).to(x.device)

        # 1. Embed tokens: (batch, seq_len) → (batch, seq_len, d_model)
        # Debug: Check input tokens
        if debug:
            if (x < 0).any() or (x >= self.token_embedding.embedding.num_embeddings).any():
                print(f"[DEBUG] Invalid token IDs in input!")
                print(f"  Token range: min={x.min().item()}, max={x.max().item()}")
                print(f"  Vocab size: {self.token_embedding.embedding.num_embeddings}")
                print(f"  Negative tokens: {(x < 0).sum().item()}")
                print(f"  Too large tokens: {(x >= self.token_embedding.embedding.num_embeddings).sum().item()}")

            # Check if embedding weights contain NaN
            if torch.isnan(self.token_embedding.embedding.weight).any():
                print(f"[DEBUG] Embedding weights contain NaN before lookup!")

        x = self.token_embedding(x)
        if debug and torch.isnan(x).any():
            print(f"NaN after token_embedding! Stats: min={x.min()}, max={x.max()}")

        # 2. Add positional encoding: (batch, seq_len, d_model) → (batch, seq_len, d_model)
        # Calculate starting position based on cache
        # If using cache, start_pos = length of cached sequence
        # Otherwise, start_pos = 0
        if use_cache and caches is not None:
            # In decode mode, get position from cache length
            # All layers should have same cache length, so check first layer
            start_pos = caches[0]['keys'].shape[2]  # Shape: (batch, num_heads, seq_len, d_k)
        else:
            # In prefill mode or no cache, start from position 0
            start_pos = 0

        x = self.pos_encoding(x, start_pos=start_pos)
        if debug and torch.isnan(x).any():
            print(f"NaN after pos_encoding! Stats: min={x.min()}, max={x.max()}")

        # 3. Pass through all transformer blocks, collecting updated caches
        new_caches = []
        for i, block in enumerate(self.blocks):
            # Get cache for this layer (None if not using cache)
            layer_cache = caches[i] if use_cache else None

            # Forward through block, get output and updated cache
            x, updated_cache = block(x, mask=mask, cache=layer_cache, debug=debug)
            new_caches.append(updated_cache)

            if debug and torch.isnan(x).any():
                print(f"NaN after block {i}! Stats: min={x.min()}, max={x.max()}")
                break

        # 4. Final layer normalization
        x = self.ln_f(x)  # (batch, seq_len, d_model) → (batch, seq_len, d_model)
        if debug and torch.isnan(x).any():
            print(f"NaN after final layernorm! Stats: min={x.min()}, max={x.max()}")

        # 5. Project to vocabulary: (batch, seq_len, d_model) → (batch, seq_len, vocab_size)
        logits = self.output_proj(x)
        if debug and torch.isnan(logits).any():
            print(f"NaN after output_proj! Stats: min={logits.min()}, max={logits.max()}")

        # Return logits and caches
        # Always return caches (they're always computed now)
        return logits, new_caches

    def create_causal_mask(self, seq_len):
        """
        Create causal mask to prevent attending to future positions.

        Args:
            seq_len: Sequence length

        Returns:
            mask: Causal mask of shape (seq_len, seq_len)
                  mask[i, j] = True means position i cannot attend to position j

        Example for seq_len=4:
            [[False, True,  True,  True ],   # Pos 0 can only see pos 0
             [False, False, True,  True ],   # Pos 1 can see pos 0-1
             [False, False, False, True ],   # Pos 2 can see pos 0-2
             [False, False, False, False]]   # Pos 3 can see pos 0-3
        """
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        return mask

    def generate(
        self,
        start_tokens,
        max_length,
        temperature=1.0,
        top_k=None,
        top_p=None,
        sampling_strategy="multinomial",
        use_cache=True,
    ):
        """
        Generate text autoregressively with advanced sampling strategies and KV-cache.

        KV-Cache for Fast Generation:
        -----------------------------
        WITHOUT cache (use_cache=False):
            for each new token:
                - Process entire sequence [tok1, tok2, ..., tokN]
                - Compute K, V for all N tokens (redundant!)
                - Time: O(N²) for generating N tokens
                - Slow for long sequences

        WITH cache (use_cache=True, DEFAULT):
            PREFILL phase:
                - Process prompt [tok1, tok2, ..., tokP]
                - Compute K, V for all P tokens
                - Cache K, V for reuse

            DECODE phase (for each new token):
                - Process only new token [tokN+1]
                - Compute K, V only for new token
                - Concatenate with cached K, V
                - Update cache
                - Time: O(N) for generating N tokens
                - 10-50x faster for long sequences!

        Generation Process:
        ------------------
        1. PREFILL: Process start_tokens, initialize cache
        2. DECODE: Generate one token at a time, updating cache
        3. Repeat DECODE until max_length reached

        Starts with start_tokens and generates new tokens one at a time,
        feeding each generated token back as input for the next step.

        Args:
            start_tokens: Starting token indices of shape (batch, start_len)
            max_length: Maximum total length to generate (including start_tokens)
            temperature: Sampling temperature (default 1.0)
                        < 1.0: more confident/deterministic
                        = 1.0: unchanged probabilities
                        > 1.0: more random/diverse
            top_k: If set, only sample from top-k most probable tokens
                   Recommended: 50 for balanced generation
            top_p: If set, sample from nucleus with cumulative probability ≥ p
                   Recommended: 0.9 or 0.95 for high-quality generation
            sampling_strategy: One of:
                - "greedy": Always pick most probable token (deterministic)
                - "multinomial": Sample from full distribution (default)
                - "top_k": Use top-k filtering (requires top_k to be set)
                - "top_p": Use nucleus sampling (requires top_p to be set)
                - "top_k_top_p": Use both (requires both top_k and top_p)
            use_cache: Whether to use KV-cache for faster generation (default: True)
                      - True: 10-50x faster, recommended for all use cases
                      - False: Slower but simpler, useful for debugging

        Returns:
            generated: Generated token indices of shape (batch, max_length)

        Examples:
            # Basic generation with KV-cache (fast!)
            start = torch.tensor([[1, 2, 3]])  # "The cat"
            generated = model.generate(start, max_length=100)  # use_cache=True by default

            # Disable cache (for debugging/comparison)
            generated = model.generate(start, max_length=100, use_cache=False)

            # Greedy decoding with cache
            generated = model.generate(
                start, max_length=100,
                sampling_strategy="greedy",
                use_cache=True
            )

            # Top-k sampling with cache (filter long tail)
            generated = model.generate(
                start, max_length=100,
                sampling_strategy="top_k", top_k=50, temperature=0.8,
                use_cache=True
            )

            # Top-p sampling with cache (adaptive nucleus)
            generated = model.generate(
                start, max_length=100,
                sampling_strategy="top_p", top_p=0.9, temperature=0.8,
                use_cache=True
            )

            # Combined (recommended for best quality AND speed)
            generated = model.generate(
                start, max_length=100,
                sampling_strategy="top_k_top_p",
                top_k=50, top_p=0.9, temperature=0.8,
                use_cache=True  # 10-50x faster!
            )

        Sampling Strategy Guide:
            Greedy: Deterministic, safe, but often repetitive
            Multinomial: Random, diverse, but can be nonsensical
            Top-k: Filters unlikely tokens, good baseline
            Top-p: Adaptive to model confidence, more natural
            Top-k + Top-p: Best of both worlds (recommended!)

        Recommended Settings by Use Case:
            - Creative writing: top_k=100, top_p=0.95, temperature=1.2, use_cache=True
            - Balanced output: top_k=50, top_p=0.9, temperature=1.0, use_cache=True
            - Focused/factual: top_k=40, top_p=0.85, temperature=0.8, use_cache=True
            - Deterministic: sampling_strategy="greedy", use_cache=True
            - Debugging: use_cache=False (to disable caching)

        Performance:
            Without cache: ~1-2 tokens/second for 100-token sequence
            With cache: ~20-50 tokens/second for 100-token sequence
            Speedup increases with sequence length!
        """
        # Import sampling functions
        from .sampling import (
            sample_greedy,
            sample_top_k,
            sample_top_p,
            sample_top_k_top_p,
        )

        # Validate sampling strategy
        valid_strategies = ["greedy", "multinomial", "top_k", "top_p", "top_k_top_p"]
        if sampling_strategy not in valid_strategies:
            raise ValueError(
                f"sampling_strategy must be one of {valid_strategies}, "
                f"got '{sampling_strategy}'"
            )

        # Validate parameters based on strategy
        if sampling_strategy == "top_k" and top_k is None:
            raise ValueError("top_k must be set when using 'top_k' strategy")
        if sampling_strategy == "top_p" and top_p is None:
            raise ValueError("top_p must be set when using 'top_p' strategy")
        if sampling_strategy == "top_k_top_p" and (top_k is None or top_p is None):
            raise ValueError(
                "Both top_k and top_p must be set when using 'top_k_top_p' strategy"
            )

        self.eval()  # Set to evaluation mode (disables dropout)

        generated = start_tokens
        batch_size = start_tokens.size(0)
        caches = None  # Will be initialized in prefill step

        with torch.no_grad():  # No gradient computation needed for generation
            for step_idx in range(max_length - start_tokens.size(1)):
                # Determine input for this step
                if use_cache and caches is not None:
                    # DECODE MODE: Only process the last token we generated
                    # This is much faster because we reuse cached K, V from previous tokens
                    current_input = generated[:, -1:]  # (batch, 1)
                else:
                    # PREFILL MODE (first step with cache) or NO CACHE mode
                    # Process the entire sequence so far
                    current_input = generated  # (batch, current_len)

                # Get logits for current input
                # If using cache after first step: current_input is (batch, 1)
                # If not using cache or first step: current_input is (batch, current_len)
                if use_cache:
                    logits, caches = self.forward(current_input, caches=caches)
                else:
                    logits, _ = self.forward(current_input, caches=None)

                # Get logits for last position only: (batch, vocab_size)
                # In cache mode after prefill, logits is already (batch, 1, vocab_size)
                # In no-cache mode, logits is (batch, current_len, vocab_size)
                next_token_logits = logits[:, -1, :]

                # Apply sampling strategy
                if sampling_strategy == "greedy":
                    next_token = sample_greedy(next_token_logits)

                elif sampling_strategy == "multinomial":
                    # Original implementation (for backward compatibility)
                    scaled_logits = next_token_logits / temperature
                    probs = torch.softmax(scaled_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                elif sampling_strategy == "top_k":
                    next_token = sample_top_k(
                        next_token_logits, k=top_k, temperature=temperature
                    )

                elif sampling_strategy == "top_p":
                    next_token = sample_top_p(
                        next_token_logits, p=top_p, temperature=temperature
                    )

                elif sampling_strategy == "top_k_top_p":
                    next_token = sample_top_k_top_p(
                        next_token_logits, k=top_k, p=top_p, temperature=temperature
                    )

                # Append to sequence
                generated = torch.cat([generated, next_token], dim=1)

        return generated
