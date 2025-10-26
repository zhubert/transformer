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

    def forward(self, x, mask=None, debug=False):
        """
        Forward pass through the transformer.

        Args:
            x: Input token indices of shape (batch, seq_len)
            mask: Optional causal mask of shape (seq_len, seq_len) or (batch, seq_len, seq_len)
                  If None, a causal mask will be created automatically
            debug: If True, print diagnostic information for NaN detection

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size)
                   These are unnormalized scores for each token in the vocabulary

        Shape flow:
            Input:  (batch, seq_len) - token IDs
                ↓ token_embedding
            (batch, seq_len, d_model)
                ↓ pos_encoding
            (batch, seq_len, d_model)
                ↓ transformer blocks (×num_layers)
            (batch, seq_len, d_model)
                ↓ final layer norm
            (batch, seq_len, d_model)
                ↓ output projection
            (batch, seq_len, vocab_size) - logits
        """
        batch_size, seq_len = x.shape

        # Create causal mask if not provided
        if mask is None:
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
        x = self.pos_encoding(x)
        if debug and torch.isnan(x).any():
            print(f"NaN after pos_encoding! Stats: min={x.min()}, max={x.max()}")

        # 3. Pass through all transformer blocks
        for i, block in enumerate(self.blocks):
            x = block(x, mask=mask, debug=debug)  # (batch, seq_len, d_model) → (batch, seq_len, d_model)
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

        return logits

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
    ):
        """
        Generate text autoregressively with advanced sampling strategies.

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

        Returns:
            generated: Generated token indices of shape (batch, max_length)

        Examples:
            # Basic generation (multinomial sampling)
            start = torch.tensor([[1, 2, 3]])  # "The cat"
            generated = model.generate(start, max_length=10)

            # Greedy decoding (most confident)
            generated = model.generate(start, max_length=10, sampling_strategy="greedy")

            # Top-k sampling (filter long tail)
            generated = model.generate(
                start, max_length=10,
                sampling_strategy="top_k", top_k=50, temperature=0.8
            )

            # Top-p sampling (adaptive nucleus)
            generated = model.generate(
                start, max_length=10,
                sampling_strategy="top_p", top_p=0.9, temperature=0.8
            )

            # Combined (recommended for best quality)
            generated = model.generate(
                start, max_length=10,
                sampling_strategy="top_k_top_p",
                top_k=50, top_p=0.9, temperature=0.8
            )

        Sampling Strategy Guide:
            Greedy: Deterministic, safe, but often repetitive
            Multinomial: Random, diverse, but can be nonsensical
            Top-k: Filters unlikely tokens, good baseline
            Top-p: Adaptive to model confidence, more natural
            Top-k + Top-p: Best of both worlds (recommended!)

        Recommended Settings by Use Case:
            - Creative writing: top_k=100, top_p=0.95, temperature=1.2
            - Balanced output: top_k=50, top_p=0.9, temperature=1.0
            - Focused/factual: top_k=40, top_p=0.85, temperature=0.8
            - Deterministic: sampling_strategy="greedy"
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

        with torch.no_grad():  # No gradient computation needed for generation
            for _ in range(max_length - start_tokens.size(1)):
                # Get logits for current sequence
                logits = self.forward(generated)

                # Get logits for last position only: (batch, vocab_size)
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
