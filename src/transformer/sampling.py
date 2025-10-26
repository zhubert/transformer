"""
Advanced sampling strategies for text generation.

This module implements various sampling methods to improve the quality of
generated text by controlling which tokens the model can select from.

What is Sampling in Text Generation?
-------------------------------------
When generating text, the model outputs logits (raw scores) for each token
in the vocabulary. We need to choose which token to generate next.

Naive approach: Always pick the highest probability token (greedy decoding)
  Problem: Repetitive, boring text

Better approach: Sample randomly according to probabilities
  Problem: Sometimes picks very unlikely tokens → nonsense

Best approach: Smart sampling strategies that balance quality and diversity!

Sampling Methods Implemented:
------------------------------

1. Top-k Sampling
   - Keep only the k most probable tokens
   - Simple and effective
   - Fixed k doesn't adapt to context

2. Top-p Sampling (Nucleus Sampling)
   - Keep smallest set of tokens with cumulative probability ≥ p
   - Adaptive: small nucleus when confident, large when uncertain
   - More natural than fixed k

3. Combined Top-k + Top-p
   - Apply Top-k first (filter long tail)
   - Then apply Top-p (adaptive selection)
   - Best of both worlds!

All methods work with temperature scaling for additional control.

Temperature:
  < 1.0: More focused/deterministic (sharper distribution)
  = 1.0: Original probabilities
  > 1.0: More random/creative (flatter distribution)

Recommended Settings by Use Case:
----------------------------------
+-------------------+--------------+---------------------------------------+
| Use Case          | Strategy     | Settings                              |
+-------------------+--------------+---------------------------------------+
| Creative Writing  | top_k_top_p  | k=100, p=0.95, temp=1.2               |
| Balanced/General  | top_k_top_p  | k=50, p=0.9, temp=1.0                 |
| Factual/Technical | top_k_top_p  | k=40, p=0.85, temp=0.7                |
| Chatbot           | top_p        | p=0.9, temp=0.8                       |
| Code Generation   | top_k        | k=20, temp=0.6                        |
| Debugging         | greedy       | (deterministic, no randomness)        |
+-------------------+--------------+---------------------------------------+

For most use cases, top_k_top_p with k=50, p=0.9, temp=0.8-1.0 is recommended!

Example Usage:
--------------
    logits = model(input_ids)[:, -1, :]  # Shape: (batch, vocab_size)

    # Method 1: Top-k sampling
    token = sample_top_k(logits, k=50, temperature=0.8)

    # Method 2: Top-p sampling
    token = sample_top_p(logits, p=0.9, temperature=0.8)

    # Method 3: Combined
    token = sample_top_k_top_p(logits, k=50, p=0.9, temperature=0.8)

Shape Requirements:
-------------------
    Input logits:  (batch_size, vocab_size) or (vocab_size,)
    Output tokens: (batch_size, 1) or (1,) depending on input shape

All functions preserve the batch dimension and handle single sequences too.
"""

import torch
import torch.nn.functional as F


def apply_temperature(logits, temperature=1.0):
    """
    Apply temperature scaling to logits.

    Temperature controls the "sharpness" of the probability distribution:
    - temperature < 1.0: Sharper (more confident, less random)
    - temperature = 1.0: Unchanged
    - temperature > 1.0: Flatter (less confident, more random)

    Args:
        logits: Logits tensor of shape (batch, vocab_size) or (vocab_size,)
        temperature: Temperature value (must be > 0)

    Returns:
        scaled_logits: Temperature-scaled logits (same shape as input)

    Example:
        Original logits: [2.0, 1.0, 0.5]
        Original probs:  [0.59, 0.24, 0.17]

        With temperature=0.5 (more focused):
            Scaled logits: [4.0, 2.0, 1.0]
            New probs:     [0.84, 0.11, 0.05]  ← More peaked!

        With temperature=2.0 (more random):
            Scaled logits: [1.0, 0.5, 0.25]
            New probs:     [0.42, 0.26, 0.22]  ← More uniform!

    Implementation Note:
        We divide by temperature: logits / T
        - T < 1: Makes logits larger → sharper distribution after softmax
        - T > 1: Makes logits smaller → flatter distribution after softmax
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    if temperature == 1.0:
        return logits  # No scaling needed

    return logits / temperature


def sample_top_k(logits, k=50, temperature=1.0):
    """
    Sample from top-k most probable tokens.

    Strategy:
    1. Apply temperature scaling
    2. Find top-k logits
    3. Set all other logits to -inf (zero probability after softmax)
    4. Sample from the filtered distribution

    Args:
        logits: Logits tensor of shape (batch, vocab_size) or (vocab_size,)
        k: Number of top tokens to keep (1 ≤ k ≤ vocab_size)
        temperature: Temperature for scaling (default 1.0)

    Returns:
        tokens: Sampled token indices of shape (batch, 1) or (1,)

    Example:
        Input logits for vocab_size=6:
            [2.3, -1.5, 4.1, 0.8, 0.2, -0.5]

        With k=3:
            Step 1: Find top-3 → [4.1, 2.3, 0.8] at indices [2, 0, 3]
            Step 2: Mask others → [2.3, -inf, 4.1, 0.8, -inf, -inf]
            Step 3: Softmax → [0.22, 0.0, 0.67, 0.11, 0.0, 0.0]
            Step 4: Sample from {0, 2, 3} according to probabilities

    When to use:
        - Want to prevent sampling very unlikely tokens
        - Need consistent nucleus size across all contexts
        - Good default: k=50 for diverse but coherent text
    """
    # Handle single sequence (add batch dimension if needed)
    original_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # (vocab_size,) → (1, vocab_size)

    # Validate k
    vocab_size = logits.size(-1)
    if k <= 0 or k > vocab_size:
        raise ValueError(f"k must be in range [1, {vocab_size}], got {k}")

    # Step 1: Apply temperature
    logits = apply_temperature(logits, temperature)

    # Step 2: Find top-k values and indices
    # topk returns: (values, indices) both of shape (batch, k)
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Step 3: Create filtered logits tensor filled with -inf
    # After softmax, -inf becomes 0 probability
    filtered_logits = torch.full_like(logits, float('-inf'))

    # Step 4: Scatter top-k values back into filtered logits
    # This sets the top-k positions to their original values, rest stay -inf
    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Step 5: Convert to probabilities
    probs = F.softmax(filtered_logits, dim=-1)

    # Step 6: Sample from the distribution
    # multinomial expects probabilities, samples 1 token per batch item
    token = torch.multinomial(probs, num_samples=1)  # (batch, 1)

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        token = token.squeeze(0)  # (1, 1) → (1,)

    return token


def sample_top_p(logits, p=0.9, temperature=1.0, min_tokens_to_keep=1):
    """
    Sample from nucleus of tokens with cumulative probability ≥ p.

    Also called "nucleus sampling". Dynamically selects number of tokens
    based on probability distribution, making it adaptive to model confidence.

    Strategy:
    1. Apply temperature scaling
    2. Convert to probabilities and sort descending
    3. Compute cumulative probabilities
    4. Find cutoff where cumulative probability exceeds p
    5. Mask tokens outside the nucleus
    6. Sample from the filtered distribution

    Args:
        logits: Logits tensor of shape (batch, vocab_size) or (vocab_size,)
        p: Cumulative probability threshold (0 < p ≤ 1)
        temperature: Temperature for scaling (default 1.0)
        min_tokens_to_keep: Minimum number of tokens to keep (default 1)

    Returns:
        tokens: Sampled token indices of shape (batch, 1) or (1,)

    Example:
        Probabilities: [0.45, 0.30, 0.15, 0.08, 0.01, 0.01]

        With p=0.9:
            Cumulative: [0.45, 0.75, 0.90, 0.98, 0.99, 1.00]
                         ✓     ✓     ✓     ✗     ✗     ✗
            Nucleus: first 3 tokens (cumulative exactly 0.90)

        With p=0.95:
            Cumulative: [0.45, 0.75, 0.90, 0.98, 0.99, 1.00]
                         ✓     ✓     ✓     ✓     ✗     ✗
            Nucleus: first 4 tokens (cumulative 0.98 ≥ 0.95)

    Adaptive behavior:
        When model is confident:
            Probs: [0.95, 0.02, 0.01, ...]
            → Nucleus has ~1 token

        When model is uncertain:
            Probs: [0.15, 0.14, 0.13, 0.12, ...]
            → Nucleus has many tokens

    When to use:
        - Want adaptive nucleus size based on model confidence
        - More natural than fixed k
        - Good default: p=0.9 or p=0.95
    """
    # Handle single sequence (add batch dimension if needed)
    original_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)  # (vocab_size,) → (1, vocab_size)

    # Validate p
    if p <= 0 or p > 1:
        raise ValueError(f"p must be in range (0, 1], got {p}")

    # Step 1: Apply temperature
    logits = apply_temperature(logits, temperature)

    # Step 2: Convert to probabilities
    probs = F.softmax(logits, dim=-1)  # (batch, vocab_size)

    # Step 3: Sort probabilities in descending order
    # sorted_probs: (batch, vocab_size) - probabilities in descending order
    # sorted_indices: (batch, vocab_size) - original indices of sorted probs
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Step 4: Compute cumulative probabilities
    # cumsum computes running sum: [a, b, c] → [a, a+b, a+b+c]
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)  # (batch, vocab_size)

    # Step 5: Find nucleus cutoff
    # We want to remove tokens where cumulative probability already exceeded p
    # But we need to keep at least the first token, so we shift by 1
    #
    # Example: p=0.9, cumulative=[0.45, 0.75, 0.90, 0.98]
    #   shifted cumulative=[0.00, 0.45, 0.75, 0.90]  (prepend 0, remove last)
    #   mask = [False, False, False, True]  (where shifted_cumulative > p)
    #   We keep first 3, remove 4th onwards

    # Shift cumulative probs to the right (prepend 0, remove last element)
    shifted_cumulative = torch.cat([
        torch.zeros_like(cumulative_probs[:, :1]),  # (batch, 1) of zeros
        cumulative_probs[:, :-1]                     # (batch, vocab_size-1)
    ], dim=-1)

    # Create mask: True for tokens to remove (where previous cumulative > p)
    mask = shifted_cumulative > p  # (batch, vocab_size)

    # Ensure we keep at least min_tokens_to_keep tokens
    # Set mask[:, :min_tokens_to_keep] = False
    if min_tokens_to_keep > 1:
        mask[:, :min_tokens_to_keep] = False

    # Step 6: Apply mask to sorted probabilities
    sorted_probs[mask] = 0.0  # Zero out probabilities outside nucleus

    # Step 7: Renormalize probabilities (so they sum to 1.0 again)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Step 8: Sample from the filtered distribution (in sorted space)
    # Note: We sample from sorted_probs, which corresponds to sorted_indices
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)  # (batch, 1)

    # Step 9: Map back to original vocabulary indices
    # We sampled an index in sorted space, need to get original vocab index
    token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)  # (batch, 1)

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        token = token.squeeze(0)  # (1, 1) → (1,)

    return token


def sample_top_k_top_p(logits, k=50, p=0.9, temperature=1.0):
    """
    Sample using both top-k and top-p filtering (combined).

    This is the recommended approach for high-quality text generation!

    Strategy:
    1. Apply temperature scaling
    2. First apply top-k filtering (removes long tail)
    3. Then apply top-p filtering (adaptive selection)
    4. Sample from the final distribution

    Args:
        logits: Logits tensor of shape (batch, vocab_size) or (vocab_size,)
        k: Number of top tokens to keep (1 ≤ k ≤ vocab_size)
        p: Cumulative probability threshold (0 < p ≤ 1)
        temperature: Temperature for scaling (default 1.0)

    Returns:
        tokens: Sampled token indices of shape (batch, 1) or (1,)

    Why combine both?
        - Top-k: Removes very unlikely tokens (long tail)
        - Top-p: Adaptively selects from remaining tokens
        - Result: Best of both worlds!

    Example:
        Original probs (100 tokens):
            [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.01, 0.01, ...]

        After top-k (k=50):
            Removes tokens 51-100 (very low probability)
            [0.40, 0.25, 0.15, 0.08, 0.05, 0.03, 0.01, 0.01, ..., 0.00001] ← cut here

        After top-p (p=0.9):
            Within top-50, keep only nucleus
            [0.40, 0.25, 0.15, 0.08] ← cumulative = 0.88 + next = 0.93 > 0.9
            Final nucleus: 4-5 tokens

    Recommended settings:
        - Balanced: k=50, p=0.9, temperature=1.0
        - More focused: k=40, p=0.85, temperature=0.8
        - More creative: k=100, p=0.95, temperature=1.2

    When to use:
        - Default choice for most generation tasks
        - Combines benefits of both methods
        - More robust than using either alone
    """
    # Handle single sequence (add batch dimension if needed)
    original_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Validate parameters
    vocab_size = logits.size(-1)
    if k <= 0 or k > vocab_size:
        raise ValueError(f"k must be in range [1, {vocab_size}], got {k}")
    if p <= 0 or p > 1:
        raise ValueError(f"p must be in range (0, 1], got {p}")

    # Step 1: Apply temperature
    logits = apply_temperature(logits, temperature)

    # Step 2: Apply top-k filtering
    # Find top-k values and indices
    top_k_logits, top_k_indices = torch.topk(logits, k, dim=-1)

    # Create filtered logits with -inf for non-top-k tokens
    filtered_logits = torch.full_like(logits, float('-inf'))
    filtered_logits.scatter_(dim=-1, index=top_k_indices, src=top_k_logits)

    # Step 3: Convert to probabilities (for top-p filtering)
    probs = F.softmax(filtered_logits, dim=-1)  # (batch, vocab_size)

    # Step 4: Apply top-p filtering on top-k tokens
    # Sort probabilities in descending order
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)

    # Compute cumulative probabilities
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Find nucleus cutoff (same logic as sample_top_p)
    shifted_cumulative = torch.cat([
        torch.zeros_like(cumulative_probs[:, :1]),
        cumulative_probs[:, :-1]
    ], dim=-1)

    mask = shifted_cumulative > p
    sorted_probs[mask] = 0.0

    # Renormalize
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)

    # Step 5: Sample from the filtered distribution
    sampled_sorted_idx = torch.multinomial(sorted_probs, num_samples=1)
    token = torch.gather(sorted_indices, dim=-1, index=sampled_sorted_idx)

    # Restore original shape if input was 1D
    if len(original_shape) == 1:
        token = token.squeeze(0)

    return token


def sample_greedy(logits):
    """
    Greedy decoding: always select the most probable token.

    This is the simplest sampling strategy but often produces repetitive text.

    Args:
        logits: Logits tensor of shape (batch, vocab_size) or (vocab_size,)

    Returns:
        tokens: Most probable token indices of shape (batch, 1) or (1,)

    Example:
        Logits: [2.3, -1.5, 4.1, 0.8]
                              ↑ highest
        Output: tensor([2])  (index of highest logit)

    When to use:
        - Debugging (deterministic output)
        - When you want most confident/safe predictions
        - NOT recommended for creative text generation
    """
    original_shape = logits.shape
    if logits.dim() == 1:
        logits = logits.unsqueeze(0)

    # Get index of maximum logit along vocab dimension
    token = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

    if len(original_shape) == 1:
        token = token.squeeze(0)

    return token
