"""
Attention Pattern Analysis

Tools for analyzing and visualizing attention patterns to understand what
each attention head is focusing on.

What is Attention Analysis?
---------------------------
Attention weights reveal what information each token is "looking at" when
making predictions. By analyzing these patterns, we can discover:

1. **Common Head Types:**
   - Previous token heads: Always attend to position i-1
   - Next token heads: Attend to position i+1
   - Beginning/end of sentence heads: Focus on start/end tokens
   - Uniform heads: Spread attention evenly (information aggregation)

2. **Task-Specific Patterns:**
   - Subject-verb agreement: Attend to subject when predicting verb
   - Coreference: "it" attends to the noun it refers to
   - Syntactic structure: Attend to grammatically related words

3. **Circuit Discovery:**
   - Which heads work together?
   - What information flows between layers?
   - How do heads compose to implement behaviors?

Example Patterns:
-----------------
Input: "The cat sat on the mat"

Head 2.3 (previous token head):
    The  cat  sat  on   the  mat
The [  -   -    -    -    -    - ]
cat [ 1.0  -    -    -    -    - ]   ← Attends to "The"
sat [ -   1.0  -    -    -    - ]   ← Attends to "cat"
on  [ -    -   1.0  -    -    - ]   ← Attends to "sat"

Head 4.1 (uniform/averaging head):
    The  cat  sat  on   the  mat
The [0.2 0.2 0.2 0.2 0.2 0.2]   ← Spread evenly
cat [0.2 0.2 0.2 0.2 0.2 0.2]
sat [0.2 0.2 0.2 0.2 0.2 0.2]

These patterns help us understand WHAT the model has learned, not just
that it works.

References:
-----------
- A Mathematical Framework for Transformer Circuits
  https://transformer-circuits.pub/2021/framework/
- In-context Learning and Induction Heads
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple


class AttentionAnalyzer:
    """
    Analyzer for attention patterns in transformer models.

    Helps understand what tokens each attention head focuses on by
    extracting and analyzing attention weights from forward passes.
    """

    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize attention analyzer.

        Args:
            model: DecoderOnlyTransformer model
            tokenizer: Tokenizer for encoding/decoding
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def analyze(
        self,
        text: str,
        layer_idx: Optional[int] = None,
        head_idx: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze attention patterns for given input text.

        Process:
        1. Tokenize input text
        2. Run forward pass with return_attention_weights=True
        3. Extract attention weights for specified layer/head
        4. Return weights along with tokens for visualization

        Args:
            text: Input text to analyze
            layer_idx: Which layer to analyze (None = all layers)
            head_idx: Which head to analyze (None = all heads in layer)

        Returns:
            Dictionary containing:
                - 'tokens': List of token strings
                - 'token_ids': List of token IDs
                - 'attention_weights': Attention weights tensor or list
                    If layer_idx and head_idx specified:
                        (seq_len, seq_len) - single head's attention
                    If only layer_idx specified:
                        (num_heads, seq_len, seq_len) - all heads in layer
                    If neither specified:
                        List of (num_heads, seq_len, seq_len) for each layer
                - 'layer_idx': Layer index (if specified)
                - 'head_idx': Head index (if specified)
                - 'num_layers': Total number of layers
                - 'num_heads': Number of attention heads per layer

        Example:
            analyzer = AttentionAnalyzer(model, tokenizer)

            # Analyze specific head
            results = analyzer.analyze("Hello world", layer_idx=2, head_idx=3)
            attn = results['attention_weights']  # (seq_len, seq_len)

            # Analyze all heads in a layer
            results = analyzer.analyze("Hello world", layer_idx=2)
            attn = results['attention_weights']  # (num_heads, seq_len, seq_len)

            # Analyze all layers
            results = analyzer.analyze("Hello world")
            attn_list = results['attention_weights']  # List[Tensor]
        """
        # Set model to evaluation mode
        self.model.eval()

        # Tokenize input
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        input_tensor = torch.tensor([token_ids], device=self.device)

        # Get attention weights from model
        logits, caches, attention_weights_list = self.model(
            input_tensor,
            return_attention_weights=True
        )

        # attention_weights_list is a list of tensors, one per layer
        # Each tensor has shape: (batch, num_heads, seq_len, seq_len)

        # Remove batch dimension (we only have batch=1)
        attention_weights_list = [attn[0] for attn in attention_weights_list]
        # Now each tensor: (num_heads, seq_len, seq_len)

        num_layers = len(attention_weights_list)
        num_heads = attention_weights_list[0].shape[0]

        # Extract requested layer/head
        if layer_idx is not None:
            if layer_idx < 0 or layer_idx >= num_layers:
                raise ValueError(f"layer_idx must be in [0, {num_layers-1}], got {layer_idx}")

            layer_attn = attention_weights_list[layer_idx]  # (num_heads, seq_len, seq_len)

            if head_idx is not None:
                if head_idx < 0 or head_idx >= num_heads:
                    raise ValueError(f"head_idx must be in [0, {num_heads-1}], got {head_idx}")

                # Return single head's attention
                attention_weights = layer_attn[head_idx]  # (seq_len, seq_len)
            else:
                # Return all heads in layer
                attention_weights = layer_attn  # (num_heads, seq_len, seq_len)
        else:
            # Return all layers
            attention_weights = attention_weights_list

        return {
            'tokens': tokens,
            'token_ids': token_ids,
            'attention_weights': attention_weights,
            'layer_idx': layer_idx,
            'head_idx': head_idx,
            'num_layers': num_layers,
            'num_heads': num_heads,
        }

    def get_head_pattern_type(
        self,
        attention_weights: torch.Tensor,
        threshold: float = 0.5
    ) -> str:
        """
        Classify the pattern type of an attention head.

        Detects common patterns:
        - "previous_token": Strongly attends to position i-1
        - "next_token": Strongly attends to position i+1 (rare in decoder)
        - "uniform": Spreads attention evenly across all positions
        - "start_token": Focuses on beginning of sequence
        - "sparse": Attends to very few positions
        - "mixed": No clear pattern

        Args:
            attention_weights: Attention matrix of shape (seq_len, seq_len)
            threshold: Minimum average attention weight to classify a pattern

        Returns:
            String describing the pattern type

        Example:
            pattern = analyzer.get_head_pattern_type(attn_matrix)
            if pattern == "previous_token":
                print("This head implements a previous token circuit!")
        """
        seq_len = attention_weights.shape[0]
        attn = attention_weights.cpu().numpy()

        # Check for previous token pattern
        # Look at diagonal offset by -1 (position i attends to i-1)
        if seq_len > 1:
            prev_tok_scores = []
            for i in range(1, seq_len):
                prev_tok_scores.append(attn[i, i-1])
            avg_prev = np.mean(prev_tok_scores)

            if avg_prev > threshold:
                return "previous_token"

        # Check for uniform/averaging pattern
        # All positions have similar attention weights
        attn_std = np.std(attn, axis=1).mean()
        if attn_std < 0.1:  # Low variance = uniform
            return "uniform"

        # Check for start token pattern
        # Strong attention to position 0 from most positions
        start_attn = attn[:, 0].mean()
        if start_attn > threshold:
            return "start_token"

        # Check for sparse pattern
        # Most attention weight concentrated on few positions
        max_attn_per_row = attn.max(axis=1)
        if max_attn_per_row.mean() > 0.7:
            return "sparse"

        return "mixed"

    def compare_heads(
        self,
        text: str,
        layer_idx: int,
        head_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Compare multiple attention heads side-by-side.

        Useful for understanding how different heads in the same layer
        process the same input differently.

        Args:
            text: Input text
            layer_idx: Which layer to analyze
            head_indices: List of head indices to compare

        Returns:
            Dictionary with:
                - 'tokens': Token strings
                - 'heads': List of dicts, one per head with:
                    - 'head_idx': Head index
                    - 'attention_weights': (seq_len, seq_len)
                    - 'pattern_type': Detected pattern type

        Example:
            # Compare heads 0, 2, and 5 in layer 3
            comparison = analyzer.compare_heads(
                "The cat sat on the mat",
                layer_idx=3,
                head_indices=[0, 2, 5]
            )

            for head_info in comparison['heads']:
                print(f"Head {head_info['head_idx']}: {head_info['pattern_type']}")
        """
        results = self.analyze(text, layer_idx=layer_idx)

        heads_data = []
        for head_idx in head_indices:
            # Get attention for this specific head
            attn = results['attention_weights'][head_idx]  # (seq_len, seq_len)

            # Classify pattern
            pattern_type = self.get_head_pattern_type(attn)

            heads_data.append({
                'head_idx': head_idx,
                'attention_weights': attn,
                'pattern_type': pattern_type,
            })

        return {
            'tokens': results['tokens'],
            'layer_idx': layer_idx,
            'heads': heads_data,
        }

    def find_heads_by_pattern(
        self,
        text: str,
        pattern_type: str,
        threshold: float = 0.5
    ) -> List[Dict[str, Any]]:
        """
        Find all attention heads matching a specific pattern type.

        Searches through all layers and heads to find those that exhibit
        the desired pattern (e.g., "previous_token", "uniform", etc.).

        Args:
            text: Input text to analyze
            pattern_type: Pattern to search for
                         ("previous_token", "uniform", "start_token", "sparse", "mixed")
            threshold: Threshold for pattern classification

        Returns:
            List of dicts with:
                - 'layer': Layer index
                - 'head': Head index
                - 'pattern_type': Confirmed pattern type
                - 'strength': How strongly it matches the pattern (0-1)

        Example:
            # Find all previous token heads
            prev_heads = analyzer.find_heads_by_pattern(
                "The quick brown fox",
                pattern_type="previous_token"
            )

            for head in prev_heads:
                print(f"Layer {head['layer']}, Head {head['head']}: "
                      f"strength={head['strength']:.2f}")
        """
        # Analyze all layers
        results = self.analyze(text)

        matching_heads = []

        for layer_idx, layer_attn in enumerate(results['attention_weights']):
            # layer_attn shape: (num_heads, seq_len, seq_len)
            num_heads = layer_attn.shape[0]

            for head_idx in range(num_heads):
                head_attn = layer_attn[head_idx]  # (seq_len, seq_len)

                # Classify this head
                detected_pattern = self.get_head_pattern_type(head_attn, threshold)

                if detected_pattern == pattern_type:
                    # Calculate strength metric (varies by pattern)
                    if pattern_type == "previous_token":
                        # Average attention to previous position
                        seq_len = head_attn.shape[0]
                        if seq_len > 1:
                            prev_scores = [head_attn[i, i-1].item() for i in range(1, seq_len)]
                            strength = np.mean(prev_scores)
                        else:
                            strength = 0.0
                    elif pattern_type == "uniform":
                        # Low variance = more uniform
                        strength = 1.0 - head_attn.cpu().numpy().std(axis=1).mean()
                    elif pattern_type == "start_token":
                        # Average attention to position 0
                        strength = head_attn[:, 0].mean().item()
                    elif pattern_type == "sparse":
                        # Average of max attention per row
                        strength = head_attn.max(dim=1)[0].mean().item()
                    else:
                        strength = 0.5  # Default for "mixed"

                    matching_heads.append({
                        'layer': layer_idx,
                        'head': head_idx,
                        'pattern_type': pattern_type,
                        'strength': strength,
                    })

        # Sort by strength (strongest first)
        matching_heads.sort(key=lambda x: x['strength'], reverse=True)

        return matching_heads
