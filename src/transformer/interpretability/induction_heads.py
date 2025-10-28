"""
Induction Head Detection

Tools for detecting and analyzing induction heads - attention heads that
implement pattern matching and copying behavior.

What are Induction Heads?
--------------------------
Induction heads are a specific **circuit** (pattern of attention heads working
together) that implements a crucial capability: **copying from context based
on pattern matching**.

The Behavior:
-------------
Given a sequence with a repeated pattern like:

    Input: "... A B C ... A B [?]"

The induction head learns to predict:

    Output: "C"  (because it saw "A B" → "C" earlier)

This is **in-context learning** - the model learns from examples in the prompt
without any parameter updates!

The Circuit:
-----------
Induction heads typically involve TWO heads working together across layers:

1. **Previous Token Head** (Layer L):
   - At position i, attends to position i-1
   - Creates a representation of "what came before"

2. **Induction Head** (Layer L+1):
   - At position i, looks for positions where the PREVIOUS token matches
   - Uses K-composition: queries for token[i-1], finds matches in the sequence
   - Attends to what came AFTER those matches
   - Predicts the next token based on the pattern

Example:
--------
Input: "A B C D A B [?]"

Step 1 - Previous Token Head (Layer 0, Head 1):
    Position 5 (second "B"):
    - Attends to position 4 (second "A")
    - Creates representation: "token after A"

Step 2 - Induction Head (Layer 1, Head 3):
    Position 6 ([?]):
    - Query: "what follows A-B sequence?"
    - Searches for other "B" tokens (finds position 1)
    - Looks at what came AFTER that B → "C" (position 2)
    - Predicts: "C"

Why They're Important:
---------------------
1. **First discovered circuit** in transformers (Olsson et al., 2022)
2. **Emerges suddenly** during training ("grokking" phenomenon)
3. **Crucial for in-context learning** - enables few-shot prompting
4. **Compositional** - shows how simple mechanisms combine for complex behavior
5. **Universal** - appears in almost all transformer language models

Detection Methods:
------------------
We use multiple tests to identify induction heads:

1. **Prefix Matching Score**:
   - Generate sequences with repeated patterns
   - Measure if attention focuses on matching prefixes
   - High score = strong induction behavior

2. **Copying Score**:
   - Test if the head successfully copies from earlier in context
   - Compare predictions on repeated vs. random sequences

3. **Attention Pattern**:
   - Check for diagonal stripe patterns in attention weights
   - Indicates attending to previous occurrences of current token

References:
-----------
- In-context Learning and Induction Heads (Anthropic, 2022)
  https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
- A Mathematical Framework for Transformer Circuits
  https://transformer-circuits.pub/2021/framework/
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import random


class InductionHeadDetector:
    """
    Detector for induction heads in transformer models.

    Induction heads are circuits that copy from context based on pattern matching,
    enabling in-context learning and few-shot capabilities.
    """

    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize induction head detector.

        Args:
            model: DecoderOnlyTransformer model
            tokenizer: Tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    def generate_repeated_sequence(
        self,
        seq_length: int = 50,
        vocab_size: Optional[int] = None
    ) -> Tuple[List[int], List[int]]:
        """
        Generate a sequence with repeated patterns for testing induction.

        Creates: [random tokens] [random tokens again]

        This tests if the model can copy from the first occurrence when
        it sees the pattern repeat.

        Args:
            seq_length: Length of the random portion (total will be 2x)
            vocab_size: Vocabulary size to sample from (default: use tokenizer vocab)

        Returns:
            Tuple of (token_ids, repeat_positions)
            - token_ids: List of token IDs (length = 2 * seq_length)
            - repeat_positions: Indices where repetition starts (for analysis)

        Example:
            tokens = [A, B, C, D, E, A, B, C, D, E]
                      ↑____________↑  Repeated pattern
        """
        if vocab_size is None:
            vocab_size = self.tokenizer.n_vocab

        # Generate random sequence (avoiding special tokens like BOS, EOS)
        # Use tokens from a safe range in the middle of the vocabulary
        min_token = 100
        max_token = min(vocab_size - 100, 10000)

        random_tokens = [random.randint(min_token, max_token) for _ in range(seq_length)]

        # Create repeated sequence
        repeated_tokens = random_tokens + random_tokens

        return repeated_tokens, list(range(seq_length, 2 * seq_length))

    @torch.no_grad()
    def compute_prefix_matching_score(
        self,
        tokens: List[int],
        layer_idx: int,
        head_idx: int
    ) -> float:
        """
        Compute prefix matching score for a head on a repeated sequence.

        Prefix Matching Test:
        --------------------
        For induction heads, when processing position i, they should attend
        to positions where the previous token matches token[i-1].

        Example:
            Sequence: [A, B, C, A, B, ?]
            Position 5 (second B):
                - Previous token: A (position 4)
                - Should attend to position 1 (other occurrence of B)
                - Because position 0 (before first B) also has A

        Args:
            tokens: Token IDs (should be a repeated sequence)
            layer_idx: Layer index to test
            head_idx: Head index to test

        Returns:
            Score between 0 and 1 indicating induction strength
            Higher score = stronger induction behavior

        Algorithm:
            1. Get attention weights for this head
            2. For each position i (in second half):
                a. Find previous token: prev = tokens[i-1]
                b. Find all positions j where tokens[j] == prev
                c. Check if attention[i] focuses on j+1
                d. High attention = good prefix matching
            3. Average across all positions
        """
        # Prepare input
        input_tensor = torch.tensor([tokens], device=self.device)

        # Get attention weights
        logits, caches, attention_weights_list = self.model(
            input_tensor,
            return_attention_weights=True
        )

        # Extract attention for this specific head
        # attention_weights_list[layer_idx] shape: (batch, num_heads, seq_len, seq_len)
        attn = attention_weights_list[layer_idx][0, head_idx]  # (seq_len, seq_len)

        seq_len = len(tokens)
        scores = []

        # Test positions in the second half (where repetition occurs)
        repeat_start = seq_len // 2

        for i in range(repeat_start + 1, seq_len):
            # Get previous token
            prev_token = tokens[i - 1]

            # Find positions where this token appears earlier
            matching_positions = []
            for j in range(i - 1):
                if tokens[j] == prev_token:
                    # Check position j+1 (what comes after the match)
                    if j + 1 < seq_len:
                        matching_positions.append(j + 1)

            if matching_positions:
                # Measure attention to matching positions
                attn_to_matches = sum(attn[i, j].item() for j in matching_positions)
                scores.append(attn_to_matches)

        if scores:
            return np.mean(scores)
        else:
            return 0.0

    @torch.no_grad()
    def compute_copying_score(
        self,
        tokens: List[int]
    ) -> Dict[int, Dict[int, float]]:
        """
        Compute copying score for all heads on a repeated sequence.

        Copying Test:
        ------------
        For a repeated sequence [A, B, C, A, B, C], induction heads should
        predict the second half correctly by copying from the first half.

        We compare:
        - Loss on repeated sequence (should be low)
        - Loss on random sequence (should be high)

        Heads with lower loss on repeated sequences are likely induction heads.

        Args:
            tokens: Repeated sequence of token IDs

        Returns:
            Dictionary mapping layer -> head -> copying_score
            Higher score = better copying performance

        Note:
            This is a simpler metric than prefix matching but very effective.
        """
        # This would require multiple forward passes and loss computation
        # For now, we'll rely on prefix matching score as the main metric
        # This is a placeholder for potential future enhancement
        pass

    def detect(
        self,
        num_sequences: int = 50,
        seq_length: int = 40
    ) -> List[Dict[str, Any]]:
        """
        Detect induction heads across all layers using multiple test sequences.

        Process:
        1. Generate multiple repeated sequences
        2. For each head in each layer:
           a. Compute prefix matching score on all sequences
           b. Average scores across sequences
        3. Rank heads by score
        4. Return top candidates

        Args:
            num_sequences: Number of random sequences to test
            seq_length: Length of each random portion (total = 2x)

        Returns:
            List of dicts sorted by score (highest first):
                - 'layer': Layer index
                - 'head': Head index
                - 'score': Induction score (0-1)
                - 'pattern_type': 'induction' if score > threshold

        Example:
            detector = InductionHeadDetector(model, tokenizer)
            results = detector.detect(num_sequences=100)

            for result in results[:5]:  # Top 5
                print(f"Layer {result['layer']}, Head {result['head']}: "
                      f"score = {result['score']:.2%}")
        """
        from src.transformer.interpretability import AttentionAnalyzer

        # Get model configuration
        num_layers = self.model.num_layers
        num_heads = self.model.blocks[0].attention.num_heads

        # Storage for scores
        all_scores = {}  # (layer, head) -> list of scores

        print(f"Testing {num_layers} layers × {num_heads} heads on {num_sequences} sequences...")

        for seq_idx in range(num_sequences):
            # Generate test sequence
            tokens, _ = self.generate_repeated_sequence(seq_length)

            # Test all heads
            for layer_idx in range(num_layers):
                for head_idx in range(num_heads):
                    key = (layer_idx, head_idx)

                    # Compute score for this head on this sequence
                    score = self.compute_prefix_matching_score(
                        tokens, layer_idx, head_idx
                    )

                    if key not in all_scores:
                        all_scores[key] = []
                    all_scores[key].append(score)

            # Progress indicator
            if (seq_idx + 1) % 10 == 0:
                print(f"  Completed {seq_idx + 1}/{num_sequences} sequences")

        # Average scores across all sequences
        results = []
        for (layer_idx, head_idx), scores in all_scores.items():
            avg_score = np.mean(scores)

            # Classify as induction head if score is high
            pattern_type = "induction" if avg_score > 0.5 else "non-induction"

            results.append({
                'layer': layer_idx,
                'head': head_idx,
                'score': avg_score,
                'pattern_type': pattern_type,
            })

        # Sort by score (highest first)
        results.sort(key=lambda x: x['score'], reverse=True)

        return results

    def analyze_induction_circuit(
        self,
        text: str,
        induction_layer: int,
        induction_head: int,
        prev_token_layer: Optional[int] = None,
        prev_token_head: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Analyze a potential induction circuit (pair of heads working together).

        The typical induction circuit has:
        1. Previous token head in earlier layer
        2. Induction head in later layer

        Args:
            text: Text to analyze
            induction_layer: Layer of the induction head
            induction_head: Head index of the induction head
            prev_token_layer: Layer of previous token head (None = auto-detect)
            prev_token_head: Head index of previous token head (None = auto-detect)

        Returns:
            Dictionary with circuit analysis:
                - 'induction_head': (layer, head)
                - 'prev_token_head': (layer, head) or None
                - 'circuit_strength': How well they work together
                - 'explanation': Human-readable description

        Example:
            circuit = detector.analyze_induction_circuit(
                "A B C A B",
                induction_layer=1,
                induction_head=3
            )
            print(circuit['explanation'])
        """
        from src.transformer.interpretability import AttentionAnalyzer

        analyzer = AttentionAnalyzer(self.model, self.tokenizer)

        # Auto-detect previous token head if not specified
        if prev_token_layer is None or prev_token_head is None:
            # Find previous token heads in earlier layers
            prev_heads = analyzer.find_heads_by_pattern(
                text, "previous_token", threshold=0.3
            )

            # Filter to layers before the induction head
            prev_heads = [h for h in prev_heads if h['layer'] < induction_layer]

            if prev_heads:
                best = prev_heads[0]
                prev_token_layer = best['layer']
                prev_token_head = best['head']

        if prev_token_layer is None:
            return {
                'induction_head': (induction_layer, induction_head),
                'prev_token_head': None,
                'circuit_strength': 0.0,
                'explanation': "No previous token head found to complete the circuit."
            }

        # Analyze both heads
        results = analyzer.analyze(text)

        # Get attention patterns
        prev_attn = results['attention_weights'][prev_token_layer][prev_token_head]
        ind_attn = results['attention_weights'][induction_layer][induction_head]

        # Check if previous token head is actually a previous token head
        prev_pattern = analyzer.get_head_pattern_type(prev_attn)

        # Compute circuit strength (simplified metric)
        if prev_pattern == "previous_token":
            circuit_strength = 0.8  # Strong evidence
        else:
            circuit_strength = 0.3  # Weak evidence

        explanation = (
            f"Potential induction circuit:\n"
            f"  1. Layer {prev_token_layer}, Head {prev_token_head}: {prev_pattern} pattern\n"
            f"  2. Layer {induction_layer}, Head {induction_head}: induction pattern\n"
            f"  Circuit strength: {circuit_strength:.2f}"
        )

        return {
            'induction_head': (induction_layer, induction_head),
            'prev_token_head': (prev_token_layer, prev_token_head),
            'circuit_strength': circuit_strength,
            'explanation': explanation,
        }
