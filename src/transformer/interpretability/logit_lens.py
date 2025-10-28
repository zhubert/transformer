"""
Logit Lens: Visualizing Predictions at Each Layer

The logit lens technique lets us "peek inside" the transformer to see what the
model is "thinking" at different depths. This reveals how predictions evolve
as information flows through layers.

What is Logit Lens?
-------------------
Normally, we only see the final output after all layers:
    Input → Layer 1 → Layer 2 → ... → Layer N → Unembed → Logits

But what if we could ask: "What would the model predict if it stopped at Layer 3?"

The logit lens does exactly this by applying the unembedding (final projection)
to the hidden states after each layer:

    Input → Layer 1 → [Unembed] → "What does Layer 1 think?"
         → Layer 2 → [Unembed] → "What does Layer 2 think?"
         → Layer 3 → [Unembed] → "What does Layer 3 think?"

Why is This Useful?
-------------------
1. **Understanding emergence**: See when the correct answer first appears
   - Early layers might predict generic tokens ("the", "a")
   - Middle layers start converging on semantic concepts
   - Final layers refine to the specific answer

2. **Debugging**: If the model gets something wrong, find which layer went astray

3. **Circuit discovery**: Identify which layers are crucial for specific tasks

Example Insight:
----------------
Input: "The capital of France is"

Layer 0: Predicts "the" (15%), "a" (12%) - just copying common words
Layer 2: Predicts "located" (20%), "Paris" (18%) - starting to understand
Layer 4: Predicts "Paris" (65%), "French" (10%) - confident, correct!
Layer 6: Predicts "Paris" (72%), "France" (8%) - final refinement

This shows the model "knows" Paris by Layer 4, and later layers just refine.

Technical Details:
------------------
The logit lens applies the unembedding matrix (W_unembed) to hidden states:

    logits_at_layer_i = hidden_states[i] @ W_unembed.T

Then we can compute probabilities:
    probs = softmax(logits_at_layer_i)

And see the top-k predictions at each layer.

Note: Tuned Lens is a refinement where we learn a small linear transformation
for each layer, since intermediate representations might not be perfectly
aligned with the output space. We implement the simpler logit lens here.

References:
-----------
- Original LessWrong post: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/
- nostalgebraist analysis: https://www.alignmentforum.org/posts/...
"""

import torch
import torch.nn as nn
from typing import List, Tuple, Dict, Any


class LogitLens:
    """
    Logit lens for analyzing predictions at each transformer layer.

    This tool helps understand how the model's predictions evolve through
    the forward pass by applying the unembedding at each layer.
    """

    def __init__(self, model: nn.Module, tokenizer):
        """
        Initialize logit lens with a model and tokenizer.

        Args:
            model: DecoderOnlyTransformer model
            tokenizer: Tokenizer for decoding token IDs to strings
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = next(model.parameters()).device

    @torch.no_grad()
    def analyze(
        self,
        text: str,
        position: int = -1,
        top_k: int = 5,
        temperature: float = 1.0
    ) -> Dict[str, Any]:
        """
        Analyze predictions at each layer for a given input text.

        Process:
        1. Tokenize input text
        2. Run forward pass with return_hidden_states=True
        3. For each layer's hidden state:
           a. Apply unembedding (output projection)
           b. Compute probabilities
           c. Extract top-k predictions
        4. Return predictions for each layer

        Args:
            text: Input text to analyze
            position: Which position to analyze predictions for
                     -1 (default) = last position (most common for next-token prediction)
                     0 = first position, 1 = second position, etc.
            top_k: Number of top predictions to return per layer (default: 5)
            temperature: Temperature for probability scaling (default: 1.0)
                        Lower = more confident, Higher = more uniform

        Returns:
            Dictionary containing:
                - 'input_text': Original input text
                - 'tokens': List of token strings
                - 'token_ids': List of token IDs
                - 'position': Position being analyzed
                - 'layer_predictions': List of predictions for each layer
                    Each element is list of (token_str, probability) tuples
                - 'layer_names': Names for each layer

        Example:
            lens = LogitLens(model, tokenizer)
            results = lens.analyze("The capital of France is")

            for layer_name, preds in zip(results['layer_names'], results['layer_predictions']):
                print(f"{layer_name}:")
                for token, prob in preds[:3]:
                    print(f"  {token}: {prob:.2%}")
        """
        # Set model to evaluation mode
        self.model.eval()

        # Tokenize input
        token_ids = self.tokenizer.encode(text)
        tokens = [self.tokenizer.decode([tid]) for tid in token_ids]
        input_tensor = torch.tensor([token_ids], device=self.device)

        # Get hidden states at each layer
        # return_hidden_states=True means we get (logits, caches, hidden_states)
        logits, caches, hidden_states = self.model(
            input_tensor,
            return_hidden_states=True
        )

        # Handle position indexing
        seq_len = input_tensor.size(1)
        if position < 0:
            position = seq_len + position  # Convert -1 to last position

        # Get the final layer norm and output projection from the model
        # These are used to convert hidden states to vocabulary logits
        final_ln = self.model.ln_f
        output_proj = self.model.output_proj

        # Analyze predictions at each layer
        layer_predictions = []
        layer_names = []

        for layer_idx, hidden_state in enumerate(hidden_states):
            # hidden_state shape: (batch, seq_len, d_model)
            # We want to analyze position `position`

            # Extract hidden state at the target position
            h = hidden_state[0, position, :]  # (d_model,)

            # Apply final layer norm (in Pre-LN, this is important!)
            # The hidden states are outputs of residual blocks, but the final LN
            # hasn't been applied yet
            h_normalized = final_ln(h)  # (d_model,)

            # Apply output projection to get logits
            layer_logits = output_proj(h_normalized)  # (vocab_size,)

            # Apply temperature scaling
            layer_logits = layer_logits / temperature

            # Get probabilities
            probs = torch.softmax(layer_logits, dim=-1)

            # Get top-k predictions
            top_probs, top_indices = torch.topk(probs, k=top_k)

            # Convert to token strings
            predictions = []
            for prob, idx in zip(top_probs.cpu(), top_indices.cpu()):
                token_str = self.tokenizer.decode([idx.item()])
                predictions.append((token_str, prob.item()))

            layer_predictions.append(predictions)
            layer_names.append(f"Layer {layer_idx}")

        # Also add the final output for comparison
        final_logits = logits[0, position, :] / temperature
        final_probs = torch.softmax(final_logits, dim=-1)
        top_probs, top_indices = torch.topk(final_probs, k=top_k)
        final_predictions = [
            (self.tokenizer.decode([idx.item()]), prob.item())
            for prob, idx in zip(top_probs.cpu(), top_indices.cpu())
        ]
        layer_predictions.append(final_predictions)
        layer_names.append("Final Output")

        return {
            'input_text': text,
            'tokens': tokens,
            'token_ids': token_ids,
            'position': position,
            'layer_predictions': layer_predictions,
            'layer_names': layer_names,
        }

    def compare_positions(
        self,
        text: str,
        positions: List[int],
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Compare predictions across multiple positions to see how context affects predictions.

        Educational Purpose:
            Shows how the model's predictions for different positions evolve
            through layers. Useful for understanding contextual effects.

        Args:
            text: Input text
            positions: List of positions to analyze
            top_k: Number of predictions per position

        Returns:
            Dictionary with results for each position

        Example:
            # See how predictions for positions 3 and 5 differ
            results = lens.compare_positions("The quick brown fox", positions=[3, 5])
        """
        results = {}
        for pos in positions:
            results[pos] = self.analyze(text, position=pos, top_k=top_k)
        return results

    def find_convergence_layer(
        self,
        text: str,
        target_token: str,
        position: int = -1,
        threshold: float = 0.5
    ) -> Tuple[int, float]:
        """
        Find the first layer where the model predicts a target token with high confidence.

        Educational Purpose:
            Helps answer: "At which layer does the model 'know' the answer?"
            This is useful for understanding when information is computed.

        Args:
            text: Input text
            target_token: The token to look for
            position: Position to analyze
            threshold: Probability threshold for "confident" (default: 0.5)

        Returns:
            Tuple of (layer_index, probability)
            Returns (-1, 0.0) if threshold never reached

        Example:
            # When does the model know the answer is "Paris"?
            layer, prob = lens.find_convergence_layer(
                "The capital of France is",
                target_token=" Paris",
                threshold=0.6
            )
            print(f"Model confidently predicts 'Paris' starting at layer {layer}")
        """
        results = self.analyze(text, position=position, top_k=10)

        for layer_idx, predictions in enumerate(results['layer_predictions']):
            # Check if target token is in top predictions
            for token_str, prob in predictions:
                if token_str == target_token and prob >= threshold:
                    return layer_idx, prob

        # Not found
        return -1, 0.0
