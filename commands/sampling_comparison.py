"""
Demonstration of different sampling strategies for text generation.

This script compares various sampling methods:
- Greedy decoding
- Basic multinomial sampling
- Top-k sampling
- Top-p (nucleus) sampling
- Combined top-k + top-p

Each method is demonstrated with different temperature settings to show
how they affect text generation quality and diversity.

What You'll Learn:
------------------
1. How different sampling strategies affect output diversity
2. The impact of temperature on generation
3. Trade-offs between quality and creativity
4. When to use each sampling method

Run this script with a trained model to see the differences!
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.sampling import (
    sample_greedy,
    sample_top_k,
    sample_top_p,
    sample_top_k_top_p,
)
from src.transformer.checkpoint_utils import load_checkpoint


def demonstrate_sampling_strategies():
    """
    Demonstrate different sampling strategies with a simple example.

    Since we may not have a trained model yet, this creates a simple
    scenario where we manually define probability distributions to show
    how each sampling method behaves.
    """
    print("=" * 70)
    print("TEXT GENERATION SAMPLING STRATEGIES DEMONSTRATION")
    print("=" * 70)
    print()

    # Create example logits representing model output for next token
    # Imagine these are scores for tokens: ["cat", "dog", "sat", "ran", "the", "purple", ...]
    # Higher logits = higher probability

    # Scenario 1: Confident model (one token much more likely)
    print("SCENARIO 1: Confident Model")
    print("-" * 70)
    print("The model is very confident about what comes next.")
    print("Logits: [10.0, 5.0, 2.0, 1.0, 0.5, 0.1, ...]")
    print()

    confident_logits = torch.tensor([10.0, 5.0, 2.0, 1.0, 0.5, 0.1, 0.05, 0.01])

    # Show probabilities
    probs = torch.softmax(confident_logits, dim=-1)
    print("Probabilities after softmax:")
    for i, p in enumerate(probs):
        print(f"  Token {i}: {p.item():.4f} ({p.item() * 100:.2f}%)")
    print()

    # Test different sampling methods
    print("Sampling results (20 samples each):")
    print()

    # Greedy
    samples = [sample_greedy(confident_logits).item() for _ in range(20)]
    print(f"Greedy:       {samples}")
    print(f"  → Always selects token 0 (deterministic)")
    print()

    # Top-k with k=3
    samples = [sample_top_k(confident_logits, k=3).item() for _ in range(20)]
    unique = len(set(samples))
    print(f"Top-k (k=3):  {samples}")
    print(f"  → Only samples from top-3 tokens (0, 1, 2)")
    print(f"  → Unique tokens: {unique}")
    print()

    # Top-p with p=0.95
    samples = [sample_top_p(confident_logits, p=0.95).item() for _ in range(20)]
    unique = len(set(samples))
    print(f"Top-p (p=0.95): {samples}")
    print(f"  → Adapts nucleus size to confidence")
    print(f"  → Unique tokens: {unique}")
    print()

    print("\n")

    # Scenario 2: Uncertain model (many tokens roughly equal)
    print("SCENARIO 2: Uncertain Model")
    print("-" * 70)
    print("The model is uncertain - many tokens seem equally likely.")
    print("Logits: [2.0, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2]")
    print()

    uncertain_logits = torch.tensor([2.0, 1.8, 1.7, 1.6, 1.5, 1.4, 1.3, 1.2])

    # Show probabilities
    probs = torch.softmax(uncertain_logits, dim=-1)
    print("Probabilities after softmax:")
    for i, p in enumerate(probs):
        print(f"  Token {i}: {p.item():.4f} ({p.item() * 100:.2f}%)")
    print()

    print("Sampling results (20 samples each):")
    print()

    # Greedy
    samples = [sample_greedy(uncertain_logits).item() for _ in range(20)]
    print(f"Greedy:         {samples}")
    print(f"  → Still deterministic, always token 0")
    print()

    # Top-k with k=3
    samples = [sample_top_k(uncertain_logits, k=3).item() for _ in range(20)]
    unique = len(set(samples))
    print(f"Top-k (k=3):    {samples}")
    print(f"  → More diverse than with confident model")
    print(f"  → Unique tokens: {unique}")
    print()

    # Top-p with p=0.95
    samples = [sample_top_p(uncertain_logits, p=0.95).item() for _ in range(20)]
    unique = len(set(samples))
    print(f"Top-p (p=0.95): {samples}")
    print(f"  → Larger nucleus due to uncertainty")
    print(f"  → Unique tokens: {unique}")
    print()

    # Combined
    samples = [sample_top_k_top_p(uncertain_logits, k=5, p=0.9).item()
               for _ in range(20)]
    unique = len(set(samples))
    print(f"Combined (k=5, p=0.9): {samples}")
    print(f"  → Best of both: filters tail + adaptive")
    print(f"  → Unique tokens: {unique}")
    print()

    print("\n")

    # Scenario 3: Temperature effects
    print("SCENARIO 3: Temperature Effects")
    print("-" * 70)
    print("Same logits, different temperatures")
    print("Logits: [5.0, 4.0, 3.0, 2.0, 1.0]")
    print()

    temp_logits = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])

    temperatures = [0.5, 1.0, 2.0]

    for temp in temperatures:
        print(f"Temperature = {temp}:")

        # Show how probabilities change
        probs = torch.softmax(temp_logits / temp, dim=-1)
        print("  Probabilities:", [f"{p.item():.3f}" for p in probs])

        # Sample
        samples = [sample_top_k(temp_logits, k=5, temperature=temp).item()
                   for _ in range(20)]
        unique = len(set(samples))

        # Count frequency of most common token
        most_common_count = max([samples.count(i) for i in range(5)])

        print(f"  Samples: {samples}")
        print(f"  Unique tokens: {unique}, "
              f"Most common appears: {most_common_count}/20 times")
        print()

    print("Observations:")
    print("  - Low temperature (0.5): More focused, less diverse")
    print("  - High temperature (2.0): More uniform, more diverse")
    print()

    print("\n")

    # Summary recommendations
    print("=" * 70)
    print("RECOMMENDATIONS FOR TEXT GENERATION")
    print("=" * 70)
    print()

    print("Use Case                    | Strategy      | Settings")
    print("-" * 70)
    print("Factual/Technical content   | top_k_top_p   | k=40, p=0.85, temp=0.7")
    print("Balanced general purpose    | top_k_top_p   | k=50, p=0.9,  temp=1.0")
    print("Creative writing            | top_k_top_p   | k=100, p=0.95, temp=1.2")
    print("Chatbot (coherent)          | top_p         | p=0.9, temp=0.8")
    print("Code generation             | top_k         | k=20, temp=0.6")
    print("Deterministic output        | greedy        | (no randomness)")
    print()

    print("Key Insights:")
    print("  1. Greedy: Fast and deterministic, but repetitive")
    print("  2. Top-k: Good baseline, but doesn't adapt to confidence")
    print("  3. Top-p: Adaptive nucleus, more natural")
    print("  4. Combined: Best quality - filters tail + adapts to confidence")
    print("  5. Temperature: Fine-tune diversity (lower=focused, higher=creative)")
    print()


def demonstrate_with_model():
    """
    Demonstrate sampling with an actual model.

    This requires a trained model. If you haven't trained one yet,
    this will show you how to use the sampling methods once you have one.
    """
    print("=" * 70)
    print("DEMONSTRATION WITH ACTUAL MODEL")
    print("=" * 70)
    print()

    print("NOTE: This demonstration requires a trained model.")
    print("      If you don't have one yet, train using: uv run python main.py train")
    print()

    # Check if a trained model exists
    import os
    model_path = "transformer_model.pt"

    if not os.path.exists(model_path):
        print(f"No trained model found at '{model_path}'")
        print("Skipping model demonstration.")
        print()
        return

    print(f"Loading model from '{model_path}'...")

    # Load model using checkpoint utilities
    result = load_checkpoint(model_path, device=torch.device('cpu'), verbose=False)
    model = result['model']
    config = result['config']
    vocab_size = config['vocab_size']

    print("Model loaded successfully!")
    print()

    # Example starting sequence (you'll need to tokenize your actual input)
    # For now, use random tokens as example
    start_tokens = torch.randint(0, vocab_size, (1, 5))

    print("Starting tokens:", start_tokens.tolist())
    print()

    # Generate with different strategies
    strategies = [
        ("Greedy", "greedy", {}),
        ("Multinomial", "multinomial", {"temperature": 1.0}),
        ("Top-k", "top_k", {"top_k": 50, "temperature": 0.8}),
        ("Top-p", "top_p", {"top_p": 0.9, "temperature": 0.8}),
        ("Combined", "top_k_top_p", {"top_k": 50, "top_p": 0.9, "temperature": 0.8}),
    ]

    max_length = 20

    for name, strategy, kwargs in strategies:
        print(f"{name} sampling:")

        generated = model.generate(
            start_tokens,
            max_length=max_length,
            sampling_strategy=strategy,
            **kwargs
        )

        print(f"  Generated tokens: {generated[0].tolist()}")
        print()

    print("NOTE: To see actual text, you need to:")
    print("  1. Use a proper tokenizer (e.g., tiktoken)")
    print("  2. Decode the generated token IDs to text")
    print()


if __name__ == "__main__":
    # First demonstrate sampling behavior without a model
    demonstrate_sampling_strategies()

    print("\n" * 2)

    # Then demonstrate with actual model if available
    demonstrate_with_model()

    print("=" * 70)
    print("DEMONSTRATION COMPLETE")
    print("=" * 70)
    print()
    print("To learn more:")
    print("  - Read src/transformer/sampling.py for implementation details")
    print("  - Check tests/test_sampling.py for edge cases and validation")
    print("  - Experiment with different parameters on your own data!")
    print()
