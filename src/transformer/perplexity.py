"""
Perplexity calculation for language model evaluation.

What is Perplexity?
-------------------
Perplexity is a metric that measures how well a language model predicts text.
It answers the question: "How confused is the model by this text?"

Intuition:
----------
Think of perplexity as the "effective vocabulary size" the model is choosing from:

- Perplexity = 1: Perfect! The model knows exactly what word comes next
- Perplexity = 10: The model is as confused as if randomly picking from 10 words
- Perplexity = 100: Very confused, like picking from 100 equally likely words
- Perplexity = vocab_size: Totally lost, random guessing

Real-world examples:
    "The capital of France is ___"
    → A good model assigns high probability to "Paris"
    → Low perplexity (confident and correct)

    "asdfgh qwerty zxcvbn ___"
    → Model has no idea what comes next
    → High perplexity (confused)

Mathematical Definition:
------------------------
Perplexity is the exponential of the average negative log-likelihood (cross-entropy):

    Perplexity = exp(CrossEntropyLoss)

    = exp(-1/N * Σ log P(actual_word_i | context))

    where:
    - N = number of tokens
    - P(actual_word_i | context) = probability model assigns to the correct word

Why Exponential?
----------------
1. Makes the metric more interpretable
2. Converts log-space loss to linear space
3. Perplexity ~50 means "model is as confused as choosing from ~50 words"

Example Calculation:
--------------------
Suppose we have 3 tokens to predict and the model assigns these probabilities
to the correct tokens:

    Token 1: P("cat") = 0.5     → log(0.5) = -0.693
    Token 2: P("sat") = 0.25    → log(0.25) = -1.386
    Token 3: P("mat") = 0.125   → log(0.125) = -2.079

    Average negative log-likelihood:
    = -1/3 * (-0.693 + -1.386 + -2.079)
    = -1/3 * (-4.158)
    = 1.386

    Perplexity = exp(1.386) = 4.0

This means on average, the model was as confused as if it had to choose
uniformly from 4 words.

Relationship to Cross-Entropy Loss:
------------------------------------
Cross-entropy loss is exactly the average negative log-likelihood:

    Loss = -1/N * Σ log P(actual_word_i | context)
    Perplexity = exp(Loss)

So if your training loss is 3.0, your perplexity is exp(3.0) ≈ 20.

This means: "The model is as confused as if choosing from ~20 words"

Typical Values:
---------------
- Perfect model: Perplexity = 1.0 (always correct with 100% confidence)
- Excellent model: Perplexity = 10-30 (GPT-2 level on good text)
- Decent model: Perplexity = 50-100
- Poor model: Perplexity = 200+
- Random guessing: Perplexity = vocab_size (e.g., 50,000)

Why Lower is Better:
--------------------
Lower perplexity means:
- Model is more confident in its predictions
- Model assigns higher probability to correct words
- Model has learned the patterns in the language better

Training Progress Example:
--------------------------
    Epoch 1: Loss = 8.0  → Perplexity = 2981 (terrible, random)
    Epoch 5: Loss = 4.0  → Perplexity = 55   (getting better)
    Epoch 10: Loss = 3.0 → Perplexity = 20   (pretty good!)
    Epoch 20: Loss = 2.5 → Perplexity = 12   (excellent!)

Perplexity vs. Other Metrics:
------------------------------
- **Loss**: More sensitive to small changes, used for optimization
- **Perplexity**: More interpretable, better for comparing models
- **Accuracy**: Only cares about top-1 prediction (too coarse)
- **Perplexity**: Considers the full probability distribution (richer)

Use Cases:
----------
1. Track training progress (should decrease over time)
2. Compare different model architectures
3. Evaluate on validation/test sets
4. Detect overfitting (train perplexity << val perplexity)
5. Compare to other models in literature
"""

import torch
import torch.nn as nn


def calculate_perplexity(logits, targets, ignore_index=-100):
    """
    Calculate perplexity from model logits and target tokens.

    Perplexity measures how well the model predicts the text. Lower is better.
    It's the exponential of the average cross-entropy loss.

    Args:
        logits: Model output logits of shape (batch, seq_len, vocab_size)
                These are the raw scores before softmax
        targets: Target token IDs of shape (batch, seq_len)
                These are the correct next tokens we want to predict
        ignore_index: Token ID to ignore in loss calculation (e.g., padding)
                     Default -100 matches PyTorch's CrossEntropyLoss default

    Returns:
        perplexity: Scalar tensor with perplexity value

    Mathematical steps:
        1. Calculate cross-entropy loss (average negative log-likelihood)
        2. Take exponential: perplexity = exp(loss)

    Example:
        >>> # Suppose we have 2 sequences of length 3, vocab size 1000
        >>> logits = torch.randn(2, 3, 1000)
        >>> targets = torch.randint(0, 1000, (2, 3))
        >>> perplexity = calculate_perplexity(logits, targets)
        >>> print(f"Perplexity: {perplexity.item():.2f}")
        Perplexity: 987.23  # High because model is random/untrained!

    Why this works:
        - CrossEntropyLoss computes: -log(P(correct_token))
        - Averaged over all tokens
        - Exponentiating gives us perplexity

    Interpretation:
        If perplexity = 50, the model is as confused as if it had to
        choose uniformly from 50 words at each step.
    """
    # Create loss function (CrossEntropyLoss computes average negative log-likelihood)
    criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    # Reshape for loss calculation
    # CrossEntropyLoss expects: (batch_size * seq_len, vocab_size) and (batch_size * seq_len)
    batch_size, seq_len, vocab_size = logits.shape
    logits_flat = logits.view(batch_size * seq_len, vocab_size)
    targets_flat = targets.view(batch_size * seq_len)

    # Calculate cross-entropy loss (average negative log-likelihood)
    loss = criterion(logits_flat, targets_flat)

    # Perplexity = exp(loss)
    # This converts from log-space to linear space for interpretability
    perplexity = torch.exp(loss)

    return perplexity


def calculate_perplexity_from_loss(loss):
    """
    Calculate perplexity directly from cross-entropy loss.

    This is a convenience function when you already have the loss computed.
    Since perplexity = exp(loss), this is a simple exponential.

    Args:
        loss: Cross-entropy loss (scalar or tensor)
              This is the average negative log-likelihood

    Returns:
        perplexity: Scalar with perplexity value

    Example:
        >>> loss = torch.tensor(3.0)
        >>> perplexity = calculate_perplexity_from_loss(loss)
        >>> print(f"Loss: {loss:.2f}, Perplexity: {perplexity:.2f}")
        Loss: 3.00, Perplexity: 20.09

    Interpretation:
        A loss of 3.0 means the model is as confused as choosing from ~20 words.
    """
    # Perplexity = exp(cross_entropy_loss)
    perplexity = torch.exp(loss)
    return perplexity


def evaluate_perplexity(model, dataloader, device='cpu', max_batches=None):
    """
    Evaluate perplexity of a model on a dataset.

    This function runs the model on a dataset and computes the average perplexity.
    Useful for evaluating on validation/test sets.

    Args:
        model: The transformer model to evaluate
        dataloader: DataLoader providing (input, target) batches
        device: Device to run evaluation on ('cpu', 'cuda', or 'mps')
        max_batches: If set, only evaluate on first N batches (for quick checks)

    Returns:
        avg_perplexity: Average perplexity across all batches
        avg_loss: Average cross-entropy loss across all batches

    Example:
        >>> from torch.utils.data import DataLoader
        >>> # Assume we have a model and dataset
        >>> val_loader = DataLoader(val_dataset, batch_size=8)
        >>> perplexity, loss = evaluate_perplexity(model, val_loader, device='cuda')
        >>> print(f"Validation Perplexity: {perplexity:.2f}")
        >>> print(f"Validation Loss: {loss:.4f}")

    Use Cases:
        1. Check validation perplexity during training
        2. Final evaluation on test set
        3. Compare different model checkpoints
        4. Detect overfitting (train vs. val perplexity)

    Training vs. Validation Perplexity:
        - Train perplexity: How well model fits training data
        - Val perplexity: How well model generalizes to new data
        - If val_perplexity >> train_perplexity: Overfitting!
        - Good models: val_perplexity ≈ train_perplexity (maybe slightly higher)
    """
    model.eval()  # Set to evaluation mode (disables dropout, etc.)

    total_loss = 0.0
    total_batches = 0

    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():  # Don't compute gradients (saves memory and time)
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Stop early if max_batches specified (useful for quick checks)
            if max_batches is not None and batch_idx >= max_batches:
                break

            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
            logits = model(inputs)

            # Calculate loss
            batch_size, seq_len, vocab_size = logits.shape
            loss = criterion(
                logits.view(batch_size * seq_len, vocab_size),
                targets.view(batch_size * seq_len)
            )

            total_loss += loss.item()
            total_batches += 1

    # Calculate averages
    avg_loss = total_loss / total_batches
    avg_perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()  # Set back to training mode

    return avg_perplexity, avg_loss


if __name__ == "__main__":
    """
    Demo: How perplexity changes with model confidence.

    This demonstrates the relationship between probabilities, loss, and perplexity.
    """
    print("=" * 80)
    print("PERPLEXITY DEMONSTRATION")
    print("=" * 80)
    print()

    # Example: Predicting 3 tokens with different confidence levels
    print("Scenario: Model predicting 3 tokens")
    print("-" * 80)

    # Case 1: Very confident (good model)
    print("\n1. CONFIDENT MODEL (assigns high probabilities to correct tokens):")
    probs = torch.tensor([0.8, 0.7, 0.9])  # High probabilities
    log_probs = torch.log(probs)
    avg_nll = -log_probs.mean()  # Negative log-likelihood
    perplexity = torch.exp(avg_nll)
    print(f"   Probabilities: {probs.tolist()}")
    print(f"   Average loss: {avg_nll.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f}")
    print(f"   → Low perplexity! Model is confident and correct.")

    # Case 2: Somewhat confident (okay model)
    print("\n2. MODERATE MODEL (medium confidence):")
    probs = torch.tensor([0.5, 0.4, 0.6])
    log_probs = torch.log(probs)
    avg_nll = -log_probs.mean()
    perplexity = torch.exp(avg_nll)
    print(f"   Probabilities: {probs.tolist()}")
    print(f"   Average loss: {avg_nll.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f}")
    print(f"   → Medium perplexity. Model is uncertain.")

    # Case 3: Low confidence (bad model)
    print("\n3. CONFUSED MODEL (low probabilities, like random guessing):")
    probs = torch.tensor([0.1, 0.05, 0.15])  # Low probabilities
    log_probs = torch.log(probs)
    avg_nll = -log_probs.mean()
    perplexity = torch.exp(avg_nll)
    print(f"   Probabilities: {probs.tolist()}")
    print(f"   Average loss: {avg_nll.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f}")
    print(f"   → High perplexity! Model is very confused.")

    # Case 4: Perfect model
    print("\n4. PERFECT MODEL (100% confidence, always correct):")
    probs = torch.tensor([1.0, 1.0, 1.0])
    log_probs = torch.log(probs)
    avg_nll = -log_probs.mean()
    perplexity = torch.exp(avg_nll)
    print(f"   Probabilities: {probs.tolist()}")
    print(f"   Average loss: {avg_nll.item():.4f}")
    print(f"   Perplexity: {perplexity.item():.2f}")
    print(f"   → Perplexity = 1! Perfect predictions!")

    print("\n" + "=" * 80)
    print("KEY INSIGHT: Lower perplexity = Better model")
    print("=" * 80)
    print()

    # Demonstrate with actual logits
    print("Example with actual model logits:")
    print("-" * 80)

    # Simulated logits and targets
    batch_size, seq_len, vocab_size = 2, 5, 100

    # Random model (untrained)
    random_logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    random_perplexity = calculate_perplexity(random_logits, targets)
    print(f"\nRandom untrained model:")
    print(f"  Perplexity: {random_perplexity.item():.2f}")
    print(f"  (Close to vocab_size={vocab_size} because model is guessing randomly)")

    # Better model (create logits that favor correct tokens)
    better_logits = torch.randn(batch_size, seq_len, vocab_size)
    for b in range(batch_size):
        for s in range(seq_len):
            # Give correct token a much higher score
            better_logits[b, s, targets[b, s]] += 5.0

    better_perplexity = calculate_perplexity(better_logits, targets)
    print(f"\nTrained model (higher scores for correct tokens):")
    print(f"  Perplexity: {better_perplexity.item():.2f}")
    print(f"  (Much lower! Model has learned patterns)")

    print()
