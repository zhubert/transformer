# Understanding Perplexity in Language Models

## What is Perplexity?

**Perplexity** is the standard metric for evaluating language models. It measures how "confused" or "perplexed" a model is when predicting text.

### The Intuition

Think of perplexity as answering this question: "On average, how many words does the model have to choose from?"

- **Perplexity = 1**: Perfect! The model always knows exactly what comes next
- **Perplexity = 50**: The model is as confused as if picking uniformly from 50 words
- **Perplexity = 10,000**: Totally lost, like random guessing over entire vocabulary

### Real-World Examples

```
Text: "The capital of France is ___"

Good model:
- Assigns P("Paris") = 0.9
- Low perplexity (~1.1)
- Confident and correct!

Bad model:
- Assigns P("Paris") = 0.01
- High perplexity (~100)
- Very confused!
```

## Mathematical Definition

Perplexity is the exponential of the average cross-entropy loss:

```
Perplexity = exp(CrossEntropyLoss)
         = exp(-1/N * Σ log P(correct_word))
```

Where:
- `N` = number of tokens
- `P(correct_word)` = probability the model assigns to the actual next word

### Example Calculation

Suppose we predict 3 tokens:

```
Token 1: "cat"  → Model assigns P("cat") = 0.5
Token 2: "sat"  → Model assigns P("sat") = 0.25
Token 3: "mat"  → Model assigns P("mat") = 0.125

Step 1: Calculate negative log-likelihoods
  -log(0.5) = 0.693
  -log(0.25) = 1.386
  -log(0.125) = 2.079

Step 2: Average them
  Average = (0.693 + 1.386 + 2.079) / 3 = 1.386

Step 3: Take exponential
  Perplexity = exp(1.386) = 4.0
```

**Interpretation**: On average, the model was as confused as choosing from 4 words.

## Typical Perplexity Values

| Perplexity | Quality | Description |
|------------|---------|-------------|
| 1.0 | Perfect | Always 100% confident and correct (impossible in practice) |
| 10-30 | Excellent | GPT-2 level performance on good text |
| 50-100 | Decent | Model has learned patterns, room for improvement |
| 200+ | Poor | Model is quite confused, needs more training |
| ~vocab_size | Random | Model is just guessing randomly |

### Training Progress Example

```
Epoch 1:  Loss = 8.0  → Perplexity = 2981  (random guessing)
Epoch 5:  Loss = 4.0  → Perplexity = 55    (learning patterns)
Epoch 10: Loss = 3.0  → Perplexity = 20    (pretty good!)
Epoch 20: Loss = 2.5  → Perplexity = 12    (excellent!)
```

## Using Perplexity in This Project

### During Training

The training script (`examples/train.py`) now automatically tracks perplexity:

```bash
uv run python examples/train.py
```

Output:
```
Batch 10/100, Loss: 3.5, Perplexity: 33.12, Avg Loss: 4.2, Avg Perplexity: 66.69, LR: 0.000300

Epoch 1 complete!
  Average Loss: 4.1234
  Average Perplexity: 61.89
  Time: 45.2s
```

### Evaluating a Trained Model

Use the evaluation script to test your model:

```bash
# Evaluate latest checkpoint
uv run python examples/evaluate_perplexity.py

# Evaluate specific checkpoint
uv run python examples/evaluate_perplexity.py --checkpoint checkpoints/model_epoch_10.pt

# Compare all checkpoints
uv run python examples/evaluate_perplexity.py --compare

# Specify evaluation text
uv run python examples/evaluate_perplexity.py --text-file my_test_data.txt
```

Example output:
```
================================================================================
EVALUATION RESULTS
================================================================================
Loss: 2.8543
Perplexity: 17.37

What does this mean?
--------------------------------------------------------------------------------
EXCELLENT! Perplexity 17.37 is GPT-2 level performance.
The model has learned the language patterns very well.

Interpretation: On average, the model is as confused as if it had to
choose uniformly from ~17 words at each step.
```

### Comparing Checkpoints

```bash
uv run python examples/evaluate_perplexity.py --compare
```

Output:
```
================================================================================
COMPARISON SUMMARY
================================================================================

Checkpoint                Epoch    Train PPL    Eval PPL     Status
--------------------------------------------------------------------------------
model_epoch_15.pt         15       18.23        19.45        ★ BEST
model_epoch_14.pt         14       19.12        20.01        Very good
model_epoch_16.pt         16       17.89        20.89
model_epoch_13.pt         13       21.45        23.12

RECOMMENDATION: Use model_epoch_15.pt (lowest perplexity)
```

## Why Perplexity Instead of Just Loss?

### 1. Interpretability

**Loss = 3.0** - What does this mean? Hard to say!

**Perplexity = 20** - The model is as confused as choosing from 20 words. Much clearer!

### 2. Comparison Across Models

You can compare perplexity across different:
- Model architectures
- Datasets
- Training configurations
- Published research results

Example: "Our model achieves perplexity of 25, compared to GPT-2's 29 on this dataset"

### 3. Connection to Information Theory

Perplexity is related to the number of bits needed to encode the text:
```
Bits per token = log₂(perplexity)
```

Lower perplexity = fewer bits needed = better compression = better model!

## Detecting Overfitting with Perplexity

Overfitting occurs when a model memorizes training data but fails to generalize.

### How to Detect

Compare training and validation perplexity:

```
Good generalization:
  Train perplexity: 18.2
  Val perplexity:   19.5
  → Small gap, model generalizes well! ✓

Overfitting:
  Train perplexity: 12.3
  Val perplexity:   45.8
  → Large gap, model memorized training data! ✗
```

### What to Do

If you see overfitting (val >> train):
1. Add more dropout
2. Reduce model size
3. Get more training data
4. Stop training earlier (early stopping)
5. Add data augmentation

## Code Examples

### Calculate Perplexity from Logits

```python
from src.transformer.perplexity import calculate_perplexity

# During training
logits = model(inputs)  # Shape: (batch, seq_len, vocab_size)
targets = ...           # Shape: (batch, seq_len)

perplexity = calculate_perplexity(logits, targets)
print(f"Perplexity: {perplexity.item():.2f}")
```

### Calculate Perplexity from Loss

```python
from src.transformer.perplexity import calculate_perplexity_from_loss

# If you already have the loss
loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
perplexity = calculate_perplexity_from_loss(loss)
print(f"Loss: {loss.item():.4f}, Perplexity: {perplexity.item():.2f}")
```

### Evaluate on a Dataset

```python
from src.transformer.perplexity import evaluate_perplexity

# Evaluate on validation set
val_perplexity, val_loss = evaluate_perplexity(
    model,
    val_dataloader,
    device='cuda'
)

print(f"Validation perplexity: {val_perplexity:.2f}")
```

## Common Questions

### Q: What's a good perplexity?

**A**: It depends on the task:
- Clean, well-formed text (books, articles): 10-30 is excellent
- Conversational text (chat, social media): 30-60 is good
- Noisy or diverse text: 60-100 is acceptable

### Q: My perplexity increased during training. What's wrong?

**A**: Perplexity should decrease during training. If it increases:
- Learning rate might be too high (gradient explosion)
- Model might be overfitting (check validation perplexity)
- Data might have changed or gotten noisier
- Bug in training code

### Q: Can perplexity be less than 1?

**A**: No! Perplexity = exp(loss), and loss is always ≥ 0, so perplexity ≥ 1.
Perplexity = 1 means perfect predictions (loss = 0).

### Q: How does perplexity relate to accuracy?

**A**:
- **Accuracy**: Only looks at top-1 prediction (binary: right or wrong)
- **Perplexity**: Considers full probability distribution (more nuanced)

You can have high accuracy but high perplexity if the model is often correct but uncertain.

### Q: Should I optimize for loss or perplexity during training?

**A**: **Always optimize for loss!** Perplexity is just exp(loss), so minimizing loss automatically minimizes perplexity. Loss has better gradient properties for optimization.

Use perplexity for **evaluation and interpretation**, not optimization.

## Further Reading

- Original paper: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Transformer architecture)
- [Language Model Evaluation](https://huggingface.co/docs/transformers/perplexity) by Hugging Face
- [Shannon's Information Theory](https://en.wikipedia.org/wiki/Perplexity) (theoretical foundation)

## Running the Demo

To see perplexity in action with concrete examples:

```bash
uv run python src/transformer/perplexity.py
```

This runs an educational demo showing how perplexity changes with model confidence.
