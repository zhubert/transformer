# Transformer from Scratch

An educational implementation of a decoder-only transformer (GPT-style) built from the ground up using PyTorch.

## Project Goal

This project implements a complete transformer architecture from scratch to understand how modern Large Language Models (LLMs) like GPT, Claude, and others actually work under the hood. Rather than using pre-built transformer modules, we build each component ourselves to gain deep insight into the architecture.

## Why Build from Scratch?

- **Deep Understanding**: Learn exactly how each component works by implementing it yourself
- **Educational Focus**: Code is heavily documented with explanations of concepts, architectures, and design decisions
- **Modern Architecture**: Uses learned positional embeddings (like GPT-2/3) rather than outdated sinusoidal encodings
- **Production Patterns**: Follows the same architectural choices as real-world LLMs

## Architecture Overview

We're building a **decoder-only transformer** (the architecture used by GPT models), which consists of:

1. **Token Embeddings** - Convert token IDs to dense vectors
2. **Positional Encodings** - Add position information using learned embeddings
3. **Transformer Blocks** (stacked multiple times):
   - **Multi-Head Self-Attention** - Allow tokens to attend to previous tokens
   - **Feed-Forward Network (MLP)** - Process each position independently
   - **Layer Normalization** - Stabilize training
   - **Residual Connections** - Enable gradient flow
4. **Output Layer** - Project to vocabulary for next-token prediction

## Implementation Status

### ✅ Completed Components

#### 1. Scaled Dot-Product Attention (`src/transformer/attention.py`)
The core attention mechanism implementing: `Attention(Q, K, V) = softmax(Q·Kᵀ / √d_k) · V`

- Computes attention weights between all positions
- Supports causal masking (for autoregressive generation)
- Returns both output and attention weights

**Tests**: 9/9 passing

#### 2. Token Embeddings (`src/transformer/embeddings.py`)
Converts token IDs to dense vector representations.

- Simple wrapper around `nn.Embedding`
- Maps vocabulary indices to learned vectors

**Tests**: 5/5 passing

#### 3. Learned Positional Encodings (`src/transformer/embeddings.py`)
Adds position information using learned embeddings (GPT-2/BERT approach).

- Uses `nn.Embedding` for position indices
- Adds positional embeddings to token embeddings
- Supports configurable max sequence length

**Tests**: 8/8 passing

#### 4. Feed-Forward Network - MLP (`src/transformer/feedforward.py`)
Position-wise 2-layer neural network (Multi-Layer Perceptron).

- Expands dimension: d_model → d_ff (typically 4x)
- GELU activation (used in GPT-2/GPT-3)
- Projects back: d_ff → d_model
- Includes dropout for regularization

**Comprehensive documentation** explaining what MLP means and why we use this architecture.

**Tests**: 10/10 passing

#### 5. Multi-Head Attention (`src/transformer/attention.py`)
Runs multiple scaled dot-product attention operations in parallel.

- Projects input to Q, K, V using learned linear transformations
- Splits into multiple heads (d_k = d_model / num_heads)
- Applies attention to each head in parallel
- Concatenates outputs and applies final projection
- Allows model to attend to different representation subspaces

**Comprehensive documentation** explaining:
- What multi-head attention is and why we use it
- How different heads learn different patterns
- Why heads don't all learn the same thing

**Tests**: 13/13 passing

#### 6. Transformer Block (`src/transformer/block.py`)
The fundamental building block that gets stacked to create the full transformer.

- Combines multi-head attention and feed-forward network
- Uses Pre-LN architecture (like GPT-2/GPT-3)
- Implements residual connections (gradient highways!)
- Applies layer normalization for training stability
- Includes dropout for regularization
- Input shape = Output shape (enables stacking)

**Extremely comprehensive documentation** explaining:
- What a transformer block is and its role
- Architecture diagram with Pre-LN approach
- **Gradient flow** and why residual connections are essential
- Mathematical proof of residual gradient properties
- Pre-LN vs Post-LN comparison
- Division of labor between attention and FFN

**Tests**: 14/14 passing

#### 7. Complete Decoder-Only Transformer Model (`src/transformer/model.py`)
The complete, working transformer that assembles all components.

- Combines token embeddings + positional encodings + transformer blocks
- Automatic causal mask creation for autoregressive generation
- Proper weight initialization for training stability
- Output projection to vocabulary logits
- Includes `generate()` method for text generation
- Supports various model sizes (configurable layers, heads, dimensions)

**Extremely comprehensive documentation** explaining:
- Complete architecture with shape flow diagrams
- What logits are and why we use them
- Causal masking for decoder-only models
- Model size comparisons (GPT-2, GPT-3)
- Autoregressive generation process

**Tests**: 15/15 passing

#### 8. Advanced Sampling Strategies (`src/transformer/sampling.py`)
Sophisticated text generation sampling methods for improved output quality.

- **Top-k Sampling**: Filters to only k most probable tokens
- **Top-p (Nucleus) Sampling**: Adaptively selects tokens based on cumulative probability
- **Combined Top-k + Top-p**: Best of both approaches
- **Temperature Scaling**: Controls randomness/creativity
- **Greedy Decoding**: Deterministic selection

**Comprehensive documentation** explaining:
- How each sampling method works
- When to use each strategy
- Trade-offs between quality and diversity
- Recommended settings for different use cases
- Mathematical details with examples

**Tests**: 27/27 passing

Integrated into `model.generate()` with easy-to-use API:
```python
# Recommended: Combined top-k + top-p
output = model.generate(
    prompt, max_length=50,
    sampling_strategy="top_k_top_p",
    top_k=50, top_p=0.9, temperature=0.8
)
```

#### 9. Training Implementation (`commands/train.py`, `src/transformer/dataset.py`)
Complete training pipeline with MPS GPU acceleration.

- BPE tokenization using tiktoken (p50k_base encoding)
- Automatic device detection (MPS/CUDA/CPU)
- Memory-optimized for M1 chips
- CrossEntropyLoss and Adam optimizer
- Checkpoint saving after each epoch
- Sample text generation during training
- Debug mode for NaN detection

**Comprehensive documentation** explaining:
- What training is and how it works
- Training loop components (forward pass, loss, backprop, optimizer)
- Key concepts (logits, loss, learning rate, batches, epochs)
- MPS GPU acceleration on Apple Silicon
- How to interpret loss values

**Successfully trains on M1 GPUs** with stable convergence!

#### 10. Learning Rate Scheduling (`src/transformer/scheduler.py`)
Warmup + cosine decay learning rate schedule for better convergence.

- Linear warmup: Gradually increases LR from 0 to peak over warmup steps
- Cosine decay: Smoothly decreases LR following cosine curve
- Prevents early training instability and improves final convergence

#### 11. Perplexity Evaluation (`src/transformer/perplexity.py`, `commands/evaluate_perplexity.py`)
Complete perplexity calculation and model evaluation system.

- Calculate perplexity from model logits or loss
- Evaluate models on datasets
- Compare multiple checkpoints to find best model
- Detect overfitting by comparing train/validation perplexity
- Comprehensive educational documentation explaining perplexity

**Tests**: Full test coverage in `tests/test_perplexity.py`

See [Understanding Perplexity](#understanding-perplexity) section for detailed explanation of this metric and how to use it.

#### 12. Text Generation Scripts (`commands/generate.py`, `commands/sampling_comparison.py`)
Complete text generation and inference capabilities.

- **Interactive generation**: Generate text with various presets (greedy, precise, balanced, creative)
- **Preset system**: Pre-configured sampling strategies for different use cases
- **Sampling comparison**: Demo showing differences between sampling strategies
- **Custom parameters**: Override presets with custom top-k, top-p, temperature settings
- Checkpoint loading and model inference

See [Text Generation with Advanced Sampling](#text-generation-with-advanced-sampling) for usage examples.

### 🚧 Next Steps

**Planned Training Improvements**:
- **Gradient Accumulation** - Simulate larger batch sizes without more memory
  - Accumulate gradients over multiple batches before updating weights
  - Larger effective batch size = more stable gradients, better learning
  - Example: 4 accumulation steps × batch 8 = effective batch 32

**Current Status**:
- ✅ All core components complete and tested
- ✅ Training pipeline with learning rate scheduling
- ✅ Advanced sampling methods (top-k, top-p, combined)
- ✅ Text generation scripts with preset strategies
- ✅ Perplexity evaluation and model comparison tools

## Project Structure

```
transformer/
├── src/transformer/          # Main implementation
│   ├── attention.py         # Scaled dot-product & multi-head attention
│   ├── embeddings.py        # Token embeddings & positional encoding
│   ├── feedforward.py       # Feed-forward network (MLP)
│   ├── block.py            # Single transformer block
│   ├── model.py            # Complete decoder-only transformer
│   ├── sampling.py         # Advanced sampling strategies (top-k, top-p)
│   ├── scheduler.py        # Learning rate scheduling
│   ├── perplexity.py       # Perplexity calculation and evaluation
│   └── dataset.py          # Dataset utilities
│
├── tests/                   # Comprehensive test suite
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_feedforward.py
│   ├── test_block.py
│   ├── test_model.py
│   ├── test_sampling.py    # 27 tests for sampling methods
│   └── test_perplexity.py  # Tests for perplexity calculation
│
├── commands/                # CLI command implementations
│   ├── train.py            # Training command
│   ├── generate.py         # Text generation command
│   ├── sampling_comparison.py    # Sampling strategy demo
│   └── evaluate_perplexity.py   # Perplexity evaluation command
│
├── CLAUDE.md               # Project planning and context
└── README.md               # This file
```

## Development Setup

This project uses `uv` for Python environment management.

### Prerequisites

- Python 3.13+
- uv (Python package manager)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd transformer

# uv will automatically create a virtual environment and install dependencies
uv sync
```

### Dependencies

- **PyTorch** - Deep learning framework
- **tiktoken** - BPE tokenization (same as GPT models)
- **NumPy** - Numerical operations
- **pytest** - Testing framework

## Running Tests

All components have comprehensive test suites to verify correctness.

```bash
# Run all tests
uv run pytest

# Run tests for a specific component
uv run pytest tests/test_attention.py -v
uv run pytest tests/test_embeddings.py -v
uv run pytest tests/test_feedforward.py -v
uv run pytest tests/test_sampling.py -v  # Test all sampling strategies

# Run with coverage
uv run pytest --cov=src/transformer
```

## Text Generation with Advanced Sampling

The model supports multiple sampling strategies for generating text:

### Quick Example

```python
from src.transformer.model import DecoderOnlyTransformer
import torch

# Load your trained model
model = DecoderOnlyTransformer(vocab_size=50000, d_model=512, num_heads=8)
# model.load_state_dict(...)

# Prepare prompt (tokenized)
prompt = torch.tensor([[1, 2, 3]])  # Your tokenized text

# Generate with different strategies:

# 1. Greedy (deterministic, safest)
output = model.generate(prompt, max_length=50, sampling_strategy="greedy")

# 2. Top-k (filters unlikely tokens)
output = model.generate(
    prompt, max_length=50,
    sampling_strategy="top_k",
    top_k=50, temperature=0.8
)

# 3. Top-p / Nucleus (adaptive)
output = model.generate(
    prompt, max_length=50,
    sampling_strategy="top_p",
    top_p=0.9, temperature=0.8
)

# 4. Combined (RECOMMENDED - best quality)
output = model.generate(
    prompt, max_length=50,
    sampling_strategy="top_k_top_p",
    top_k=50, top_p=0.9, temperature=0.8
)
```

### Recommended Settings by Use Case

| Use Case | Strategy | Settings |
|----------|----------|----------|
| **Creative Writing** | `top_k_top_p` | `k=100, p=0.95, temp=1.2` |
| **Balanced/General** | `top_k_top_p` | `k=50, p=0.9, temp=1.0` |
| **Factual/Technical** | `top_k_top_p` | `k=40, p=0.85, temp=0.7` |
| **Chatbot** | `top_p` | `p=0.9, temp=0.8` |
| **Code Generation** | `top_k` | `k=20, temp=0.6` |
| **Deterministic** | `greedy` | (no parameters) |

### Sampling Strategy Comparison

Run the demonstration script to see how different strategies behave:

```bash
uv run python commands/sampling_comparison.py
```

This shows:
- How each sampling method works
- Impact of temperature on diversity
- When to use each strategy
- Statistical behavior with different distributions

## Understanding Perplexity

**Perplexity** is the standard metric for evaluating language models. It measures how "confused" or "perplexed" a model is when predicting text.

### The Intuition

Think of perplexity as answering: "On average, how many words does the model have to choose from?"

- **Perplexity = 1**: Perfect! The model always knows exactly what comes next
- **Perplexity = 50**: The model is as confused as if picking uniformly from 50 words
- **Perplexity = 10,000**: Totally lost, like random guessing over entire vocabulary

### Real-World Example

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

### Mathematical Definition

Perplexity is the exponential of the average cross-entropy loss:

```
Perplexity = exp(CrossEntropyLoss)
         = exp(-1/N * Σ log P(correct_word))
```

**Example Calculation**: If the model predicts 3 tokens with probabilities [0.5, 0.25, 0.125]:
```
Average loss = (0.693 + 1.386 + 2.079) / 3 = 1.386
Perplexity = exp(1.386) = 4.0
→ On average, the model was as confused as choosing from 4 words
```

### Typical Perplexity Values

| Perplexity | Quality | Description |
|------------|---------|-------------|
| 1.0 | Perfect | Always 100% confident and correct (impossible in practice) |
| 10-30 | Excellent | GPT-2 level performance on good text |
| 50-100 | Decent | Model has learned patterns, room for improvement |
| 200+ | Poor | Model is quite confused, needs more training |
| ~vocab_size | Random | Model is just guessing randomly |

### Why Perplexity Instead of Just Loss?

**Loss = 3.0** - What does this mean? Hard to say!

**Perplexity = 20** - The model is as confused as choosing from 20 words. Much clearer!

Perplexity is more interpretable and allows you to compare models across different architectures, datasets, and research papers.

### Detecting Overfitting

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

If you see overfitting (val >> train):
1. Add more dropout
2. Reduce model size
3. Get more training data
4. Stop training earlier
5. Add data augmentation

### Evaluating Your Model

Use the evaluation script to test your model:

```bash
# Evaluate latest checkpoint
uv run python main.py evaluate

# Evaluate specific checkpoint
uv run python main.py evaluate --checkpoint checkpoints/model_epoch_10.pt

# Compare all checkpoints
uv run python main.py compare

# Specify evaluation text
uv run python main.py evaluate --text-file my_test_data.txt
```

**Example output**:
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

### Using Perplexity Programmatically

```python
from src.transformer.perplexity import calculate_perplexity, evaluate_perplexity

# During training - calculate from logits
logits = model(inputs)  # Shape: (batch, seq_len, vocab_size)
targets = ...           # Shape: (batch, seq_len)
perplexity = calculate_perplexity(logits, targets)
print(f"Perplexity: {perplexity.item():.2f}")

# Evaluate on a dataset
val_perplexity, val_loss = evaluate_perplexity(
    model, val_dataloader, device='cuda'
)
print(f"Validation perplexity: {val_perplexity:.2f}")
```

## Training the Model

The transformer can be trained on any text file using the training script.

### Quick Start

```bash
# Train on your text file (default: CPU)
uv run python main.py train
```

### Training Configuration

The training script (`commands/train.py`) defaults to CPU for stability:

- **Model size**: 6 layers, 256 dimensions, 4 attention heads (~30M parameters)
- **Batch size**: 8
- **Sequence length**: 128 tokens
- **Learning rate**: 1e-4 (conservative for stability)
- **Tokenization**: BPE using tiktoken `p50k_base` (same as GPT-3)
- **Device**: CPU by default, CUDA if available, MPS opt-in (see Known Issues)

### What to Expect

**Initial Training Output**:
```
Loading dataset...
Loaded text file: Singular.txt
Text length: 430,297 characters
Tokenized into 101,895 tokens
Vocabulary size: 50,281 tokens
Created 796 training sequences of length 128

Device: CPU
Model parameters: 30,598,761

Epoch 1/3
--------------------------------------------------------------------------------
  Batch 10/99, Loss: 8.2451, Avg Loss: 8.7234
  Batch 20/99, Loss: 7.1923, Avg Loss: 7.9102
  ...
```

**Loss Progression**:
- **Epoch 1**: Loss starts around 9-10 (random guessing), drops to ~5-6
- **Epoch 2**: Loss continues dropping to ~3-4 (learning patterns)
- **Epoch 3**: Loss reaches ~2-3 (decent predictions)

**Training Time** (on M1 MacBook Pro, CPU mode):
- ~10-15 minutes per epoch
- ~30-45 minutes total for 3 epochs

### Known Issues: MPS (Apple Silicon GPU)

**MPS backend has known NaN training issues** (PyTorch bugs [#107294](https://github.com/pytorch/pytorch/issues/107294), [#109457](https://github.com/pytorch/pytorch/issues/109457)):

- Training randomly fails with NaN loss (typically after a few batches)
- Caused by asynchronous execution bugs in PyTorch's MPS backend
- Affects transformer models specifically (attention + layer norm)
- No official fix as of PyTorch 2.9.0

**Workarounds**:
- ✅ **Use CPU** (default) - stable, 100% reliable
- ⚠️ **Try MPS** with `--mps` flag - 5-10x faster but may crash
- 🐛 **Debug mode** helps MPS: `--mps --debug` (forces synchronization)

```bash
# Recommended: CPU mode (stable)
uv run python main.py train

# Experimental: MPS mode (faster but unstable)
uv run python main.py train --mps

# If MPS fails: debug mode forces sync (slower but more stable)
uv run python main.py train --mps --debug
```

**Note**: CUDA (NVIDIA GPUs) does not have these issues.

### Debug Mode

Enable debug mode for detailed NaN diagnostics:

```bash
uv run python main.py train --debug
```

This prints diagnostic information at each step to help identify numerical stability issues.

### Using Your Own Text

1. Place your text file in the project root
2. Update `TEXT_FILE` in `commands/train.py` (line 151)
3. Run training as above

**Recommended text size**: 100KB - 10MB (smaller trains faster, larger learns better)

### Generated Text Examples

After training completes, the script automatically generates sample text:

```
Prompt: 'The'
Generated: 'The singularity was approaching faster than anyone had anticipated...'
```

The quality improves significantly after each epoch!

### Saved Checkpoints

Training saves model checkpoints after each epoch to `checkpoints/`:
```
checkpoints/
├── model_epoch_1.pt
├── model_epoch_2.pt
└── model_epoch_3.pt
```

Each checkpoint includes:
- Model weights
- Optimizer state
- Training loss
- Model configuration

## Implementation Approach

We're building **bottom-up**, starting with the simplest components and working our way up to the complete model:

1. ✅ **Scaled Dot-Product Attention** - The fundamental mechanism
2. ✅ **Embeddings** - Token and positional representations
3. ✅ **Feed-Forward Network** - Position-wise MLP
4. ✅ **Multi-Head Attention** - Parallel attention heads
5. ✅ **Transformer Block** - Combining all components
6. ✅ **Complete Model** - Decoder-only transformer with generation capability
7. ✅ **Training** - Full training pipeline with MPS GPU support
8. ✅ **Advanced Sampling** - Top-k, Top-p, and combined strategies for high-quality generation
9. ✅ **Learning Rate Scheduling** - Warmup + cosine decay for better convergence
10. ✅ **Perplexity Evaluation** - Standard metric for evaluating language model quality
11. ✅ **Text Generation Scripts** - Interactive generation with preset sampling strategies

Each component is:
- Fully implemented from scratch (no pre-built transformer modules)
- Heavily documented with explanations
- Thoroughly tested with comprehensive test suites
- Committed separately for clear progression

## Key Design Decisions

### Why Learned Positional Embeddings?
We use learned positional embeddings (like GPT-2/3 and BERT) instead of the original sinusoidal encodings from "Attention is All You Need" because:
- Simpler to implement and understand
- Used in modern production systems
- Works well in practice

### Why GELU Activation?
We use GELU (Gaussian Error Linear Unit) instead of ReLU because:
- Used in GPT-2, GPT-3, and BERT
- Smoother gradients than ReLU
- Better performance in practice

### Why Decoder-Only?
We implement a decoder-only architecture (like GPT) instead of encoder-decoder because:
- Simpler architecture
- What modern LLMs actually use
- Sufficient for autoregressive generation

## Learning Resources

### Papers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - Original transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2 paper

### Concepts
- **Attention Mechanism**: How tokens "attend to" other tokens
- **Multi-Layer Perceptron (MLP)**: Simple feedforward neural network
- **Layer Normalization**: Normalize activations for stable training
- **Residual Connections**: Skip connections that help gradients flow

## Contributing

This is an educational project built for learning. The code prioritizes clarity and documentation over performance.

## License

MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is part of a learning series on understanding LLMs from the ground up.
