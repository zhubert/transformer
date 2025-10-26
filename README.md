# Transformer - Incrementally Built with AI

An educational GPT-style transformer incrementally built with AI in PyTorch. Every component is implemented by Claude's hand with comprehensive documentation to understand how modern LLMs work under the hood.

## Quick Start

```bash
# Install dependencies
uv sync

# Run all tests
uv run pytest

# Train on FineWeb (100M tokens per epoch)
uv run python main.py train

# Quick training (smaller model, 10M tokens/epoch)
uv run python main.py train --quick

# Generate text
uv run python main.py generate --checkpoint checkpoints/model_epoch_10_p50k.pt
```

## What's Inside

This project implements a complete decoder-only transformer (GPT architecture) with:

- **Core Components** - Attention, embeddings, feed-forward networks, transformer blocks
- **Training Pipeline** - FineWeb dataset streaming, learning rate scheduling, checkpointing
- **Text Generation** - Advanced sampling strategies (greedy, top-k, top-p, combined)
- **Evaluation** - Perplexity calculation and model comparison tools
- **Comprehensive Tests** - 86/86 tests passing across all components

**All components include extensive educational documentation** - read the source files to learn!

## Project Structure

```
src/transformer/
├── attention.py        # Scaled dot-product & multi-head attention mechanisms
├── embeddings.py       # Token embeddings & learned positional encodings
├── feedforward.py      # Feed-forward networks (MLP) with GELU activation
├── block.py            # Transformer blocks with Pre-LN architecture
├── model.py            # Complete decoder-only transformer with generation
├── sampling.py         # Advanced sampling strategies (top-k, top-p, combined)
├── perplexity.py       # Perplexity calculation and evaluation metrics
├── scheduler.py        # Learning rate scheduling (warmup + cosine decay)
├── dataset.py          # Dataset utilities
└── fineweb_dataset.py  # FineWeb streaming dataset with smart caching

commands/
├── train.py            # Training command - see file for complete guide
├── generate.py         # Text generation with preset strategies
├── sampling_comparison.py   # Demo of different sampling strategies
└── evaluate_perplexity.py   # Model evaluation and comparison

tests/                  # Comprehensive test suite (86 tests)
```

## Learning Path

Want to understand transformers deeply? Read the code in this order:

1. **`src/transformer/attention.py`** - Start here! The core self-attention mechanism
2. **`src/transformer/embeddings.py`** - How tokens and positions are represented
3. **`src/transformer/feedforward.py`** - Position-wise neural networks
4. **`src/transformer/block.py`** - How components combine (with gradient flow explanation!)
5. **`src/transformer/model.py`** - The complete architecture
6. **`src/transformer/sampling.py`** - How to generate high-quality text
7. **`src/transformer/perplexity.py`** - How to evaluate language models
8. **`commands/train.py`** - How to train the model

Each file has extensive documentation explaining concepts, design decisions, and mathematical details.

## Architecture Overview

**Decoder-Only Transformer** (GPT-style):

```
Input Token IDs
    ↓
[Token Embeddings] → Convert IDs to vectors
    ↓
[Positional Encodings] → Add position information (learned, not sinusoidal)
    ↓
[Transformer Block] → ┐
[Transformer Block] → ├─ Stacked N times (6 layers by default)
[Transformer Block] → ┘
    ↓
[Output Projection] → Project to vocabulary logits
    ↓
Next Token Predictions
```

**Each Transformer Block**:
- Multi-head self-attention (4 heads, causal masking)
- Feed-forward network (4x dimension expansion)
- Layer normalization (Pre-LN architecture like GPT-2/3)
- Residual connections (gradient highways!)
- Dropout for regularization

See `src/transformer/block.py` for detailed architecture diagrams and gradient flow explanation.

## Training

### Quick Start

```bash
# Default: 100M tokens/epoch, 6 layers, d_model=256, CPU
uv run python main.py train

# Quick mode: 10M tokens/epoch, 4 layers, d_model=128
uv run python main.py train --quick

# Use larger vocabulary (100K tokens vs 50K)
uv run python main.py train --encoding cl100k_base

# Apple Silicon GPU (experimental - see note below)
uv run python main.py train --mps
```

### Dataset: FineWeb

We use HuggingFace's [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (sample-10BT):
- **10 billion tokens** of high-quality web content
- **Streaming**: Downloads shards on-demand (no huge upfront download)
- **Smart caching**: Keeps 5 recent shards (~2GB), automatically cleans up old ones
- **Configurable**: Default 100M tokens per epoch, adjust as needed

### What to Expect

**Training Progress** (quick mode, CPU):
```
Epoch 1:  Loss ~8.0, Perplexity ~3000  (random guessing)
Epoch 3:  Loss ~5.0, Perplexity ~150   (learning patterns)
Epoch 5:  Loss ~4.0, Perplexity ~55    (getting decent)
Epoch 10: Loss ~3.0, Perplexity ~20    (pretty good!)
```

**Timing** (M1 MacBook Pro):
- **CPU**: ~10-15 min/epoch (quick mode)
- **MPS**: ~2-3 min/epoch (if stable)

### Apple Silicon GPU (MPS) - Known Issues

⚠️ **MPS has known NaN training issues** due to PyTorch bugs ([#107294](https://github.com/pytorch/pytorch/issues/107294), [#109457](https://github.com/pytorch/pytorch/issues/109457)):

- Training may randomly fail with NaN loss
- Affects transformer models (attention + layer norm interaction)
- No official fix as of PyTorch 2.9.0

**Recommendations**:
- ✅ **Use CPU** (default) - 100% reliable, good for learning
- ⚠️ **Try MPS** with `--mps` flag - 5-10x faster but may crash
- 🐛 **Debug mode**: `--mps --debug` forces synchronization (more stable, slower)

See `commands/train.py` for complete training documentation.

## Text Generation

### Quick Examples

```bash
# Use preset strategies
uv run python main.py generate --checkpoint checkpoints/model_epoch_10_p50k.pt --preset creative
uv run python main.py generate --checkpoint checkpoints/model_epoch_10_p50k.pt --preset precise

# Custom parameters
uv run python main.py generate \
    --checkpoint checkpoints/model_epoch_10_p50k.pt \
    --strategy top_k_top_p \
    --top-k 50 --top-p 0.9 --temperature 0.8
```

### Sampling Strategies

We implement four sampling strategies:

1. **Greedy** - Always pick most probable token (deterministic, repetitive)
2. **Top-k** - Sample from k most probable tokens (simple, effective)
3. **Top-p (Nucleus)** - Adaptive sampling based on cumulative probability
4. **Combined Top-k + Top-p** - **RECOMMENDED** - Best quality and diversity

**When to use which strategy?**

| Use Case | Strategy | Settings |
|----------|----------|----------|
| Creative writing | `top_k_top_p` | `k=100, p=0.95, temp=1.2` |
| Balanced/general | `top_k_top_p` | `k=50, p=0.9, temp=1.0` |
| Factual/technical | `top_k_top_p` | `k=40, p=0.85, temp=0.7` |
| Debugging | `greedy` | (deterministic) |

For detailed explanations of how each strategy works, see **`src/transformer/sampling.py`**.

## Evaluation: Perplexity

**Perplexity** measures how "confused" a model is when predicting text. Lower is better.

**Intuition**: Perplexity of 20 means "the model is as confused as choosing from 20 words at each step."

| Perplexity | Quality | Interpretation |
|------------|---------|----------------|
| 1.0 | Perfect | Always correct with 100% confidence (impossible!) |
| 10-30 | Excellent | GPT-2 level performance |
| 50-100 | Decent | Model has learned patterns, room for improvement |
| 200+ | Poor | Model is quite confused |
| ~50,000 | Random | Just guessing (vocab_size) |

### Evaluate Your Model

```bash
# Evaluate latest checkpoint
uv run python main.py evaluate

# Evaluate specific checkpoint
uv run python main.py evaluate --checkpoint checkpoints/model_epoch_10_p50k.pt

# Compare all checkpoints to find best model
uv run python main.py compare
```

For the complete perplexity tutorial (math, examples, overfitting detection), see **`src/transformer/perplexity.py`**.

## Development

### Requirements

- Python 3.13+
- uv (Python package manager)
- Dependencies: PyTorch, tiktoken, NumPy, pytest

### Running Tests

```bash
# All tests (86 tests)
uv run pytest

# Specific components
uv run pytest tests/test_attention.py -v
uv run pytest tests/test_sampling.py -v
uv run pytest tests/test_perplexity.py -v

# With coverage
uv run pytest --cov=src/transformer
```

### Project Configuration

This project uses `uv` for Python environment management:

```bash
# Install dependencies
uv sync

# Add new dependency
uv add <package-name>

# Run any command in the environment
uv run <command>
```

## Design Decisions

### Why Learned Positional Embeddings?
We use learned positional embeddings (like GPT-2/3, BERT) instead of sinusoidal encodings because:
- Simpler to implement and understand
- Used in modern production systems (GPT-2, GPT-3, BERT)
- Works well in practice

### Why GELU Activation?
We use GELU instead of ReLU because:
- Used in GPT-2, GPT-3, and BERT
- Smoother gradients than ReLU
- Better performance in transformers

### Why Decoder-Only?
We implement decoder-only (GPT) instead of encoder-decoder because:
- Simpler architecture (easier to learn)
- What modern LLMs actually use (GPT, Claude, etc.)
- Sufficient for autoregressive generation

### Why Pre-LN Architecture?
We use Pre-LN (layer norm before attention/FFN) instead of Post-LN because:
- More stable training (used in GPT-2, GPT-3)
- Better gradient flow
- Industry standard for large models

See `src/transformer/block.py` for detailed explanation with gradient flow math.

## Learning Resources

### Papers
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017) - Original transformer
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - GPT-2

### Key Concepts
- **Self-Attention** - How tokens "attend to" each other
- **Multi-Head Attention** - Running attention in parallel with different learned projections
- **Residual Connections** - Skip connections that create "gradient highways"
- **Layer Normalization** - Stabilizes training by normalizing activations
- **Causal Masking** - Prevents looking at future tokens during training

## Contributing

This is an educational project - code prioritizes clarity and documentation over performance. Feel free to use it for learning!

## License

MIT License - see LICENSE file for details.
