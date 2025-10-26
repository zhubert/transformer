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

### 🚧 In Progress

- Complete Decoder-Only Transformer Model
- Training script
- Text generation

## Project Structure

```
transformer/
├── src/transformer/          # Main implementation
│   ├── attention.py         # Scaled dot-product & multi-head attention
│   ├── embeddings.py        # Token embeddings & positional encoding
│   ├── feedforward.py       # Feed-forward network (MLP)
│   ├── block.py            # Single transformer block
│   └── model.py            # Complete decoder-only transformer
│
├── tests/                   # Comprehensive test suite
│   ├── test_attention.py
│   ├── test_embeddings.py
│   ├── test_feedforward.py
│   ├── test_block.py
│   └── test_model.py
│
├── examples/                # Training and generation scripts
│   ├── train.py
│   └── generate.py
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

# Run with coverage
uv run pytest --cov=src/transformer
```

## Implementation Approach

We're building **bottom-up**, starting with the simplest components and working our way up to the complete model:

1. ✅ **Scaled Dot-Product Attention** - The fundamental mechanism
2. ✅ **Embeddings** - Token and positional representations
3. ✅ **Feed-Forward Network** - Position-wise MLP
4. ✅ **Multi-Head Attention** - Parallel attention heads
5. ✅ **Transformer Block** - Combining all components
6. 🚧 **Complete Model** - Stacking blocks
7. 🚧 **Training** - Training loop with a toy dataset
8. 🚧 **Generation** - Autoregressive text generation

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

[Add your license here]

## Acknowledgments

This implementation is part of a learning series on understanding LLMs from the ground up.
