# Transformer - Educational Implementation

## Project Purpose
This is an **educational codebase** that teaches how transformers work by implementing a complete decoder-only transformer (GPT-style) from scratch in PyTorch. Every component prioritizes **clarity and learning** over performance.

## Core Principles

### 1. Educational First
- **Comprehensive documentation**: Every file has detailed docstrings explaining concepts, not just implementation
- **Teaching through code**: Inline comments explain the "why" behind design decisions
- **Mathematical explanations**: Include formulas and complexity analysis where relevant
- **Visual aids**: index.html contains interactive diagrams and step-by-step tutorials
- **Expected outcomes**: Tell users what they should see/learn from each component

### 2. No Magic
- Implement all transformer components from scratch (no `nn.Transformer` or similar)
- Use PyTorch only for basic operations (tensors, autograd, optimizers, nn.Module)
- Make all architectural decisions explicit and documented

### 3. Accuracy Over Simplification
- Use modern best practices (Pre-LN, learned positional embeddings, GELU)
- Include production optimizations (KV-cache, gradient accumulation)
- Train on realistic data (FineWeb 10BT) not toy datasets

## Architecture Overview

**Model Type**: Decoder-only transformer (GPT architecture)

**Key Decisions**:
- **Positional encoding**: Learned embeddings (not sinusoidal) - matches GPT-2/3/BERT
- **Layer normalization**: Pre-LN architecture for training stability
- **Activation**: GELU (not ReLU) - modern standard
- **Attention**: Causal masking for autoregressive generation
- **Optimization**: KV-cache for 2-50x faster generation

**Default Hyperparameters**:
- Layers: 6 (configurable)
- Model dimension: 256 (d_model)
- Attention heads: 4
- FFN expansion: 4x
- Dropout: 0.1
- Vocabulary: ~100K tokens (tiktoken cl100k_base)
- Context length: 128 tokens (training), 5000 max (generation)

## Project Structure

```
src/transformer/
├── attention.py        # Scaled dot-product & multi-head attention with KV-cache
├── embeddings.py       # Token + learned positional embeddings
├── feedforward.py      # Position-wise FFN with GELU
├── block.py            # Transformer block (Pre-LN: norm → attn/ffn → residual)
├── model.py            # Complete decoder-only model + generation
├── sampling.py         # Text generation strategies (greedy, top-k, top-p)
├── perplexity.py       # Model evaluation metrics
├── scheduler.py        # Learning rate scheduling (warmup + cosine)
├── training_utils.py   # Gradient accumulation utilities
├── dataset.py          # Dataset base classes
└── fineweb_dataset.py  # FineWeb streaming with train/val split

commands/               # CLI scripts for training, generation, evaluation
tests/                  # Test suite for core components
main.py                 # Main CLI entry point (python main.py train/generate/etc)
index.html              # Educational documentation with diagrams
```

## Implementation Patterns

### Code Organization
- **One concept per file**: Each file focuses on a single component
- **Standalone modules**: Components can be understood independently
- **Clear hierarchies**: Basic → complex (embeddings → attention → blocks → model)

### Documentation Style
```python
def function(x):
    """
    Brief one-line summary.

    Detailed explanation of what this does and WHY it's designed this way.
    Include relevant formulas, complexity analysis, or architectural notes.

    Args:
        x: Clear description with shape information (batch, seq_len, d_model)

    Returns:
        Clear description with shape information

    Example:
        # Show actual usage
        result = function(input)
    """
```

### Educational Comments
- Explain **why** not just **what**
- Include **complexity analysis** for algorithms
- Note **design decisions** and alternatives considered
- Reference **papers or standards** when applicable

## Key Features

### Training Infrastructure
- **FineWeb dataset**: 10B tokens, streaming with smart caching (5 shards ~2GB)
- **Gradient accumulation**: Simulate large batches (16x default) for stability
- **Train/val split**: 90/10 deterministic split for overfitting detection
- **Device support**: Auto-detect CUDA > MPS > CPU
- **Checkpointing**: Full state save/resume

### Generation Optimizations
- **KV-cache**: Cache Key/Value projections across generation steps (O(n²) → O(n))
- **Sampling strategies**: Greedy, top-k, top-p, combined
- **Batched generation**: Support for generating multiple sequences

### Quality Assurance
- **Test suite**: Core components have unit tests
- **Type hints**: Where they improve clarity
- **Validation**: Shape checks and assertions in critical paths

## Important Implementation Details

### Weight Tying
The model uses **weight tying** between token embeddings and output projection - a standard practice in GPT-2, GPT-3, and BERT:

```python
# Both layers share the same weight matrix
self.output_proj.weight = self.token_embedding.embedding.weight
```

**Why this works**:
- **Embedding**: Maps token ID → vector (lookup row from matrix E)
- **Output**: Maps vector → token scores (multiply by E^T)
- These are inverse operations, so sharing the matrix makes conceptual sense

**Benefits**:
- **50% parameter reduction** for embedding/output layers
  - Without: vocab_size × d_model × 2 = 51.2M params (default config)
  - With: vocab_size × d_model = 25.6M params (50% savings!)
- **Better generalization**: Regularization effect from shared weights
- **Consistent representations**: Forces token → vector → token consistency
- **Improved perplexity**: Empirically shown to improve by 5-15%

**Configurable**:
```python
# Enable (default, recommended)
model = DecoderOnlyTransformer(..., tie_weights=True)

# Disable (for ablation studies)
model = DecoderOnlyTransformer(..., tie_weights=False)
```

**Compatibility with embedding scaling**: Weight tying and sqrt(d_model) scaling work together perfectly. The scaling happens during the forward pass (embedding lookup), not in the parameters, so there's no conflict.

### KV-Cache Gotcha
When using KV-cache, positional encodings must account for the current position:
```python
# WRONG: Always encodes position 0 for new tokens
pos_encoding(new_token, start_pos=0)

# RIGHT: Track cache length to get correct position
pos_encoding(new_token, start_pos=cache_length)
```

### Pre-LN Architecture
Layer norm comes **before** the sub-layer, not after:
```python
# Pre-LN (what we use - more stable)
x = x + attention(norm(x))
x = x + ffn(norm(x))

# Post-LN (original paper - less stable)
x = norm(x + attention(x))
x = norm(x + ffn(x))
```

### Gradient Accumulation
Don't forget to scale loss by accumulation steps:
```python
loss = loss / accumulation_steps  # Scale before backward
loss.backward()                    # Accumulate gradients
# Only update weights every N steps
```

## When Making Changes

### Always Preserve
- Educational focus and comprehensive documentation
- Code clarity over performance optimizations
- Detailed explanations in comments and docstrings
- Visual diagrams in index.html
- Test coverage for core components

### Feel Free to Modify
- Hyperparameters (model size, learning rate, etc.)
- CLI interfaces and command options
- Dataset configuration
- Checkpoint formats
- Utility scripts

### Requires Careful Consideration
- Core architecture changes (affects educational content)
- Documentation structure (index.html, README.md, CLAUDE.md must stay aligned)
- Test suite modifications (maintain coverage)

## Quick Start for Development

```bash
# Install dependencies
make install

# Run tests
make test

# Train model (100M tokens/epoch)
make train

# Quick training (10M tokens/epoch, smaller model)
make train-quick

# Generate text
make generate
```

## Documentation Sync

Three docs must stay consistent:
- **README.md**: User-facing, quick start, API reference
- **index.html**: Educational tutorial with diagrams
- **CLAUDE.md**: AI context (this file) - architecture and patterns

When changing architecture or features, update all three.
