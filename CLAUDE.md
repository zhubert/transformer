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
- Use modern best practices (Pre-LN, RoPE/learned position encodings, GELU)
- Include production optimizations (KV-cache, gradient accumulation)
- Train on realistic data (FineWeb 10BT) not toy datasets

## Architecture Overview

**Model Type**: Decoder-only transformer (GPT architecture)

**Key Decisions**:
- **Positional encoding**: RoPE (Rotary Position Embeddings) by default - modern standard (LLaMA, Mistral)
  - Alternative: Learned embeddings (GPT-2/3 style) - still supported via `position_encoding_type='learned'`
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
├── embeddings.py       # Token + positional embeddings (RoPE and learned)
├── feedforward.py      # Position-wise FFN with GELU
├── block.py            # Transformer block (Pre-LN: norm → attn/ffn → residual)
├── model.py            # Complete decoder-only model + generation
├── sampling.py         # Text generation strategies (greedy, top-k, top-p)
├── perplexity.py       # Model evaluation metrics
├── scheduler.py        # Learning rate scheduling (warmup + cosine)
├── training_utils.py   # Gradient accumulation utilities
├── dataset.py          # Dataset base classes
├── fineweb_dataset.py  # FineWeb streaming with train/val split
├── checkpoint_utils.py # Checkpoint loading/saving utilities (DRY)
├── dataset_utils.py    # Dataset configuration utilities (DRY)
└── device_utils.py     # Device initialization and management

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

### Shared Utilities (DRY Principle)
To avoid code duplication and maintain consistency, common functionality is extracted into shared utility modules:

**checkpoint_utils.py** - Checkpoint loading/saving
- `load_checkpoint()` - Universal checkpoint loader for all commands
- `detect_encoding()` - Encoding detection with backward compatibility
- `get_encoding_short_name()` - Convert encoding to short form for filenames
- `strip_compile_prefix()` - Remove torch.compile() `_orig_mod.` prefix
- `infer_max_seq_len()` - Infer max_seq_len from positional embedding shape

**dataset_utils.py** - Dataset configuration
- `calculate_optimal_cache_size()` - FineWeb shard cache calculation

**device_utils.py** - Device management
- `init_device()` - Initialize device (CUDA/MPS/CPU) with proper setup
- `get_autocast_context()` - Mixed precision context manager
- `get_synchronize_fn()` - Device synchronization function
- `get_memory_stats_fn()` - Memory statistics function

These utilities eliminate ~230 lines of duplicated code across 8 command files while maintaining:
- ✅ Single source of truth for common operations
- ✅ Easier maintenance (fix once, works everywhere)
- ✅ Consistent behavior across all commands
- ✅ Better discoverability for developers

## Key Features

### Training Infrastructure
- **FineWeb dataset**: 10B tokens, streaming with dynamic cache sizing
  - Cache automatically sized per mode (1-10 GB) to hold all shards
  - Pre-download option: `make download-medium` for offline training
  - Epoch 1: Downloads shards on-demand, builds cache
  - Epochs 2+: Pure local I/O → 2-4x speedup
- **Gradient accumulation**: Simulate large batches (16x default) for stability
- **Train/val split**: 90/10 deterministic split for overfitting detection
- **Device support**: Auto-detect CUDA (NVIDIA/AMD via ROCm) > MPS > CPU
  - AMD GPUs use HIP compatibility layer (torch.cuda API)
  - Automatic backend detection displays correct vendor (NVIDIA vs AMD)
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

### Position Encoding: RoPE vs Learned Embeddings

The model supports two position encoding approaches, with **RoPE as the modern default**:

#### RoPE (Rotary Position Embeddings) - Default
**The modern standard** used in LLaMA, LLaMA 2, Mistral, and most 2023-2024 LLMs.

```python
# Create model with RoPE (default)
model = DecoderOnlyTransformer(..., position_encoding_type='rope')
```

**How it works**:
- Instead of adding position information, **rotates** Q and K vectors by angle proportional to position
- Rotation happens in attention mechanism, not in embeddings
- Each pair of dimensions rotated by different frequency (logarithmically spaced)

**Key advantages**:
- **Zero parameters**: Purely mathematical, no weights to learn
  - Saves 1.28M parameters for default config (max_seq_len=5000, d_model=256)
- **Relative positions**: Encodes "3 tokens apart" not "at position 47"
  - Better for language understanding (relationships matter more than absolute position)
- **Length extrapolation**: Trained on 128 tokens? Generate 500+ tokens naturally!
  - Learned embeddings hit hard wall at max_seq_len
- **Theoretically grounded**: Based on geometric rotation properties
  - Dot product of rotated vectors encodes relative position automatically

**Implementation notes**:
- Applied to Q and K in attention, NOT to V (V carries content, not position)
- Works seamlessly with KV-cache (each new token rotated by its absolute position)
- Precomputed sin/cos values cached for efficiency (~1.2 MB for default config)
- Compatible with all other features (torch.compile, mixed precision, etc.)

#### Learned Position Embeddings - Alternative
Traditional approach from GPT-2, GPT-3, BERT (2018-2020).

```python
# Create model with learned embeddings
model = DecoderOnlyTransformer(..., position_encoding_type='learned')
```

**How it works**:
- Learnable embedding for each position (like token embeddings)
- Added to token embeddings before transformer blocks
- Encodes absolute positions

**Trade-offs**:
- **Parameters**: Requires max_seq_len × d_model parameters
  - Default config: 5000 × 256 = 1.28M parameters
- **Absolute positions**: "This is position 47" not "3 tokens apart"
- **Fixed length**: Cannot extrapolate beyond max_seq_len
- **Still works well**: Proven approach, just older

**When to use**:
- Reproducing GPT-2/GPT-3 style models exactly
- Ablation studies comparing position encoding methods
- Educational comparison to understand the difference

#### Comparison

| Feature | RoPE (Default) | Learned Embeddings |
|---------|----------------|-------------------|
| Parameters | 0 | max_seq_len × d_model |
| Position Type | Relative | Absolute |
| Extrapolation | Excellent | Poor |
| Memory | Minimal cache | Embedding table |
| Speed | ~5-10% overhead | Negligible |
| Used In | LLaMA, Mistral (2023+) | GPT-2, GPT-3 (2018-2020) |
| Educational Value | Modern standard | Historical context |

**Recommendation**: Use RoPE (default) unless you specifically need GPT-2 compatibility.

### KV-Cache Gotcha
When using KV-cache, positional encodings must account for the current position. This applies to both position encoding types:

**For RoPE**:
```python
# WRONG: Always rotates by position 0
rope(q, k, start_pos=0)

# RIGHT: Track cache length for correct absolute position
rope(q, k, start_pos=cache_length)
```

**For Learned Embeddings**:
```python
# WRONG: Always encodes position 0
pos_encoding(new_token, start_pos=0)

# RIGHT: Track cache length to get correct position
pos_encoding(new_token, start_pos=cache_length)
```

Both implementations handle this automatically by checking the cache length in the forward pass.

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

### Gradient Clipping
Transformers need gradient clipping to prevent exploding gradients in deep networks:
```python
# Clip gradients to prevent explosions
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Why transformers need this**:
- Deep networks (6+ layers) with residual connections can accumulate large gradients
- Attention mechanisms can produce extreme values early in training
- Without clipping, loss can suddenly spike to NaN

**Implementation** (GPT-2/GPT-3 standard):
- max_norm=1.0 - clips gradient norm if it exceeds this threshold
- Tracks gradient norms for monitoring training stability
- Applied after backward pass, before optimizer step

### Gradient Accumulation
Don't forget to scale loss by accumulation steps:
```python
loss = loss / accumulation_steps  # Scale before backward
loss.backward()                    # Accumulate gradients
# Only update weights every N steps
```

### Optimizer: AdamW with Selective Weight Decay
We use **AdamW** (not Adam) with selective weight decay, following modern transformer best practices:

```python
# Separate parameters into groups
decay_params = []      # Weights (get regularized)
no_decay_params = []   # Biases and LayerNorms (no regularization)

for name, param in model.named_parameters():
    # Exclude biases and all LayerNorm parameters (weight and bias)
    if 'bias' in name.lower() or 'norm' in name.lower():
        no_decay_params.append(param)
    else:
        decay_params.append(param)

# Create optimizer with parameter groups
optimizer = torch.optim.AdamW([
    {'params': decay_params, 'weight_decay': 0.01},
    {'params': no_decay_params, 'weight_decay': 0.0}
], lr=3e-4, betas=(0.9, 0.999), eps=1e-8)
```

**Why AdamW over Adam?**
- Decouples weight decay from gradient-based updates
- Better generalization for transformers (3-5% improvement)
- Used in GPT-2, GPT-3, BERT, LLaMA, and all modern LLMs

**Why selective weight decay?**
- Biases are just offsets → don't benefit from regularization
- LayerNorm parameters are already constrained → regularization can hurt
- Linear weights and embeddings → benefit from L2 regularization
- Matches GPT-2/GPT-3 implementation (2-5% improvement)

**Configuration** (GPT-2/GPT-3 standard):
- lr=3e-4 for small models, 6e-5 to 1e-4 for large models
- weight_decay=0.01 for weights only
- betas=(0.9, 0.999) - standard momentum parameters
- eps=1e-8 for numerical stability

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
make install              # NVIDIA CUDA or CPU (default)
make install-rocm         # AMD ROCm (Linux only)

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
