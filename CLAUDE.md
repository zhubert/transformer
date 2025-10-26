# Transformer Implementation - Educational Project

## Project Goal
Implement a decoder-only transformer incrementally with AI for educational purposes, building on our series about learning LLMs.

## What is a Transformer?
A transformer is a neural network architecture introduced in "Attention is All You Need" (Vaswani et al., 2017). It revolutionized NLP and is the foundation of modern LLMs like GPT, BERT, and Claude.

### Key Components
1. **Self-Attention Mechanism** - Allows the model to weigh the importance of different words when processing each word
2. **Multi-Head Attention** - Runs multiple attention mechanisms in parallel, each learning different relationships
3. **Position Encodings** - Adds positional information since transformers process all tokens in parallel
4. **Feed-Forward Networks** - Applied to each position independently
5. **Layer Normalization & Residual Connections** - Helps with training stability
6. **Decoder-Only Structure** - We'll implement this (like GPT), simpler than encoder-decoder

## Implementation Approach
- Incrementally built with AI to understand internals
- Use PyTorch for:
  - Automatic differentiation (autograd)
  - GPU acceleration
  - Tensor operations
  - BUT avoid pre-built transformer modules
- Use `uv` for Python environment management

## Planned Components
1. Positional encoding (sinusoidal)
2. Scaled dot-product attention
3. Multi-head attention with causal masking
4. Feed-forward network (MLP)
5. Transformer block (attention + FFN + residuals + LayerNorm)
6. Full decoder-only transformer model
7. Training script with toy dataset
8. Text generation (inference) capability

## Current Status
✅ All core components implemented:
- Token embeddings with positional encoding
- Scaled dot-product attention with causal masking
- Multi-head attention
- Feed-forward networks
- Complete transformer blocks
- Full decoder-only transformer model
- Training script with learning rate scheduling
- **Gradient accumulation for stable training (Phase 1)**
- **Train/validation split for overfitting detection (Phase 1)**
- Multiple sampling strategies (greedy, top-k, top-p)
- Text generation capabilities
- Perplexity evaluation

## Training Data: FineWeb

We use HuggingFace's FineWeb dataset for large-scale pretraining:

**What is FineWeb?**
- Large-scale web crawl dataset from HuggingFace
- We use the sample-10BT split (10 billion tokens)
- High-quality, filtered web content
- Perfect for realistic language model training

**Key Features:**
- **Streaming**: Downloads shards on-demand (no huge upfront download)
- **Caching**: Saves downloaded shards locally for fast reuse
- **LRU Cleanup**: Automatically manages disk space by removing old shards
- **Scalable**: Train on 100M+ tokens per epoch without storing entire dataset

**Architecture:**
```
HuggingFace FineWeb (10BT)
    ↓ (stream on-demand)
FineWebDataset
    ↓ (cache shards)
data/fineweb_cache/
├── shard_00000.parquet
├── shard_00001.parquet
└── cache_metadata.json
```

**Usage:**
```bash
# Test FineWeb integration
uv run python commands/test_fineweb.py

# Train with FineWeb (100M tokens per epoch, default)
uv run python commands/train.py

# Optional: Use Apple Silicon GPU (experimental)
uv run python commands/train.py --mps
```

**Benefits:**
- 10 billion tokens available (no overfitting!)
- Diverse web content (better generalization)
- Realistic pretraining experience
- Configurable tokens per epoch (100M default)
- Cache management keeps disk usage low (~2-5GB)

**Configuration:**
- Default: 100M tokens per epoch
- Cache: Keeps 5 most recent shards (~2GB)
- Sequence length: 128 tokens
- Tokenizer: tiktoken p50k_base (~50K vocab)

## Phase 1: Training at Scale (Educational Focus)

We've implemented two critical training improvements with comprehensive educational documentation:

### 1. Gradient Accumulation

**What:** Simulate large batch training without memory overhead by accumulating gradients over multiple small batches.

**Why:** Small batches (8 sequences) produce noisy gradients → unstable training. Large batches (128+ sequences) are too memory-intensive for hobby hardware.

**How it works:**
- Process 16 small batches (8 sequences each)
- Accumulate gradients without updating weights
- Update weights once using averaged gradients
- **Result:** Effective batch of 128 sequences (16x more stable!) with same memory as batch=8

**Mathematical basis:**
```
∇(L₁ + L₂ + ... + L₁₆) = ∇L₁ + ∇L₂ + ... + ∇L₁₆
```

**Expected improvements:**
- 20-30% lower final loss
- Smoother training curves
- Faster convergence
- Better generalization

**Implementation:** See `src/transformer/training_utils.py` for detailed explanation with ASCII diagrams.

**Usage:**
```bash
# Default: 16x accumulation (effective batch = 128 sequences)
uv run python commands/train.py

# Custom accumulation (32x = even more stable)
uv run python commands/train.py --accumulation-steps 32
```

### 2. Train/Validation Split

**What:** Separate 10% of data for validation to detect overfitting/underfitting.

**Why:** Without validation, we don't know if the model is truly learning patterns or just memorizing training data.

**How it works:**
- Deterministic hash-based split: 90% train, 10% validation
- No data leakage (validation shards never appear in training)
- Evaluate on validation after each epoch
- Compare train vs validation metrics to assess learning

**Interpreting results:**
```
Good:          Train ↓, Val ↓         (Both improving)
Underfitting:  Train flat, Val flat   (Not learning enough)
Overfitting:   Train ↓, Val ↑         (Memorizing training data!)
```

**Implementation:** See `src/transformer/fineweb_dataset.py` for detailed explanation of splitting strategy.

**What you'll see:**
```
Epoch 5 Summary:
  Train Loss: 3.2  |  Train Perplexity: 24.5
  Val Loss:   3.4  |  Val Perplexity:   29.8
  Status: ✓ Model is learning (val slightly > train, normal)
```

### Educational Documentation

All Phase 1 features include:
- **Comprehensive docstrings** explaining concepts, not just implementation
- **Mathematical explanations** for why techniques work
- **Visual diagrams** in index.html showing processes
- **Inline comments** teaching throughout the code
- **Expected outcomes** so users know what to look for

**Where to learn more:**
- **index.html**: Step 7 "Training at Scale" with interactive diagrams
- **src/transformer/training_utils.py**: Full gradient accumulation explanation
- **src/transformer/fineweb_dataset.py**: Train/val split strategy
- **commands/train.py**: Implementation with extensive comments
