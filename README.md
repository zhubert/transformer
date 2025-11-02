<div align="center">
  <img src="assets/mascot.png" alt="Transformer Mascot" width="300">
</div>

# Transformer - Built with AI

An educational GPT-style transformer built with AI in PyTorch. Every component includes comprehensive documentation to understand how modern LLMs work under the hood.

## Quick Start

### Requirements & Installation

**Prerequisites:**
- **Python 3.12+** - Specified in `.python-version`
- **UV** - Fast Python package manager ([docs](https://docs.astral.sh/uv/))

**Install UV:**
```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Alternative: via pip
pip install uv
```

**Install Project Dependencies:**
```bash
# Install dependencies (PyTorch, tiktoken, NumPy, datasets, rich, questionary)
make install              # Default: NVIDIA CUDA or CPU
make install-rocm         # For AMD GPUs with ROCm (Linux only)

# Run all tests
make test
```

**Automatic Virtual Environment Activation (Recommended):**

This project includes a `.envrc` file for automatic virtual environment activation using [direnv](https://direnv.net/). Once set up, you can use `python` and `pytest` commands directly without the `uv run` prefix.

```bash
# Install direnv (one-time setup)
# macOS
brew install direnv

# Linux (Ubuntu/Debian)
sudo apt install direnv

# Linux (Arch)
sudo pacman -S direnv

# Add direnv hook to your shell (one-time setup)
# For bash, add to ~/.bashrc:
eval "$(direnv hook bash)"

# For zsh, add to ~/.zshrc:
eval "$(direnv hook zsh)"

# For fish, add to ~/.config/fish/config.fish:
direnv hook fish | source

# Allow direnv in this directory (one-time per project)
direnv allow

# Now the virtual environment activates automatically when you cd into the project!
# You can use python/pytest directly instead of 'uv run python'/'uv run pytest'
```

**Without direnv:**
If you prefer not to use direnv, you can still manually activate the virtual environment:
```bash
source .venv/bin/activate
```

Or continue using the `uv run` prefix for commands:
```bash
uv run python main.py
uv run pytest tests/
```

### Interactive CLI (Recommended!)

The easiest way to use this project is through the interactive CLI:

```bash
# Launch interactive mode - no flags to memorize!
python main.py
```

The interactive CLI provides a beautiful, arrow-key navigated menu system that lets you:
- ✨ **Train models** - Choose Quick/Medium/Full modes with smart defaults
- 🎨 **Generate text** - Select checkpoints, presets, and settings interactively
- 📊 **Evaluate models** - Compare checkpoints or calculate perplexity
- 🔍 **Analyze internals** - Explore attention patterns, logit lens, induction heads
- ⬇️ **Download data** - Pre-download training shards for offline use
- 🤖 **Download pretrained models** - Get Phi-2 (2.7B) and other models

**Features:**
- Auto-detects available checkpoints and shows status
- Colorful tables and progress indicators
- Helpful explanations for each option
- No need to remember complex command-line flags!

### Advanced: Command-Line Interface

For power users and automation, all operations are also available via traditional CLI commands:

```bash
# Training
python main.py train               # Train with default settings
python main.py train --resume      # Resume training

# Generation
python main.py generate checkpoints/model_epoch_15.pt --preset balanced

# Evaluation
python main.py evaluate --checkpoint checkpoints/model_epoch_15.pt

# Interpretability
python main.py interpret logit-lens checkpoints/model_epoch_15.pt

# Download data
python main.py download --tokens 50000000  # Download 50M tokens (~5 GB)
```

## Pretrained Models

### Phi-2 (2.7B Parameter Model)

You can download and use Microsoft's Phi-2 model - a state-of-the-art 2.7B parameter transformer that's competitive with much larger models. This is perfect for:

- **Fine-tuning** on custom datasets
- **Learning** from a production-quality architecture
- **Experimentation** with a powerful but manageable model size
- **Comparison** with your trained models

**Download Phi-2:**

```bash
# Interactive mode (recommended)
python main.py
# → Select "🤖 Download pretrained models" → "phi-2"

# Or directly via command line
python commands/download_phi2.py
```

**Download size:** ~5.5 GB (model weights in fp32)
**Checkpoint size:** ~10.8 GB (includes optimizer state)
**Output:** `checkpoints/phi2_pretrained_cl100k.pt`

**Memory Requirements:**
- **Generation:** ~6GB GPU memory (or CPU, but slower)
- **Training/Fine-tuning:** ~11GB GPU memory (mixed precision) or ~22GB (fp32)
- **Recommendation:** At least 8GB VRAM for comfortable generation, 12GB+ for fine-tuning

**Phi-2 Architecture Highlights:**
- **2.7 billion parameters** - Large enough to be powerful, small enough to be practical
- **32 layers, 2560 model dimension** - Deep architecture with rich representations
- **RoPE with partial rotation** (40% of dimensions) - Modern position encoding
- **CodeGen tokenizer** - Optimized for code and technical content (auto-converted to cl100k_base)
- **2048 token context** - Substantial context window

**After downloading, you can:**

```bash
# Fine-tune on your data
python main.py train --resume

# Generate text
python main.py generate checkpoints/phi2_pretrained_cl100k.pt --preset creative

# Evaluate perplexity
python main.py evaluate --checkpoint checkpoints/phi2_pretrained_cl100k.pt

# Analyze interpretability
python main.py interpret logit-lens checkpoints/phi2_pretrained_cl100k.pt
```

**Requirements:**
The download requires the `transformers` library (automatically added to dependencies).

**Architecture Compatibility:**
Our implementation fully supports Phi-2's architecture including:
- ✅ Partial RoPE rotation (`partial_rotary_factor=0.4`)
- ✅ Weight tying between embeddings and output
- ✅ All 32 layers with proper weight mapping
- ✅ Compatible with all our generation and interpretability tools

## What's Inside

This project implements a complete decoder-only transformer (GPT architecture) with:

- **Core Components** - Attention, embeddings, feed-forward networks, transformer blocks
- **Training Pipeline** - FineWeb dataset streaming, gradient accumulation, train/val split, learning rate scheduling
- **Text Generation** - KV-Cache optimization (2-50x faster!), advanced sampling strategies (greedy, top-k, top-p)
- **Interpretability** - Logit lens, attention analysis, induction heads, activation patching
- **Evaluation** - Perplexity calculation and model comparison tools
- **Testing** - Test suite covering core components and functionality

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
├── training_utils.py   # Gradient accumulation for stable training
├── dataset.py          # Dataset utilities
├── fineweb_dataset.py  # FineWeb streaming with caching & train/val split
└── interpretability/   # Mechanistic interpretability tools
    ├── logit_lens.py          # Visualize predictions at each layer
    ├── attention_analysis.py  # Analyze attention patterns (Phase 2)
    ├── induction_heads.py     # Detect induction circuits (Phase 3)
    ├── activation_patching.py # Causal interventions (Phase 4)
    └── visualizations.py      # Rich-based terminal visualizations

commands/
├── train.py                 # Training command - see file for complete guide
├── generate.py              # Text generation with preset strategies
├── interpret.py             # Interpretability tools (logit lens, attention, etc.)
├── sampling_comparison.py   # Demo of different sampling strategies
├── evaluate_perplexity.py   # Model evaluation and comparison
├── benchmark_generation.py  # KV-cache speedup benchmarking
├── analyze_checkpoints.py   # Checkpoint analysis utilities
└── test_fineweb.py          # FineWeb dataset testing

tests/                  # Test suite for core components
```

## Learning Path

Want to understand transformers deeply? Read the code in this order:

### Core Architecture (Start Here!)
1. **[`src/transformer/attention.py`](src/transformer/attention.py)** - Start here! The core self-attention mechanism with KV-cache optimization
2. **[`src/transformer/embeddings.py`](src/transformer/embeddings.py)** - How tokens and positions are represented (learned vs sinusoidal)
3. **[`src/transformer/feedforward.py`](src/transformer/feedforward.py)** - Position-wise neural networks with GELU activation
4. **[`src/transformer/block.py`](src/transformer/block.py)** - How components combine (Pre-LN architecture + gradient flow explanation)
5. **[`src/transformer/model.py`](src/transformer/model.py)** - The complete decoder-only transformer architecture

### Training & Evaluation
6. **[`src/transformer/scheduler.py`](src/transformer/scheduler.py)** - Learning rate scheduling (warmup + cosine decay)
7. **[`src/transformer/training_utils.py`](src/transformer/training_utils.py)** - Gradient accumulation for stable training
8. **[`src/transformer/fineweb_dataset.py`](src/transformer/fineweb_dataset.py)** - Streaming dataset with smart caching
9. **[`commands/train.py`](commands/train.py)** - Complete training pipeline with device auto-detection
10. **[`src/transformer/perplexity.py`](src/transformer/perplexity.py)** - How to evaluate and compare language models

### Generation & Sampling
11. **[`src/transformer/sampling.py`](src/transformer/sampling.py)** - Advanced sampling strategies (greedy, top-k, top-p)
12. **[`commands/generate.py`](commands/generate.py)** - Interactive text generation with presets
13. **[`commands/benchmark_generation.py`](commands/benchmark_generation.py)** - KV-cache speedup demonstration (2-50x faster!)

### Interpretability (Advanced)
14. **[`src/transformer/interpretability/logit_lens.py`](src/transformer/interpretability/logit_lens.py)** - Visualize how predictions evolve through layers
15. **[`src/transformer/interpretability/attention_analysis.py`](src/transformer/interpretability/attention_analysis.py)** - Discover attention patterns and head behaviors
16. **[`src/transformer/interpretability/induction_heads.py`](src/transformer/interpretability/induction_heads.py)** - Detect pattern-matching circuits
17. **[`src/transformer/interpretability/activation_patching.py`](src/transformer/interpretability/activation_patching.py)** - Causal interventions to test component importance

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

See [`src/transformer/block.py`](src/transformer/block.py) for detailed architecture diagrams and gradient flow explanation.

## Training

### Quick Start

```bash
# Default: 50M tokens/epoch, 4 layers, d_model=256, 16x gradient accumulation
# Auto-detects best device (CUDA > MPS > CPU)
python main.py train

# Resume training from latest checkpoint
python main.py train --resume

# Standard training with cl100k_base tokenizer (100K vocab)
python main.py train

# Custom gradient accumulation (higher = more stable training)
python main.py train --accumulation-steps 32

# Force specific device (optional - auto-detect is recommended)
python main.py train --mps    # Apple Silicon GPU
```

### Training Features

- **Gradient accumulation**: Simulate large batch sizes (16x default) for stable training
- **Gradient clipping**: Prevents exploding gradients (max_norm=1.0, GPT-2/GPT-3 standard)
- **Learning rate scheduling**: Warmup (5%) + cosine decay for optimal convergence
- **Train/val split**: 90/10 split for monitoring overfitting

### Dataset: FineWeb

We use HuggingFace's [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) dataset (sample-10BT):
- **10 billion tokens** of high-quality web content
- **Streaming**: Downloads shards on-demand (no huge upfront download)
- **Smart caching**: Dynamic cache sizing per training mode
  - Quick: ~1 GB cache (26 shards)
  - Medium: ~5 GB cache (132 shards)
  - Default: ~10 GB cache (264 shards)
- **Performance**: After epoch 1, all shards are cached → 2-4x speedup for epochs 2+
- **Configurable**: Default 100M tokens per epoch, adjust as needed

### Pre-downloading Data (Optional)

You can pre-download all training data before starting training. This is useful for:
- **Offline training**: Download once, train anytime without internet
- **Bandwidth control**: Download when network is fast/cheap
- **Uninterrupted training**: No network issues during training runs

```bash
# Pre-download training data (default: 50M tokens, ~5 GB)
make download

# Or use the CLI directly with custom token count
python main.py download --tokens 50000000   # 50M tokens (~5 GB)
python main.py download --tokens 10000000   # 10M tokens (~1 GB)
python main.py download --tokens 100000000  # 100M tokens (~10 GB)
```

After downloading, training will run at full speed from epoch 1 (instead of building the cache during the first epoch).

### Device Support

The training script automatically detects and uses the best available device:

1. **CUDA** (NVIDIA GPUs) - Preferred for maximum performance
   - Automatic mixed precision (bfloat16) for ~2x speedup
   - Memory tracking and synchronization utilities
   - Install with: `make install`

2. **ROCm** (AMD GPUs, Linux only) - Full GPU acceleration
   - Uses HIP compatibility layer (torch.cuda API)
   - Same performance optimizations as CUDA
   - Install with: `make install-rocm`

3. **MPS** (Apple Silicon) - Excellent for Mac users
   - Native GPU acceleration on M1/M2/M3 chips
   - Significantly faster than CPU (5-10x)
   - Install with: `make install`

4. **CPU** - Universal fallback
   - Works everywhere, good for learning and debugging

The device is selected automatically. The code automatically detects whether you're using NVIDIA or AMD GPUs and displays the appropriate backend information.

### Known Issue: hipBLASLt Warning (AMD GPUs)

If you're using AMD GPUs with ROCm, you may see this warning during training:

```
UserWarning: Attempting to use hipBLASLt on an unsupported architecture!
Overriding blas backend to hipblas
```

**This is harmless and expected behavior.** Here's why:

- **Root cause**: Pre-built PyTorch binaries don't include hipBLASLt support (a newer high-performance BLAS library)
- **What happens**: PyTorch tries to use hipBLASLt, fails, and automatically falls back to `hipblas` (the standard BLAS library)
- **Performance impact**: None - `hipblas` works perfectly fine for training and generation
- **Hardware support**: Modern AMD GPUs (RDNA 2/3, e.g., RX 6000/7000 series) support hipBLASLt at the hardware level, but PyTorch binaries aren't compiled with it

**This warning can be safely ignored.** It appears once per run and doesn't affect training quality, speed, or stability. The fallback to `hipblas` is automatic and transparent.

To eliminate the warning entirely, you'd need to build PyTorch from source with hipBLASLt support (complex setup), but this provides no practical benefit for this educational project.

See [`commands/train.py`](commands/train.py) for complete training documentation.

## Text Generation

### Quick Examples

```bash
# Use preset strategies
python main.py generate --checkpoint checkpoints/model_epoch_10_cl100k.pt --preset creative
python main.py generate --checkpoint checkpoints/model_epoch_10_cl100k.pt --preset precise

# Custom parameters
python main.py generate \
    --checkpoint checkpoints/model_epoch_10_cl100k.pt \
    --strategy top_k_top_p \
    --top-k 50 --top-p 0.9 --temperature 0.8
```

See [`src/transformer/attention.py`](src/transformer/attention.py) for KV-Cache implementation details and [`commands/benchmark_generation.py`](commands/benchmark_generation.py) to benchmark the speedup.

## Model Interpretability

Understand what your transformer has learned using mechanistic interpretability tools.

### Logit Lens

Visualize how predictions evolve through layers:

```bash
# Demo mode with educational examples
python main.py interpret logit-lens checkpoints/model.pt --demo

# Analyze specific text
python main.py interpret logit-lens checkpoints/model.pt \
    --text "The capital of France is"

# Interactive mode
python main.py interpret logit-lens checkpoints/model.pt --interactive
```

**What it shows:** How the model's predictions change at each layer, revealing when the "correct answer" emerges.

**Example:** For "The capital of France is", you might see:
- Layer 0: Predicts generic tokens ("the", "a")
- Layer 3: Starting to converge ("Paris", "located")
- Layer 6: Confident final answer ("Paris")

### Attention Analysis

Understand what tokens each attention head focuses on:

```bash
# Demo mode - find common patterns
python main.py interpret attention checkpoints/model.pt --demo

# Analyze specific layer and head
python main.py interpret attention checkpoints/model.pt \
    --text "The cat sat on the mat" --layer 2 --head 3

# Interactive mode
python main.py interpret attention checkpoints/model.pt --interactive

# Find all heads matching a pattern
python main.py interpret attention checkpoints/model.pt \
    --text "Hello world"  # Shows pattern summary
```

**What it shows:** Which tokens each head attends to, revealing learned patterns like:
- **Previous token heads**: Always look at position i-1
- **Uniform heads**: Spread attention evenly (averaging)
- **Start token heads**: Focus on beginning of sequence
- **Sparse heads**: Concentrate on few key tokens

**Example patterns you might discover:**
- Head 2.3 implements a "previous token" circuit
- Head 4.1 averages information uniformly
- Head 1.0 focuses on the start token

### Induction Head Detection

Find and analyze induction heads - circuits that implement pattern matching and in-context learning:

```bash
# Detect induction heads across all layers
python main.py interpret induction-heads checkpoints/model.pt

# Custom detection parameters
python main.py interpret induction-heads checkpoints/model.pt \
    --num-sequences 50 --seq-length 30 --top-k 5
```

**What it shows:** Ranks attention heads by their ability to perform pattern matching (induction):
- **Prefix matching score**: How well the head attends to positions where the previous token matches
- **Circuit analysis**: Identifies pairs of heads working together (previous token head + induction head)
- **Pattern strength**: Quantitative measure of induction behavior

**What are induction heads?**
Induction heads are a key circuit discovered in transformer models that enable in-context learning. Given a repeated pattern like `A B C ... A B [?]`, the induction head learns to predict `C` by copying from the earlier occurrence. This involves two heads working together:
1. **Previous token head** (earlier layer): Attends to position i-1
2. **Induction head** (later layer): Finds matching prefixes and copies what came after

This circuit is crucial for few-shot learning and emerges suddenly during training ("grokking").

### Activation Patching

Test which components are **causally responsible** for specific behaviors through intervention experiments:

```bash
# Test which layers are critical for a prediction
python main.py interpret patch checkpoints/model.pt \
    --clean "The Eiffel Tower is in" \
    --corrupted "The Empire State Building is in" \
    --target "Paris"
```

**What it shows:** Recovery rates showing how much each layer contributes to the behavior:
- **Recovery Rate**: % of correct behavior restored when patching that layer
- **Ranked Results**: Layers sorted by causal importance
- **Color-coded Assessment**: Critical (>70%) / Important (>40%) / Moderate (>20%) / Minimal (<20%)

**How it works:**
1. Run model on "clean" input (correct behavior) and "corrupted" input (incorrect behavior)
2. For each layer, estimate what would happen if we swapped clean activations into corrupted run
3. Measure how much this restores the correct prediction
4. High recovery = that layer is causally important for this behavior

**Example interpretation:** If patching Layer 4 gives 85% recovery, that layer is critical for predicting "Paris" after "The Eiffel Tower". This is **causal** evidence, not just correlation!

---

See [`src/transformer/interpretability/`](src/transformer/interpretability/) for implementation details and [`commands/interpret.py`](commands/interpret.py) for all available tools.

## Development

### Running Tests

```bash
# All tests
pytest

# Specific components
pytest tests/test_attention.py -v
pytest tests/test_sampling.py -v
pytest tests/test_perplexity.py -v

# With coverage
pytest --cov=src/transformer
```

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
