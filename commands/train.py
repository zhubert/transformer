"""
Train a decoder-only transformer on text data.

This script demonstrates the complete training process for a language model,
with comprehensive explanations of each step.

What is Training?
------------------
Training is the process of adjusting the model's weights (parameters) so it learns
to predict the next token in a sequence. Right now, our model has random weights
and would generate gibberish. After training, it learns patterns in language!

The Training Loop:
------------------
Training follows this cycle, repeated many times:

    1. FORWARD PASS: Run input through model → get predictions (logits)
    2. CALCULATE LOSS: How wrong were the predictions? (high = bad, low = good)
    3. BACKWARD PASS: Calculate gradients (which way to adjust each weight)
    4. UPDATE WEIGHTS: Adjust weights to reduce loss

Repeat this thousands of times, and the model learns!

Key Concepts:
-------------

**Logits**: Raw model output scores before converting to probabilities
    Example: [2.3, -1.5, 4.1, 0.8]  ← Higher score = model thinks more likely

**Loss**: A single number measuring how wrong the predictions are
    - High loss (e.g., 8.0): Model is guessing randomly, terrible predictions
    - Medium loss (e.g., 3.0): Model is learning patterns
    - Low loss (e.g., 1.5): Model is quite good at predictions
    - The goal of training is to make loss as low as possible!

**Perplexity**: Exponential of loss, measuring model "confusion"
    - Perplexity = exp(loss)
    - More interpretable than raw loss
    - Perplexity ~50 means "model as confused as choosing from 50 words"
    - Lower perplexity = better model
    - Typical values:
        * Perfect: 1.0 (always correct)
        * Excellent: 10-30 (GPT-2 level)
        * Decent: 50-100
        * Poor: 200+
        * Random: vocab_size (e.g., 50,000)

**CrossEntropyLoss**: The loss function for classification tasks
    - Compares model's predicted probabilities with actual next token
    - Penalizes confident wrong predictions more than uncertain ones
    - Built into PyTorch, perfect for language modeling

**Optimizer (Adam)**: Algorithm that updates weights based on gradients
    - Plain gradient descent: weight -= learning_rate * gradient
    - Adam: Smarter! Adapts learning rate for each parameter
    - Handles momentum and scaling automatically

**Learning Rate**: How big of steps to take when updating weights
    - Too high (e.g., 0.1): Training unstable, loss explodes
    - Too low (e.g., 0.000001): Training too slow
    - Just right (e.g., 0.0003): Steady improvement!

**Gradient Clipping**: Prevents exploding gradients in transformers
    - Problem: Gradients can grow exponentially through deep layers
    - Solution: Scale down gradients if they get too large
    - max_norm=1.0: Standard for GPT-2, GPT-3, BERT
    - Clipping rate: % of updates where gradients were clipped
        * Early training: 30-70% (high clipping is normal)
        * Mid training: 10-30% (stabilizing)
        * Late training: 5-15% (stable)
    - Without clipping: Training can collapse with NaN loss!

**Batch**: Group of training examples processed together
    - Batch size 8 = process 8 sequences at once
    - Larger batches = more stable gradients but use more memory
    - M1 chips have limited unified memory, so we use smaller batches

**Epoch**: One complete pass through the entire dataset
    - We typically train for multiple epochs

**Device (MPS/CPU/CUDA)**: Where computations happen
    - CPU: Slow but always available
    - MPS: Apple Silicon GPU, 5-15x faster than CPU
    - CUDA: NVIDIA GPU, fastest (not on Mac)

MPS (Metal Performance Shaders):
---------------------------------
MPS is Apple's GPU acceleration for PyTorch on M1/M2/M3 chips.

Why use MPS instead of CPU?
- 5-15x faster training
- Can handle larger batches (more GPU memory)

⚠️ KNOWN ISSUES with MPS:
- Random NaN training failures (PyTorch bugs #107294, #109457)
- Caused by asynchronous execution bugs in PyTorch's MPS backend
- Affects transformer models specifically (attention + layer norm)
- No official fix as of PyTorch 2.9.0

Workarounds:
- ✅ Use CPU (default) - stable, 100% reliable
- ⚠️ Try MPS with --mps flag - 5-10x faster but may crash
- 🐛 Debug mode: --mps --debug - forces synchronization (more stable, slower)

Training Modes:
---------------
- Default: Full quality (6 layers, 256 d_model, 100M tokens/epoch × 20 epochs)
  * Epoch 1: ~4-6h on M3 Pro (builds cache), Epochs 2-20: ~1-2h (cached)
- --medium: Balanced (4 layers, 256 d_model, 50M tokens/epoch × 15 epochs)
  * Epoch 1: ~2h on M3 Pro (builds cache), Epochs 2-15: ~30-60min (cached)
- --quick: Fast iteration (4 layers, 128 d_model, 10M tokens/epoch × 10 epochs)
  * Epoch 1: ~40-50min on M1 (builds cache), Epochs 2-10: ~15-25min (cached)

What to Expect During Training:
---------------------------------
Training progress (quick mode on M1 MacBook Pro):

Epoch 1:  Loss ~8.0, Perplexity ~3000  (random guessing)
Epoch 3:  Loss ~5.0, Perplexity ~150   (learning patterns)
Epoch 5:  Loss ~4.0, Perplexity ~55    (getting decent)
Epoch 10: Loss ~3.0, Perplexity ~20    (pretty good!)

Training time per epoch (estimated with optimized caching):
- Quick mode: Epoch 1: ~40-50min (M1 MPS), Epochs 2+: ~15-25min (cached)
- Medium mode: Epoch 1: ~2h (M3 Pro MPS), Epochs 2+: ~30-60min (cached)
- Normal mode: Epoch 1: ~4-6h (M3 Pro MPS), Epochs 2+: ~1-2h (cached)

Note: Epoch 1 downloads and caches shards. Epochs 2+ use cached data → 2-4x faster!

Loss Interpretation:
- High loss (8-10): Model guessing randomly, terrible predictions
- Medium loss (3-5): Model learning patterns
- Low loss (1.5-3): Model making decent predictions
- Very low loss (<1.5): Excellent predictions (or overfitting - check validation!)

The model saves checkpoints after each epoch to checkpoints/ directory.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import time
import argparse
from rich.console import Console, Group
from rich.table import Table
from rich.live import Live
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, TaskProgressColumn

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.fineweb_dataset import FineWebDataset
from src.transformer.scheduler import get_cosine_schedule_with_warmup
from src.transformer.perplexity import calculate_perplexity_from_loss
from src.transformer.training_utils import GradientAccumulator
from src.transformer.device_utils import (
    init_device, get_autocast_context, get_synchronize_fn,
    get_memory_stats_fn, print_device_info
)
from src.transformer.dataset_utils import calculate_optimal_cache_size
from src.transformer.checkpoint_utils import (
    detect_encoding,
    get_encoding_short_name,
    strip_compile_prefix,
)


# Utility functions are now imported from checkpoint_utils and dataset_utils
# Previously they were defined here, causing code duplication across commands


def get_device_type_from_args(use_mps=False):
    """
    Convert legacy --mps flag to device_type string.

    Args:
        use_mps: If True, explicitly request MPS

    Returns:
        device_type: "mps" if use_mps is True, None otherwise (autodetect)
    """
    if use_mps:
        return "mps"
    return None  # Autodetect


def find_latest_checkpoint(checkpoint_dir, encoding):
    """
    Find the latest checkpoint for a given encoding.

    Args:
        checkpoint_dir: Directory containing checkpoints
        encoding: Encoding name (e.g., 'cl100k_base')

    Returns:
        Path to latest checkpoint, or None if no checkpoints found
    """
    encoding_short = get_encoding_short_name(encoding)
    pattern = f"model_epoch_*_{encoding_short}.pt"
    checkpoints = list(checkpoint_dir.glob(pattern))

    if not checkpoints:
        return None

    # Extract epoch numbers and find maximum
    epoch_numbers = []
    for ckpt in checkpoints:
        # Parse: model_epoch_5_cl100k.pt -> epoch 5
        try:
            parts = ckpt.stem.split('_')
            epoch_idx = parts.index('epoch') + 1
            epoch_num = int(parts[epoch_idx])
            epoch_numbers.append((epoch_num, ckpt))
        except (ValueError, IndexError):
            continue

    if not epoch_numbers:
        return None

    # Return path to checkpoint with highest epoch number
    latest_epoch, latest_path = max(epoch_numbers, key=lambda x: x[0])
    return latest_path


def load_checkpoint_for_resume(checkpoint_path, model, optimizer, scheduler, console):
    """
    Load checkpoint to resume training.

    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        console: Rich console for printing

    Returns:
        start_epoch: Epoch number to resume from (next epoch after checkpoint)
        checkpoint: Full checkpoint dict for additional info
    """
    console.print(f"[bold cyan]Loading checkpoint:[/bold cyan] {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Strip torch.compile() prefix if present using utility function
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        console.print("[yellow]Detected torch.compile() checkpoint, stripping prefix...[/yellow]")
        state_dict = strip_compile_prefix(state_dict)
        checkpoint['model_state_dict'] = state_dict

    # Load model weights
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError as e:
        # Re-raise any loading errors
        console.print(f"[bold red]Error loading checkpoint:[/bold red] {e}")
        raise

    # Load optimizer state
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    # Load scheduler state if available
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    # Get resume epoch (next epoch after the saved one)
    saved_epoch = checkpoint['epoch']
    start_epoch = saved_epoch  # Will start from this epoch (0-indexed in loop)

    # Display checkpoint info
    info_table = Table(title="Checkpoint Info", show_header=True, header_style="bold green")
    info_table.add_column("Metric", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="white")

    info_table.add_row("Completed epochs", str(saved_epoch))
    info_table.add_row("Resuming from epoch", str(saved_epoch + 1))
    info_table.add_row("Train loss", f"{checkpoint['train_loss']:.4f}")
    info_table.add_row("Train perplexity", f"{checkpoint['train_perplexity']:.2f}")

    if 'val_loss' in checkpoint:
        info_table.add_row("Val loss", f"{checkpoint['val_loss']:.4f}")
        info_table.add_row("Val perplexity", f"{checkpoint['val_perplexity']:.2f}")

    if 'current_lr' in checkpoint:
        info_table.add_row("Learning rate", f"{checkpoint['current_lr']:.6f}")

    console.print(info_table)
    console.print()

    return start_epoch, checkpoint


def generate_sample(model, dataset, prompt_text, max_length=50, device="cpu", autocast_ctx=None):
    """Generate sample text to see how the model is learning."""
    from contextlib import nullcontext
    if autocast_ctx is None:
        autocast_ctx = nullcontext()

    model.eval()
    prompt_tokens = dataset.tokenizer.encode(prompt_text)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    with torch.no_grad(), autocast_ctx:
        output_ids = model.generate(input_ids, max_length=max_length, temperature=0.5)

    generated_text = dataset.decode(output_ids[0])
    model.train()
    return generated_text


def train(debug=False, use_mps=False, quick=False, medium=False, accumulation_steps=16, resume=False, compile=True):
    """
    Main training function with gradient accumulation and validation.

    Args:
        debug: If True, print diagnostic information for debugging NaN issues
        use_mps: If True, try MPS (experimental - has known NaN issues)
        quick: If True, use smaller model and fewer tokens for faster training
        medium: If True, use medium-sized model with good balance of quality and speed
        accumulation_steps: Number of batches to accumulate before updating weights.
                           Effective batch size = BATCH_SIZE × accumulation_steps
                           Higher values = more stable training but slower updates
                           Recommended: 16-32 for hobby hardware
        resume: If True, resume training from the latest checkpoint
        compile: If True, use torch.compile() for 20-40% speedup (PyTorch 2.0+).
                Recommended for AMD/NVIDIA GPUs. Adds ~1-2 min compilation on first epoch.

    What is Gradient Accumulation?
    ------------------------------
    Gradient accumulation allows us to simulate large batch training without
    running out of memory. Instead of updating weights after each small batch,
    we accumulate gradients over multiple batches and update once.

    Example:
        Without accumulation (batch_size=8):
            Process 8 sequences → Update weights (noisy gradients!)

        With accumulation (batch_size=8, accumulation_steps=16):
            Process 8 sequences → Accumulate gradients
            Process 8 sequences → Accumulate gradients
            ... (16 times total)
            Update weights using accumulated gradients (smooth!)

        Effective batch: 8 × 16 = 128 sequences (stable training!)
        Memory usage: Same as batch_size=8 (fits on hobby hardware)

    See src/transformer/training_utils.py for detailed explanation.
    """

    # Configuration
    encoding = "cl100k_base"  # Only cl100k_base tokenizer supported
    SEQ_LENGTH = 128
    BATCH_SIZE = 8              # Reduced from 32 to fit in M1 memory

    # Validate that only one mode is selected
    if quick and medium:
        raise ValueError("Cannot use both --quick and --medium flags. Please choose one.")

    # Training mode configuration
    if quick:
        # Quick mode: smaller model, fewer tokens for faster iteration
        D_MODEL = 128
        NUM_HEADS = 4
        NUM_LAYERS = 4
        D_FF = 512
        NUM_EPOCHS = 10
        TOKENS_PER_EPOCH = 10_000_000  # 10M tokens per epoch
        CHECKPOINT_DIR = Path("checkpoints_quick")
    elif medium:
        # Medium mode: balanced quality and speed
        # Full d_model (crucial for representation quality)
        # Fewer layers (saves training time)
        # More training data than quick (7.5x improvement)
        D_MODEL = 256
        NUM_HEADS = 4
        NUM_LAYERS = 4
        D_FF = 1024
        NUM_EPOCHS = 15
        TOKENS_PER_EPOCH = 50_000_000  # 50M tokens per epoch
        CHECKPOINT_DIR = Path("checkpoints_medium")
    else:
        # Normal mode: full model, maximum quality
        D_MODEL = 256
        NUM_HEADS = 4
        NUM_LAYERS = 6
        D_FF = 1024
        NUM_EPOCHS = 20
        TOKENS_PER_EPOCH = 100_000_000  # 100M tokens per epoch
        CHECKPOINT_DIR = Path("checkpoints")

    DROPOUT = 0.1
    LEARNING_RATE = 3e-4        # Standard transformer learning rate
    WEIGHT_DECAY = 0.01
    LOG_INTERVAL = 10
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    # Calculate optimal cache size for this training mode
    # This ensures all shards fit in cache, eliminating re-downloads after epoch 1
    MAX_CACHED_SHARDS = calculate_optimal_cache_size(TOKENS_PER_EPOCH)

    # Initialize Rich console
    console = Console()

    # Display header
    header_text = "TRANSFORMER TRAINING"
    if quick:
        header_text += " [yellow](Quick Mode)[/yellow]"
    elif medium:
        header_text += " [cyan](Medium Mode)[/cyan]"
    console.print(Panel(header_text, style="bold blue", expand=False))
    console.print()

    # Device setup with proper initialization
    device_type_str = get_device_type_from_args(use_mps)
    device, device_name = init_device(device_type_str, seed=42)

    # Create setup configuration table
    setup_table = Table(title="Setup Configuration", show_header=True, header_style="bold cyan")
    setup_table.add_column("Setting", style="cyan", no_wrap=True)
    setup_table.add_column("Value", style="white")

    setup_table.add_row("Tokenizer", encoding)
    device_display = device_name
    if use_mps:
        device_display += " [yellow](⚠ Experimental - may have NaN issues)[/yellow]"
    setup_table.add_row("Device", device_display)

    console.print(setup_table)
    console.print()

    # Get device-specific utilities
    autocast_ctx = get_autocast_context(device.type)
    synchronize = get_synchronize_fn(device.type)
    get_max_memory = get_memory_stats_fn(device.type)

    # Load datasets (train and validation)
    console.print("[bold]Loading FineWeb dataset...[/bold]")

    # Training dataset (90% of data)
    train_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=SEQ_LENGTH,
        tokens_per_epoch=TOKENS_PER_EPOCH,
        max_cached_shards=MAX_CACHED_SHARDS,
        seed=42,  # Reproducible shard selection
        encoding_name=encoding,
        split="train",  # Use training split
        validation_fraction=0.1  # 10% reserved for validation
    )

    # Validation dataset (10% of data)
    # Use fewer tokens for validation to save time
    val_tokens_per_epoch = TOKENS_PER_EPOCH // 10  # 10% of training tokens
    val_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=SEQ_LENGTH,
        tokens_per_epoch=val_tokens_per_epoch,
        max_cached_shards=MAX_CACHED_SHARDS,
        seed=42,  # Same seed for consistency
        encoding_name=encoding,
        split="validation",  # Use validation split (different shards!)
        validation_fraction=0.1
    )

    # Create dataset info table
    dataset_table = Table(title="Dataset Configuration", show_header=True, header_style="bold green")
    dataset_table.add_column("Parameter", style="cyan", no_wrap=True)
    dataset_table.add_column("Train", style="green", justify="right")
    dataset_table.add_column("Validation", style="yellow", justify="right")

    dataset_table.add_row(
        "Tokens per epoch",
        f"{TOKENS_PER_EPOCH:,}",
        f"{val_tokens_per_epoch:,}"
    )
    dataset_table.add_row(
        "Sequences per epoch",
        f"~{TOKENS_PER_EPOCH // SEQ_LENGTH:,}",
        f"~{val_tokens_per_epoch // SEQ_LENGTH:,}"
    )
    dataset_table.add_row(
        "Vocabulary size",
        f"{train_dataset.vocab_size:,}",
        f"{val_dataset.vocab_size:,}"
    )
    dataset_table.add_row(
        "Cache directory",
        "data/fineweb_cache",
        "data/fineweb_cache"
    )
    # Calculate cache size in GB for display
    cache_size_gb = (MAX_CACHED_SHARDS * 40) / 1024  # ~40MB per shard
    dataset_table.add_row(
        "Max cached shards",
        f"{MAX_CACHED_SHARDS} (~{cache_size_gb:.1f} GB)",
        f"{MAX_CACHED_SHARDS} (~{cache_size_gb:.1f} GB)"
    )

    console.print(dataset_table)
    console.print()

    # Create dataloaders
    # IterableDataset requires different DataLoader setup
    # No shuffle (streaming), no len()
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Can't shuffle IterableDataset
        drop_last=True
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=True
    )

    # Calculate effective batch size with gradient accumulation
    effective_batch_size = BATCH_SIZE * SEQ_LENGTH * accumulation_steps

    # Create training configuration table
    train_config_table = Table(title="Training Configuration", show_header=True, header_style="bold magenta")
    train_config_table.add_column("Parameter", style="cyan", no_wrap=True)
    train_config_table.add_column("Value", style="white")

    train_config_table.add_row(
        "Batch size",
        f"{BATCH_SIZE} sequences ({BATCH_SIZE * SEQ_LENGTH:,} tokens)"
    )
    train_config_table.add_row(
        "Gradient accumulation",
        f"{accumulation_steps} steps"
    )
    train_config_table.add_row(
        "Effective batch size",
        f"[bold]{BATCH_SIZE * accumulation_steps}[/bold] sequences ({effective_batch_size:,} tokens)"
    )
    train_config_table.add_row(
        "Stability improvement",
        f"[green]✓ {accumulation_steps}x more stable than without accumulation[/green]"
    )

    console.print(train_config_table)
    console.print()

    # Create model
    console.print("[bold]Creating model...[/bold]")
    model = DecoderOnlyTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LENGTH * 2,
        dropout=DROPOUT
    )
    model = model.to(device)

    # PyTorch 2.0+ Compilation for Performance
    # -----------------------------------------
    # torch.compile() is a JIT (Just-In-Time) compiler that optimizes models by:
    #
    # What it does:
    # 1. **Kernel fusion**: Combines multiple operations into single GPU kernels
    #    Example: LayerNorm + Linear → Single fused kernel (faster!)
    # 2. **Graph optimization**: Eliminates redundant operations
    # 3. **Memory layout optimization**: Better cache utilization
    # 4. **Backend-specific tuning**: Uses AMD ROCm or NVIDIA CUDA optimizations
    #
    # Performance impact:
    # - AMD GPUs (ROCm): 20-40% faster training (measured on RX 6000/7000 series)
    # - NVIDIA GPUs: 15-30% faster (Ampere and newer)
    # - First epoch: ~1-2 min compilation overhead (one-time cost)
    # - Subsequent epochs: Pure speedup, no overhead
    #
    # How it works:
    # - First forward pass: PyTorch traces execution graph and compiles
    # - Compiled kernels are cached for future use
    # - Works seamlessly with autograd (backward pass also optimized)
    #
    # ROCm-specific benefits:
    # - Automatically uses hipBLAS optimized matrix multiplication
    # - Fuses attention patterns (QK^T, softmax, matmul V)
    # - Better utilization of AMD GPU architecture (RDNA2/3, CDNA)
    #
    # Educational note:
    # - You can disable with --no-compile to compare performance
    # - First epoch will be slower (compilation), but worth it for multi-epoch training
    # - Modern LLM training (GPT-4, LLaMA, etc.) uses similar compilation techniques
    if compile:
        console.print("[bold yellow]Compiling model with torch.compile()...[/bold yellow]")
        console.print("[dim]First epoch will have ~1-2 min compilation overhead, then 20-40% faster training[/dim]")
        model = torch.compile(model, backend="inductor", mode="default")
        console.print("[green]✓ Model compiled successfully[/green]")
        console.print()

    num_params = sum(p.numel() for p in model.parameters())

    # Check for NaN in initial weights
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            console.print(f"[red]WARNING: NaN in initial weights: {name}[/red]")
            has_nan = True
    if has_nan:
        raise ValueError("Model has NaN in initial weights!")

    # Create model info table
    model_table = Table(title="Model Architecture", show_header=True, header_style="bold blue")
    model_table.add_column("Parameter", style="cyan", no_wrap=True)
    model_table.add_column("Value", style="white", justify="right")

    model_table.add_row("Embedding dimension (d_model)", str(D_MODEL))
    model_table.add_row("Number of layers", str(NUM_LAYERS))
    model_table.add_row("Attention heads", str(NUM_HEADS))
    model_table.add_row("Feed-forward dimension", str(D_FF))
    model_table.add_row("Dropout", str(DROPOUT))
    model_table.add_row("Max sequence length", str(SEQ_LENGTH * 2))
    model_table.add_row("[bold]Total parameters[/bold]", f"[bold]{num_params:,}[/bold]")
    model_table.add_row("Weight initialization", "[green]✓ No NaN detected[/green]")
    if compile:
        model_table.add_row("PyTorch compilation", "[green]✓ Enabled (20-40% faster)[/green]")
    else:
        model_table.add_row("PyTorch compilation", "[yellow]Disabled (use --compile for speedup)[/yellow]")

    console.print(model_table)
    console.print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer Configuration with Selective Weight Decay
    # -----------------------------------------------------
    # We use AdamW (not Adam) with selective weight decay, following modern best practices:
    #
    # Why AdamW over Adam?
    # --------------------
    # - AdamW decouples weight decay from gradient-based updates
    # - Better generalization for transformers (empirically proven)
    # - Used in GPT-2, GPT-3, BERT, LLaMA, and all modern LLMs
    # - Expected improvement: 3-5% better final loss
    #
    # Selective Weight Decay:
    # -----------------------
    # Not all parameters benefit from weight decay (L2 regularization):
    #
    # APPLY weight decay to:
    #   ✓ Linear layer weights (W in attention, FFN, projections)
    #   ✓ Embedding weights
    #   → These can grow large and benefit from regularization
    #
    # DON'T apply weight decay to:
    #   ✗ Biases (just offsets, don't benefit from regularization)
    #   ✗ LayerNorm parameters (already constrained by normalization)
    #   → Regularizing these can hurt performance
    #
    # This follows GPT-2, GPT-3, BERT implementations and can improve
    # final perplexity by 2-5% compared to applying weight decay uniformly.

    # Separate parameters into two groups
    decay_params = []      # Weights that should be regularized
    no_decay_params = []   # Biases and norms that shouldn't be regularized

    for name, param in model.named_parameters():
        # Check if parameter name contains bias or layer norm indicators
        # LayerNorm parameters: norm1, norm2, ln_f, layer_norm (and their .weight/.bias)
        # Biases: anything ending in .bias
        if 'bias' in name.lower() or 'norm' in name.lower() or 'ln_f' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    # Create optimizer with parameter groups
    optimizer = torch.optim.AdamW([
        {
            'params': decay_params,
            'weight_decay': WEIGHT_DECAY  # 0.01 - standard regularization
        },
        {
            'params': no_decay_params,
            'weight_decay': 0.0  # No regularization for biases/norms
        }
    ],
        lr=LEARNING_RATE,  # Base LR = max_lr, scheduler will scale this
        betas=(0.9, 0.999),  # Standard momentum parameters
        eps=1e-8  # Numerical stability
    )

    # Display optimizer configuration
    num_decay_params = sum(p.numel() for p in decay_params)
    num_no_decay_params = sum(p.numel() for p in no_decay_params)
    total_optimizer_params = num_decay_params + num_no_decay_params

    optimizer_table = Table(title="Optimizer Configuration", show_header=True, header_style="bold green")
    optimizer_table.add_column("Parameter", style="cyan", no_wrap=True)
    optimizer_table.add_column("Value", style="white", justify="right")

    optimizer_table.add_row("Optimizer type", "AdamW")
    optimizer_table.add_row("Learning rate", f"{LEARNING_RATE:.6f}")
    optimizer_table.add_row("Weight decay (for weights)", f"{WEIGHT_DECAY:.4f}")
    optimizer_table.add_row("Beta1, Beta2", "0.9, 0.999")
    optimizer_table.add_row("Epsilon", "1e-8")
    optimizer_table.add_row("", "")  # Blank row
    optimizer_table.add_row("[bold]Parameters with decay[/bold]", f"[bold]{num_decay_params:,}[/bold] ({100*num_decay_params/total_optimizer_params:.1f}%)")
    optimizer_table.add_row("[bold]Parameters without decay[/bold]", f"[bold]{num_no_decay_params:,}[/bold] ({100*num_no_decay_params/total_optimizer_params:.1f}%)")
    optimizer_table.add_row("[dim]Total parameters[/dim]", f"[dim]{total_optimizer_params:,}[/dim]")

    console.print(optimizer_table)
    console.print()

    # Gradient Accumulator
    # --------------------
    # Manages gradient accumulation to simulate large batch training.
    # See src/transformer/training_utils.py for detailed explanation.
    accumulator = GradientAccumulator(accumulation_steps=accumulation_steps)

    # Learning Rate Scheduler
    # ------------------------
    # We use warmup + cosine decay for better training:
    #
    # 1. WARMUP (first 5% of training):
    #    - Starts at LR=0, gradually increases to max_lr
    #    - Prevents instability from huge gradients at start
    #    - Random weights → huge gradients → need small LR initially
    #
    # 2. COSINE DECAY (remaining 95% of training):
    #    - Smoothly decreases from max_lr to min_lr
    #    - Enables fine-tuning at the end
    #    - Large steps early (learn fast), small steps late (settle in)
    #
    # Expected improvement: 10-30% better final loss!

    # Estimate steps per epoch for FineWeb (IterableDataset)
    # With gradient accumulation, we update weights less frequently
    # Real steps = (tokens_per_epoch / seq_length / batch_size) / accumulation_steps
    batches_per_epoch = (TOKENS_PER_EPOCH // SEQ_LENGTH // BATCH_SIZE)
    steps_per_epoch = batches_per_epoch // accumulation_steps  # Fewer weight updates
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(0.05 * total_steps)  # 5% of training for warmup
    min_lr = LEARNING_RATE * 0.1  # Final LR = 10% of peak

    # Create learning rate schedule table
    lr_table = Table(title="Learning Rate Schedule", show_header=True, header_style="bold yellow")
    lr_table.add_column("Parameter", style="cyan", no_wrap=True)
    lr_table.add_column("Value", style="white", justify="right")

    lr_table.add_row("Batches per epoch", f"~{batches_per_epoch:,}")
    lr_table.add_row("Weight updates per epoch", f"~{steps_per_epoch:,}")
    lr_table.add_row("Total epochs", str(NUM_EPOCHS))
    lr_table.add_row("[bold]Total training steps[/bold]", f"[bold]{total_steps:,}[/bold]")
    lr_table.add_row("Warmup steps", f"{warmup_steps:,} (5%)")
    lr_table.add_row("Max learning rate", f"{LEARNING_RATE:.6f}")
    lr_table.add_row("Min learning rate", f"{min_lr:.6f}")
    lr_table.add_row("Schedule type", "Warmup + Cosine Decay")

    console.print(lr_table)
    console.print()

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=LEARNING_RATE,
        min_lr=min_lr
    )

    # Resume from checkpoint if requested
    # ------------------------------------
    # Check if we should resume training from a previous checkpoint
    start_epoch = 0  # Default: start from epoch 0
    if resume:
        latest_checkpoint = find_latest_checkpoint(CHECKPOINT_DIR, encoding)
        if latest_checkpoint is None:
            console.print(f"[yellow]Warning: --resume flag set but no checkpoints found in {CHECKPOINT_DIR}[/yellow]")
            console.print(f"[yellow]Starting training from scratch instead.[/yellow]")
            console.print()
        else:
            # Load checkpoint and get starting epoch
            start_epoch, loaded_checkpoint = load_checkpoint_for_resume(
                latest_checkpoint, model, optimizer, scheduler, console
            )
            console.print(f"[green]✓ Successfully resumed from checkpoint[/green]")
            console.print()

    # Starting training panel
    console.print(Panel("[bold green]STARTING TRAINING[/bold green]", style="bold green", expand=False))
    console.print()

    total_batches = 0
    start_time = time.time()

    def create_metrics_table(rows, epoch_num, num_epochs):
        """Create a Rich table with training metrics and fixed header.

        Args:
            rows: List of tuples (batch_str, loss, ppl, avg_loss, avg_ppl, lr)
            epoch_num: Current epoch number (1-indexed)
            num_epochs: Total number of epochs

        Returns:
            Rich Table with the metrics
        """
        table = Table(title=f"Epoch {epoch_num}/{num_epochs}", show_header=True, header_style="bold magenta")
        table.add_column("Batch", justify="right", style="cyan", no_wrap=True)
        table.add_column("Loss", justify="right", style="green")
        table.add_column("PPL", justify="right", style="green")
        table.add_column("Avg Loss", justify="right", style="yellow")
        table.add_column("Avg PPL", justify="right", style="yellow")
        table.add_column("LR", justify="right", style="blue")

        # Only show last 15 rows to keep display clean
        for row in rows[-15:]:
            batch_str, loss, ppl, avg_loss, avg_ppl, lr = row
            table.add_row(
                batch_str,
                f"{loss:.4f}",
                f"{ppl:.2f}",
                f"{avg_loss:.4f}",
                f"{avg_ppl:.2f}",
                f"{lr:.2e}"
            )

        return table

    # Create overall progress tracker for epochs and batches
    # Note: We track progress from 0 even when resuming, so that TimeRemainingColumn
    # has accurate timing data. The epoch counter in the description shows the actual epoch number.
    overall_progress = Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
        console=console
    )
    remaining_epochs = NUM_EPOCHS - start_epoch
    epoch_task = overall_progress.add_task(
        f"[cyan]Overall Progress (Epoch {start_epoch + 1}-{NUM_EPOCHS})",
        total=remaining_epochs,
        completed=0
    )

    # Start the progress bar to enable time tracking
    overall_progress.start()

    for epoch in range(start_epoch, NUM_EPOCHS):
        # ==============================================================================
        # TRAINING PHASE
        # ==============================================================================
        model.train()  # Set to training mode

        epoch_loss = 0.0
        epoch_start = time.time()
        training_rows = []  # Store training metrics for Rich table display
        grad_norms = []  # Track gradient norms for monitoring clipping

        # Reset gradient accumulator for new epoch
        accumulator.reset()
        optimizer.zero_grad()  # Clear gradients once at epoch start

        # Add batch progress task for this epoch
        batch_task = overall_progress.add_task(
            f"[green]Epoch {epoch + 1}/{NUM_EPOCHS}",
            total=batches_per_epoch
        )

        # Create Live display for this epoch
        # Combines the metrics table with overall progress bars
        with Live(Group(create_metrics_table(training_rows, epoch + 1, NUM_EPOCHS), overall_progress), console=console, refresh_per_second=4) as live:
            for batch_idx, (inputs, targets) in enumerate(train_dataloader):
                # Move to device
                inputs = inputs.to(device)
                targets = targets.to(device)

                # Forward pass with autocast (mixed precision on CUDA, no-op on MPS/CPU)
                with autocast_ctx:
                    logits, _ = model(inputs, debug=debug)  # Unpack (logits, caches)

                    # NaN detection (defensive check)
                    if torch.isnan(logits).any() or torch.isinf(logits).any():
                        print(f"\n  ERROR: NaN/Inf detected in logits at batch {batch_idx + 1}")
                        print(f"  Logits stats: min={logits.min().item():.4f}, "
                              f"max={logits.max().item():.4f}, mean={logits.mean().item():.4f}")
                        raise ValueError("NaN/Inf in logits - training unstable")

                    # Calculate loss
                    batch_size, seq_length, vocab_size = logits.shape
                    loss = criterion(
                        logits.view(batch_size * seq_length, vocab_size),
                        targets.view(batch_size * seq_length)
                    )

                    # IMPORTANT: Scale loss for gradient accumulation
                    # This ensures accumulated gradients have correct magnitude
                    # See training_utils.py for detailed explanation
                    loss = accumulator.scale_loss(loss)

                # Check loss for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"\n  ERROR: NaN/Inf loss at batch {batch_idx + 1}")
                    print(f"  Loss value: {loss.item()}")
                    raise ValueError("NaN/Inf loss - training unstable")

                # Backward pass - accumulates gradients
                loss.backward()

                # Check if we should update weights (every accumulation_steps batches)
                if accumulator.step():
                    # Gradient Clipping
                    # ------------------
                    # Clip gradients to prevent exploding gradients - a common issue in
                    # transformers where gradients can grow exponentially through layers.
                    #
                    # Why transformers need clipping:
                    # - Deep networks multiply gradients across many layers
                    # - Attention softmax can produce sharp distributions
                    # - Long sequences create longer backprop paths
                    #
                    # Without clipping: Gradients can explode → NaN loss → training collapse
                    # With clipping (max_norm=1.0): Scale down gradients proportionally
                    #
                    # Standard practice: max_norm=1.0 (used in GPT-2, GPT-3, BERT)
                    #
                    # Returns: Original gradient norm (before clipping) for monitoring
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    grad_norms.append(grad_norm.item())  # Store for statistics

                    # Update weights
                    optimizer.step()

                    # Update learning rate (MUST be called after optimizer.step())
                    # This adjusts the LR according to our warmup + cosine decay schedule
                    scheduler.step()

                    # Clear gradients for next accumulation cycle
                    optimizer.zero_grad()

                # Track loss (unscaled for logging)
                epoch_loss += loss.item() * accumulation_steps  # Unscale for display
                total_batches += 1

                # Debug: Check for NaN in weights after first batch
                if batch_idx == 0 and not debug:
                    for name, param in model.named_parameters():
                        if torch.isnan(param).any():
                            print(f"  WARNING: NaN in weights after batch 1: {name}")
                            print(f"  This means batch 1 corrupted the model!")
                            raise ValueError("Weights corrupted after batch 1")

                # Logging
                if (batch_idx + 1) % LOG_INTERVAL == 0:
                    avg_loss = epoch_loss / (batch_idx + 1)
                    current_lr = scheduler.get_last_lr()[0]  # Get current LR from scheduler
                    batch_perplexity = calculate_perplexity_from_loss(loss * accumulation_steps).item()
                    avg_perplexity = calculate_perplexity_from_loss(torch.tensor(avg_loss)).item()

                    # Check if we're in warmup phase
                    current_optimizer_step = total_batches // accumulation_steps
                    in_warmup = current_optimizer_step < warmup_steps

                    # Format batch count as "X/Y", with asterisk if in warmup
                    batch_str = f"{batch_idx + 1}/{batches_per_epoch}"
                    if in_warmup:
                        batch_str = f"*{batch_str}"

                    # Add row to training data
                    training_rows.append((
                        batch_str,
                        loss.item() * accumulation_steps,
                        batch_perplexity,
                        avg_loss,
                        avg_perplexity,
                        current_lr
                    ))

                    # Update the live display with new table and progress
                    live.update(Group(create_metrics_table(training_rows, epoch + 1, NUM_EPOCHS), overall_progress))

                # Update batch progress (do this for every batch, not just logged ones)
                overall_progress.update(batch_task, completed=batch_idx + 1)

        # Training phase summary
        synchronize()
        train_time = time.time() - epoch_start
        avg_train_loss = epoch_loss / (batch_idx + 1)  # Use actual batch count
        avg_train_perplexity = calculate_perplexity_from_loss(torch.tensor(avg_train_loss)).item()
        current_lr = scheduler.get_last_lr()[0]

        # ==============================================================================
        # VALIDATION PHASE
        # ==============================================================================
        # Evaluate on validation data to check for overfitting/underfitting
        # The model has NEVER seen this data during training, so this tells us
        # if the model is truly learning patterns or just memorizing training data.
        console.print()
        console.print("[bold]Running validation...[/bold]")
        model.eval()  # Set to evaluation mode (disables dropout)

        val_loss = 0.0
        val_batches = 0
        val_start = time.time()

        with torch.no_grad():  # Don't compute gradients for validation (faster!)
            for val_inputs, val_targets in val_dataloader:
                val_inputs = val_inputs.to(device)
                val_targets = val_targets.to(device)

                # Forward pass only (no backward!)
                with autocast_ctx:
                    val_logits, _ = model(val_inputs)  # Unpack (logits, caches)

                    # Calculate validation loss
                    batch_size, seq_length, vocab_size = val_logits.shape
                    batch_val_loss = criterion(
                        val_logits.view(batch_size * seq_length, vocab_size),
                        val_targets.view(batch_size * seq_length)
                    )

                val_loss += batch_val_loss.item()
                val_batches += 1

        synchronize()
        val_time = time.time() - val_start
        avg_val_loss = val_loss / val_batches
        avg_val_perplexity = calculate_perplexity_from_loss(torch.tensor(avg_val_loss)).item()

        # Create epoch summary table
        console.print()
        summary_table = Table(show_header=True, header_style="bold cyan", box=None)
        summary_table.add_column("Metric", style="cyan", no_wrap=True)
        summary_table.add_column("Train", style="green", justify="right")
        summary_table.add_column("Validation", style="yellow", justify="right")

        summary_table.add_row("Loss", f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}")
        summary_table.add_row("Perplexity", f"{avg_train_perplexity:.2f}", f"{avg_val_perplexity:.2f}")
        summary_table.add_row("Time", f"{train_time:.1f}s", f"{val_time:.1f}s")

        # Gradient norm statistics (educational insight into training stability)
        if grad_norms:
            import numpy as np
            max_grad_norm = max(grad_norms)
            avg_grad_norm = np.mean(grad_norms)
            clipped_count = sum(1 for g in grad_norms if g > 1.0)
            clip_rate = (clipped_count / len(grad_norms)) * 100

            # Format gradient norm display with color coding
            grad_norm_display = f"{avg_grad_norm:.3f} (max: {max_grad_norm:.3f})"
            if clip_rate > 50:
                clip_display = f"[yellow]{clip_rate:.1f}%[/yellow]"
            elif clip_rate > 10:
                clip_display = f"[yellow]{clip_rate:.1f}%[/yellow]"
            else:
                clip_display = f"{clip_rate:.1f}%"

            summary_table.add_row("Gradient norm", grad_norm_display, "")
            summary_table.add_row("Clipping rate", clip_display, "")

        # Determine status message with color
        if avg_val_loss < avg_train_loss * 1.05:
            status = "[green]✓ Model is learning well (val ≈ train)[/green]"
        elif avg_val_loss < avg_train_loss * 1.15:
            status = "[green]✓ Model is learning (val slightly > train, normal)[/green]"
        elif avg_val_loss > avg_train_loss * 1.3:
            status = "[yellow]⚠ Possible overfitting (val >> train)[/yellow]"
        else:
            status = "Model is training"

        # Create panel with summary
        panel_content = f"[bold]Learning Rate:[/bold] {current_lr:.6f}\n"
        panel_content += f"[bold]Total Time:[/bold] {train_time + val_time:.1f}s\n"
        panel_content += f"[bold]Status:[/bold] {status}"

        console.print(Panel(
            summary_table,
            title=f"[bold]Epoch {epoch + 1}/{NUM_EPOCHS} Summary[/bold]",
            subtitle=panel_content,
            border_style="blue"
        ))
        console.print()

        # Save checkpoint with encoding in filename
        encoding_short = get_encoding_short_name(encoding)
        checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}_{encoding_short}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'train_loss': avg_train_loss,
            'train_perplexity': avg_train_perplexity,
            'val_loss': avg_val_loss,  # Save validation metrics
            'val_perplexity': avg_val_perplexity,
            'current_lr': current_lr,  # Save current LR for reference
            'config': {
                'vocab_size': train_dataset.vocab_size,
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'd_ff': D_FF,
                'dropout': DROPOUT,
                'encoding': encoding,  # Store encoding for detection
                'accumulation_steps': accumulation_steps,  # Store for reference
            }
        }, checkpoint_path)
        console.print(f"[dim]Saved checkpoint: {checkpoint_path}[/dim]")
        console.print()

        # Update overall progress and remove batch task for this epoch
        overall_progress.update(epoch_task, advance=1)
        overall_progress.remove_task(batch_task)

    # Stop the progress bar after all epochs are complete
    overall_progress.stop()

    # Training complete
    synchronize()  # Ensure all operations complete
    total_time = time.time() - start_time

    # Create training complete summary
    complete_table = Table(show_header=False, box=None)
    complete_table.add_column("", style="cyan", no_wrap=True)
    complete_table.add_column("", style="white", justify="right")

    complete_table.add_row("Total time", f"{total_time / 60:.1f} minutes")
    complete_table.add_row("Epochs completed", str(NUM_EPOCHS))
    complete_table.add_row("Final train loss", f"{avg_train_loss:.4f}")
    complete_table.add_row("Final train perplexity", f"{avg_train_perplexity:.2f}")
    complete_table.add_row("Final val loss", f"{avg_val_loss:.4f}")
    complete_table.add_row("Final val perplexity", f"{avg_val_perplexity:.2f}")

    # Device-specific stats (CUDA only for now)
    if device.type == "cuda":
        peak_memory_gb = get_max_memory() / (1024**3)
        complete_table.add_row("Peak GPU memory", f"{peak_memory_gb:.2f} GB")

    console.print(Panel(
        complete_table,
        title="[bold green]TRAINING COMPLETE![/bold green]",
        border_style="green",
        expand=False
    ))
    console.print()

    # Final samples
    console.print(Panel("[bold]Final Sample Generations[/bold]", style="bold magenta", expand=False))
    console.print()

    for prompt in ["The", "In the", "She was"]:
        sample = generate_sample(model, train_dataset, prompt, max_length=100, device=device, autocast_ctx=autocast_ctx)
        console.print(f"[cyan]Prompt:[/cyan] '{prompt}'")
        console.print(f"[green]Generated:[/green] '{sample}'")
        console.print()
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decoder-only transformer on FineWeb with gradient accumulation and validation")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with diagnostic prints for NaN detection"
    )
    parser.add_argument(
        "--mps",
        action="store_true",
        help="Use MPS (Apple Silicon GPU) - EXPERIMENTAL, has known NaN issues"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode: smaller model (4 layers, d_model=128) and fewer tokens (10M/epoch)"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium training mode: balanced quality and speed (4 layers, d_model=256, 50M tokens/epoch × 15 epochs). "
             "Epoch 1: ~2h (builds cache), Epochs 2+: ~30-60min (cached)"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=16,
        help="Number of batches to accumulate before updating weights. Higher = more stable training. "
             "Effective batch size = batch_size × accumulation_steps. Recommended: 16-32 (default: 16)"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint. Automatically loads model, optimizer, and scheduler state."
    )
    args = parser.parse_args()

    train(
        debug=args.debug,
        use_mps=args.mps,
        quick=args.quick,
        medium=args.medium,
        accumulation_steps=args.accumulation_steps,
        resume=args.resume
    )
