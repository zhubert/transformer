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

What to Expect During Training:
---------------------------------
Training progress (quick mode on M1 MacBook Pro):

Epoch 1:  Loss ~8.0, Perplexity ~3000  (random guessing)
Epoch 3:  Loss ~5.0, Perplexity ~150   (learning patterns)
Epoch 5:  Loss ~4.0, Perplexity ~55    (getting decent)
Epoch 10: Loss ~3.0, Perplexity ~20    (pretty good!)

Training time per epoch:
- CPU: ~10-15 minutes (quick mode)
- MPS: ~2-3 minutes (if stable)

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

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.fineweb_dataset import FineWebDataset
from src.transformer.scheduler import get_cosine_schedule_with_warmup
from src.transformer.perplexity import calculate_perplexity_from_loss
from src.transformer.device_utils import (
    init_device, get_autocast_context, get_synchronize_fn,
    get_memory_stats_fn, print_device_info
)


def detect_encoding_from_checkpoint(checkpoint):
    """
    Detect encoding from checkpoint, with backward compatibility.

    Args:
        checkpoint: Loaded checkpoint dict

    Returns:
        encoding_name: String like 'p50k_base' or 'cl100k_base'
    """
    # Preferred: use stored encoding (new checkpoints)
    if 'encoding' in checkpoint.get('config', {}):
        return checkpoint['config']['encoding']

    # Fallback: infer from vocab_size (backward compatibility with old checkpoints)
    vocab_size = checkpoint['config']['vocab_size']
    if vocab_size == 50281:
        return 'p50k_base'
    elif vocab_size == 100277:
        return 'cl100k_base'
    else:
        raise ValueError(f"Unknown vocab size: {vocab_size}. Cannot detect encoding.")


def get_encoding_short_name(encoding):
    """
    Convert full encoding name to short version for filenames.

    Args:
        encoding: Full encoding name like 'p50k_base' or 'cl100k_base'

    Returns:
        Short name like 'p50k' or 'cl100k'
    """
    if encoding == 'p50k_base':
        return 'p50k'
    elif encoding == 'cl100k_base':
        return 'cl100k'
    else:
        # Fallback: just remove '_base' suffix if present
        return encoding.replace('_base', '')


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


def train(debug=False, use_mps=False, encoding="p50k_base", quick=False):
    """
    Main training function.

    Args:
        debug: If True, print diagnostic information for debugging NaN issues
        use_mps: If True, try MPS (experimental - has known NaN issues)
        encoding: Tokenizer encoding to use ("p50k_base" or "cl100k_base")
        quick: If True, use smaller model and fewer tokens for faster training
    """

    # Configuration
    SEQ_LENGTH = 128
    BATCH_SIZE = 8              # Reduced from 32 to fit in M1 memory

    # Quick mode: smaller model, fewer tokens for faster iteration
    if quick:
        D_MODEL = 128
        NUM_HEADS = 4
        NUM_LAYERS = 4
        D_FF = 512
        NUM_EPOCHS = 10
        TOKENS_PER_EPOCH = 10_000_000  # 10M tokens per epoch
        CHECKPOINT_DIR = Path("checkpoints_quick")
    else:
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
    MAX_CACHED_SHARDS = 5           # Keep 5 shards in cache (~2GB)
    LOG_INTERVAL = 10
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("TRANSFORMER TRAINING")
    print("=" * 80)
    print()

    if quick:
        print("Quick mode enabled: smaller model, fewer tokens")
        print()

    print(f"Tokenizer encoding: {encoding}")
    print()

    # Device setup with proper initialization
    device_type_str = get_device_type_from_args(use_mps)
    device, device_name = init_device(device_type_str, seed=42)
    print(f"Device: {device_name}")
    if use_mps:
        print("  WARNING: MPS has known NaN issues. Use --debug if training fails.")
    print()

    # Get device-specific utilities
    autocast_ctx = get_autocast_context(device.type)
    synchronize = get_synchronize_fn(device.type)
    get_max_memory = get_memory_stats_fn(device.type)

    # Load dataset
    print("Loading FineWeb dataset...")
    dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=SEQ_LENGTH,
        tokens_per_epoch=TOKENS_PER_EPOCH,
        max_cached_shards=MAX_CACHED_SHARDS,
        seed=42,  # Reproducible shard selection
        encoding_name=encoding
    )
    print()

    # IterableDataset requires different DataLoader setup
    # No shuffle (streaming), no len()
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Can't shuffle IterableDataset
        drop_last=True
    )

    print(f"Training configuration:")
    print(f"  Tokens per epoch: {TOKENS_PER_EPOCH:,}")
    print(f"  Approximate sequences per epoch: ~{TOKENS_PER_EPOCH // SEQ_LENGTH:,}")
    print(f"  Batch size: {BATCH_SIZE}")
    print()

    # Create model
    print("Creating model...")
    model = DecoderOnlyTransformer(
        vocab_size=dataset.vocab_size,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        d_ff=D_FF,
        max_seq_len=SEQ_LENGTH * 2,
        dropout=DROPOUT
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    # Check for NaN in initial weights
    has_nan = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"  WARNING: NaN in initial weights: {name}")
            has_nan = True
    if has_nan:
        raise ValueError("Model has NaN in initial weights!")
    print("  Initial weights: OK (no NaN)")
    print()

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()

    # Optimizer with placeholder LR (scheduler will control actual LR)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1.0,  # Placeholder - scheduler overrides this
        weight_decay=WEIGHT_DECAY
    )

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
    # tokens_per_epoch / seq_length / batch_size
    steps_per_epoch = (TOKENS_PER_EPOCH // SEQ_LENGTH // BATCH_SIZE)
    total_steps = steps_per_epoch * NUM_EPOCHS
    warmup_steps = int(0.05 * total_steps)  # 5% of training for warmup
    min_lr = LEARNING_RATE * 0.1  # Final LR = 10% of peak

    print(f"Learning rate schedule:")
    print(f"  Steps per epoch: ~{steps_per_epoch:,}")
    print(f"  Total training steps: {total_steps:,}")
    print(f"  Warmup steps: {warmup_steps:,} (5% of training)")
    print(f"  Max learning rate: {LEARNING_RATE:.6f}")
    print(f"  Min learning rate: {min_lr:.6f}")
    print()

    scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        max_lr=LEARNING_RATE,
        min_lr=min_lr
    )

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    total_batches = 0
    start_time = time.time()

    def print_table_header():
        """Print the table header for training metrics."""
        print(f"{'Batch':>10} {'Loss':>8} {'PPL':>8} {'Avg Loss':>9} {'Avg PPL':>9} {'LR':>10} {'Phase':>10}")
        print("-" * 90)

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 90)

        # Print initial table header
        print_table_header()

        epoch_loss = 0.0
        epoch_start = time.time()
        log_lines_printed = 0  # Track how many log lines we've printed

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass with autocast (mixed precision on CUDA, no-op on MPS/CPU)
            with autocast_ctx:
                logits = model(inputs, debug=debug)

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

            # Check loss for NaN
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"\n  ERROR: NaN/Inf loss at batch {batch_idx + 1}")
                print(f"  Loss value: {loss.item()}")
                raise ValueError("NaN/Inf loss - training unstable")

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Update weights
            optimizer.step()

            # Update learning rate (MUST be called after optimizer.step())
            # This adjusts the LR according to our warmup + cosine decay schedule
            scheduler.step()

            epoch_loss += loss.item()
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
                batch_perplexity = calculate_perplexity_from_loss(loss).item()
                avg_perplexity = calculate_perplexity_from_loss(torch.tensor(avg_loss)).item()

                # Repeat header every 30 log lines to keep it visible
                if log_lines_printed > 0 and log_lines_printed % 30 == 0:
                    print()
                    print_table_header()

                # Determine phase (warmup or training)
                phase = "warmup" if total_batches < warmup_steps else "learning"

                # Format batch count as "X/Y"
                batch_str = f"{batch_idx + 1}/{steps_per_epoch}"
                print(f"{batch_str:>10} {loss.item():8.4f} {batch_perplexity:8.2f} "
                      f"{avg_loss:9.4f} {avg_perplexity:9.2f} {current_lr:10.6f} {phase:>10}")
                log_lines_printed += 1

        # Epoch summary (synchronize for accurate timing on CUDA)
        synchronize()
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / (batch_idx + 1)  # Use actual batch count
        avg_epoch_perplexity = calculate_perplexity_from_loss(torch.tensor(avg_epoch_loss)).item()
        current_lr = scheduler.get_last_lr()[0]
        print()
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Avg Loss: {avg_epoch_loss:.4f}  |  Avg Perplexity: {avg_epoch_perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}  |  Time: {epoch_time:.1f}s")
        print()

        # Save checkpoint with encoding in filename
        encoding_short = get_encoding_short_name(encoding)
        checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}_{encoding_short}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),  # Save scheduler state
            'loss': avg_epoch_loss,
            'perplexity': avg_epoch_perplexity,  # Save perplexity for comparison
            'current_lr': current_lr,  # Save current LR for reference
            'config': {
                'vocab_size': dataset.vocab_size,
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'd_ff': D_FF,
                'dropout': DROPOUT,
                'encoding': encoding,  # Store encoding for detection
            }
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        print()

    # Training complete
    synchronize()  # Ensure all operations complete
    total_time = time.time() - start_time
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")

    # Device-specific stats (CUDA only for now)
    if device.type == "cuda":
        peak_memory_gb = get_max_memory() / (1024**3)
        print(f"Peak GPU memory: {peak_memory_gb:.2f} GB")

    print()

    # Final samples
    print("Final sample generations:")
    print("-" * 80)
    for prompt in ["The", "In the", "She was"]:
        sample = generate_sample(model, dataset, prompt, max_length=100, device=device, autocast_ctx=autocast_ctx)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{sample}'")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decoder-only transformer on FineWeb")
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
        "--encoding",
        type=str,
        default="cl100k_base",
        choices=["p50k_base", "cl100k_base"],
        help="Tokenizer encoding to use (default: cl100k_base, ~100K vocab; p50k_base: ~50K vocab)"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode: smaller model (4 layers, d_model=128) and fewer tokens (10M/epoch)"
    )
    args = parser.parse_args()

    train(debug=args.debug, use_mps=args.mps, encoding=args.encoding, quick=args.quick)
