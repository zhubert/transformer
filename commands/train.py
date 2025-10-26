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
from src.transformer.training_utils import GradientAccumulator
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


def train(debug=False, use_mps=False, encoding="p50k_base", quick=False, accumulation_steps=16):
    """
    Main training function with gradient accumulation and validation.

    Args:
        debug: If True, print diagnostic information for debugging NaN issues
        use_mps: If True, try MPS (experimental - has known NaN issues)
        encoding: Tokenizer encoding to use ("p50k_base" or "cl100k_base")
        quick: If True, use smaller model and fewer tokens for faster training
        accumulation_steps: Number of batches to accumulate before updating weights.
                           Effective batch size = BATCH_SIZE × accumulation_steps
                           Higher values = more stable training but slower updates
                           Recommended: 16-32 for hobby hardware

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

    # Load datasets (train and validation)
    print("Loading FineWeb dataset...")
    print()

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
    print()

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
    print()

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

    print(f"Training configuration:")
    print(f"  Tokens per epoch (train): {TOKENS_PER_EPOCH:,}")
    print(f"  Tokens per epoch (val): {val_tokens_per_epoch:,}")
    print(f"  Approximate sequences per epoch: ~{TOKENS_PER_EPOCH // SEQ_LENGTH:,}")
    print(f"  Batch size: {BATCH_SIZE} sequences ({BATCH_SIZE * SEQ_LENGTH:,} tokens)")
    print(f"  Gradient accumulation steps: {accumulation_steps}")
    print(f"  Effective batch size: {BATCH_SIZE * accumulation_steps} sequences ({effective_batch_size:,} tokens)")
    print(f"  → {accumulation_steps}x more stable than without accumulation!")
    print()

    # Create model
    print("Creating model...")
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

    print(f"Learning rate schedule:")
    print(f"  Batches per epoch: ~{batches_per_epoch:,}")
    print(f"  Weight updates per epoch: ~{steps_per_epoch:,} (with accumulation)")
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

        # ==============================================================================
        # TRAINING PHASE
        # ==============================================================================
        model.train()  # Set to training mode

        # Print initial table header
        print_table_header()

        epoch_loss = 0.0
        epoch_start = time.time()
        log_lines_printed = 0  # Track how many log lines we've printed

        # Reset gradient accumulator for new epoch
        accumulator.reset()
        optimizer.zero_grad()  # Clear gradients once at epoch start

        for batch_idx, (inputs, targets) in enumerate(train_dataloader):
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
                # Clip gradients to prevent explosion
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

                # Repeat header every 30 log lines to keep it visible
                if log_lines_printed > 0 and log_lines_printed % 30 == 0:
                    print()
                    print_table_header()

                # Determine phase (warmup or training)
                step_in_cycle, total_in_cycle = accumulator.get_progress()
                phase = "warmup" if total_batches < (warmup_steps * accumulation_steps) else "learning"

                # Format batch count as "X/Y"
                batch_str = f"{batch_idx + 1}/{batches_per_epoch}"
                print(f"{batch_str:>10} {loss.item() * accumulation_steps:8.4f} {batch_perplexity:8.2f} "
                      f"{avg_loss:9.4f} {avg_perplexity:9.2f} {current_lr:10.6f} {phase:>10}")
                log_lines_printed += 1

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
        print()
        print("Running validation...")
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
                    val_logits = model(val_inputs)

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

        # Print epoch summary with both train and validation metrics
        print()
        print(f"Epoch {epoch + 1} Summary:")
        print(f"  Train Loss: {avg_train_loss:.4f}  |  Train Perplexity: {avg_train_perplexity:.2f}")
        print(f"  Val Loss:   {avg_val_loss:.4f}  |  Val Perplexity:   {avg_val_perplexity:.2f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        print(f"  Time: {train_time:.1f}s train + {val_time:.1f}s val = {train_time + val_time:.1f}s total")

        # Interpretation help for users
        if avg_val_loss < avg_train_loss * 1.05:
            # Validation loss is close to training loss (good!)
            print(f"  Status: ✓ Model is learning well (val ≈ train)")
        elif avg_val_loss < avg_train_loss * 1.15:
            # Validation loss is slightly higher (normal)
            print(f"  Status: ✓ Model is learning (val slightly > train, normal)")
        elif avg_val_loss > avg_train_loss * 1.3:
            # Validation loss much higher (possible overfitting)
            print(f"  Status: ⚠ Possible overfitting (val >> train)")
        else:
            # In between
            print(f"  Status: Model is training")
        print()

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
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=16,
        help="Number of batches to accumulate before updating weights. Higher = more stable training. "
             "Effective batch size = batch_size × accumulation_steps. Recommended: 16-32 (default: 16)"
    )
    args = parser.parse_args()

    train(
        debug=args.debug,
        use_mps=args.mps,
        encoding=args.encoding,
        quick=args.quick,
        accumulation_steps=args.accumulation_steps
    )
