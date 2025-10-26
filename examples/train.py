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
from src.transformer.dataset import TextDataset


def get_device(use_mps=False):
    """
    Detect best available device for training.

    Defaults to CPU for stability. MPS (Apple Silicon GPU) has known
    issues with NaN during training (PyTorch bugs #107294, #109457).

    Args:
        use_mps: If True, try to use MPS despite known issues
    """
    if use_mps and torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon GPU) - EXPERIMENTAL"
    elif torch.cuda.is_available():
        return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    else:
        return torch.device("cpu"), "CPU"


def generate_sample(model, dataset, prompt_text, max_length=50, device="cpu"):
    """Generate sample text to see how the model is learning."""
    model.eval()
    prompt_tokens = dataset.tokenizer.encode(prompt_text)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    with torch.no_grad():
        output_ids = model.generate(input_ids, max_length=max_length, temperature=0.5)

    generated_text = dataset.decode(output_ids[0])
    model.train()
    return generated_text


def train(debug=False, use_mps=False):
    """
    Main training function.

    Args:
        debug: If True, print diagnostic information for debugging NaN issues
        use_mps: If True, try MPS (experimental - has known NaN issues)
    """

    # Set random seed for reproducibility
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(42)

    # Configuration
    TEXT_FILE = "Singular.txt"
    SEQ_LENGTH = 128
    BATCH_SIZE = 8              # Reduced from 32 to fit in M1 memory
    D_MODEL = 256
    NUM_HEADS = 4
    NUM_LAYERS = 6
    D_FF = 1024
    DROPOUT = 0.1
    NUM_EPOCHS = 20             # Increased from 3 - more epochs for better learning
    LEARNING_RATE = 3e-4        # Increased from 1e-4 - standard transformer learning rate
    WEIGHT_DECAY = 0.01
    LOG_INTERVAL = 10
    SAMPLE_INTERVAL = 100
    SAMPLE_PROMPT = "The"
    CHECKPOINT_DIR = Path("checkpoints")
    CHECKPOINT_DIR.mkdir(exist_ok=True)

    print("=" * 80)
    print("TRANSFORMER TRAINING")
    print("=" * 80)
    print()

    # Device setup
    device, device_name = get_device(use_mps=use_mps)
    print(f"Device: {device_name}")
    if use_mps:
        print("  WARNING: MPS has known NaN issues. Use --debug if training fails.")
    print()

    # Load dataset
    print("Loading dataset...")
    dataset = TextDataset(TEXT_FILE, seq_length=SEQ_LENGTH)
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    print(f"Training configuration:")
    print(f"  Sequences per epoch: {len(dataset):,}")
    print(f"  Batches per epoch: {len(dataloader):,}")
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
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    print("=" * 80)
    print("STARTING TRAINING")
    print("=" * 80)
    print()

    total_batches = 0
    start_time = time.time()

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 80)

        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, (inputs, targets) in enumerate(dataloader):
            # Move to device
            inputs = inputs.to(device)
            targets = targets.to(device)

            # Forward pass
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

            epoch_loss += loss.item()
            total_batches += 1

            # Debug: Check for NaN in weights after first batch
            if batch_idx == 0 and not debug:
                for name, param in model.named_parameters():
                    if torch.isnan(param).any():
                        print(f"  WARNING: NaN in weights after batch 1: {name}")
                        print(f"  This means batch 1 corrupted the model!")
                        raise ValueError("Weights corrupted after batch 1")
                print("  After batch 1: Weights still OK")

            # Logging
            if (batch_idx + 1) % LOG_INTERVAL == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"  Batch {batch_idx + 1}/{len(dataloader)}, "
                      f"Loss: {loss.item():.4f}, Avg: {avg_loss:.4f}")

            # Sample generation
            if (batch_idx + 1) % SAMPLE_INTERVAL == 0:
                print(f"\n  Sample (after {total_batches} batches):")
                sample = generate_sample(model, dataset, SAMPLE_PROMPT,
                                       max_length=50, device=device)
                print(f"  '{sample}'")
                print()

        # Epoch summary
        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"\n  Epoch {epoch + 1} complete!")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Time: {epoch_time:.1f}s")
        print()

        # Save checkpoint
        checkpoint_path = CHECKPOINT_DIR / f"model_epoch_{epoch + 1}.pt"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_epoch_loss,
            'config': {
                'vocab_size': dataset.vocab_size,
                'd_model': D_MODEL,
                'num_heads': NUM_HEADS,
                'num_layers': NUM_LAYERS,
                'd_ff': D_FF,
                'dropout': DROPOUT,
            }
        }, checkpoint_path)
        print(f"  Saved checkpoint: {checkpoint_path}")
        print()

    # Training complete
    total_time = time.time() - start_time
    print("=" * 80)
    print("TRAINING COMPLETE!")
    print("=" * 80)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print()

    # Final samples
    print("Final sample generations:")
    print("-" * 80)
    for prompt in ["The", "In the", "She was"]:
        sample = generate_sample(model, dataset, prompt, max_length=100, device=device)
        print(f"\nPrompt: '{prompt}'")
        print(f"Generated: '{sample}'")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a decoder-only transformer on text data")
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
    args = parser.parse_args()

    train(debug=args.debug, use_mps=args.mps)
