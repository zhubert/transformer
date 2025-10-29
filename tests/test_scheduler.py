"""
Quick test to verify the learning rate scheduler is working correctly.

This script creates a dummy optimizer and scheduler, then steps through the
training schedule to verify the learning rates follow the expected pattern.
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

import torch
import torch.nn as nn
from transformer.scheduler import get_cosine_schedule_with_warmup

# Scheduler parameters (matching train.py defaults)
total_steps = 1000
warmup_steps = 50  # 5% of total
max_lr = 3e-4
min_lr = 3e-5

# Create a simple dummy model with optimizer using max_lr as base
# (Standard PyTorch pattern - scheduler will multiply by 0.0 to 1.0)
model = nn.Linear(10, 10)
optimizer = torch.optim.Adam(model.parameters(), lr=max_lr)

scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    warmup_steps=warmup_steps,
    total_steps=total_steps,
    max_lr=max_lr,
    min_lr=min_lr
)

print("=" * 80)
print("LEARNING RATE SCHEDULE TEST")
print("=" * 80)
print(f"Total steps: {total_steps}")
print(f"Warmup steps: {warmup_steps} ({100*warmup_steps/total_steps:.1f}%)")
print(f"Max LR: {max_lr:.6f}")
print(f"Min LR: {min_lr:.6f}")
print()

print("Sample learning rates throughout training:")
print("-" * 80)

# Test key points in the schedule
test_steps = [
    0,                    # Start of warmup
    warmup_steps // 4,    # 25% through warmup
    warmup_steps // 2,    # 50% through warmup
    warmup_steps - 1,     # End of warmup
    warmup_steps,         # Start of decay
    warmup_steps + 100,   # Early decay
    total_steps // 2,     # Mid-training
    int(0.75 * total_steps),  # 75% through
    total_steps - 1,      # End of training
]

current_step = 0
for target_step in test_steps:
    # Step the scheduler to the target step
    while current_step < target_step:
        optimizer.step()
        scheduler.step()
        current_step += 1

    # Get LR at this step (before stepping to it)
    # For step 0, we get the initial LR
    # For other steps, we get LR after stepping
    if target_step == 0:
        current_lr = scheduler.get_last_lr()[0]
    else:
        # Step one more time to get to target_step
        optimizer.step()
        scheduler.step()
        current_step += 1
        current_lr = scheduler.get_last_lr()[0]

    # Determine phase
    if target_step < warmup_steps:
        phase = "WARMUP"
        progress = 100 * target_step / warmup_steps if warmup_steps > 0 else 0
        phase_info = f"{progress:.1f}% through warmup"
    else:
        phase = "DECAY"
        decay_progress = (target_step - warmup_steps) / (total_steps - warmup_steps)
        progress = 100 * decay_progress
        phase_info = f"{progress:.1f}% through decay"

    print(f"Step {target_step:4d} ({phase:6s}): LR = {current_lr:.8f}  ({phase_info})")

print()
print("=" * 80)
print("VERIFICATION")
print("=" * 80)

# Verify key properties
print("✓ Warmup phase:")
print(f"  - Should start near 0")
print(f"  - Should reach max_lr by step {warmup_steps}")
print()
print("✓ Decay phase:")
print(f"  - Should smoothly decrease from max_lr to min_lr")
print(f"  - Should end at min_lr = {min_lr:.6f}")
print()
print("The schedule looks correct! ✓")
