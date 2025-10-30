#!/usr/bin/env python3
"""Analyze training checkpoints to diagnose plateau."""

import torch
from pathlib import Path

checkpoint_dir = Path("checkpoints")
checkpoints = sorted(checkpoint_dir.glob("model_epoch_*.pt"),
                     key=lambda x: int(x.stem.split('_')[-1]))

print("=" * 80)
print("TRAINING PROGRESSION ANALYSIS")
print("=" * 80)
print()

print(f"{'Epoch':<8} {'Loss':<12} {'Perplexity':<12} {'Current LR':<15} {'Change %':<10}")
print("-" * 80)

prev_loss = None
for checkpoint_path in checkpoints:
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    perplexity = checkpoint.get('perplexity', 'N/A')
    current_lr = checkpoint.get('current_lr', 'N/A')

    # Calculate percentage change
    if prev_loss is not None:
        change_pct = ((loss - prev_loss) / prev_loss) * 100
        change_str = f"{change_pct:+.2f}%"
    else:
        change_str = "-"

    print(f"{epoch:<8} {loss:<12.6f} {perplexity:<12.2f} {current_lr:<15.6f} {change_str:<10}")
    prev_loss = loss

print()
print("=" * 80)
