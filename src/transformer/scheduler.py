"""
Learning Rate Scheduling: Warmup + Cosine Decay

This module implements a learning rate schedule that combines two important techniques:
1. Linear warmup at the start of training
2. Cosine annealing (decay) for the rest of training

================================================================================
WHY DO WE NEED LEARNING RATE SCHEDULING?
================================================================================

The Problem: The optimal learning rate CHANGES during training!

Early Training (Random Weights):
    - Model outputs are random garbage
    - Gradients are huge and chaotic (10x-100x normal)
    - Loss landscape is steep and unpredictable
    → Need: SMALL learning rate to avoid instability

Mid Training (Learning Patterns):
    - Model starts capturing patterns
    - Gradients become more reliable
    - Loss is decreasing steadily
    → Need: LARGE learning rate to learn quickly

Late Training (Fine-tuning):
    - Model has learned major patterns
    - Trying to optimize details
    - Near a good minimum
    → Need: SMALL learning rate to "settle in" precisely

A constant learning rate can't satisfy all three phases!

================================================================================
PART 1: WARMUP - Preventing Early Training Instability
================================================================================

The Problem at Step 1
---------------------
When training starts:
    - Weights are randomly initialized (essentially noise)
    - First forward pass produces random outputs
    - Loss is huge (cross-entropy of random guessing ~10)
    - Gradients can be EXTREMELY large (10x-100x normal)

If we use the target LR (e.g., 3e-4) immediately:
    weight_update = 3e-4 × huge_gradient = MASSIVE_CHANGE

This causes:
    ❌ Weights jump wildly
    ❌ Model outputs become NaN
    ❌ Training explodes before it starts
    ❌ Or: model jumps to terrible region and never recovers

The Warmup Solution
-------------------
Start with tiny LR and gradually increase to target:

    Step 1:     LR = 0.00001  (1% of target)
    Step 100:   LR = 0.0001   (33% of target)
    Step 200:   LR = 0.0002   (67% of target)
    Step 300:   LR = 0.0003   (100% of target) ← warmup complete!

Why This Works:
    Step 1-100 (LR very small):
        - Tiny weight updates despite huge gradients
        - Model starts to produce less random outputs
        - Gradients normalize
        - Loss begins dropping: ~10 → ~8

    Step 100-200 (LR growing):
        - Gradients are more reasonable
        - Can safely take bigger steps
        - Learning accelerates
        - Loss drops: ~8 → ~5

    Step 200-300 (approaching target):
        - Model is in reasonable region
        - Gradients are well-behaved
        - Can use full LR safely
        - Ready for main training

Warmup Duration:
    - Typical: 1-10% of total training steps
    - Too short: May still see early instability
    - Too long: Waste time at suboptimal LR
    - For small models: 500-2000 steps
    - For large models (GPT-scale): 2000-10000 steps

================================================================================
PART 2: COSINE DECAY - Fine-tuning at the End
================================================================================

The Problem in Late Training
----------------------------
After training for a while:
    - Model has learned major patterns
    - Loss is much lower (e.g., 10 → 2)
    - Near a good minimum
    - But still using high LR (e.g., 3e-4)

The Issue: Large LR causes "bouncing"

Imagine hiking downhill toward a valley:
    Early on: Large steps are fine (far from bottom)
    Near valley: Large steps make you overshoot back and forth
    Result: OSCILLATE around minimum instead of settling into it

At a minimum, the loss landscape looks like a bowl:

    Loss
      │
      │   ╱  ╲      ← With high LR: bounce between sides
      │  ╱    ╲
      │ ╱      ╲    ← With low LR: settle at bottom
      └─────────── Weights
          ↑
       minimum

The Cosine Decay Solution
--------------------------
Gradually reduce LR as training progresses, using a cosine curve.

Why Cosine?
    ✓ Smooth, no sudden jumps (unlike step decay)
    ✓ Stays near max_lr early (allows fast learning)
    ✓ Decreases faster at end (helps fine-tuning)
    ✓ Mathematically elegant and proven effective

The Formula:
    LR = min_lr + (max_lr - min_lr) × 0.5 × (1 + cos(π × t / T))

    Where:
        t = current step (since warmup ended)
        T = total steps (after warmup)
        max_lr = peak learning rate (e.g., 3e-4)
        min_lr = final learning rate (e.g., 3e-5, which is 10% of max)

The Curve:

    LR
     │
    3e-4│  ╭──────╮           ← Stays high for first ~30% (fast learning)
         │ ╱        ╲
         │╱          ╲        ← Smooth decrease in middle
         │            ╲___    ← Faster decrease at end (fine-tuning)
    1e-4│                 ╲___
         │                     ╲___
    3e-5│________________________
         └─────────────────────────── Training Steps
         warmup              max_steps

Math Breakdown:
    When t = 0 (start of decay):
        cos(π × 0 / T) = cos(0) = 1
        0.5 × (1 + 1) = 1.0
        LR = min_lr + (max_lr - min_lr) × 1.0 = max_lr ✓

    When t = T/2 (halfway):
        cos(π × (T/2) / T) = cos(π/2) = 0
        0.5 × (1 + 0) = 0.5
        LR = (max_lr + min_lr) / 2 ✓

    When t = T (end):
        cos(π × T / T) = cos(π) = -1
        0.5 × (1 + (-1)) = 0.0
        LR = min_lr ✓

================================================================================
COMBINING WARMUP + COSINE DECAY
================================================================================

The Complete Schedule:

    LR
     │
    3e-4│     ╭───────╮                ← Peak LR maintained for a while
         │    ╱         ╲
         │   ╱           ╲___          ← Smooth cosine decay
         │  ╱                ╲___
    1e-4│ ╱                      ╲___
         │╱                           ╲___
    3e-5│________________________________
         └────────────────────────────────── Steps
         0   warmup=1k              total=16k

    Phase 1: Warmup (steps 0-1000)
             Linear increase: 0 → 3e-4

    Phase 2: Cosine Decay (steps 1000-16000)
             Smooth decrease: 3e-4 → 3e-5

================================================================================
EXPECTED IMPROVEMENTS
================================================================================

Before (Constant LR):
    Epoch 1:  Loss 10.0 → 6.5
    Epoch 5:  Loss 4.2 → 3.8
    Epoch 10: Loss 3.1 → 2.9  ← bouncing, not improving much
    Epoch 15: Loss 2.8 → 2.8  ← stuck, oscillating
    Epoch 20: Loss 2.7 → 2.8  ← may even go up!
    Final loss: ~2.7-2.8

After (Warmup + Cosine Decay):
    Epoch 1:  Loss 10.0 → 7.0  ← warmup prevents explosion
    Epoch 5:  Loss 5.0 → 4.0   ← learning at full speed
    Epoch 10: Loss 3.0 → 2.5   ← still improving
    Epoch 15: Loss 2.3 → 2.0   ← LR decaying, fine-tuning
    Epoch 20: Loss 1.8 → 1.7   ← settled into better minimum
    Final loss: ~1.7-1.8 (20-30% better!)

Why the Improvement?
    1. Warmup prevents early instability
       - No explosion from huge early gradients
       - Model starts in better region

    2. Cosine decay enables fine-tuning
       - Doesn't get stuck bouncing
       - Can optimize details
       - Reaches better final solution

    3. Smoother convergence
       - Cleaner loss curves
       - More predictable training
       - Better generalization

================================================================================
USAGE EXAMPLE
================================================================================

    from transformer.scheduler import get_lr_scheduler

    # Setup (before training loop)
    optimizer = torch.optim.Adam(model.parameters(), lr=1.0)  # lr=1.0 is placeholder

    total_steps = len(dataloader) * num_epochs
    scheduler = get_lr_scheduler(
        optimizer=optimizer,
        warmup_steps=1000,      # 5-10% of total steps
        total_steps=total_steps,
        max_lr=3e-4,           # Peak learning rate
        min_lr=3e-5            # Final LR (10% of max)
    )

    # Training loop
    for epoch in range(num_epochs):
        for batch in dataloader:
            # ... forward pass, loss calculation ...

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update learning rate (call AFTER optimizer.step())
            scheduler.step()

            # Optional: Log current LR
            current_lr = scheduler.get_last_lr()[0]
            print(f"Step {step}, LR: {current_lr:.6f}, Loss: {loss.item():.4f}")

================================================================================
"""

import math
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    max_lr: float = 3e-4,
    min_lr: float = 3e-5,
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.

    This is the industry-standard learning rate schedule used by modern transformers
    (GPT-2, GPT-3, BERT, etc.). It combines:
        1. Linear warmup: Gradually increase LR from 0 to max_lr
        2. Cosine decay: Smoothly decrease LR from max_lr to min_lr

    Learning Rate Schedule:
    ----------------------

        LR
         │
      max_lr│     ╭───────╮
             │    ╱         ╲
             │   ╱           ╲___
             │  ╱                ╲___
             │ ╱                      ╲___
      min_lr │╱___________________________
             └────────────────────────────── Steps
             0    warmup            total_steps

    Args:
        optimizer: PyTorch optimizer (e.g., Adam)
        warmup_steps: Number of steps for linear warmup phase
                     Typical: 1-10% of total_steps
                     Example: 1000 steps for 10k total steps

        total_steps: Total number of training steps
                    Calculate as: len(dataloader) × num_epochs

        max_lr: Peak learning rate (reached after warmup)
               Typical for small transformers: 1e-4 to 5e-4
               GPT-2 used: 2.5e-4
               GPT-3 used: 6e-5 (larger models need smaller LR)

        min_lr: Final learning rate (reached at end of training)
               Typical: 10% of max_lr
               Example: 3e-5 if max_lr is 3e-4
               Don't use 0 (keep some learning happening)

    Returns:
        LambdaLR scheduler - call scheduler.step() after each optimizer.step()

    Implementation Details:
    ----------------------
    We use PyTorch's LambdaLR scheduler with a custom lambda function that
    returns a learning rate multiplier for each step. The optimizer's base LR
    is set to max_lr, and our lambda returns a multiplier (0.0 to 1.0).

    The lambda function implements:
        - If step < warmup_steps: Linear warmup
            multiplier = step / warmup_steps  (0.0 → 1.0)
            Actual LR = max_lr × multiplier

        - If step >= warmup_steps: Cosine decay
            progress = (step - warmup_steps) / (total_steps - warmup_steps)
            decay = 0.5 × (1 + cos(π × progress))  (1.0 → 0.0)
            multiplier = (min_lr/max_lr) + (1 - min_lr/max_lr) × decay
            Actual LR = max_lr × multiplier

    Why LambdaLR?
    ------------
    LambdaLR allows us to define a custom schedule using a simple function.
    The function takes the current step and returns the LR multiplier.
    This is cleaner than manually updating optimizer.param_groups['lr'].

    Example Usage:
    -------------
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)  # Set base LR to max_lr
    >>> scheduler = get_cosine_schedule_with_warmup(
    ...     optimizer=optimizer,
    ...     warmup_steps=1000,
    ...     total_steps=10000,
    ...     max_lr=3e-4,  # Should match optimizer's lr
    ...     min_lr=3e-5
    ... )
    >>>
    >>> for step in range(total_steps):
    ...     loss.backward()
    ...     optimizer.step()
    ...     scheduler.step()  # Update LR
    ...
    ...     if step % 100 == 0:
    ...         current_lr = scheduler.get_last_lr()[0]
    ...         print(f"Step {step}, LR: {current_lr:.6f}")

    Learning Rate at Key Points:
    ---------------------------
    Step 0:           LR ≈ 0           (warmup start)
    Step warmup/2:    LR = max_lr/2    (mid-warmup)
    Step warmup:      LR = max_lr      (warmup complete, decay starts)
    Step total/2:     LR ≈ (max+min)/2 (mid-training)
    Step total:       LR = min_lr      (training complete)
    """

    def lr_lambda(current_step: int) -> float:
        """
        Calculate learning rate multiplier for the current step.

        This function is called by LambdaLR at each step to determine the LR.

        Args:
            current_step: Current training step (0-indexed)

        Returns:
            Learning rate multiplier (will be multiplied by optimizer's base LR)

        The optimizer's base LR should be set to max_lr. This function returns
        a multiplier between 0.0 and 1.0 that scales the base LR.
        """

        # Phase 1: Linear Warmup (steps 0 to warmup_steps)
        if current_step < warmup_steps:
            # Linear increase from 0 to 1.0
            # At step 0: multiplier = 0.0 → LR = max_lr × 0.0 = 0
            # At step warmup_steps/2: multiplier = 0.5 → LR = max_lr × 0.5
            # At step warmup_steps: multiplier = 1.0 → LR = max_lr × 1.0 = max_lr
            warmup_factor = float(current_step) / float(max(1, warmup_steps))
            return warmup_factor

        # Phase 2: Cosine Decay (steps warmup_steps to total_steps)
        else:
            # Calculate progress through decay phase (0.0 to 1.0)
            decay_steps = total_steps - warmup_steps
            current_decay_step = current_step - warmup_steps
            progress = float(current_decay_step) / float(max(1, decay_steps))

            # Cosine decay factor: starts at 1.0, ends at 0.0
            # Uses cosine curve for smooth transition
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))

            # Calculate multiplier that interpolates between min_lr/max_lr and 1.0
            # At progress=0.0: cosine_decay=1.0 → multiplier = 1.0 → LR = max_lr
            # At progress=0.5: cosine_decay=0.5 → multiplier = 0.55 → LR ≈ (max_lr + min_lr) / 2
            # At progress=1.0: cosine_decay=0.0 → multiplier = min_lr/max_lr → LR = min_lr
            min_lr_ratio = min_lr / max_lr
            multiplier = min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
            return multiplier

    # Create the scheduler
    # Note: Optimizer's base LR should be set to max_lr before calling this function
    scheduler = LambdaLR(optimizer, lr_lambda)

    return scheduler


# Alias for convenience
get_lr_scheduler = get_cosine_schedule_with_warmup
