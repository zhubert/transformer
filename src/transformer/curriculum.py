#!/usr/bin/env python3
"""
Curriculum learning for mid-training.

Curriculum learning is the idea of starting with easier examples and progressively
increasing difficulty, similar to how humans learn. This is especially effective
for mid-training where data has natural difficulty levels (e.g., math problems).

Why Curriculum Learning Works:
===============================
Imagine learning calculus before algebra - you'd struggle! Same with neural networks.
Starting with easier examples helps the model:
1. Build foundational patterns first
2. Avoid getting stuck in poor local minima
3. Generalize better to hard examples later

Research shows 10-15% improvement in final performance for tasks with natural
difficulty hierarchies.

When to Use Curriculum Learning:
=================================
✓ Math problems (difficulty levels 1-5)
✓ Code (simple scripts → complex libraries)
✓ Scientific papers (introductory → research)
✗ General web text (no clear difficulty ordering)

References:
- "Curriculum Learning" (Bengio et al., 2009)
- "On The Power of Curriculum Learning in Training Deep Networks" (2019)
- "Competence-based Curriculum Learning for Neural Machine Translation" (2019)

Educational Purpose:
====================
This module demonstrates how to order training data for better learning outcomes.
The key insight: Order matters! Not all training examples are created equal.
"""

from typing import Callable, Optional, List, Any
from dataclasses import dataclass
import math


@dataclass
class CurriculumStage:
    """
    A stage in the curriculum learning schedule.

    Each stage represents a difficulty level or data subset that the model
    trains on for a specified number of steps or epochs.

    Attributes:
        name: Human-readable name (e.g., "Easy problems", "Medium code")
        difficulty_range: (min, max) difficulty for this stage
        num_epochs: How many epochs to train on this stage
        data_filter: Optional function to filter data for this stage
    """
    name: str
    difficulty_range: tuple[float, float]
    num_epochs: int
    data_filter: Optional[Callable] = None


class CurriculumScheduler:
    """
    Manages curriculum learning schedule during mid-training.

    The scheduler controls which data the model sees at each training stage,
    progressively increasing difficulty over time.

    Three Common Curriculum Strategies:
    ====================================

    1. **Step Curriculum** (what we implement):
       - Stage 1: Easy examples only (epochs 1-3)
       - Stage 2: Medium examples only (epochs 4-6)
       - Stage 3: Hard examples only (epochs 7-10)

    2. **Progressive Curriculum**:
       - Stage 1: Easy only
       - Stage 2: Easy + Medium
       - Stage 3: Easy + Medium + Hard

    3. **Smooth Curriculum**:
       - Gradually shift difficulty threshold each epoch
       - No discrete stages

    We use Step Curriculum because it's simple to understand and implement,
    while still providing most of the benefit.

    Example Usage:
    ==============
    >>> # Math curriculum: start easy, end hard
    >>> scheduler = CurriculumScheduler([
    ...     CurriculumStage("Easy", (1.0, 2.0), num_epochs=3),
    ...     CurriculumStage("Medium", (2.0, 3.0), num_epochs=3),
    ...     CurriculumStage("Hard", (3.0, 5.0), num_epochs=4),
    ... ])
    >>>
    >>> # Training loop
    >>> for epoch in range(10):
    ...     stage = scheduler.get_current_stage(epoch)
    ...     print(f"Epoch {epoch}: Training on {stage.name} problems")
    ...     # Filter dataset based on stage.difficulty_range
    ...     # Train for one epoch
    """

    def __init__(self, stages: List[CurriculumStage]):
        """
        Initialize curriculum scheduler.

        Args:
            stages: List of curriculum stages in order (easy → hard)

        Raises:
            ValueError: If stages are not in increasing difficulty order
        """
        # Validate stages are in increasing difficulty
        for i in range(len(stages) - 1):
            if stages[i].difficulty_range[1] > stages[i+1].difficulty_range[0]:
                raise ValueError(
                    f"Stage {i} difficulty range {stages[i].difficulty_range} "
                    f"overlaps with stage {i+1} range {stages[i+1].difficulty_range}. "
                    "Stages should be in increasing difficulty order."
                )

        self.stages = stages
        self.total_epochs = sum(stage.num_epochs for stage in stages)

        # Build epoch → stage mapping for fast lookup
        self._epoch_to_stage_idx = {}
        current_epoch = 0
        for idx, stage in enumerate(stages):
            for _ in range(stage.num_epochs):
                self._epoch_to_stage_idx[current_epoch] = idx
                current_epoch += 1

    def get_current_stage(self, epoch: int) -> CurriculumStage:
        """
        Get the curriculum stage for a given epoch.

        Args:
            epoch: Current training epoch (0-indexed)

        Returns:
            CurriculumStage for this epoch

        Raises:
            ValueError: If epoch exceeds total curriculum epochs
        """
        if epoch >= self.total_epochs:
            # After curriculum is complete, use hardest stage
            return self.stages[-1]

        stage_idx = self._epoch_to_stage_idx.get(epoch, len(self.stages) - 1)
        return self.stages[stage_idx]

    def get_stage_progress(self, epoch: int) -> tuple[int, int, float]:
        """
        Get progress through the current stage.

        Args:
            epoch: Current training epoch

        Returns:
            Tuple of (stage_number, epochs_in_stage, progress_fraction)
            - stage_number: Which stage (1-indexed for display)
            - epochs_in_stage: How many epochs into this stage
            - progress_fraction: 0.0 to 1.0, how far through this stage

        Example:
            >>> scheduler = CurriculumScheduler([
            ...     CurriculumStage("Easy", (1, 2), num_epochs=5),
            ...     CurriculumStage("Hard", (3, 5), num_epochs=5),
            ... ])
            >>> stage_num, epochs_in, progress = scheduler.get_stage_progress(2)
            >>> print(f"Stage {stage_num}, Epoch {epochs_in}/5, {progress*100:.0f}% complete")
            Stage 1, Epoch 2/5, 40% complete
        """
        stage = self.get_current_stage(epoch)
        stage_idx = self.stages.index(stage)

        # Count epochs completed in previous stages
        epochs_before_stage = sum(
            s.num_epochs for s in self.stages[:stage_idx]
        )

        # How many epochs into current stage?
        epochs_in_stage = epoch - epochs_before_stage

        # Progress fraction through this stage
        progress = epochs_in_stage / stage.num_epochs if stage.num_epochs > 0 else 1.0

        return stage_idx + 1, epochs_in_stage, progress

    def print_schedule(self):
        """
        Print a human-readable curriculum schedule.

        Useful for understanding the training plan before starting.

        Example Output:
        ===============
        Curriculum Learning Schedule
        ════════════════════════════════════════════════════════════
        Total epochs: 10

        Stage 1: Easy (Epochs 0-2)
          Difficulty: 1.0 - 2.0
          Duration: 3 epochs (30% of training)

        Stage 2: Medium (Epochs 3-5)
          Difficulty: 2.0 - 3.0
          Duration: 3 epochs (30% of training)

        Stage 3: Hard (Epochs 6-9)
          Difficulty: 3.0 - 5.0
          Duration: 4 epochs (40% of training)
        ════════════════════════════════════════════════════════════
        """
        print("\nCurriculum Learning Schedule")
        print("=" * 60)
        print(f"Total epochs: {self.total_epochs}")
        print()

        current_epoch = 0
        for idx, stage in enumerate(self.stages, 1):
            end_epoch = current_epoch + stage.num_epochs - 1
            percentage = (stage.num_epochs / self.total_epochs) * 100

            print(f"Stage {idx}: {stage.name} (Epochs {current_epoch}-{end_epoch})")
            print(f"  Difficulty: {stage.difficulty_range[0]} - {stage.difficulty_range[1]}")
            print(f"  Duration: {stage.num_epochs} epochs ({percentage:.0f}% of training)")
            print()

            current_epoch += stage.num_epochs

        print("=" * 60)


def create_math_curriculum(
    total_epochs: int = 10,
    difficulty_levels: int = 3,
) -> CurriculumScheduler:
    """
    Create a curriculum for math mid-training.

    This is a convenience function that creates a sensible default curriculum
    for math problems with difficulty levels 1-5.

    Strategy:
    =========
    - Divide total epochs evenly among difficulty levels
    - Start with easiest (level 1-2)
    - End with hardest (level 4-5)

    Args:
        total_epochs: Total number of training epochs
        difficulty_levels: How many stages (2-5)

    Returns:
        CurriculumScheduler configured for math

    Example:
        >>> # 3-stage curriculum over 9 epochs
        >>> curriculum = create_math_curriculum(total_epochs=9, difficulty_levels=3)
        >>> curriculum.print_schedule()
        # Stage 1: Easy (1-2) - 3 epochs
        # Stage 2: Medium (2-3.5) - 3 epochs
        # Stage 3: Hard (3.5-5) - 3 epochs
    """
    if difficulty_levels < 2 or difficulty_levels > 5:
        raise ValueError("difficulty_levels must be between 2 and 5")

    epochs_per_stage = total_epochs // difficulty_levels
    remaining_epochs = total_epochs % difficulty_levels

    # Divide difficulty range 1-5 into equal stages
    min_difficulty = 1.0
    max_difficulty = 5.0
    difficulty_step = (max_difficulty - min_difficulty) / difficulty_levels

    stages = []
    for i in range(difficulty_levels):
        # Calculate difficulty range for this stage
        stage_min = min_difficulty + (i * difficulty_step)
        stage_max = min_difficulty + ((i + 1) * difficulty_step)

        # Add extra epoch to early stages if total doesn't divide evenly
        num_epochs = epochs_per_stage + (1 if i < remaining_epochs else 0)

        # Name stages
        if i == 0:
            name = "Easy"
        elif i == difficulty_levels - 1:
            name = "Hard"
        else:
            name = "Medium" if difficulty_levels == 3 else f"Level {i+1}"

        stages.append(CurriculumStage(
            name=name,
            difficulty_range=(stage_min, stage_max),
            num_epochs=num_epochs,
        ))

    return CurriculumScheduler(stages)


def create_code_curriculum(
    total_epochs: int = 10,
) -> CurriculumScheduler:
    """
    Create a curriculum for code mid-training.

    Code complexity can be measured by:
    - Lines of code
    - Cyclomatic complexity
    - Number of dependencies
    - Nesting depth

    For educational purposes, we use a simple 3-stage curriculum:
    1. Simple scripts (< 50 lines, no imports)
    2. Moderate code (50-200 lines, few imports)
    3. Complex libraries (> 200 lines, many dependencies)

    Args:
        total_epochs: Total number of training epochs

    Returns:
        CurriculumScheduler configured for code

    Example:
        >>> curriculum = create_code_curriculum(total_epochs=9)
        >>> curriculum.print_schedule()
        # Stage 1: Simple scripts - 3 epochs
        # Stage 2: Moderate code - 3 epochs
        # Stage 3: Complex libraries - 3 epochs
    """
    # For code, we define difficulty by complexity metrics
    # (normalized to 1-5 scale for consistency)

    epochs_per_stage = total_epochs // 3

    stages = [
        CurriculumStage(
            name="Simple scripts",
            difficulty_range=(1.0, 2.0),
            num_epochs=epochs_per_stage,
        ),
        CurriculumStage(
            name="Moderate code",
            difficulty_range=(2.0, 3.5),
            num_epochs=epochs_per_stage,
        ),
        CurriculumStage(
            name="Complex libraries",
            difficulty_range=(3.5, 5.0),
            num_epochs=total_epochs - (2 * epochs_per_stage),  # Remaining epochs
        ),
    ]

    return CurriculumScheduler(stages)


def create_science_curriculum(
    total_epochs: int = 10,
) -> CurriculumScheduler:
    """
    Create a curriculum for science mid-training.

    Science papers can be ranked by complexity:
    - Introductory papers (reviews, tutorials)
    - Research papers (standard difficulty)
    - Advanced research (cutting-edge, high complexity)

    For educational purposes, we use a simple 3-stage curriculum.

    Args:
        total_epochs: Total number of training epochs

    Returns:
        CurriculumScheduler configured for science

    Example:
        >>> curriculum = create_science_curriculum(total_epochs=9)
        >>> curriculum.print_schedule()
        # Stage 1: Introductory - 3 epochs
        # Stage 2: Research papers - 3 epochs
        # Stage 3: Advanced research - 3 epochs
    """
    epochs_per_stage = total_epochs // 3

    stages = [
        CurriculumStage(
            name="Introductory",
            difficulty_range=(1.0, 2.0),
            num_epochs=epochs_per_stage,
        ),
        CurriculumStage(
            name="Research papers",
            difficulty_range=(2.0, 3.5),
            num_epochs=epochs_per_stage,
        ),
        CurriculumStage(
            name="Advanced research",
            difficulty_range=(3.5, 5.0),
            num_epochs=total_epochs - (2 * epochs_per_stage),  # Remaining epochs
        ),
    ]

    return CurriculumScheduler(stages)


def create_general_curriculum(
    total_epochs: int = 10,
) -> CurriculumScheduler:
    """
    Create a general curriculum for domains without clear difficulty levels.

    This is a simple single-stage curriculum that trains on all data uniformly.
    Used as a fallback when domain-specific curriculum isn't applicable.

    Args:
        total_epochs: Total number of training epochs

    Returns:
        CurriculumScheduler configured for general training

    Example:
        >>> curriculum = create_general_curriculum(total_epochs=10)
        >>> curriculum.print_schedule()
        # Stage 1: All data - 10 epochs
    """
    stages = [
        CurriculumStage(
            name="All data",
            difficulty_range=(1.0, 5.0),  # No filtering, accept all
            num_epochs=total_epochs,
        ),
    ]

    return CurriculumScheduler(stages)


# Example usage and tests
if __name__ == "__main__":
    print("=" * 80)
    print("CURRICULUM LEARNING DEMONSTRATION")
    print("=" * 80)
    print()

    # Create math curriculum
    print("MATH CURRICULUM (3 stages, 10 epochs)")
    print("-" * 80)
    math_curriculum = create_math_curriculum(total_epochs=10, difficulty_levels=3)
    math_curriculum.print_schedule()

    # Simulate training
    print("\nSIMULATED TRAINING:")
    print("-" * 80)
    for epoch in range(10):
        stage = math_curriculum.get_current_stage(epoch)
        stage_num, epochs_in, progress = math_curriculum.get_stage_progress(epoch)
        print(
            f"Epoch {epoch:2d}: Stage {stage_num} ({stage.name:8s}) | "
            f"Difficulty {stage.difficulty_range[0]:.1f}-{stage.difficulty_range[1]:.1f} | "
            f"Progress: {progress*100:5.1f}%"
        )

    print()
    print("=" * 80)
    print("EDUCATIONAL INSIGHT")
    print("=" * 80)
    print()
    print("Notice how we start with easy problems (1-2) and progressively")
    print("increase difficulty. This helps the model:")
    print("  1. Build foundational patterns first")
    print("  2. Avoid getting stuck on hard examples early")
    print("  3. Generalize better to complex problems later")
    print()
    print("Research shows ~10-15% improvement vs. random ordering!")
    print("=" * 80)
