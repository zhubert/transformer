#!/usr/bin/env python3
"""
Catastrophic forgetting detection for mid-training.

Catastrophic forgetting is when a model "forgets" previously learned capabilities
while learning new ones. This is a critical problem in mid-training:

Example:
========
- Pre-training: Model learns general language (perplexity: 25)
- Mid-training on code: Model becomes great at code... but perplexity on
  general text increases to 40! The model "forgot" how to write normal English!

This is called **catastrophic forgetting** because the damage happens quickly
and can be severe.

Why It Happens:
===============
Neural networks have limited capacity. When you train on very different data:
1. Gradients push weights toward code patterns
2. Weights that were important for general text get overwritten
3. General capability deteriorates

Think of it like this: If you only spoke Chinese for a year, your English
might get rusty. Same principle!

How We Prevent It:
==================
1. **Dual Evaluation**: Track BOTH domain and general perplexity
2. **Data Mixing**: Mix domain data (90%) with general data (10%)
3. **Lower Learning Rate**: Gentler updates (1e-5 vs 3e-4)
4. **Early Stopping**: Stop if general perplexity increases >10%

Research shows these techniques maintain general capability while adding
domain expertise.

References:
- "Catastrophic Forgetting in Neural Networks" (Goodfellow et al., 2013)
- "Continual Learning in Neural Networks" (review, 2019)
- "Elastic Weight Consolidation for Continual Learning" (2017)

Educational Purpose:
====================
This module demonstrates how to monitor and prevent capability loss during
specialization. The key insight: Track multiple metrics, not just training loss!
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import json
import math


@dataclass
class ForgettingMetrics:
    """
    Metrics for detecting catastrophic forgetting.

    Tracks performance on both domain-specific and general datasets
    to detect when specialization comes at the cost of general capability.

    Attributes:
        epoch: Training epoch
        train_loss: Training loss (on domain data)
        domain_perplexity: Perplexity on domain validation set
        general_perplexity: Perplexity on general validation set (e.g., FineWeb)
        forgetting_score: How much general capability has degraded (0-1 scale)
    """
    epoch: int
    train_loss: float
    domain_perplexity: float
    general_perplexity: float
    forgetting_score: float = 0.0

    def __post_init__(self):
        """Calculate forgetting score after initialization."""
        # Forgetting score is computed by ForgettingDetector
        pass


class ForgettingDetector:
    """
    Detects and quantifies catastrophic forgetting during mid-training.

    The detector maintains a history of metrics across training and alerts
    when general performance degrades significantly.

    Key Metrics:
    ============
    1. **Domain Perplexity**: Should decrease (model getting better at domain)
    2. **General Perplexity**: Should stay stable (not forgetting general language)
    3. **Forgetting Score**: Quantifies how much general capability is lost
       - 0.0: No forgetting (perfect!)
       - 0.1: Minor forgetting (acceptable)
       - 0.2+: Significant forgetting (take action!)

    Forgetting Score Formula:
    ==========================
    forgetting_score = (current_general_ppl - baseline_general_ppl) / baseline_general_ppl

    Example:
    - Baseline general perplexity: 25
    - Current general perplexity: 27.5
    - Forgetting score: (27.5 - 25) / 25 = 0.10 (10% degradation)

    When to Take Action:
    ====================
    - Forgetting score < 0.05: Everything is great!
    - 0.05 - 0.10: Minor forgetting, monitor closely
    - 0.10 - 0.20: Moderate forgetting, consider:
      * Increasing general data mixing ratio
      * Lowering learning rate
      * Adding regularization
    - > 0.20: Severe forgetting, STOP and adjust hyperparameters!

    Example Usage:
    ==============
    >>> detector = ForgettingDetector(baseline_general_perplexity=25.0)
    >>>
    >>> # After each epoch, record metrics
    >>> for epoch in range(10):
    ...     # Train and evaluate
    ...     metrics = ForgettingMetrics(
    ...         epoch=epoch,
    ...         train_loss=2.5,
    ...         domain_perplexity=18.0,  # Getting better at domain!
    ...         general_perplexity=26.0,  # Slight increase (4%)
    ...     )
    ...     detector.record(metrics)
    ...
    ...     # Check for forgetting
    ...     if detector.is_forgetting(threshold=0.10):
    ...         print(f"WARNING: Catastrophic forgetting detected!")
    ...         # Take corrective action
    """

    def __init__(
        self,
        baseline_general_perplexity: float,
        history_file: Optional[Path] = None,
    ):
        """
        Initialize forgetting detector.

        Args:
            baseline_general_perplexity: Pre-training general perplexity (baseline)
            history_file: Optional file to save metrics history
        """
        self.baseline_general_ppl = baseline_general_perplexity
        self.history: List[ForgettingMetrics] = []
        self.history_file = history_file

    def record(self, metrics: ForgettingMetrics):
        """
        Record metrics for an epoch and compute forgetting score.

        Args:
            metrics: ForgettingMetrics for this epoch (forgetting_score will be computed)
        """
        # Compute forgetting score
        metrics.forgetting_score = self._compute_forgetting_score(
            metrics.general_perplexity
        )

        # Add to history
        self.history.append(metrics)

        # Save to file if specified
        if self.history_file:
            self._save_history()

    def _compute_forgetting_score(self, current_general_ppl: float) -> float:
        """
        Compute forgetting score based on general perplexity degradation.

        Args:
            current_general_ppl: Current general validation perplexity

        Returns:
            Forgetting score (0.0 = no forgetting, >0.2 = severe)
        """
        # How much has general perplexity increased?
        degradation = current_general_ppl - self.baseline_general_ppl

        # Normalize by baseline (percentage change)
        forgetting_score = degradation / self.baseline_general_ppl

        # Clamp to [0, 1] range (negative = improvement, cap at 100% degradation)
        return max(0.0, min(1.0, forgetting_score))

    def is_forgetting(self, threshold: float = 0.10) -> bool:
        """
        Check if catastrophic forgetting is occurring.

        Args:
            threshold: Forgetting score threshold (default: 0.10 = 10% degradation)

        Returns:
            True if latest forgetting score exceeds threshold
        """
        if not self.history:
            return False

        latest = self.history[-1]
        return latest.forgetting_score > threshold

    def get_recommendations(self) -> List[str]:
        """
        Get recommended actions based on forgetting severity.

        Returns:
            List of recommended actions to mitigate forgetting

        Example:
            >>> detector = ForgettingDetector(baseline_general_perplexity=25.0)
            >>> # ... training ...
            >>> recommendations = detector.get_recommendations()
            >>> for rec in recommendations:
            ...     print(f"- {rec}")
        """
        if not self.history:
            return ["No metrics recorded yet"]

        latest = self.history[-1]
        score = latest.forgetting_score

        if score < 0.05:
            return [
                "✓ No significant forgetting detected",
                "✓ Current training approach is working well",
                "✓ Continue with current hyperparameters",
            ]
        elif score < 0.10:
            return [
                "⚠ Minor forgetting detected (5-10% degradation)",
                "→ Monitor closely in next epoch",
                "→ Consider increasing general data mix ratio",
                "→ Ensure learning rate isn't too high",
            ]
        elif score < 0.20:
            return [
                "⚠ Moderate forgetting detected (10-20% degradation)",
                "→ INCREASE general data mix ratio (e.g., 10% → 20%)",
                "→ LOWER learning rate by 50% (e.g., 1e-5 → 5e-6)",
                "→ Consider adding regularization (L2, dropout)",
                "→ Reduce number of remaining epochs",
            ]
        else:
            return [
                "❌ SEVERE forgetting detected (>20% degradation)",
                "→ STOP training immediately",
                "→ Restore from previous checkpoint",
                "→ SIGNIFICANTLY increase general data (e.g., 30-40%)",
                "→ REDUCE learning rate by 5-10x",
                "→ Consider different domain dataset (may be too different)",
            ]

    def print_summary(self):
        """
        Print a human-readable summary of forgetting metrics.

        Example Output:
        ===============
        Catastrophic Forgetting Detection Summary
        ════════════════════════════════════════════════════════════
        Baseline General Perplexity: 25.0
        Total Epochs: 10

        Epoch  Domain PPL  General PPL  Forgetting Score  Status
        ─────  ──────────  ───────────  ────────────────  ──────
            0       22.0         25.2              0.01   ✓ OK
            1       20.5         25.8              0.03   ✓ OK
            2       19.0         26.5              0.06   ⚠ Minor
            3       17.5         27.5              0.10   ⚠ Moderate
            4       16.8         28.2              0.13   ❌ Action needed
        ────────────────────────────────────────────────────────────

        Latest Recommendations:
        → INCREASE general data mix ratio
        → LOWER learning rate by 50%
        """
        if not self.history:
            print("No metrics recorded yet")
            return

        print("\nCatastrophic Forgetting Detection Summary")
        print("=" * 80)
        print(f"Baseline General Perplexity: {self.baseline_general_ppl:.1f}")
        print(f"Total Epochs: {len(self.history)}")
        print()

        print(f"{'Epoch':>5}  {'Domain PPL':>10}  {'General PPL':>11}  "
              f"{'Forgetting':>10}  {'Status':>8}")
        print("─" * 80)

        for metrics in self.history:
            # Determine status icon
            if metrics.forgetting_score < 0.05:
                status = "✓ OK"
            elif metrics.forgetting_score < 0.10:
                status = "⚠ Minor"
            elif metrics.forgetting_score < 0.20:
                status = "⚠ Action"
            else:
                status = "❌ SEVERE"

            print(
                f"{metrics.epoch:5d}  "
                f"{metrics.domain_perplexity:10.1f}  "
                f"{metrics.general_perplexity:11.1f}  "
                f"{metrics.forgetting_score:10.2f}  "
                f"{status:>8}"
            )

        print("=" * 80)
        print()

        # Show recommendations
        print("Recommendations:")
        for rec in self.get_recommendations():
            print(f"  {rec}")
        print()

    def plot_metrics(self) -> Tuple[List[int], Dict[str, List[float]]]:
        """
        Get data for plotting forgetting metrics.

        Returns:
            Tuple of (epochs, metrics_dict) where metrics_dict contains:
            - 'domain_ppl': List of domain perplexities
            - 'general_ppl': List of general perplexities
            - 'forgetting_score': List of forgetting scores

        Example:
            >>> epochs, metrics = detector.plot_metrics()
            >>> import matplotlib.pyplot as plt
            >>> plt.plot(epochs, metrics['domain_ppl'], label='Domain')
            >>> plt.plot(epochs, metrics['general_ppl'], label='General')
            >>> plt.legend()
            >>> plt.show()
        """
        epochs = [m.epoch for m in self.history]
        return epochs, {
            'domain_ppl': [m.domain_perplexity for m in self.history],
            'general_ppl': [m.general_perplexity for m in self.history],
            'forgetting_score': [m.forgetting_score for m in self.history],
        }

    def _save_history(self):
        """Save metrics history to JSON file."""
        if not self.history_file:
            return

        data = {
            'baseline_general_perplexity': self.baseline_general_ppl,
            'history': [
                {
                    'epoch': m.epoch,
                    'train_loss': m.train_loss,
                    'domain_perplexity': m.domain_perplexity,
                    'general_perplexity': m.general_perplexity,
                    'forgetting_score': m.forgetting_score,
                }
                for m in self.history
            ],
        }

        self.history_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.history_file, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load_from_file(cls, history_file: Path) -> 'ForgettingDetector':
        """
        Load forgetting detector from saved history file.

        Args:
            history_file: Path to saved history JSON

        Returns:
            ForgettingDetector with loaded history
        """
        with open(history_file, 'r') as f:
            data = json.load(f)

        detector = cls(
            baseline_general_perplexity=data['baseline_general_perplexity'],
            history_file=history_file,
        )

        for record in data['history']:
            metrics = ForgettingMetrics(**record)
            detector.history.append(metrics)

        return detector


# Example usage and demonstration
if __name__ == "__main__":
    print("=" * 80)
    print("CATASTROPHIC FORGETTING DETECTION DEMONSTRATION")
    print("=" * 80)
    print()

    # Simulate mid-training with forgetting
    print("SCENARIO: Mid-training on code with forgetting")
    print("-" * 80)
    print()

    detector = ForgettingDetector(baseline_general_perplexity=25.0)

    # Simulate 10 epochs of training
    print("Simulating training...")
    print()

    for epoch in range(10):
        # Domain perplexity improves (good!)
        domain_ppl = 22.0 - (epoch * 0.5)

        # General perplexity degrades (bad!)
        general_ppl = 25.0 + (epoch * 0.3)

        metrics = ForgettingMetrics(
            epoch=epoch,
            train_loss=3.0 - (epoch * 0.1),
            domain_perplexity=domain_ppl,
            general_perplexity=general_ppl,
        )

        detector.record(metrics)

    # Show summary
    detector.print_summary()

    print("=" * 80)
    print("EDUCATIONAL INSIGHT")
    print("=" * 80)
    print()
    print("Notice how domain perplexity improves (22 → 17.5) while general")
    print("perplexity degrades (25 → 27.7). This is catastrophic forgetting!")
    print()
    print("In practice, you would:")
    print("  1. Mix more general data into training (10% → 20%)")
    print("  2. Lower the learning rate (1e-5 → 5e-6)")
    print("  3. Monitor closely and stop if it gets worse")
    print()
    print("The goal: Gain domain expertise WITHOUT losing general capability!")
    print("=" * 80)
