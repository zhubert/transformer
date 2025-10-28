"""
Interpretability Tools for Understanding Transformer Internals

This module provides tools for mechanistic interpretability - understanding what
a trained transformer has learned by analyzing its internal representations and
mechanisms.

Key Capabilities:
    - Logit Lens: See predictions at each layer to understand how answers emerge
    - Attention Analysis: Visualize what tokens each attention head focuses on
    - Induction Head Detection: Find circuits that implement pattern matching
    - Activation Patching: Causally test which components matter for behaviors

Educational Purpose:
    These tools connect to cutting-edge mechanistic interpretability research,
    helping you understand not just how transformers work architecturally, but
    what they actually learn during training.

Example Usage:
    # Via CLI (recommended for exploration)
    python main.py interpret logit-lens --checkpoint model.pt --text "The capital of France is"
    python main.py interpret attention --checkpoint model.pt --text "Hello world"
    python main.py interpret induction-heads --checkpoint model.pt

    # Via Python API (for custom analysis)
    from transformer.interpretability import LogitLens, InductionHeadDetector

    lens = LogitLens(model, tokenizer)
    results = lens.analyze("The Eiffel Tower is in Paris")

References:
    - Logit Lens: https://www.lesswrong.com/posts/AcKRB8wDpdaN6v6ru/
    - Induction Heads: https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/
    - Activation Patching: https://www.neelnanda.io/mechanistic-interpretability/activation-patching
"""

from .logit_lens import LogitLens
from .attention_analysis import AttentionAnalyzer
from .induction_heads import InductionHeadDetector
from .activation_patching import ActivationPatcher
from .visualizations import (
    visualize_logit_lens,
    visualize_attention_pattern,
    visualize_induction_scores,
    visualize_patching_results,
    visualize_layer_patching_results,
)

__all__ = [
    "LogitLens",
    "AttentionAnalyzer",
    "InductionHeadDetector",
    "ActivationPatcher",
    "visualize_logit_lens",
    "visualize_attention_pattern",
    "visualize_induction_scores",
    "visualize_patching_results",
    "visualize_layer_patching_results",
]
