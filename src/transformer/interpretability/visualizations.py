"""
Terminal Visualizations for Interpretability Tools

This module provides Rich-based terminal visualizations for understanding
transformer internals. All visualizations are designed to work in a standard
terminal with colors and formatting.

Why Terminal Visualizations?
----------------------------
- Immediate feedback during analysis (no need to open separate tools)
- Works over SSH and in any terminal environment
- Integrates seamlessly with CLI workflow
- Uses Rich library for beautiful, informative output

Visualization Types:
--------------------
1. **Logit Lens**: Show top predictions at each layer as a table
2. **Attention Patterns**: Heatmaps showing which tokens attend to which
3. **Induction Scores**: Ranked table of heads by induction capability
4. **Patching Results**: Before/after comparison of predictions

All visualizations use color-coding, tables, and panels for clarity.
"""

import torch
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from typing import List, Dict, Any, Optional


def visualize_logit_lens(
    layer_predictions: List[List[tuple]],
    layer_names: List[str],
    input_text: str,
    console: Optional[Console] = None,
    top_k: int = 5
):
    """
    Visualize logit lens results showing how predictions evolve through layers.

    Educational Purpose:
        This visualization shows how the model's "thoughts" evolve as information
        flows through deeper layers. Early layers might predict generic tokens,
        while deeper layers converge on the correct answer.

    Args:
        layer_predictions: List of predictions for each layer
                          Each element is a list of (token_str, prob) tuples
                          Shape: [num_layers][top_k]
        layer_names: Names for each layer (e.g., ["Layer 0", "Layer 1", ...])
        input_text: The input text being analyzed
        console: Rich console object (creates new one if None)
        top_k: Number of top predictions to show per layer

    Example Output:
        ┌─ Logit Lens: Predictions at Each Layer ────────────┐
        │ Input: "The capital of France is"                  │
        │                                                      │
        │ Layer 0:  1. the    (15.2%)                        │
        │           2. a      (12.1%)                        │
        │                                                      │
        │ Layer 3:  1. Paris  (45.3%)  ← Correct!            │
        │           2. France (8.2%)                         │
        └─────────────────────────────────────────────────────┘
    """
    if console is None:
        console = Console()

    # Create header
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Input:[/bold cyan] {input_text}",
        title="[bold]Logit Lens: How Predictions Evolve Through Layers[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Create table
    table = Table(title="Top Predictions by Layer", show_header=True, header_style="bold magenta")
    table.add_column("Layer", style="cyan", width=12)

    # Add columns for each rank
    for i in range(top_k):
        table.add_column(f"#{i+1}", justify="left")

    # Add rows
    for layer_name, predictions in zip(layer_names, layer_predictions):
        row = [layer_name]
        for token_str, prob in predictions[:top_k]:
            # Color-code by probability
            if prob > 0.5:
                color = "bright_green"
            elif prob > 0.3:
                color = "green"
            elif prob > 0.1:
                color = "yellow"
            else:
                color = "white"

            row.append(f"[{color}]{token_str}[/{color}] ({prob:.1%})")

        table.add_row(*row)

    console.print(table)
    console.print()


def visualize_attention_pattern(
    tokens: List[str],
    attention_weights: torch.Tensor,
    layer_idx: int,
    head_idx: int,
    console: Optional[Console] = None,
    threshold: float = 0.1
):
    """
    Visualize attention pattern showing which tokens attend to which.

    Educational Purpose:
        Attention weights reveal what information each token is "looking at"
        when making predictions. High attention = strong connection.

    Args:
        tokens: List of token strings
        attention_weights: Attention matrix of shape (seq_len, seq_len)
                          attention_weights[i, j] = attention from token i to token j
        layer_idx: Which layer this attention is from
        head_idx: Which head this attention is from
        console: Rich console object
        threshold: Only show attention values above this threshold (default: 0.1)

    Example Output:
        ┌─ Attention Pattern: Layer 2, Head 3 ───────────────┐
        │                                                      │
        │     The    cat    sat    on     the    mat          │
        │ The [████] [░░░░] [░░░░] [░░░░] [░░░░] [░░░░]      │
        │ cat [████] [████] [░░░░] [░░░░] [░░░░] [░░░░]      │
        │ sat [████] [████] [████] [██░░] [░░░░] [░░░░]      │
        │ ...                                                  │
        └─────────────────────────────────────────────────────┘
    """
    if console is None:
        console = Console()

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Analyzing attention pattern[/bold cyan]",
        title=f"[bold]Layer {layer_idx}, Head {head_idx}[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Convert to numpy for easier manipulation
    attn = attention_weights.cpu().numpy()
    seq_len = len(tokens)

    # Create table
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("From →", style="cyan", width=12)

    # Add column for each token
    for token in tokens:
        table.add_column(token[:8], justify="center", width=8)

    # Add rows
    for i, from_token in enumerate(tokens):
        row = [from_token[:12]]
        for j in range(seq_len):
            attn_value = attn[i, j]

            # Create visual representation
            if attn_value < threshold:
                display = "[dim]·[/dim]"
            else:
                # Use color intensity based on attention strength
                if attn_value > 0.5:
                    color = "bright_red"
                    bar = "█"
                elif attn_value > 0.3:
                    color = "red"
                    bar = "▓"
                elif attn_value > 0.1:
                    color = "yellow"
                    bar = "▒"
                else:
                    color = "white"
                    bar = "░"

                display = f"[{color}]{bar}[/{color}] {attn_value:.2f}"

            row.append(display)

        table.add_row(*row)

    console.print(table)
    console.print()

    # Add interpretation
    console.print("[dim]Legend: █ > 50% | ▓ > 30% | ▒ > 10% | · < threshold[/dim]")
    console.print()


def visualize_attention_text(
    tokens: List[str],
    attention_weights: torch.Tensor,
    focus_position: int,
    layer_idx: int,
    head_idx: int,
    console: Optional[Console] = None
):
    """
    Visualize attention by highlighting tokens with color intensity.

    Educational Purpose:
        Shows attention pattern for a specific query position by highlighting
        which tokens it attends to. More vivid color = stronger attention.

    Args:
        tokens: List of token strings
        attention_weights: Attention matrix of shape (seq_len, seq_len)
        focus_position: Which position's attention to visualize (the query)
        layer_idx: Which layer
        head_idx: Which head
        console: Rich console object

    Example Output:
        Position 5 ("is") attends to:

        The capital of France is Paris
        ░░░ ███████ ██ ██████ ░░ ░░░░░
                    ↑
              Strong attention to "France"
    """
    if console is None:
        console = Console()

    console.print()
    console.print(f"[bold]Layer {layer_idx}, Head {head_idx}[/bold]")
    console.print(f"[cyan]Position {focus_position} ([/cyan][bold]{tokens[focus_position]}[/bold][cyan]) attends to:[/cyan]")
    console.print()

    # Get attention weights for this position
    attn = attention_weights[focus_position].cpu().numpy()

    # Create colored text
    text = Text()
    for i, token in enumerate(tokens):
        weight = attn[i]

        # Color based on attention strength
        if weight > 0.5:
            style = "bold bright_red"
        elif weight > 0.3:
            style = "bold red"
        elif weight > 0.2:
            style = "bold yellow"
        elif weight > 0.1:
            style = "yellow"
        else:
            style = "dim"

        text.append(token, style=style)
        text.append(" ")

    console.print(text)
    console.print()


def visualize_induction_scores(
    scores: List[Dict[str, Any]],
    console: Optional[Console] = None,
    top_k: int = 10
):
    """
    Visualize induction head detection scores as a ranked table.

    Educational Purpose:
        Shows which attention heads exhibit induction behavior (pattern matching).
        Higher scores = stronger evidence of induction circuit.

    Args:
        scores: List of dicts with keys: 'layer', 'head', 'score', 'pattern_type'
                Sorted by score (highest first)
        console: Rich console object
        top_k: Number of top heads to display

    Example Output:
        ┌─ Induction Head Detection ──────────────────────────┐
        │                                                       │
        │ Rank │ Layer │ Head │ Score │ Pattern Type          │
        │ ─────┼───────┼──────┼───────┼───────────────────    │
        │  1   │   4   │  2   │ 0.87  │ Strong Induction ✓    │
        │  2   │   5   │  1   │ 0.72  │ Moderate Induction    │
        │  3   │   3   │  3   │ 0.45  │ Weak Induction        │
        └──────────────────────────────────────────────────────┘
    """
    if console is None:
        console = Console()

    console.print()
    console.print(Panel.fit(
        "[bold cyan]Detecting attention heads that implement pattern matching[/bold cyan]",
        title="[bold]Induction Head Analysis[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Create table
    table = Table(title=f"Top {top_k} Induction Heads", show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan", width=6)
    table.add_column("Layer", justify="right", style="yellow", width=7)
    table.add_column("Head", justify="right", style="yellow", width=6)
    table.add_column("Score", justify="right", style="green", width=8)
    table.add_column("Assessment", justify="left")

    # Add rows
    for rank, score_dict in enumerate(scores[:top_k], 1):
        layer = score_dict['layer']
        head = score_dict['head']
        score = score_dict['score']

        # Determine assessment and color
        if score > 0.7:
            assessment = "[bright_green]Strong Induction ✓[/bright_green]"
        elif score > 0.5:
            assessment = "[green]Moderate Induction[/green]"
        elif score > 0.3:
            assessment = "[yellow]Weak Induction[/yellow]"
        else:
            assessment = "[dim]Minimal Induction[/dim]"

        # Color-code score
        if score > 0.7:
            score_str = f"[bright_green]{score:.3f}[/bright_green]"
        elif score > 0.5:
            score_str = f"[green]{score:.3f}[/green]"
        elif score > 0.3:
            score_str = f"[yellow]{score:.3f}[/yellow]"
        else:
            score_str = f"[dim]{score:.3f}[/dim]"

        table.add_row(
            str(rank),
            str(layer),
            str(head),
            score_str,
            assessment
        )

    console.print(table)
    console.print()
    console.print("[dim]Score > 0.7: Strong evidence of induction behavior[/dim]")
    console.print("[dim]Score > 0.5: Moderate evidence[/dim]")
    console.print("[dim]Score > 0.3: Weak evidence[/dim]")
    console.print()


def visualize_patching_results(
    clean_text: str,
    corrupted_text: str,
    component_name: str,
    clean_pred: str,
    corrupted_pred: str,
    patched_pred: str,
    clean_prob: float,
    corrupted_prob: float,
    patched_prob: float,
    console: Optional[Console] = None
):
    """
    Visualize activation patching results to show causal impact.

    Educational Purpose:
        Shows how patching a specific component's activation from a "clean" run
        into a "corrupted" run affects the model's prediction. If patching
        recovers the clean prediction, that component is causally important!

    Args:
        clean_text: The clean input text
        corrupted_text: The corrupted input text
        component_name: Name of patched component (e.g., "Layer 4, Head 2")
        clean_pred: Prediction on clean input
        corrupted_pred: Prediction on corrupted input
        patched_pred: Prediction after patching
        clean_prob: Probability of clean prediction
        corrupted_prob: Probability of corrupted prediction
        patched_prob: Probability of patched prediction
        console: Rich console object

    Example Output:
        ┌─ Activation Patching Results ───────────────────────┐
        │                                                       │
        │ Clean:      "Paris is in France"    → Paris (85%)   │
        │ Corrupted:  "Paris is in Germany"   → Germany (72%) │
        │ Patched:    [Layer 4, Head 2]       → Paris (81%)   │
        │                                                       │
        │ ✓ Patching recovered clean prediction!              │
        │   Layer 4, Head 2 is causally important.            │
        └──────────────────────────────────────────────────────┘
    """
    if console is None:
        console = Console()

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Testing causal importance of {component_name}[/bold cyan]",
        title="[bold]Activation Patching Experiment[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Create comparison table
    table = Table(show_header=True, header_style="bold magenta", box=None)
    table.add_column("Condition", style="cyan", width=20)
    table.add_column("Input", width=40)
    table.add_column("Prediction", width=20)

    table.add_row(
        "[green]Clean[/green]",
        clean_text,
        f"[green]{clean_pred}[/green] ({clean_prob:.1%})"
    )
    table.add_row(
        "[red]Corrupted[/red]",
        corrupted_text,
        f"[red]{corrupted_pred}[/red] ({corrupted_prob:.1%})"
    )
    table.add_row(
        f"[yellow]Patched[/yellow]\n[dim]({component_name})[/dim]",
        corrupted_text,
        f"[yellow]{patched_pred}[/yellow] ({patched_prob:.1%})"
    )

    console.print(table)
    console.print()

    # Determine outcome
    if patched_pred == clean_pred:
        recovery = (patched_prob - corrupted_prob) / (clean_prob - corrupted_prob) if clean_prob != corrupted_prob else 1.0
        console.print(f"[bold green]✓ Patching recovered clean prediction![/bold green]")
        console.print(f"[green]  Recovery: {recovery:.1%}[/green]")
        console.print(f"[green]  {component_name} is causally important for this behavior.[/green]")
    elif patched_pred == corrupted_pred:
        console.print(f"[bold red]✗ Patching did not change prediction.[/bold red]")
        console.print(f"[red]  {component_name} may not be important for this behavior.[/red]")
    else:
        console.print(f"[bold yellow]~ Patching changed prediction to something else.[/bold yellow]")
        console.print(f"[yellow]  {component_name} has some influence, but not deterministic.[/yellow]")

    console.print()


def visualize_layer_patching_results(
    results: List,
    clean_text: str,
    corrupted_text: str,
    target_token: str,
    console: Optional[Console] = None,
    top_k: int = 5
):
    """
    Visualize results from patching all layers.

    Shows which layers are most causally important for predicting the target token.

    Args:
        results: List of PatchResult objects from patch_all_layers()
        clean_text: Clean input text
        corrupted_text: Corrupted input text
        target_token: Target token we're trying to predict
        console: Rich console object
        top_k: Number of top layers to highlight
    """
    if console is None:
        console = Console()

    console.print()
    console.print(Panel.fit(
        f"[bold cyan]Layer-by-Layer Causal Analysis[/bold cyan]\n\n"
        f"Clean: \"{clean_text}\"\n"
        f"Corrupted: \"{corrupted_text}\"\n"
        f"Target: \"{target_token}\"\n\n"
        "Testing which layers are causally responsible for the prediction.",
        title="[bold]Activation Patching: All Layers[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Create results table
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Rank", justify="right", style="cyan", width=6)
    table.add_column("Layer", justify="center", width=7)
    table.add_column("Recovery", justify="right", width=12)
    table.add_column("Clean Prob", justify="right", width=12)
    table.add_column("Corrupted Prob", justify="right", width=15)
    table.add_column("Assessment", width=20)

    for rank, result in enumerate(results[:top_k], 1):
        recovery_pct = result.recovery_rate * 100

        # Color code by recovery rate
        if recovery_pct > 70:
            assessment = "[bold green]Critical[/bold green]"
            recovery_color = "bold green"
        elif recovery_pct > 40:
            assessment = "[green]Important[/green]"
            recovery_color = "green"
        elif recovery_pct > 20:
            assessment = "[yellow]Moderate[/yellow]"
            recovery_color = "yellow"
        else:
            assessment = "[dim]Minimal[/dim]"
            recovery_color = "dim"

        table.add_row(
            f"#{rank}",
            f"{result.layer}",
            f"[{recovery_color}]{recovery_pct:>5.1f}%[/{recovery_color}]",
            f"{result.clean_prob:.2%}",
            f"{result.corrupted_prob:.2%}",
            assessment
        )

    console.print(table)
    console.print()

    # Interpretation guide
    console.print("[bold]Interpretation:[/bold]")
    console.print("  • [bold green]Critical (>70%)[/bold green]: Layer is essential for this behavior")
    console.print("  • [green]Important (>40%)[/green]: Layer significantly contributes")
    console.print("  • [yellow]Moderate (>20%)[/yellow]: Layer has some effect")
    console.print("  • [dim]Minimal (<20%)[/dim]: Layer is not important")
    console.print()
