"""
Model interpretability tools for understanding transformer internals.

This command provides access to various interpretability techniques:
- logit-lens: Visualize predictions at each layer
- attention: Analyze attention patterns (Phase 2)
- induction-heads: Detect pattern-matching circuits (Phase 3)
- patch: Causal intervention experiments (Phase 4)

Educational Purpose:
--------------------
These tools help you understand WHAT your transformer has learned, not just
HOW it's architected. You'll discover:
- When does the model "know" the answer? (logit lens)
- What information is each head using? (attention analysis)
- Which circuits implement specific capabilities? (induction heads, patching)

This connects to cutting-edge mechanistic interpretability research from
Anthropic, OpenAI, and academic labs.

Usage Examples:
---------------
    # Logit lens: See how predictions evolve through layers
    uv run python main.py interpret logit-lens checkpoints/model.pt \
        --text "The capital of France is"

    # Logit lens with demo mode (educational examples)
    uv run python main.py interpret logit-lens checkpoints/model.pt --demo

    # Interactive mode
    uv run python main.py interpret logit-lens checkpoints/model.pt --interactive

    # Attention analysis (coming in Phase 2)
    uv run python main.py interpret attention checkpoints/model.pt \
        --text "Hello world" --layer 3 --head 2

    # Induction head detection (coming in Phase 3)
    uv run python main.py interpret induction-heads checkpoints/model.pt

    # Activation patching (coming in Phase 4)
    uv run python main.py interpret patch checkpoints/model.pt \
        --clean "Paris is in France" \
        --corrupted "Paris is in Germany" \
        --component "layer.4.head.2"
"""

import torch
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.device_utils import init_device
from src.transformer.interpretability import (
    LogitLens,
    visualize_logit_lens,
    visualize_attention_pattern,
    visualize_induction_scores,
)
from src.transformer.checkpoint_utils import load_checkpoint as load_checkpoint_util
import tiktoken
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        Tuple of (model, config dict, tokenizer)
    """
    console = Console()

    console.print(f"\n[cyan]Loading checkpoint from:[/cyan] {checkpoint_path}")

    # Use checkpoint utilities for loading (verbose=False to control output)
    result = load_checkpoint_util(checkpoint_path, device=device, verbose=False)

    model = result['model']
    config = result['config']
    encoding_name = result['encoding']

    console.print(f"[cyan]Detected encoding:[/cyan] {encoding_name}")

    # Initialize tokenizer
    tokenizer = tiktoken.get_encoding(encoding_name)

    console.print(f"[green]✓ Model loaded successfully[/green]")
    console.print(f"[dim]Layers: {config['num_layers']}, "
                 f"d_model: {config['d_model']}, "
                 f"heads: {config['num_heads']}[/dim]\n")

    return model, config, tokenizer


# ============================================================================
# LOGIT LENS COMMAND
# ============================================================================

def cmd_logit_lens(args):
    """
    Logit lens: Visualize how predictions evolve through layers.

    This command shows what the model would predict if we stopped at each layer,
    revealing how the model's "thoughts" evolve as information flows deeper.
    """
    console = Console()

    # Initialize device
    device, device_name = init_device()
    console.print(f"[cyan]Using device:[/cyan] {device_name}\n")

    # Load model
    model, config, tokenizer = load_model(args.checkpoint, device)

    # Create logit lens analyzer
    lens = LogitLens(model, tokenizer)

    # Demo mode: Show educational examples
    if args.demo:
        console.print(Panel.fit(
            "[bold cyan]Demo Mode: Educational Examples[/bold cyan]\n\n"
            "These examples demonstrate how predictions evolve through layers.\n"
            "Watch how early layers predict generic tokens, while deeper layers\n"
            "converge on semantically meaningful answers.",
            title="[bold]Logit Lens Demo[/bold]",
            border_style="cyan"
        ))
        console.print()

        demo_examples = [
            "The Eiffel Tower is located in",
            "To be or not to",
            "The capital of France is",
        ]

        for example in demo_examples:
            console.print(f"[bold]Analyzing:[/bold] \"{example}\"")
            console.print("=" * 60)

            results = lens.analyze(
                text=example,
                position=-1,
                top_k=args.top_k,
                temperature=args.temperature
            )

            visualize_logit_lens(
                layer_predictions=results['layer_predictions'],
                layer_names=results['layer_names'],
                input_text=results['input_text'],
                console=console,
                top_k=args.top_k
            )

            console.print()

        return

    # Interactive mode
    if args.interactive:
        console.print(Panel.fit(
            "[bold cyan]Interactive Logit Lens[/bold cyan]\n\n"
            "Enter text to analyze how predictions evolve through layers.\n"
            "Type 'quit' or 'exit' to exit.",
            title="[bold]Interactive Mode[/bold]",
            border_style="cyan"
        ))
        console.print()

        while True:
            text = Prompt.ask("\n[bold cyan]Enter text to analyze[/bold cyan]")

            if text.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Exiting...[/yellow]")
                break

            if not text.strip():
                continue

            console.print()
            results = lens.analyze(
                text=text,
                position=-1,
                top_k=args.top_k,
                temperature=args.temperature
            )

            visualize_logit_lens(
                layer_predictions=results['layer_predictions'],
                layer_names=results['layer_names'],
                input_text=results['input_text'],
                console=console,
                top_k=args.top_k
            )

        return

    # Single text analysis
    if args.text:
        results = lens.analyze(
            text=args.text,
            position=-1,
            top_k=args.top_k,
            temperature=args.temperature
        )

        visualize_logit_lens(
            layer_predictions=results['layer_predictions'],
            layer_names=results['layer_names'],
            input_text=results['input_text'],
            console=console,
            top_k=args.top_k
        )
    else:
        console.print("[red]Error:[/red] Please provide --text, --demo, or --interactive")
        sys.exit(1)


# ============================================================================
# ATTENTION ANALYSIS COMMAND (Phase 2)
# ============================================================================

def cmd_attention(args):
    """
    Attention pattern analysis: Visualize what tokens each head focuses on.

    This command shows attention weights to understand what information
    each attention head is "looking at" when making predictions.
    """
    console = Console()

    # Initialize device
    device, device_name = init_device()
    console.print(f"[cyan]Using device:[/cyan] {device_name}\n")

    # Load model
    model, config, tokenizer = load_model(args.checkpoint, device)

    # Create attention analyzer
    from src.transformer.interpretability import AttentionAnalyzer
    analyzer = AttentionAnalyzer(model, tokenizer)

    # Demo mode: Show educational examples
    if args.demo:
        console.print(Panel.fit(
            "[bold cyan]Demo Mode: Attention Pattern Examples[/bold cyan]\n\n"
            "These examples show common attention patterns that emerge during training.\n"
            "Different heads learn to focus on different types of information.",
            title="[bold]Attention Analysis Demo[/bold]",
            border_style="cyan"
        ))
        console.print()

        demo_text = "The quick brown fox jumps over the lazy dog"

        console.print(f"[bold]Analyzing:[/bold] \"{demo_text}\"")
        console.print("=" * 60)
        console.print()

        # Analyze all layers and find patterns
        results = analyzer.analyze(demo_text)

        # Find heads by pattern type
        console.print("[bold cyan]Searching for common attention patterns...[/bold cyan]\n")

        for pattern_type in ["previous_token", "uniform", "start_token", "sparse"]:
            heads = analyzer.find_heads_by_pattern(demo_text, pattern_type)

            if heads:
                console.print(f"[bold]{pattern_type.replace('_', ' ').title()} Heads:[/bold]")
                for head in heads[:3]:  # Show top 3
                    console.print(f"  Layer {head['layer']}, Head {head['head']}: "
                                f"strength = {head['strength']:.2%}")
                console.print()

        # Show specific head visualization
        if results['num_layers'] > 0:
            layer_idx = min(2, results['num_layers'] - 1)
            head_idx = 0

            console.print(f"[bold]Visualizing Layer {layer_idx}, Head {head_idx}:[/bold]\n")

            head_results = analyzer.analyze(demo_text, layer_idx=layer_idx, head_idx=head_idx)

            visualize_attention_pattern(
                tokens=head_results['tokens'],
                attention_weights=head_results['attention_weights'],
                layer_idx=layer_idx,
                head_idx=head_idx,
                console=console
            )

        return

    # Interactive mode
    if args.interactive:
        console.print(Panel.fit(
            "[bold cyan]Interactive Attention Analysis[/bold cyan]\n\n"
            "Enter text to analyze attention patterns.\n"
            "Type 'quit' or 'exit' to exit.",
            title="[bold]Interactive Mode[/bold]",
            border_style="cyan"
        ))
        console.print()

        while True:
            text = Prompt.ask("\n[bold cyan]Enter text to analyze[/bold cyan]")

            if text.lower() in ['quit', 'exit', 'q']:
                console.print("[yellow]Exiting...[/yellow]")
                break

            if not text.strip():
                continue

            # Get layer and head from user
            layer_idx = Prompt.ask(
                f"[cyan]Layer to analyze (0-{config['num_layers']-1}, or 'all')[/cyan]",
                default="0"
            )

            if layer_idx.lower() == 'all':
                layer_idx = None
            else:
                layer_idx = int(layer_idx)

            if layer_idx is not None:
                head_idx = Prompt.ask(
                    f"[cyan]Head to analyze (0-{config['num_heads']-1}, or 'all')[/cyan]",
                    default="0"
                )

                if head_idx.lower() == 'all':
                    head_idx = None
                else:
                    head_idx = int(head_idx)
            else:
                head_idx = None

            console.print()

            # Analyze
            results = analyzer.analyze(text, layer_idx=layer_idx, head_idx=head_idx)

            if layer_idx is not None and head_idx is not None:
                # Single head visualization
                visualize_attention_pattern(
                    tokens=results['tokens'],
                    attention_weights=results['attention_weights'],
                    layer_idx=layer_idx,
                    head_idx=head_idx,
                    console=console
                )
            elif layer_idx is not None:
                # All heads in layer
                console.print(f"[bold]Layer {layer_idx} - All Heads:[/bold]\n")
                for h_idx in range(results['num_heads']):
                    attn = results['attention_weights'][h_idx]
                    pattern = analyzer.get_head_pattern_type(attn)
                    console.print(f"  Head {h_idx}: [cyan]{pattern}[/cyan]")
                console.print()
            else:
                # All layers
                console.print("[bold]Pattern Summary Across All Layers:[/bold]\n")
                for l_idx, layer_attn in enumerate(results['attention_weights']):
                    console.print(f"[yellow]Layer {l_idx}:[/yellow]")
                    for h_idx in range(results['num_heads']):
                        attn = layer_attn[h_idx]
                        pattern = analyzer.get_head_pattern_type(attn)
                        console.print(f"  Head {h_idx}: {pattern}")
                    console.print()

        return

    # Single text analysis
    if args.text:
        layer_idx = args.layer
        head_idx = args.head

        results = analyzer.analyze(args.text, layer_idx=layer_idx, head_idx=head_idx)

        if layer_idx is not None and head_idx is not None:
            # Add introductory text
            console.print(Panel.fit(
                f"[bold cyan]Analyzing Attention Head {layer_idx}.{head_idx}[/bold cyan]\n\n"
                f"Input text: \"{args.text}\"\n\n"
                "The attention matrix below shows where each token (row) is focusing.\n"
                "Each row sums to 100% - the model distributes attention across positions.",
                title="[bold]Attention Pattern Analysis[/bold]",
                border_style="cyan"
            ))
            console.print()

            # Single head visualization
            visualize_attention_pattern(
                tokens=results['tokens'],
                attention_weights=results['attention_weights'],
                layer_idx=layer_idx,
                head_idx=head_idx,
                console=console
            )

            # Show pattern classification with explanation
            pattern = analyzer.get_head_pattern_type(results['attention_weights'])
            console.print(f"\n[bold]Detected Pattern:[/bold] [cyan]{pattern}[/cyan]")

            # Add pattern-specific explanations
            pattern_explanations = {
                "previous_token": "This head focuses on the immediately preceding token (position i-1).\n"
                                 "→ Useful for capturing local dependencies and sequence relationships.",
                "uniform": "This head spreads attention evenly across all positions.\n"
                          "→ Acts as an averaging mechanism, gathering broad context.",
                "start_token": "This head focuses primarily on the beginning of the sequence.\n"
                              "→ May be retrieving global context or sentence-level information.",
                "sparse": "This head concentrates attention on very few key positions.\n"
                         "→ Suggests selective information retrieval from specific tokens.",
                "mixed": "This head shows no clear single pattern.\n"
                        "→ May implement task-specific attention or multiple behaviors."
            }

            if pattern in pattern_explanations:
                console.print(f"[dim]{pattern_explanations[pattern]}[/dim]\n")

        elif layer_idx is not None:
            # All heads in layer
            console.print(Panel.fit(
                f"[bold cyan]Analyzing All Heads in Layer {layer_idx}[/bold cyan]\n\n"
                f"Input text: \"{args.text}\"\n\n"
                "Classifying attention patterns for each head in this layer.\n"
                "Different heads often specialize in different types of information.",
                title="[bold]Layer-Wide Pattern Analysis[/bold]",
                border_style="cyan"
            ))
            console.print()

            console.print(f"[bold]Pattern Summary:[/bold]\n")
            for h_idx in range(results['num_heads']):
                attn = results['attention_weights'][h_idx]
                pattern = analyzer.get_head_pattern_type(attn)
                console.print(f"  Head {h_idx}: [cyan]{pattern}[/cyan]")

            console.print()
            console.print("[dim]💡 Tip: Use --head <N> to see detailed attention matrix for a specific head[/dim]\n")
        else:
            # All layers
            console.print(Panel.fit(
                f"[bold cyan]Pattern Discovery Across All Layers[/bold cyan]\n\n"
                f"Input text: \"{args.text}\"\n\n"
                "Scanning all attention heads to find specialized patterns.\n"
                "Heads are ranked by how strongly they exhibit each pattern type.",
                title="[bold]Model-Wide Pattern Search[/bold]",
                border_style="cyan"
            ))
            console.print()

            console.print("[bold]Discovered Attention Patterns:[/bold]\n")

            pattern_descriptions = {
                "previous_token": "Focuses on the immediately preceding token",
                "uniform": "Distributes attention evenly (averaging)",
                "start_token": "Focuses on the beginning of the sequence",
                "sparse": "Concentrates on very few specific tokens"
            }

            found_any = False
            for pattern_type in ["previous_token", "uniform", "start_token", "sparse"]:
                heads = analyzer.find_heads_by_pattern(args.text, pattern_type)

                if heads:
                    found_any = True
                    console.print(f"[bold yellow]{pattern_type.replace('_', ' ').title()}:[/bold yellow] "
                                f"[dim]{pattern_descriptions[pattern_type]}[/dim]")
                    for head in heads[:5]:
                        console.print(f"  Layer {head['layer']}, Head {head['head']}: "
                                    f"{head['strength']:.2%}")
                    console.print()

            if not found_any:
                console.print("[yellow]No strong patterns detected with this input.[/yellow]")
                console.print("[dim]Try a longer or more structured text sequence.[/dim]\n")
            else:
                console.print("[dim]💡 Tip: Use --layer <N> --head <M> to visualize a specific head's attention[/dim]\n")
    else:
        console.print("[red]Error:[/red] Please provide --text, --demo, or --interactive")
        sys.exit(1)


# ============================================================================
# INDUCTION HEAD DETECTION (Phase 3)
# ============================================================================

def cmd_induction_heads(args):
    """
    Induction head detection: Find heads that implement pattern matching.

    Induction heads are circuits that copy from context based on seeing
    repeated patterns, enabling in-context learning capabilities.
    """
    console = Console()

    # Initialize device
    device, device_name = init_device()
    console.print(f"[cyan]Using device:[/cyan] {device_name}\n")

    # Load model
    model, config, tokenizer = load_model(args.checkpoint, device)

    # Create induction head detector
    from src.transformer.interpretability import InductionHeadDetector
    detector = InductionHeadDetector(model, tokenizer)

    console.print(Panel.fit(
        "[bold cyan]Induction Head Detection[/bold cyan]\n\n"
        "Searching for attention heads that implement pattern matching and copying.\n"
        "This tests each head on repeated sequences to measure induction behavior.\n\n"
        f"[yellow]Testing: {args.num_sequences} sequences, {args.seq_length} tokens each[/yellow]",
        title="[bold]Analyzing Model[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Run detection
    results = detector.detect(
        num_sequences=args.num_sequences,
        seq_length=args.seq_length
    )

    console.print()
    console.print("[bold green]✓ Detection complete![/bold green]\n")

    # Visualize results
    visualize_induction_scores(
        scores=results,
        console=console,
        top_k=args.top_k
    )

    # Show detailed analysis of top induction head
    if results and results[0]['score'] > 0.3:
        top_result = results[0]
        console.print(f"\n[bold]Analyzing Top Induction Head:[/bold] "
                     f"Layer {top_result['layer']}, Head {top_result['head']}\n")

        # Analyze the induction circuit
        test_text = "A B C D E A B C D E"
        circuit = detector.analyze_induction_circuit(
            text=test_text,
            induction_layer=top_result['layer'],
            induction_head=top_result['head']
        )

        console.print(Panel(
            circuit['explanation'],
            title="[bold]Circuit Analysis[/bold]",
            border_style="green" if circuit['circuit_strength'] > 0.5 else "yellow"
        ))
        console.print()


# ============================================================================
# ACTIVATION PATCHING (Phase 4)
# ============================================================================

def cmd_patch(args):
    """
    Activation patching: Causal intervention to identify important components.

    Tests which parts of the model are causally responsible for specific behaviors
    by swapping activations between "clean" and "corrupted" runs.
    """
    console = Console()

    # Initialize device
    device, device_name = init_device()
    console.print(f"[cyan]Using device:[/cyan] {device_name}\n")

    # Load model
    model, config, tokenizer = load_model(args.checkpoint, device)

    # Create activation patcher
    from src.transformer.interpretability import ActivationPatcher, visualize_layer_patching_results
    patcher = ActivationPatcher(model, tokenizer)

    # Check required arguments
    if not args.clean or not args.corrupted or not args.target:
        console.print("[red]Error:[/red] Activation patching requires --clean, --corrupted, and --target arguments")
        console.print("\n[yellow]Example:[/yellow]")
        console.print('  uv run python main.py interpret patch checkpoints/model.pt \\')
        console.print('    --clean "The Eiffel Tower is in" \\')
        console.print('    --corrupted "The Empire State Building is in" \\')
        console.print('    --target "Paris"')
        sys.exit(1)

    # Display experiment setup
    console.print(Panel.fit(
        "[bold cyan]Activation Patching Experiment[/bold cyan]\n\n"
        "Testing which layers are causally responsible for the prediction.\n\n"
        f"Clean: \"{args.clean}\"\n"
        f"Corrupted: \"{args.corrupted}\"\n"
        f"Target: \"{args.target}\"\n\n"
        "This may take a moment...",
        title="[bold]Causal Analysis[/bold]",
        border_style="cyan"
    ))
    console.print()

    # Get baseline predictions
    console.print("[bold]Step 1:[/bold] Running clean and corrupted inputs...")
    clean_logits, _ = patcher.run_with_cache(args.clean)
    corrupted_logits, _ = patcher.run_with_cache(args.corrupted)

    # Show baseline predictions
    clean_preds = patcher.get_top_predictions(clean_logits, position=-1, top_k=3)
    corrupted_preds = patcher.get_top_predictions(corrupted_logits, position=-1, top_k=3)

    console.print(f"  Clean predictions: {', '.join([f'{tok} ({prob:.1%})' for tok, prob in clean_preds[:3]])}")
    console.print(f"  Corrupted predictions: {', '.join([f'{tok} ({prob:.1%})' for tok, prob in corrupted_preds[:3]])}")
    console.print()

    # Run patching experiment
    console.print("[bold]Step 2:[/bold] Patching each layer individually...")
    results = patcher.patch_all_layers(
        args.clean,
        args.corrupted,
        args.target,
        position=-1
    )
    console.print(f"  Tested {len(results)} layers")
    console.print()

    # Visualize results
    console.print("[bold]Step 3:[/bold] Results:\n")
    visualize_layer_patching_results(
        results=results,
        clean_text=args.clean,
        corrupted_text=args.corrupted,
        target_token=args.target,
        console=console,
        top_k=min(len(results), args.top_k if hasattr(args, 'top_k') else 5)
    )


# ============================================================================
# MAIN ARGUMENT PARSER
# ============================================================================

def setup_parser(parser):
    """
    Set up argument parser for interpret command.

    This creates subcommands for each interpretability tool.
    """
    subparsers = parser.add_subparsers(dest='subcommand', help='Interpretability tool to use')
    subparsers.required = True

    # --------------------------------------------------------------------
    # LOGIT LENS subcommand
    # --------------------------------------------------------------------
    logit_lens_parser = subparsers.add_parser(
        'logit-lens',
        help='Visualize predictions at each layer',
        description='See how the model\'s predictions evolve through layers'
    )
    logit_lens_parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    logit_lens_parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze'
    )
    logit_lens_parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode with educational examples'
    )
    logit_lens_parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    logit_lens_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top predictions to show per layer (default: 5)'
    )
    logit_lens_parser.add_argument(
        '--temperature',
        type=float,
        default=1.0,
        help='Temperature for probability scaling (default: 1.0)'
    )
    logit_lens_parser.set_defaults(func=cmd_logit_lens)

    # --------------------------------------------------------------------
    # ATTENTION subcommand (Phase 2)
    # --------------------------------------------------------------------
    attention_parser = subparsers.add_parser(
        'attention',
        help='Analyze attention patterns',
        description='Visualize what tokens each attention head focuses on'
    )
    attention_parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    attention_parser.add_argument(
        '--text',
        type=str,
        help='Text to analyze'
    )
    attention_parser.add_argument(
        '--layer',
        type=int,
        help='Layer index to analyze (e.g., 0, 1, 2, ...)'
    )
    attention_parser.add_argument(
        '--head',
        type=int,
        help='Head index to analyze (e.g., 0, 1, 2, ...)'
    )
    attention_parser.add_argument(
        '--demo',
        action='store_true',
        help='Run demo mode with educational examples'
    )
    attention_parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    attention_parser.set_defaults(func=cmd_attention)

    # --------------------------------------------------------------------
    # INDUCTION HEADS subcommand (Phase 3)
    # --------------------------------------------------------------------
    induction_parser = subparsers.add_parser(
        'induction-heads',
        help='Detect induction heads (Phase 3)',
        description='Find attention heads that implement pattern matching'
    )
    induction_parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    induction_parser.add_argument(
        '--num-sequences',
        type=int,
        default=100,
        help='Number of test sequences (default: 100)'
    )
    induction_parser.add_argument(
        '--seq-length',
        type=int,
        default=40,
        help='Length of each random portion in test sequences (default: 40)'
    )
    induction_parser.add_argument(
        '--top-k',
        type=int,
        default=10,
        help='Number of top induction heads to show (default: 10)'
    )
    induction_parser.set_defaults(func=cmd_induction_heads)

    # --------------------------------------------------------------------
    # PATCH subcommand (Phase 4)
    # --------------------------------------------------------------------
    patch_parser = subparsers.add_parser(
        'patch',
        help='Activation patching (Phase 4)',
        description='Causal intervention to find important components'
    )
    patch_parser.add_argument(
        'checkpoint',
        type=str,
        help='Path to model checkpoint'
    )
    patch_parser.add_argument(
        '--clean',
        type=str,
        help='Clean input text'
    )
    patch_parser.add_argument(
        '--corrupted',
        type=str,
        help='Corrupted input text'
    )
    patch_parser.add_argument(
        '--target',
        type=str,
        help='Target token to predict'
    )
    patch_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Number of top layers to show (default: 5)'
    )
    patch_parser.set_defaults(func=cmd_patch)


def main(args):
    """Main entry point for interpret command."""
    # Check if called from argparse (has func attribute) or interactive CLI (has analysis attribute)
    if hasattr(args, 'func'):
        # Called from argparse - use the subcommand function
        args.func(args)
    elif hasattr(args, 'analysis'):
        # Called from interactive CLI - map analysis type to function
        console = Console()

        # Map analysis types to command functions
        analysis_map = {
            'attention': cmd_attention,
            'logit-lens': cmd_logit_lens,
            'induction-heads': cmd_induction_heads,
            'patch': cmd_patch,
        }

        # Set up args for the command functions
        # Interactive CLI provides: checkpoint, analysis, prompt, output_dir, device
        # Command functions expect: checkpoint, text, demo, interactive, layer, head, etc.

        if args.analysis == 'all':
            # Run all available analyses
            console.print("[bold cyan]Running all available analyses...[/bold cyan]\n")

            # Run attention analysis
            if args.prompt:
                args.text = args.prompt
                args.demo = False
                args.interactive = False
                args.layer = None
                args.head = None
                console.print("[bold]1. Attention Pattern Analysis[/bold]")
                cmd_attention(args)
                console.print()

            # Run logit lens
            args.text = args.prompt if args.prompt else None
            args.demo = False
            args.interactive = False
            console.print("[bold]2. Logit Lens Analysis[/bold]")
            cmd_logit_lens(args)
            console.print()

            console.print("[green]✓ All analyses complete![/green]")

        elif args.analysis in analysis_map:
            # Run specific analysis
            func = analysis_map[args.analysis]

            # Set up common args
            args.text = args.prompt if hasattr(args, 'prompt') else None
            args.demo = False
            args.interactive = False

            # Attention-specific args
            if args.analysis == 'attention':
                args.layer = None
                args.head = None

            func(args)

        elif args.analysis in ['embeddings', 'neurons']:
            # These analyses are not yet implemented
            console.print(f"[yellow]Note: '{args.analysis}' analysis is not yet implemented.[/yellow]")
            console.print("[dim]Available analyses: attention, logit-lens, induction-heads, patch[/dim]")
            console.print()
            console.print("[cyan]Running attention analysis instead...[/cyan]\n")

            # Fall back to attention analysis
            args.text = args.prompt if hasattr(args, 'prompt') else None
            args.demo = False
            args.interactive = False
            args.layer = None
            args.head = None
            cmd_attention(args)

        else:
            console.print(f"[red]Unknown analysis type: {args.analysis}[/red]")
            console.print("[dim]Available: attention, logit-lens, induction-heads, patch, all[/dim]")
    else:
        raise ValueError("args object must have either 'func' or 'analysis' attribute")
