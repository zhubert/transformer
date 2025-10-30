#!/usr/bin/env python3
"""
Interactive CLI for Transformer operations.

This module provides a user-friendly interactive interface for all transformer
operations, making it easy to train, generate, evaluate, and analyze models
without memorizing command-line flags.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict

import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from commands.train import train
from commands.download_shards import download_shards
from commands.generate import main as generate_main
from commands.evaluate_perplexity import evaluate_checkpoint, compare_checkpoints
from commands.sampling_comparison import demonstrate_sampling_strategies, demonstrate_with_model
from commands import interpret
from src.transformer.device_utils import init_device, get_autocast_context

# Initialize Rich console for pretty output
console = Console()

# Custom style for questionary prompts
custom_style = Style([
    ('qmark', 'fg:#673ab7 bold'),       # Question mark
    ('question', 'bold'),                # Question text
    ('answer', 'fg:#2196f3 bold'),      # Selected answer
    ('pointer', 'fg:#673ab7 bold'),     # Selection pointer
    ('highlighted', 'fg:#673ab7 bold'), # Highlighted choice
    ('selected', 'fg:#2196f3'),         # Selected choice
    ('separator', 'fg:#666666'),        # Separator
    ('instruction', 'fg:#666666'),      # Instructions
])


class CheckpointScanner:
    """Scans for available model checkpoints across different directories."""

    def __init__(self):
        self.checkpoint_dirs = {
            'default': Path('checkpoints'),
            'medium': Path('checkpoints_medium'),
            'quick': Path('checkpoints_quick'),
        }
        self.checkpoints: Dict[str, List[Path]] = {}
        self.scan()

    def scan(self):
        """Scan all checkpoint directories for model files."""
        for mode, dir_path in self.checkpoint_dirs.items():
            if dir_path.exists():
                checkpoints = sorted(dir_path.glob('model_epoch_*.pt'))
                if checkpoints:
                    self.checkpoints[mode] = checkpoints

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.checkpoints) > 0

    def get_all_checkpoints(self) -> List[tuple[str, Path]]:
        """Get all checkpoints with their mode labels."""
        all_checkpoints = []
        for mode, checkpoints in self.checkpoints.items():
            for checkpoint in checkpoints:
                all_checkpoints.append((mode, checkpoint))
        return all_checkpoints

    def get_latest(self) -> Optional[tuple[str, Path]]:
        """Get the most recent checkpoint across all directories."""
        all_checkpoints = self.get_all_checkpoints()
        if not all_checkpoints:
            return None
        # Sort by modification time
        return max(all_checkpoints, key=lambda x: x[1].stat().st_mtime)

    def display_summary(self):
        """Display a summary table of available checkpoints."""
        if not self.has_checkpoints():
            console.print("[yellow]No checkpoints found. Train a model first![/yellow]")
            return

        table = Table(title="Available Checkpoints", box=box.ROUNDED)
        table.add_column("Mode", style="cyan", no_wrap=True)
        table.add_column("Checkpoint", style="green")
        table.add_column("Size", justify="right", style="magenta")

        for mode, checkpoints in self.checkpoints.items():
            for i, checkpoint in enumerate(checkpoints):
                size_mb = checkpoint.stat().st_size / (1024 * 1024)
                mode_label = mode.upper() if i == 0 else ""
                table.add_row(mode_label, checkpoint.name, f"{size_mb:.1f} MB")

        console.print(table)

        latest = self.get_latest()
        if latest:
            mode, path = latest
            console.print(f"\n[bold green]Latest:[/bold green] {path.name} ({mode} mode)")


def show_welcome():
    """Display welcome message and system status."""
    console.clear()

    welcome_text = """[bold cyan]Transformer Interactive CLI[/bold cyan]

Welcome to the educational transformer implementation!
This interactive interface makes it easy to:
  • Train new models or continue training
  • Generate text with various sampling strategies
  • Evaluate model performance
  • Analyze model internals (interpretability)
  • Download training data for offline use

Use arrow keys to navigate menus, Enter to select."""

    console.print(Panel(welcome_text, border_style="cyan", box=box.ROUNDED))
    console.print()


def main_menu(scanner: CheckpointScanner) -> str:
    """Display main menu and get user choice."""
    choices = ["🎓 Train new model"]

    if scanner.has_checkpoints():
        choices.extend([
            "▶️  Continue training",
            "✨ Generate text",
            "📊 Evaluate models",
            "🔍 Interpretability analysis",
        ])

    choices.extend([
        "⬇️  Download training data",
        "❌ Exit",
    ])

    return questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style,
    ).ask()


def train_menu() -> dict:
    """Training configuration menu."""
    console.print("\n[bold cyan]Training Configuration[/bold cyan]\n")

    # Select training mode
    mode = questionary.select(
        "Select training mode:",
        choices=[
            "Quick (10M tokens/epoch × 10, 4 layers, ~40min first epoch)",
            "Medium (50M tokens/epoch × 15, 4 layers, ~2h first epoch)",
            "Full (100M tokens/epoch × 20, 6 layers, ~4h first epoch)",
        ],
        style=custom_style,
    ).ask()

    # Parse mode
    if mode.startswith("Quick"):
        quick, medium = True, False
    elif mode.startswith("Medium"):
        quick, medium = False, True
    else:
        quick, medium = False, False

    # Resume option
    resume = questionary.confirm(
        "Resume from latest checkpoint (if available)?",
        default=False,
        style=custom_style,
    ).ask()

    # Advanced options
    show_advanced = questionary.confirm(
        "Show advanced options?",
        default=False,
        style=custom_style,
    ).ask()

    debug = False
    use_mps = False
    compile = True

    if show_advanced:
        debug = questionary.confirm(
            "Enable debug mode (verbose NaN detection)?",
            default=False,
            style=custom_style,
        ).ask()

        use_mps = questionary.confirm(
            "Use MPS (Apple Silicon GPU)? [EXPERIMENTAL]",
            default=False,
            style=custom_style,
        ).ask()

        compile = questionary.confirm(
            "Use torch.compile() for 20-40% speedup?",
            default=True,
            style=custom_style,
        ).ask()

    return {
        'quick': quick,
        'medium': medium,
        'resume': resume,
        'debug': debug,
        'use_mps': use_mps,
        'compile': compile,
    }


def continue_training_menu(scanner: CheckpointScanner) -> dict:
    """Continue training menu - auto-detects mode from checkpoint."""
    console.print("\n[bold cyan]Continue Training[/bold cyan]\n")

    # Get latest checkpoint
    latest = scanner.get_latest()
    if not latest:
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    mode, checkpoint = latest

    console.print(f"Latest checkpoint: [green]{checkpoint.name}[/green] ({mode} mode)")

    confirm = questionary.confirm(
        f"Continue training from this checkpoint?",
        default=True,
        style=custom_style,
    ).ask()

    if not confirm:
        return None

    # Auto-detect mode settings
    if mode == 'quick':
        quick, medium = True, False
    elif mode == 'medium':
        quick, medium = False, True
    else:
        quick, medium = False, False

    return {
        'quick': quick,
        'medium': medium,
        'resume': True,
        'debug': False,
        'use_mps': False,
        'compile': True,
    }


def generate_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Text generation menu."""
    console.print("\n[bold cyan]Text Generation[/bold cyan]\n")

    # Select checkpoint
    all_checkpoints = scanner.get_all_checkpoints()
    if not all_checkpoints:
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Create choices with mode labels
    checkpoint_choices = [
        f"{checkpoint.name} ({mode} mode)"
        for mode, checkpoint in all_checkpoints
    ]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    # Find the selected checkpoint path
    selected_idx = checkpoint_choices.index(selected)
    mode, checkpoint_path = all_checkpoints[selected_idx]

    # Select preset
    preset = questionary.select(
        "Select generation preset:",
        choices=[
            "greedy - Deterministic, picks most likely tokens",
            "precise - Low randomness (temp=0.7, top-k=50)",
            "balanced - Moderate creativity (temp=0.8, top-k=50, top-p=0.9) [DEFAULT]",
            "creative - High creativity (temp=1.0, top-k=100, top-p=0.95)",
            "very-creative - Maximum creativity (temp=1.2, top-k=100, top-p=0.95)",
        ],
        style=custom_style,
    ).ask()

    preset_name = preset.split(' - ')[0]

    # Interactive or single prompt
    mode_choice = questionary.select(
        "Generation mode:",
        choices=[
            "Interactive - Multiple prompts in a loop",
            "Single prompt - Generate once and exit",
        ],
        style=custom_style,
    ).ask()

    prompt = None
    if mode_choice.startswith("Single"):
        prompt = questionary.text(
            "Enter your prompt:",
            style=custom_style,
        ).ask()

    # Max length
    max_length = questionary.text(
        "Maximum tokens to generate:",
        default="100",
        style=custom_style,
    ).ask()

    return {
        'checkpoint': str(checkpoint_path),
        'preset': preset_name,
        'prompt': prompt,
        'max_length': int(max_length),
    }


def evaluate_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Model evaluation menu."""
    console.print("\n[bold cyan]Model Evaluation[/bold cyan]\n")

    if not scanner.has_checkpoints():
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Evaluate single or compare all
    action = questionary.select(
        "Evaluation type:",
        choices=[
            "Evaluate single checkpoint (perplexity)",
            "Compare all checkpoints",
        ],
        style=custom_style,
    ).ask()

    if action.startswith("Evaluate single"):
        # Select checkpoint
        all_checkpoints = scanner.get_all_checkpoints()
        checkpoint_choices = [
            f"{checkpoint.name} ({mode} mode)"
            for mode, checkpoint in all_checkpoints
        ]

        selected = questionary.select(
            "Select checkpoint:",
            choices=checkpoint_choices,
            style=custom_style,
        ).ask()

        selected_idx = checkpoint_choices.index(selected)
        mode, checkpoint_path = all_checkpoints[selected_idx]

        return {
            'mode': 'single',
            'checkpoint': str(checkpoint_path),
            'seq_length': 128,
            'batch_size': 8,
            'device': None,  # Auto-detect device
            'tokens_per_epoch': 10_000_000,
        }
    else:
        # Compare all - ask which directory
        mode = questionary.select(
            "Compare checkpoints from which training mode?",
            choices=list(scanner.checkpoints.keys()),
            style=custom_style,
        ).ask()

        checkpoint_dir = scanner.checkpoint_dirs[mode]

        return {
            'mode': 'compare',
            'checkpoint_dir': str(checkpoint_dir),
            'seq_length': 128,
            'device': None,  # Auto-detect device
            'tokens_per_epoch': 10_000_000,
        }


def interpret_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Interpretability analysis menu."""
    console.print("\n[bold cyan]Interpretability Analysis[/bold cyan]\n")

    if not scanner.has_checkpoints():
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Select checkpoint
    all_checkpoints = scanner.get_all_checkpoints()
    checkpoint_choices = [
        f"{checkpoint.name} ({mode} mode)"
        for mode, checkpoint in all_checkpoints
    ]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    selected_idx = checkpoint_choices.index(selected)
    mode, checkpoint_path = all_checkpoints[selected_idx]

    # Select analysis type
    analysis = questionary.select(
        "Select analysis:",
        choices=[
            "attention - Visualize attention patterns",
            "embeddings - Analyze token embeddings",
            "neurons - Analyze individual neurons",
            "all - Run all analyses",
        ],
        style=custom_style,
    ).ask()

    analysis_type = analysis.split(' - ')[0]

    # Get prompt for attention analysis
    prompt = None
    if analysis_type in ['attention', 'all']:
        prompt = questionary.text(
            "Enter text to analyze (for attention visualization):",
            default="The quick brown fox jumps over the lazy dog",
            style=custom_style,
        ).ask()

    return {
        'checkpoint': str(checkpoint_path),
        'analysis': analysis_type,
        'prompt': prompt,
    }


def download_menu() -> dict:
    """Data download menu."""
    console.print("\n[bold cyan]Download Training Data[/bold cyan]\n")

    mode = questionary.select(
        "Select dataset size:",
        choices=[
            "Quick (10M tokens, ~1 GB)",
            "Medium (50M tokens, ~5 GB)",
            "Full (100M tokens, ~10 GB)",
        ],
        style=custom_style,
    ).ask()

    # Parse mode
    if mode.startswith("Quick"):
        quick, medium = True, False
    elif mode.startswith("Medium"):
        quick, medium = False, True
    else:
        quick, medium = False, False

    return {
        'quick': quick,
        'medium': medium,
    }


def run_train(config: dict):
    """Execute training with given configuration."""
    console.print("\n[bold green]Starting training...[/bold green]\n")
    console.print("=" * 80)
    print()  # Spacing for train output

    train(
        debug=config['debug'],
        use_mps=config['use_mps'],
        quick=config['quick'],
        medium=config['medium'],
        resume=config['resume'],
        compile=config['compile'],
    )


def run_generate(config: dict):
    """Execute text generation with given configuration."""
    console.print("\n[bold green]Starting generation...[/bold green]\n")
    console.print("=" * 80)
    print()

    # Build argv for generate script
    original_argv = sys.argv
    try:
        new_argv = ["generate", config['checkpoint']]
        new_argv.extend(["--preset", config['preset']])
        new_argv.extend(["--max-length", str(config['max_length'])])

        if config['prompt']:
            new_argv.extend(["--prompt", config['prompt']])

        sys.argv = new_argv
        generate_main()
    finally:
        sys.argv = original_argv


def run_evaluate(config: dict):
    """Execute model evaluation with given configuration."""
    console.print("\n[bold green]Starting evaluation...[/bold green]\n")
    console.print("=" * 80)
    print()

    # Initialize device with proper setup
    try:
        device, device_name = init_device(config['device'], seed=42)
        autocast_ctx = get_autocast_context(device.type)
    except RuntimeError as e:
        console.print(f"[yellow]Warning: {e}[/yellow]")
        console.print("[yellow]Falling back to CPU[/yellow]")
        device, device_name = init_device("cpu", seed=42)
        autocast_ctx = get_autocast_context(device.type)
    print()

    if config['mode'] == 'single':
        evaluate_checkpoint(
            config['checkpoint'],
            seq_length=config['seq_length'],
            batch_size=config['batch_size'],
            device=device,
            autocast_ctx=autocast_ctx,
            tokens_per_epoch=config['tokens_per_epoch'],
            device_name=device_name,
        )
    else:  # compare
        compare_checkpoints(
            config['checkpoint_dir'],
            seq_length=config['seq_length'],
            device=device,
            autocast_ctx=autocast_ctx,
            tokens_per_epoch=config['tokens_per_epoch'],
            device_name=device_name,
        )


def run_interpret(config: dict):
    """Execute interpretability analysis with given configuration."""
    console.print("\n[bold green]Starting interpretability analysis...[/bold green]\n")
    console.print("=" * 80)
    print()

    # Build args object for interpret module
    class Args:
        pass

    args = Args()
    args.checkpoint = config['checkpoint']
    args.analysis = config['analysis']
    args.prompt = config.get('prompt')
    args.output_dir = "interpretability_output"
    args.device = "cpu"

    interpret.main(args)


def run_download(config: dict):
    """Execute data download with given configuration."""
    console.print("\n[bold green]Starting download...[/bold green]\n")
    console.print("=" * 80)
    print()

    download_shards(
        quick=config['quick'],
        medium=config['medium'],
    )


def interactive_main():
    """Main interactive loop."""
    show_welcome()

    # Scan for checkpoints
    scanner = CheckpointScanner()
    scanner.display_summary()
    console.print()

    # Main loop
    while True:
        choice = main_menu(scanner)

        if not choice or choice.startswith("❌"):
            console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
            break

        elif choice.startswith("🎓"):  # Train new model
            config = train_menu()
            if config:
                run_train(config)
                scanner.scan()  # Rescan for new checkpoints

        elif choice.startswith("▶️"):  # Continue training
            config = continue_training_menu(scanner)
            if config:
                run_train(config)
                scanner.scan()  # Rescan for updated checkpoints

        elif choice.startswith("✨"):  # Generate text
            config = generate_menu(scanner)
            if config:
                run_generate(config)

        elif choice.startswith("📊"):  # Evaluate
            config = evaluate_menu(scanner)
            if config:
                run_evaluate(config)

        elif choice.startswith("🔍"):  # Interpretability
            config = interpret_menu(scanner)
            if config:
                run_interpret(config)

        elif choice.startswith("⬇️"):  # Download
            config = download_menu()
            if config:
                run_download(config)

        # Ask if user wants to do something else
        console.print()
        continue_session = questionary.confirm(
            "Do something else?",
            default=True,
            style=custom_style,
        ).ask()

        if not continue_session:
            console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
            break

        # Rescan checkpoints and show summary for next action
        console.print("\n" + "=" * 80 + "\n")
        scanner.scan()
        scanner.display_summary()
        console.print()


if __name__ == "__main__":
    try:
        interactive_main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
