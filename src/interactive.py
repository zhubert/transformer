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
from commands.download_wikitext import download_wikitext
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
    """Scans for available model checkpoints in the checkpoints directory."""

    def __init__(self):
        self.checkpoint_dir = Path('checkpoints')
        self.checkpoints: List[Path] = []
        self.scan()
        self._check_old_directories()

    def scan(self):
        """Scan checkpoint directory for model files."""
        if self.checkpoint_dir.exists():
            # New format: model_epoch_5_fineweb.pt
            self.checkpoints = sorted(
                self.checkpoint_dir.glob('model_epoch_*_*.pt'),
                key=lambda x: int(x.stem.split('_')[2])
            )

    def _check_old_directories(self):
        """Check for old checkpoint directories and warn user to migrate."""
        old_dirs = [
            Path('checkpoints_quick'),
            Path('checkpoints_medium'),
        ]

        existing_old_dirs = [d for d in old_dirs if d.exists() and list(d.glob('model_epoch_*_*.pt'))]

        if existing_old_dirs:
            console.print()
            console.print(Panel(
                "[yellow bold]⚠️  Old Checkpoint Directories Detected[/yellow bold]\n\n"
                "The following old checkpoint directories were found:\n" +
                "\n".join(f"  • {d}/" for d in existing_old_dirs) +
                "\n\n[white]Please consolidate to 'checkpoints/' directory:[/white]\n" +
                "\n".join(f"  mv {d}/* checkpoints/" for d in existing_old_dirs) +
                "\n\nAfter moving, you can remove the old directories:\n" +
                "\n".join(f"  rmdir {d}" for d in existing_old_dirs),
                border_style="yellow",
                box=box.ROUNDED
            ))
            console.print()

    def has_checkpoints(self) -> bool:
        """Check if any checkpoints exist."""
        return len(self.checkpoints) > 0

    def get_all_checkpoints(self) -> List[Path]:
        """Get all checkpoints."""
        return self.checkpoints

    def get_latest(self) -> Optional[Path]:
        """Get the most recent checkpoint."""
        if not self.checkpoints:
            return None
        # Sort by modification time
        return max(self.checkpoints, key=lambda x: x.stat().st_mtime)

    def display_summary(self):
        """Display a summary table of available checkpoints."""
        if not self.has_checkpoints():
            console.print("[yellow]No checkpoints found. Train a model first![/yellow]")
            return

        table = Table(title="Available Checkpoints", box=box.ROUNDED)
        table.add_column("Checkpoint", style="green")
        table.add_column("Size", justify="right", style="magenta")

        for checkpoint in self.checkpoints:
            size_mb = checkpoint.stat().st_size / (1024 * 1024)
            table.add_row(checkpoint.name, f"{size_mb:.1f} MB")

        console.print(table)

        latest = self.get_latest()
        if latest:
            console.print(f"\n[bold green]Latest:[/bold green] {latest.name}")


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

    # Select dataset
    dataset_choice = questionary.select(
        "Select dataset:",
        choices=[
            "fineweb - FineWeb 10B tokens (realistic web text, harder) [DEFAULT]",
            "wikitext - WikiText-103 100M tokens (clean Wikipedia, easier)",
        ],
        default="fineweb - FineWeb 10B tokens (realistic web text, harder) [DEFAULT]",
        style=custom_style,
    ).ask()

    dataset = dataset_choice.split(' - ')[0]

    # Configuration approach: preset or custom
    config_approach = questionary.select(
        "Configuration approach:",
        choices=[
            "Use recommended preset (recommended for beginners)",
            "Customize parameters (advanced users)",
        ],
        default="Use recommended preset (recommended for beginners)",
        style=custom_style,
    ).ask()

    # Training parameters
    if "preset" in config_approach:
        # Offer presets
        preset_choice = questionary.select(
            "Select training preset:",
            choices=[
                "Beginner - Fast iteration (10M tokens/epoch, 4 layers, d_model=128, 10 epochs)",
                "Intermediate - Balanced quality (50M tokens/epoch, 4 layers, d_model=256, 15 epochs)",
                "Advanced - Full quality (100M tokens/epoch, 6 layers, d_model=256, 20 epochs)",
            ],
            default="Intermediate - Balanced quality (50M tokens/epoch, 4 layers, d_model=256, 15 epochs)",
            style=custom_style,
        ).ask()

        # Parse preset
        if "Beginner" in preset_choice:
            tokens_per_epoch = 10_000_000
            num_layers = 4
            d_model = 128
            num_epochs = 10
        elif "Intermediate" in preset_choice:
            tokens_per_epoch = 50_000_000
            num_layers = 4
            d_model = 256
            num_epochs = 15
        else:  # Advanced
            tokens_per_epoch = 100_000_000
            num_layers = 6
            d_model = 256
            num_epochs = 20
    else:
        # Custom configuration - ask for each parameter
        console.print("\n[dim]Customize training parameters:[/dim]\n")

        # Tokens per epoch
        tokens_choice = questionary.select(
            "Tokens per epoch:",
            choices=[
                "10M tokens (fast iteration)",
                "50M tokens (balanced)",
                "100M tokens (full quality)",
                "Custom value",
            ],
            default="50M tokens (balanced)",
            style=custom_style,
        ).ask()

        if "10M" in tokens_choice:
            tokens_per_epoch = 10_000_000
        elif "50M" in tokens_choice:
            tokens_per_epoch = 50_000_000
        elif "100M" in tokens_choice:
            tokens_per_epoch = 100_000_000
        else:
            tokens_per_epoch = int(questionary.text(
                "Enter tokens per epoch (e.g., 25000000 for 25M):",
                default="50000000",
                style=custom_style,
            ).ask())

        # Number of layers
        layers_choice = questionary.select(
            "Number of transformer layers:",
            choices=[
                "4 layers (faster training)",
                "6 layers (better quality)",
                "8 layers (high quality)",
                "Custom value",
            ],
            default="4 layers (faster training)",
            style=custom_style,
        ).ask()

        if "4 layers" in layers_choice:
            num_layers = 4
        elif "6 layers" in layers_choice:
            num_layers = 6
        elif "8 layers" in layers_choice:
            num_layers = 8
        else:
            num_layers = int(questionary.text(
                "Enter number of layers (2-12):",
                default="4",
                style=custom_style,
            ).ask())

        # Model dimension
        d_model_choice = questionary.select(
            "Model dimension (d_model):",
            choices=[
                "128 (smallest, fastest)",
                "256 (balanced)",
                "512 (larger, slower)",
                "Custom value",
            ],
            default="256 (balanced)",
            style=custom_style,
        ).ask()

        if "128" in d_model_choice:
            d_model = 128
        elif "256" in d_model_choice:
            d_model = 256
        elif "512" in d_model_choice:
            d_model = 512
        else:
            d_model = int(questionary.text(
                "Enter d_model (64-1024, must be divisible by num_heads):",
                default="256",
                style=custom_style,
            ).ask())

        # Number of epochs
        epochs_choice = questionary.select(
            "Number of training epochs:",
            choices=[
                "10 epochs (quick)",
                "15 epochs (balanced)",
                "20 epochs (thorough)",
                "Custom value",
            ],
            default="15 epochs (balanced)",
            style=custom_style,
        ).ask()

        if "10 epochs" in epochs_choice:
            num_epochs = 10
        elif "15 epochs" in epochs_choice:
            num_epochs = 15
        elif "20 epochs" in epochs_choice:
            num_epochs = 20
        else:
            num_epochs = int(questionary.text(
                "Enter number of epochs (5-50):",
                default="15",
                style=custom_style,
            ).ask())

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
    position_encoding_type = 'alibi'  # Default

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

        # Position encoding type selection
        position_encoding_choice = questionary.select(
            "Position encoding type:",
            choices=[
                "alibi - ALiBi (Attention with Linear Biases) - RECOMMENDED",
                "rope - RoPE (Rotary Position Embeddings) - Also excellent",
                "learned - Learned embeddings (GPT-2/GPT-3 style)",
            ],
            default="alibi - ALiBi (Attention with Linear Biases) - RECOMMENDED",
            style=custom_style,
        ).ask()

        position_encoding_type = position_encoding_choice.split(' - ')[0]

    return {
        'dataset': dataset,
        'tokens_per_epoch': tokens_per_epoch,
        'num_layers': num_layers,
        'd_model': d_model,
        'num_epochs': num_epochs,
        'd_ff': None,  # Will be auto-calculated as d_model * 4
        'resume': resume,
        'debug': debug,
        'use_mps': use_mps,
        'compile': compile,
        'position_encoding_type': position_encoding_type,
    }


def continue_training_menu(scanner: CheckpointScanner) -> dict:
    """Continue training menu - loads all parameters from checkpoint."""
    console.print("\n[bold cyan]Continue Training[/bold cyan]\n")

    # Get latest checkpoint
    latest = scanner.get_latest()
    if not latest:
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    console.print(f"Latest checkpoint: [green]{latest.name}[/green]")
    console.print("[dim]All training parameters will be loaded from the checkpoint.[/dim]\n")

    confirm = questionary.confirm(
        f"Continue training from this checkpoint?",
        default=True,
        style=custom_style,
    ).ask()

    if not confirm:
        return None

    # All parameters will be auto-inferred from checkpoint in train()
    # We just return resume=True to signal that we want to continue
    return {
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

    # Create choices from checkpoint names
    checkpoint_choices = [checkpoint.name for checkpoint in all_checkpoints]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    # Find the selected checkpoint path
    selected_idx = checkpoint_choices.index(selected)
    checkpoint_path = all_checkpoints[selected_idx]

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
        checkpoint_choices = [checkpoint.name for checkpoint in all_checkpoints]

        selected = questionary.select(
            "Select checkpoint:",
            choices=checkpoint_choices,
            style=custom_style,
        ).ask()

        selected_idx = checkpoint_choices.index(selected)
        checkpoint_path = all_checkpoints[selected_idx]

        return {
            'mode': 'single',
            'checkpoint': str(checkpoint_path),
            'seq_length': 128,
            'batch_size': 8,
            'device': None,  # Auto-detect device
            'tokens_per_epoch': 10_000_000,
        }
    else:
        # Compare all checkpoints
        return {
            'mode': 'compare',
            'checkpoint_dir': 'checkpoints',
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
    checkpoint_choices = [checkpoint.name for checkpoint in all_checkpoints]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    selected_idx = checkpoint_choices.index(selected)
    checkpoint_path = all_checkpoints[selected_idx]

    # Select analysis type
    analysis = questionary.select(
        "Select analysis:",
        choices=[
            "attention - Visualize attention patterns",
            "logit-lens - See how predictions evolve through layers",
            "induction-heads - Detect pattern-matching circuits",
            "patch - Causal intervention experiments",
            "all - Run all analyses",
        ],
        style=custom_style,
    ).ask()

    analysis_type = analysis.split(' - ')[0]

    # Get prompt for analyses that need text input
    prompt = None
    if analysis_type in ['attention', 'logit-lens', 'all']:
        prompt_text = {
            'attention': "Enter text to analyze (for attention visualization):",
            'logit-lens': "Enter text to analyze (for logit lens):",
            'all': "Enter text to analyze (for attention and logit lens):",
        }
        prompt = questionary.text(
            prompt_text.get(analysis_type, "Enter text to analyze:"),
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

    # Select dataset type
    dataset_choice = questionary.select(
        "Select dataset to download:",
        choices=[
            "fineweb - FineWeb 10B tokens (realistic web text) [DEFAULT]",
            "wikitext - WikiText-103 100M tokens (clean Wikipedia)",
        ],
        default="fineweb - FineWeb 10B tokens (realistic web text) [DEFAULT]",
        style=custom_style,
    ).ask()

    dataset = dataset_choice.split(' - ')[0]

    # For FineWeb, ask about size
    if dataset == 'fineweb':
        mode = questionary.select(
            "Select dataset size:",
            choices=[
                "10M tokens (~1 GB)",
                "50M tokens (~5 GB)",
                "100M tokens (~10 GB)",
            ],
            style=custom_style,
        ).ask()

        # Parse tokens_per_epoch from selection
        if mode.startswith("10M"):
            tokens = 10_000_000
        elif mode.startswith("50M"):
            tokens = 50_000_000
        else:  # 100M
            tokens = 100_000_000

        return {
            'dataset': 'fineweb',
            'tokens': tokens,
        }
    else:
        # WikiText - no size selection needed (always downloads full dataset)
        return {
            'dataset': 'wikitext',
        }


def run_train(config: dict):
    """Execute training with given configuration."""
    console.print("\n[bold green]Starting training...[/bold green]\n")
    console.print("=" * 80)
    print()  # Spacing for train output

    train(
        tokens_per_epoch=config.get('tokens_per_epoch'),
        num_layers=config.get('num_layers'),
        d_model=config.get('d_model'),
        num_epochs=config.get('num_epochs'),
        d_ff=config.get('d_ff'),
        debug=config['debug'],
        use_mps=config['use_mps'],
        resume=config['resume'],
        compile=config['compile'],
        position_encoding_type=config.get('position_encoding_type', 'alibi'),
        dataset=config.get('dataset', 'fineweb'),
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

    # Set attributes required by interpret commands
    # These match the defaults from argparse in interpret.py
    args.text = args.prompt  # Commands expect 'text' not 'prompt'
    args.demo = False
    args.interactive = False
    args.top_k = 5
    args.temperature = 1.0
    args.layer = None
    args.head = None

    # Attributes for induction-heads command
    args.num_sequences = 100
    args.seq_length = 40

    # Attributes for patch command
    args.clean = None
    args.corrupted = None
    args.target = None

    interpret.main(args)


def run_download(config: dict):
    """Execute data download with given configuration."""
    console.print("\n[bold green]Starting download...[/bold green]\n")
    console.print("=" * 80)
    print()

    dataset = config.get('dataset', 'fineweb')

    if dataset == 'fineweb':
        tokens_per_epoch = config.get('tokens', 50_000_000)  # Default to 50M
        download_shards(
            tokens_per_epoch=tokens_per_epoch,
        )
    elif dataset == 'wikitext':
        download_wikitext()


def interactive_main():
    """Main interactive loop."""
    show_welcome()

    # Scan for checkpoints
    scanner = CheckpointScanner()

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
