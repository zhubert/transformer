#!/usr/bin/env python3
"""
Interactive CLI for Transformer operations.

This module provides a user-friendly interactive interface for the complete
transformer training pipeline: pre-training → mid-training → fine-tuning.

The interface guides users through the three-stage approach used by modern LLMs
like GPT-4, Claude, and Llama 3, making it easy to build production-ready models
without memorizing command-line flags.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import json

import questionary
from questionary import Style
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.text import Text

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from commands.train import train
from commands.download_shards import download_shards
from commands.download_wikitext import download_wikitext
from commands.generate import main as generate_main
from commands.evaluate_perplexity import evaluate_checkpoint, compare_checkpoints
from commands.sampling_comparison import demonstrate_sampling_strategies, demonstrate_with_model
from commands import interpret
from commands.midtrain_stub import demonstrate_midtraining_concepts
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


class CheckpointMetadata:
    """Manages checkpoint metadata for tracking training stages and lineage."""

    def __init__(self, checkpoint_path: Path):
        self.checkpoint_path = checkpoint_path
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict:
        """Load metadata from checkpoint file or infer from filename."""
        # For now, infer from filename until we implement metadata saving
        # Format: model_epoch_N_dataset_encoding.pt (pretrain)
        # Future: model_epoch_N_domain_from_base.pt (midtrain)
        # Future: model_epoch_N_task_sft.pt (finetune)

        stem = self.checkpoint_path.stem
        parts = stem.split('_')

        # Default metadata
        metadata = {
            'training_stage': 'pretrain',  # Default to pretrain
            'epoch': 0,
            'dataset': 'unknown',
            'base_model': None,
            'domain': None,
            'task': None,
        }

        # Parse filename to extract metadata
        if 'epoch' in stem:
            try:
                epoch_idx = parts.index('epoch')
                metadata['epoch'] = int(parts[epoch_idx + 1])
            except (ValueError, IndexError):
                pass

        # Detect stage from filename patterns
        if '_sft' in stem or '_finetune' in stem:
            metadata['training_stage'] = 'finetune'
            # Extract task if present
            if len(parts) > 3:
                metadata['task'] = parts[3]
        elif '_from_' in stem:
            metadata['training_stage'] = 'midtrain'
            # Extract domain if present
            if len(parts) > 3:
                metadata['domain'] = parts[3]
        else:
            # Pretrain - extract dataset
            if len(parts) > 3:
                metadata['dataset'] = parts[3]

        return metadata

    def get_stage(self) -> str:
        """Get the training stage (pretrain/midtrain/finetune)."""
        return self.metadata['training_stage']

    def get_display_name(self) -> str:
        """Get a human-readable display name for the checkpoint."""
        stage = self.metadata['training_stage']
        epoch = self.metadata['epoch']

        if stage == 'pretrain':
            dataset = self.metadata.get('dataset', 'unknown')
            return f"Pre-trained (Epoch {epoch}, {dataset})"
        elif stage == 'midtrain':
            domain = self.metadata.get('domain', 'unknown')
            return f"Mid-trained (Epoch {epoch}, {domain} domain)"
        elif stage == 'finetune':
            task = self.metadata.get('task', 'unknown')
            return f"Fine-tuned (Epoch {epoch}, {task})"
        else:
            return f"Epoch {epoch}"


class CheckpointScanner:
    """Scans for available model checkpoints and organizes by training stage."""

    def __init__(self):
        self.checkpoint_dir = Path('checkpoints')
        self.pretrain_dir = self.checkpoint_dir / 'pretrain'
        self.midtrain_dir = self.checkpoint_dir / 'midtrain'
        self.finetune_dir = self.checkpoint_dir / 'finetune'

        # Checkpoints organized by stage
        self.pretrain_checkpoints: List[Tuple[Path, CheckpointMetadata]] = []
        self.midtrain_checkpoints: List[Tuple[Path, CheckpointMetadata]] = []
        self.finetune_checkpoints: List[Tuple[Path, CheckpointMetadata]] = []

        self.scan()

    def scan(self):
        """Scan checkpoint directories for model files."""
        # Scan old-style checkpoints in root directory (pretrain)
        if self.checkpoint_dir.exists():
            root_checkpoints = sorted(
                self.checkpoint_dir.glob('model_epoch_*_*.pt'),
                key=lambda x: int(x.stem.split('_')[2]) if len(x.stem.split('_')) > 2 else 0
            )
            for ckpt in root_checkpoints:
                metadata = CheckpointMetadata(ckpt)
                self.pretrain_checkpoints.append((ckpt, metadata))

        # Scan new organized structure
        if self.pretrain_dir.exists():
            for ckpt in sorted(self.pretrain_dir.glob('model_epoch_*.pt')):
                metadata = CheckpointMetadata(ckpt)
                self.pretrain_checkpoints.append((ckpt, metadata))

        if self.midtrain_dir.exists():
            for ckpt in sorted(self.midtrain_dir.rglob('model_epoch_*.pt')):
                metadata = CheckpointMetadata(ckpt)
                self.midtrain_checkpoints.append((ckpt, metadata))

        if self.finetune_dir.exists():
            for ckpt in sorted(self.finetune_dir.rglob('model_epoch_*.pt')):
                metadata = CheckpointMetadata(ckpt)
                self.finetune_checkpoints.append((ckpt, metadata))

    def has_pretrain_checkpoints(self) -> bool:
        """Check if any pre-training checkpoints exist."""
        return len(self.pretrain_checkpoints) > 0

    def has_midtrain_checkpoints(self) -> bool:
        """Check if any mid-training checkpoints exist."""
        return len(self.midtrain_checkpoints) > 0

    def has_finetune_checkpoints(self) -> bool:
        """Check if any fine-tuning checkpoints exist."""
        return len(self.finetune_checkpoints) > 0

    def has_any_checkpoints(self) -> bool:
        """Check if any checkpoints exist at any stage."""
        return (self.has_pretrain_checkpoints() or
                self.has_midtrain_checkpoints() or
                self.has_finetune_checkpoints())

    def get_pretrain_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """Get all pre-training checkpoints."""
        return self.pretrain_checkpoints

    def get_midtrain_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """Get all mid-training checkpoints."""
        return self.midtrain_checkpoints

    def get_finetune_checkpoints(self) -> List[Tuple[Path, CheckpointMetadata]]:
        """Get all fine-tuning checkpoints."""
        return self.finetune_checkpoints

    def get_latest_pretrain(self) -> Optional[Path]:
        """Get the most recent pre-training checkpoint."""
        if not self.pretrain_checkpoints:
            return None
        # Sort by modification time
        return max(self.pretrain_checkpoints, key=lambda x: x[0].stat().st_mtime)[0]

    def get_latest_midtrain(self) -> Optional[Path]:
        """Get the most recent mid-training checkpoint."""
        if not self.midtrain_checkpoints:
            return None
        return max(self.midtrain_checkpoints, key=lambda x: x[0].stat().st_mtime)[0]

    def get_latest_finetune(self) -> Optional[Path]:
        """Get the most recent fine-tuning checkpoint."""
        if not self.finetune_checkpoints:
            return None
        return max(self.finetune_checkpoints, key=lambda x: x[0].stat().st_mtime)[0]

    def get_current_stage(self) -> str:
        """Determine the current training stage based on available checkpoints."""
        if self.has_finetune_checkpoints():
            return 'finetune'
        elif self.has_midtrain_checkpoints():
            return 'midtrain'
        elif self.has_pretrain_checkpoints():
            return 'pretrain'
        else:
            return 'none'

    def display_summary(self):
        """Display a summary table of available checkpoints by stage."""
        if not self.has_any_checkpoints():
            console.print("[yellow]No checkpoints found. Start with pre-training![/yellow]")
            return

        # Create summary table
        table = Table(title="Training Pipeline Progress", box=box.ROUNDED)
        table.add_column("Stage", style="cyan bold")
        table.add_column("Checkpoints", justify="center")
        table.add_column("Latest", style="green")
        table.add_column("Status", style="bold")

        # Pre-training row
        pretrain_count = len(self.pretrain_checkpoints)
        pretrain_latest = self.get_latest_pretrain()
        pretrain_status = "✓ Complete" if pretrain_count > 0 else "○ Not started"
        table.add_row(
            "1️⃣  Pre-Training",
            str(pretrain_count),
            pretrain_latest.name if pretrain_latest else "-",
            pretrain_status
        )

        # Mid-training row
        midtrain_count = len(self.midtrain_checkpoints)
        midtrain_latest = self.get_latest_midtrain()
        if midtrain_count > 0:
            midtrain_status = "✓ Complete"
        elif pretrain_count > 0:
            midtrain_status = "○ Ready to start"
        else:
            midtrain_status = "⊗ Needs pre-training"
        table.add_row(
            "2️⃣  Mid-Training",
            str(midtrain_count),
            midtrain_latest.name if midtrain_latest else "-",
            midtrain_status
        )

        # Fine-tuning row
        finetune_count = len(self.finetune_checkpoints)
        finetune_latest = self.get_latest_finetune()
        if finetune_count > 0:
            finetune_status = "✓ Complete"
        elif pretrain_count > 0 or midtrain_count > 0:
            finetune_status = "○ Ready to start"
        else:
            finetune_status = "⊗ Needs base model"
        table.add_row(
            "3️⃣  Fine-Tuning",
            str(finetune_count),
            finetune_latest.name if finetune_latest else "-",
            finetune_status
        )

        console.print(table)
        console.print()


def show_welcome():
    """Display welcome message with pipeline overview."""
    console.clear()

    welcome_text = """[bold cyan]Transformer Training Pipeline[/bold cyan]

Build production-ready language models through [bold]three sequential stages[/bold]:

  [bold green]1️⃣  Pre-Training[/bold green]   → General language understanding
  [bold blue]2️⃣  Mid-Training[/bold blue]   → Domain expertise & specialization
  [bold magenta]3️⃣  Fine-Tuning[/bold magenta]    → Task-specific behavior

This is the same approach used by [bold]GPT-4, Claude, Llama 3[/bold], and other
state-of-the-art language models.

[dim]Use arrow keys to navigate, Enter to select, Ctrl+C to exit anytime.[/dim]"""

    console.print(Panel(welcome_text, border_style="cyan", box=box.ROUNDED))
    console.print()


def show_pipeline_education():
    """Display educational screen about the three-stage pipeline."""
    console.clear()

    education_text = """[bold cyan]THE THREE-STAGE TRAINING PIPELINE[/bold cyan]

Modern LLMs (GPT-4, Claude, Llama) use this approach:

  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐
  │ PRE-TRAINING │ ──▶ │ MID-TRAINING │ ──▶ │ FINE-TUNING  │
  └──────────────┘     └──────────────┘     └──────────────┘
   Base Model          Domain Expert        Task Specialist

[bold green]📚 STAGE 1: PRE-TRAINING[/bold green]
   • Goal: Learn general language patterns
   • Data: Billions of tokens (web text, books)
   • Time: Days-weeks on GPU
   • Result: "Can predict text, knows grammar & facts"

[bold blue]🔬 STAGE 2: MID-TRAINING (Continued Pre-Training)[/bold blue]
   • Goal: Become expert in specific domain
   • Data: Millions of curated domain tokens
   • Time: Hours-days on GPU
   • Result: "Code/math/science specialist"
   • Key challenge: Don't forget general skills!

[bold magenta]🎯 STAGE 3: FINE-TUNING (Supervised Fine-Tuning)[/bold magenta]
   • Goal: Learn specific behavior patterns
   • Data: Thousands of instruction-response pairs
   • Time: Minutes-hours on GPU
   • Result: "Follows instructions, helpful assistant"
   • Tip: Use LoRA for efficient multi-task variants

[bold yellow]💡 KEY INSIGHTS:[/bold yellow]
   • Same architecture used across all stages
   • Learning rate [bold]decreases[/bold] at each stage (3e-4 → 1e-5 → 1e-6)
   • Data quality matters more than quantity (mid/fine)
   • Most capability comes from pre + mid training
   • Fine-tuning is lightweight (good for iteration)

[bold cyan]📖 EDUCATIONAL RESOURCES:[/bold cyan]
   • Pre-training: See CLAUDE.md for architecture details
   • Mid-training: Prevents catastrophic forgetting via dual evaluation
   • Fine-tuning: Loss computed on response tokens only!"""

    console.print(Panel(education_text, border_style="cyan", box=box.DOUBLE))
    console.print("\n[dim]Press Enter to return to main menu...[/dim]")
    input()


def main_menu(scanner: CheckpointScanner) -> str:
    """Display main menu organized by training pipeline stages."""

    # Display pipeline progress summary
    scanner.display_summary()

    # Build menu choices based on current state
    choices = []

    # STAGE 1: PRE-TRAINING
    choices.append("─── STAGE 1: PRE-TRAINING ───")
    choices.append("🎓 Start pre-training (build base model)")
    if scanner.has_pretrain_checkpoints():
        choices.append("▶️  Continue pre-training")

    # STAGE 2: MID-TRAINING
    choices.append("─── STAGE 2: MID-TRAINING ───")
    if scanner.has_pretrain_checkpoints():
        choices.append("🔬 Start mid-training (domain adaptation)")
        if scanner.has_midtrain_checkpoints():
            choices.append("▶️  Continue mid-training")
    else:
        choices.append("[Locked] 🔬 Mid-training (needs base model)")

    # STAGE 3: FINE-TUNING
    choices.append("─── STAGE 3: FINE-TUNING ───")
    if scanner.has_pretrain_checkpoints() or scanner.has_midtrain_checkpoints():
        choices.append("🎯 Start fine-tuning (instruction following)")
        if scanner.has_finetune_checkpoints():
            choices.append("▶️  Continue fine-tuning")
    else:
        choices.append("[Locked] 🎯 Fine-tuning (needs base model)")

    # MODEL OPERATIONS (always available if checkpoints exist)
    if scanner.has_any_checkpoints():
        choices.append("─── MODEL OPERATIONS ───")
        choices.append("✨ Generate text (test any model)")
        choices.append("📊 Evaluate models (perplexity & benchmarks)")
        choices.append("🔍 Analyze internals (interpretability)")

    # UTILITIES
    choices.append("─── UTILITIES ───")
    choices.append("⬇️  Download training data")
    choices.append("❓ Learn about the pipeline")
    choices.append("❌ Exit")

    return questionary.select(
        "What would you like to do?",
        choices=choices,
        style=custom_style,
    ).ask()


def pretrain_menu() -> dict:
    """Pre-training configuration menu (enhanced with educational context)."""
    console.print("\n[bold green]═══ STAGE 1: PRE-TRAINING ═══[/bold green]\n")

    info_text = """[dim]Purpose:[/dim] Build a base model with general language ability
[dim]Data:[/dim]    Billions of tokens from diverse web text
[dim]Loss:[/dim]    Next-token prediction on all tokens
[dim]Result:[/dim]  Foundation model ready for specialization"""

    console.print(Panel(info_text, border_style="green", box=box.ROUNDED))
    console.print()

    # Select dataset
    dataset_choice = questionary.select(
        "Select dataset:",
        choices=[
            "fineweb - FineWeb 10B tokens (realistic web text) [RECOMMENDED]",
            "wikitext - WikiText-103 100M tokens (clean Wikipedia, benchmark)",
        ],
        default="fineweb - FineWeb 10B tokens (realistic web text) [RECOMMENDED]",
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
        'stage': 'pretrain',
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


def midtrain_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Mid-training configuration menu."""
    console.print("\n[bold blue]═══ STAGE 2: MID-TRAINING ═══[/bold blue]\n")

    info_text = """[dim]Purpose:[/dim] Specialize your base model for specific domains
[dim]Data:[/dim]    Millions-billions of curated domain-specific tokens
[dim]Loss:[/dim]    Next-token prediction (lower learning rate than pre-training)
[dim]Result:[/dim]  Domain-adapted model (code/math/science expert)

[yellow]⚠️  IMPORTANT:[/yellow] Monitor catastrophic forgetting!
    We'll track both domain AND general performance"""

    console.print(Panel(info_text, border_style="blue", box=box.ROUNDED))
    console.print()

    # Select base model from pre-training
    pretrain_checkpoints = scanner.get_pretrain_checkpoints()
    if not pretrain_checkpoints:
        console.print("[yellow]No pre-trained models found! Complete pre-training first.[/yellow]")
        return None

    checkpoint_choices = [f"{ckpt.name} - {meta.get_display_name()}"
                         for ckpt, meta in pretrain_checkpoints]

    selected = questionary.select(
        "Select base model (from pre-training):",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    selected_idx = checkpoint_choices.index(selected)
    base_checkpoint = pretrain_checkpoints[selected_idx][0]

    console.print(f"\n[green]✓ Base model selected:[/green] {base_checkpoint.name}")
    console.print("[dim]Architecture, tokenizer, and weights will be loaded automatically[/dim]\n")

    # Select domain
    domain_choice = questionary.select(
        "Select domain to specialize in:",
        choices=[
            "code - Python, JavaScript, documentation, Stack Overflow [COMING SOON]",
            "math - Proofs, textbooks, problem-solution pairs [COMING SOON]",
            "science - Papers, textbooks, encyclopedias [COMING SOON]",
            "custom - Provide your own dataset [COMING SOON]",
        ],
        style=custom_style,
    ).ask()

    domain = domain_choice.split(' - ')[0]

    console.print(f"\n[green]✓ Configuration complete![/green]")
    console.print(f"[dim]Will demonstrate {domain} domain specialization from {base_checkpoint.name}[/dim]\n")

    # Return configuration for mid-training demonstration
    return {
        'stage': 'midtrain',
        'base_checkpoint': str(base_checkpoint),
        'domain': domain,
        'debug': False,
        'use_mps': False,
        'compile': True,
    }


def finetune_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Fine-tuning configuration menu."""
    console.print("\n[bold magenta]═══ STAGE 3: FINE-TUNING ═══[/bold magenta]\n")

    info_text = """[dim]Purpose:[/dim] Teach specific behaviors and response formats
[dim]Data:[/dim]    Thousands of instruction-response pairs
[dim]Loss:[/dim]    Next-token prediction (response tokens only!)
[dim]Result:[/dim]  Task-specific model (instruction-following, chat)

[yellow]💡 TIP:[/yellow] Use LoRA for parameter-efficient fine-tuning
    (trains 94% fewer parameters, same quality!)"""

    console.print(Panel(info_text, border_style="magenta", box=box.ROUNDED))
    console.print()

    # Select base model (can be pretrain or midtrain)
    all_base_checkpoints = []

    pretrain_ckpts = scanner.get_pretrain_checkpoints()
    for ckpt, meta in pretrain_ckpts:
        all_base_checkpoints.append((ckpt, meta, "pretrain"))

    midtrain_ckpts = scanner.get_midtrain_checkpoints()
    for ckpt, meta in midtrain_ckpts:
        all_base_checkpoints.append((ckpt, meta, "midtrain"))

    if not all_base_checkpoints:
        console.print("[yellow]No base models found! Complete pre-training first.[/yellow]")
        return None

    checkpoint_choices = [f"[{stage}] {ckpt.name} - {meta.get_display_name()}"
                         for ckpt, meta, stage in all_base_checkpoints]

    selected = questionary.select(
        "Select base model:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    selected_idx = checkpoint_choices.index(selected)
    base_checkpoint = all_base_checkpoints[selected_idx][0]

    console.print(f"\n[green]✓ Base model selected:[/green] {base_checkpoint.name}")
    console.print("[dim]Architecture, tokenizer, and weights will be loaded automatically[/dim]\n")

    # Select fine-tuning task
    task_choice = questionary.select(
        "Select fine-tuning task:",
        choices=[
            "instruction - Instruction following (Alpaca-style Q&A) [COMING SOON]",
            "chat - Chat / dialogue (conversational assistant) [COMING SOON]",
            "code - Code completion (GitHub Copilot-style) [COMING SOON]",
            "summarization - Article → summary [COMING SOON]",
            "custom - Provide your own instruction dataset [COMING SOON]",
        ],
        style=custom_style,
    ).ask()

    task = task_choice.split(' - ')[0]

    # For now, show coming soon message
    console.print(f"\n[yellow]Fine-tuning infrastructure is coming soon![/yellow]")
    console.print(f"[dim]Selected: {task} fine-tuning from {base_checkpoint.name}[/dim]\n")

    return None  # Will implement full functionality later


def continue_training_menu(scanner: CheckpointScanner, stage: str) -> dict:
    """Continue training menu for any stage."""
    stage_names = {
        'pretrain': 'Pre-Training',
        'midtrain': 'Mid-Training',
        'finetune': 'Fine-Tuning',
    }

    console.print(f"\n[bold cyan]Continue {stage_names.get(stage, 'Training')}[/bold cyan]\n")

    # Get latest checkpoint for this stage
    if stage == 'pretrain':
        latest = scanner.get_latest_pretrain()
    elif stage == 'midtrain':
        latest = scanner.get_latest_midtrain()
    elif stage == 'finetune':
        latest = scanner.get_latest_finetune()
    else:
        latest = None

    if not latest:
        console.print(f"[yellow]No {stage} checkpoints found![/yellow]")
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

    return {
        'stage': stage,
        'resume': True,
        'debug': False,
        'use_mps': False,
        'compile': True,
    }


def generate_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Text generation menu (works with any checkpoint)."""
    console.print("\n[bold cyan]✨ Text Generation[/bold cyan]\n")

    # Collect all checkpoints from all stages
    all_checkpoints = []

    for ckpt, meta in scanner.get_pretrain_checkpoints():
        all_checkpoints.append((ckpt, meta, "pretrain"))
    for ckpt, meta in scanner.get_midtrain_checkpoints():
        all_checkpoints.append((ckpt, meta, "midtrain"))
    for ckpt, meta in scanner.get_finetune_checkpoints():
        all_checkpoints.append((ckpt, meta, "finetune"))

    if not all_checkpoints:
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Create choices with stage labels
    checkpoint_choices = [f"[{stage}] {ckpt.name}" for ckpt, meta, stage in all_checkpoints]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    # Find the selected checkpoint path
    selected_idx = checkpoint_choices.index(selected)
    checkpoint_path = all_checkpoints[selected_idx][0]

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
    """Model evaluation menu (enhanced for multi-stage evaluation)."""
    console.print("\n[bold cyan]📊 Model Evaluation[/bold cyan]\n")

    if not scanner.has_any_checkpoints():
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Evaluate single or compare
    action = questionary.select(
        "Evaluation type:",
        choices=[
            "Evaluate single checkpoint (perplexity)",
            "Compare all checkpoints",
            "Compare by stage (pretrain vs midtrain vs finetune) [COMING SOON]",
        ],
        style=custom_style,
    ).ask()

    if action.startswith("Evaluate single"):
        # Collect all checkpoints
        all_checkpoints = []
        for ckpt, meta in scanner.get_pretrain_checkpoints():
            all_checkpoints.append((ckpt, meta, "pretrain"))
        for ckpt, meta in scanner.get_midtrain_checkpoints():
            all_checkpoints.append((ckpt, meta, "midtrain"))
        for ckpt, meta in scanner.get_finetune_checkpoints():
            all_checkpoints.append((ckpt, meta, "finetune"))

        checkpoint_choices = [f"[{stage}] {ckpt.name}" for ckpt, meta, stage in all_checkpoints]

        selected = questionary.select(
            "Select checkpoint:",
            choices=checkpoint_choices,
            style=custom_style,
        ).ask()

        selected_idx = checkpoint_choices.index(selected)
        checkpoint_path = all_checkpoints[selected_idx][0]

        return {
            'mode': 'single',
            'checkpoint': str(checkpoint_path),
            'seq_length': 128,
            'batch_size': 8,
            'device': None,  # Auto-detect
            'tokens_per_epoch': 10_000_000,
        }
    elif "Compare by stage" in action:
        console.print("\n[yellow]Stage comparison coming soon![/yellow]\n")
        return None
    else:
        # Compare all checkpoints
        return {
            'mode': 'compare',
            'checkpoint_dir': 'checkpoints',
            'seq_length': 128,
            'device': None,  # Auto-detect
            'tokens_per_epoch': 10_000_000,
        }


def interpret_menu(scanner: CheckpointScanner) -> Optional[dict]:
    """Interpretability analysis menu."""
    console.print("\n[bold cyan]🔍 Interpretability Analysis[/bold cyan]\n")

    # Collect all checkpoints
    all_checkpoints = []
    for ckpt, meta in scanner.get_pretrain_checkpoints():
        all_checkpoints.append((ckpt, meta, "pretrain"))
    for ckpt, meta in scanner.get_midtrain_checkpoints():
        all_checkpoints.append((ckpt, meta, "midtrain"))
    for ckpt, meta in scanner.get_finetune_checkpoints():
        all_checkpoints.append((ckpt, meta, "finetune"))

    if not all_checkpoints:
        console.print("[yellow]No checkpoints found![/yellow]")
        return None

    # Select checkpoint
    checkpoint_choices = [f"[{stage}] {ckpt.name}" for ckpt, meta, stage in all_checkpoints]

    selected = questionary.select(
        "Select checkpoint:",
        choices=checkpoint_choices,
        style=custom_style,
    ).ask()

    selected_idx = checkpoint_choices.index(selected)
    checkpoint_path = all_checkpoints[selected_idx][0]

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
    console.print("\n[bold cyan]⬇️  Download Training Data[/bold cyan]\n")

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
        # WikiText - no size selection needed
        return {
            'dataset': 'wikitext',
        }


def run_train(config: dict):
    """Execute training with given configuration."""
    stage = config.get('stage', 'pretrain')
    stage_names = {
        'pretrain': 'PRE-TRAINING',
        'midtrain': 'MID-TRAINING',
        'finetune': 'FINE-TUNING',
    }

    console.print(f"\n[bold green]Starting {stage_names.get(stage, 'TRAINING')}...[/bold green]\n")
    console.print("=" * 80)
    print()

    # Execute appropriate training stage
    if stage == 'pretrain':
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
    elif stage == 'midtrain':
        # Mid-training demonstration (infrastructure ready, full implementation next step)
        console.print("[bold blue]Mid-Training Concepts Demonstration[/bold blue]\n")
        console.print("[dim]Infrastructure complete. Demonstrating how mid-training works...[/dim]\n")
        demonstrate_midtraining_concepts()
        console.print("\n[bold green]✓ Demonstration complete![/bold green]")
        console.print("\n[dim]Next step: Integrate HuggingFace datasets for full mid-training.[/dim]")
    else:
        # Fine-tuning
        console.print(f"[yellow]{stage_names.get(stage)} infrastructure coming soon![/yellow]")


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
        tokens_per_epoch = config.get('tokens', 50_000_000)
        download_shards(tokens_per_epoch=tokens_per_epoch)
    elif dataset == 'wikitext':
        download_wikitext()


def interactive_main():
    """Main interactive loop with pipeline-aware UI."""
    show_welcome()

    # Scan for checkpoints
    scanner = CheckpointScanner()

    # Main loop
    while True:
        choice = main_menu(scanner)

        if not choice or choice.startswith("❌"):
            console.print("\n[bold cyan]Goodbye! 👋[/bold cyan]")
            break

        # Handle separator choices (skip)
        if choice.startswith("───") or choice.startswith("[Locked]"):
            continue

        # PRE-TRAINING
        elif choice.startswith("🎓"):  # Start pre-training
            config = pretrain_menu()
            if config:
                run_train(config)
                scanner.scan()

        elif choice.startswith("▶️") and "pre-training" in choice:  # Continue pre-training
            config = continue_training_menu(scanner, 'pretrain')
            if config:
                run_train(config)
                scanner.scan()

        # MID-TRAINING
        elif choice.startswith("🔬"):  # Start mid-training
            config = midtrain_menu(scanner)
            if config:
                run_train(config)
                scanner.scan()

        elif choice.startswith("▶️") and "mid-training" in choice:  # Continue mid-training
            config = continue_training_menu(scanner, 'midtrain')
            if config:
                run_train(config)
                scanner.scan()

        # FINE-TUNING
        elif choice.startswith("🎯"):  # Start fine-tuning
            config = finetune_menu(scanner)
            if config:
                run_train(config)
                scanner.scan()

        elif choice.startswith("▶️") and "fine-tuning" in choice:  # Continue fine-tuning
            config = continue_training_menu(scanner, 'finetune')
            if config:
                run_train(config)
                scanner.scan()

        # MODEL OPERATIONS
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

        # UTILITIES
        elif choice.startswith("⬇️"):  # Download
            config = download_menu()
            if config:
                run_download(config)

        elif choice.startswith("❓"):  # Learn about pipeline
            show_pipeline_education()

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

        # Rescan checkpoints for next action
        console.print("\n" + "=" * 80 + "\n")
        scanner.scan()


if __name__ == "__main__":
    try:
        interactive_main()
    except KeyboardInterrupt:
        console.print("\n\n[yellow]Interrupted by user[/yellow]")
        sys.exit(0)
