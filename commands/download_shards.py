"""
Pre-download FineWeb shards for offline training.

This script downloads and caches all shards needed for a training run,
allowing you to train later without network access or interruptions.

Why Pre-download?
-----------------
- **Bandwidth control**: Download when network is fast/cheap
- **Offline training**: Train later without internet connection
- **No interruptions**: Avoid network issues during training
- **Time management**: Download overnight, train tomorrow

What This Does:
---------------
1. Creates a FineWebDataset for the specified mode
2. Iterates through all sequences to trigger shard downloads
3. Shards are cached to data/fineweb_cache/
4. Progress is displayed with Rich progress bars

After downloading, training will run at full speed from epoch 1!

Cache Sizes:
------------
- Quick mode: ~1 GB (26 shards)
- Medium mode: ~5 GB (132 shards)
- Default mode: ~10 GB (264 shards)
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, DownloadColumn, TransferSpeedColumn
from rich.panel import Panel
from rich.table import Table
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.fineweb_dataset import FineWebDataset
from src.transformer.dataset_utils import calculate_optimal_cache_size


def download_shards(quick=False, medium=False):
    """
    Download all shards needed for training.

    Args:
        quick: If True, download shards for quick mode (10M tokens)
        medium: If True, download shards for medium mode (50M tokens)
    """
    encoding = "cl100k_base"
    # Configuration based on mode
    if quick and medium:
        raise ValueError("Cannot use both --quick and --medium flags. Please choose one.")

    if quick:
        TOKENS_PER_EPOCH = 10_000_000
        mode_name = "Quick"
    elif medium:
        TOKENS_PER_EPOCH = 50_000_000
        mode_name = "Medium"
    else:
        TOKENS_PER_EPOCH = 100_000_000
        mode_name = "Default"

    SEQ_LENGTH = 128
    MAX_CACHED_SHARDS = calculate_optimal_cache_size(TOKENS_PER_EPOCH)

    console = Console()

    # Display header
    console.print(Panel(
        f"[bold blue]FINEWEB SHARD DOWNLOADER[/bold blue]\n"
        f"[cyan]{mode_name} Mode[/cyan]",
        style="bold blue",
        expand=False
    ))
    console.print()

    # Configuration table
    config_table = Table(title="Download Configuration", show_header=True, header_style="bold cyan")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="white")

    cache_size_gb = (MAX_CACHED_SHARDS * 40) / 1024
    config_table.add_row("Training mode", mode_name)
    config_table.add_row("Tokenizer", encoding)
    config_table.add_row("Tokens per epoch", f"{TOKENS_PER_EPOCH:,}")
    config_table.add_row("Shards to download", f"{MAX_CACHED_SHARDS} (~{cache_size_gb:.1f} GB)")
    config_table.add_row("Cache directory", "data/fineweb_cache/")

    console.print(config_table)
    console.print()

    # Create datasets (train + validation)
    console.print("[bold]Initializing datasets...[/bold]")

    train_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=SEQ_LENGTH,
        tokens_per_epoch=TOKENS_PER_EPOCH,
        max_cached_shards=MAX_CACHED_SHARDS,
        seed=42,
        encoding_name=encoding,
        split="train",
        validation_fraction=0.1
    )

    val_tokens_per_epoch = TOKENS_PER_EPOCH // 10
    val_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=SEQ_LENGTH,
        tokens_per_epoch=val_tokens_per_epoch,
        max_cached_shards=MAX_CACHED_SHARDS,
        seed=42,
        encoding_name=encoding,
        split="validation",
        validation_fraction=0.1
    )

    console.print("[green]✓[/green] Datasets initialized")
    console.print()

    # Download training shards
    console.print(Panel("[bold]Downloading Training Shards[/bold]", style="bold green", expand=False))
    console.print()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} sequences"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        total_train_sequences = TOKENS_PER_EPOCH // SEQ_LENGTH
        train_task = progress.add_task(
            "[cyan]Training data",
            total=total_train_sequences
        )

        for idx, (input_seq, target_seq) in enumerate(train_dataset):
            progress.update(train_task, advance=1)

            # Optional: Add a small status update every 1000 sequences
            if (idx + 1) % 1000 == 0:
                progress.update(train_task, description=f"[cyan]Training data ({idx + 1:,} sequences)")

    console.print("[green]✓[/green] Training shards downloaded")
    console.print()

    # Download validation shards
    console.print(Panel("[bold]Downloading Validation Shards[/bold]", style="bold yellow", expand=False))
    console.print()

    with Progress(
        TextColumn("[bold blue]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("•"),
        TextColumn("{task.completed}/{task.total} sequences"),
        TimeRemainingColumn(),
        console=console
    ) as progress:

        total_val_sequences = val_tokens_per_epoch // SEQ_LENGTH
        val_task = progress.add_task(
            "[yellow]Validation data",
            total=total_val_sequences
        )

        for idx, (input_seq, target_seq) in enumerate(val_dataset):
            progress.update(val_task, advance=1)

    console.print("[green]✓[/green] Validation shards downloaded")
    console.print()

    # Summary
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("", style="cyan", no_wrap=True)
    summary_table.add_column("", style="white", justify="right")

    summary_table.add_row("Training sequences", f"{total_train_sequences:,}")
    summary_table.add_row("Validation sequences", f"{total_val_sequences:,}")
    summary_table.add_row("Total tokens", f"{TOKENS_PER_EPOCH + val_tokens_per_epoch:,}")
    summary_table.add_row("Cache directory", "data/fineweb_cache/")
    summary_table.add_row("Disk space used", f"~{cache_size_gb:.1f} GB")

    console.print(Panel(
        summary_table,
        title="[bold green]DOWNLOAD COMPLETE![/bold green]",
        subtitle="[dim]You can now train offline with full-speed epochs[/dim]",
        border_style="green",
        expand=False
    ))
    console.print()

    # Next steps
    console.print("[bold cyan]Next steps:[/bold cyan]")
    if quick:
        console.print("  [white]make train-quick[/white]      # Train in quick mode")
    elif medium:
        console.print("  [white]make train-medium[/white]     # Train in medium mode")
    else:
        console.print("  [white]make train[/white]            # Train in default mode")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download FineWeb shards for offline training")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Download shards for quick mode (10M tokens, ~1 GB)"
    )
    parser.add_argument(
        "--medium",
        action="store_true",
        help="Download shards for medium mode (50M tokens, ~5 GB)"
    )
    args = parser.parse_args()

    download_shards(
        quick=args.quick,
        medium=args.medium
    )
