"""
Pre-download WikiText-103 dataset for offline training.

This script downloads and caches the WikiText-103 dataset,
allowing you to train later without network access.

Why Pre-download?
-----------------
- **Bandwidth control**: Download when network is fast/cheap
- **Offline training**: Train later without internet connection
- **No interruptions**: Avoid network issues during training
- **Fast first epoch**: Dataset is already cached

What This Does:
---------------
1. Downloads WikiText-103 from HuggingFace
2. Caches to ~/.cache/huggingface/datasets/
3. Verifies all splits (train, validation, test)
4. Shows cache location and size

After downloading, training will run at full speed from epoch 1!

Dataset Info:
-------------
- Total size: ~500 MB (compressed), ~517 MB (raw text)
- Train split: ~100M tokens
- Validation split: ~217K tokens
- Test split: ~245K tokens
"""

import sys
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, SpinnerColumn
from rich.panel import Panel
from rich.table import Table
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.wikitext_dataset import WikiTextDataset


def download_wikitext():
    """
    Download WikiText-103 dataset and cache it locally.
    """
    encoding = "cl100k_base"
    SEQ_LENGTH = 128

    console = Console()

    # Display header
    console.print(Panel(
        f"[bold blue]WIKITEXT-103 DOWNLOADER[/bold blue]\n"
        f"[cyan]High-quality Wikipedia text dataset[/cyan]",
        style="bold blue",
        expand=False
    ))
    console.print()

    # Configuration table
    config_table = Table(title="Download Configuration", show_header=True, header_style="bold cyan")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="white")

    config_table.add_row("Dataset", "WikiText-103-raw-v1")
    config_table.add_row("Tokenizer", encoding)
    config_table.add_row("Train tokens", "~100M")
    config_table.add_row("Validation tokens", "~217K")
    config_table.add_row("Test tokens", "~245K")
    config_table.add_row("Download size", "~500 MB")
    config_table.add_row("Cache location", "~/.cache/huggingface/datasets/")

    console.print(config_table)
    console.print()

    # Download all splits
    console.print("[bold]Downloading WikiText-103 dataset...[/bold]")
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:

        # Download train split
        train_task = progress.add_task("[cyan]Downloading train split...", total=None)
        train_dataset = WikiTextDataset(
            seq_length=SEQ_LENGTH,
            encoding_name=encoding,
            split="train",
        )
        progress.update(train_task, description="[green]✓ Train split downloaded")

        # Download validation split
        val_task = progress.add_task("[cyan]Downloading validation split...", total=None)
        val_dataset = WikiTextDataset(
            seq_length=SEQ_LENGTH,
            encoding_name=encoding,
            split="validation",
        )
        progress.update(val_task, description="[green]✓ Validation split downloaded")

        # Download test split
        test_task = progress.add_task("[cyan]Downloading test split...", total=None)
        test_dataset = WikiTextDataset(
            seq_length=SEQ_LENGTH,
            encoding_name=encoding,
            split="test",
        )
        progress.update(test_task, description="[green]✓ Test split downloaded")

    console.print()
    console.print("[green]✓[/green] All splits downloaded and cached")
    console.print()

    # Verify datasets by getting lengths
    console.print("[bold]Verifying datasets...[/bold]")
    console.print()

    verification_table = Table(show_header=True, header_style="bold green")
    verification_table.add_column("Split", style="cyan", no_wrap=True)
    verification_table.add_column("Status", style="green")
    verification_table.add_column("Estimated Sequences", justify="right", style="white")

    verification_table.add_row("Train", "✓ Ready", f"~{len(train_dataset):,}")
    verification_table.add_row("Validation", "✓ Ready", f"~{len(val_dataset):,}")
    verification_table.add_row("Test", "✓ Ready", f"~{len(test_dataset):,}")

    console.print(verification_table)
    console.print()

    # Summary
    import os
    from pathlib import Path

    # Find cache location
    cache_dir = Path.home() / ".cache" / "huggingface" / "datasets" / "wikitext"

    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("", style="cyan", no_wrap=True)
    summary_table.add_column("", style="white", justify="right")

    summary_table.add_row("Dataset", "WikiText-103-raw-v1")
    summary_table.add_row("Vocab size", f"{train_dataset.vocab_size:,} tokens")
    summary_table.add_row("Sequence length", f"{SEQ_LENGTH}")

    # Calculate cache size if directory exists
    if cache_dir.exists():
        cache_size_mb = sum(f.stat().st_size for f in cache_dir.rglob('*') if f.is_file()) / (1024 * 1024)
        summary_table.add_row("Cache size", f"~{cache_size_mb:.1f} MB")
        summary_table.add_row("Cache location", str(cache_dir))
    else:
        summary_table.add_row("Cache location", "~/.cache/huggingface/datasets/")

    console.print(Panel(
        summary_table,
        title="[bold green]DOWNLOAD COMPLETE![/bold green]",
        subtitle="[dim]Dataset is cached and ready for offline training[/dim]",
        border_style="green",
        expand=False
    ))
    console.print()

    # Next steps
    console.print("[bold cyan]Next steps:[/bold cyan]")
    console.print("  [white]python main.py train --dataset wikitext --quick[/white]")
    console.print("  [dim]or[/dim]")
    console.print("  [white]make train-quick DATASET=wikitext[/white]")
    console.print()
    console.print("[dim]Note: WikiText is automatically cached by HuggingFace.[/dim]")
    console.print("[dim]No re-download needed for future training runs![/dim]")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-download WikiText-103 dataset for offline training")
    args = parser.parse_args()

    download_wikitext()
