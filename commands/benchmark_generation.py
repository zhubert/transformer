"""
Benchmark text generation with and without KV-cache.

Compares generation speed for different sequence lengths to demonstrate
the O(n²) vs O(n) time complexity difference.

Expected Results:
-----------------
- Short sequences (10-20 tokens): 2-5x speedup
- Medium sequences (50-100 tokens): 10-20x speedup
- Long sequences (200+ tokens): 20-50x speedup

The speedup increases with sequence length because:
- Without cache: Time = O(n²) where n = generated tokens
- With cache: Time = O(n) where n = generated tokens
"""

import torch
import time
import sys
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer


def benchmark_generation(
    model,
    start_tokens,
    max_length,
    use_cache,
    num_runs=3
):
    """
    Benchmark generation speed.

    Args:
        model: Transformer model
        start_tokens: Starting tokens (batch, start_len)
        max_length: Total length to generate
        use_cache: Whether to use KV-cache
        num_runs: Number of runs for averaging

    Returns:
        avg_time: Average time in seconds
        tokens_per_second: Average tokens generated per second
    """
    model.eval()
    times = []

    # Warm-up run (not counted)
    with torch.no_grad():
        _ = model.generate(
            start_tokens,
            max_length=max_length,
            sampling_strategy="greedy",
            use_cache=use_cache
        )

    # Timed runs
    for _ in range(num_runs):
        start_time = time.time()

        with torch.no_grad():
            generated = model.generate(
                start_tokens,
                max_length=max_length,
                sampling_strategy="greedy",
                use_cache=use_cache
            )

        end_time = time.time()
        times.append(end_time - start_time)

    avg_time = sum(times) / len(times)
    tokens_generated = max_length - start_tokens.size(1)
    tokens_per_second = tokens_generated / avg_time if avg_time > 0 else 0

    return avg_time, tokens_per_second


def main():
    """Run benchmark comparing cached vs non-cached generation."""
    console = Console()

    # Display header
    console.print(Panel(
        "[bold blue]KV-Cache Generation Benchmark[/bold blue]\n"
        "Comparing generation speed with and without KV-cache",
        style="bold blue",
        expand=False
    ))
    console.print()

    # Model configuration (small model for faster benchmarking)
    vocab_size = 1000
    d_model = 128
    num_heads = 4
    num_layers = 4
    batch_size = 1
    start_len = 5

    console.print("[bold]Model Configuration:[/bold]")
    console.print(f"  d_model: {d_model}")
    console.print(f"  num_heads: {num_heads}")
    console.print(f"  num_layers: {num_layers}")
    console.print(f"  vocab_size: {vocab_size}")
    console.print()

    # Create model
    console.print("[bold]Creating model...[/bold]")
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=512,
        max_seq_len=1000,
        dropout=0.0  # No dropout for deterministic benchmarking
    )
    model.eval()

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    console.print(f"[dim]Model parameters: {num_params:,}[/dim]")
    console.print()

    # Test different sequence lengths
    sequence_lengths = [10, 25, 50, 100, 200]

    console.print("[bold]Running benchmarks...[/bold]")
    console.print()

    # Create results table
    results_table = Table(title="Generation Speed Comparison", show_header=True, header_style="bold cyan")
    results_table.add_column("Sequence Length", justify="right", style="cyan")
    results_table.add_column("Tokens Generated", justify="right", style="white")
    results_table.add_column("Without Cache", justify="right", style="red")
    results_table.add_column("With Cache", justify="right", style="green")
    results_table.add_column("Speedup", justify="right", style="bold yellow")

    for max_length in sequence_lengths:
        tokens_to_generate = max_length - start_len

        console.print(f"[dim]Benchmarking {max_length} tokens (generating {tokens_to_generate} new tokens)...[/dim]")

        # Create starting tokens
        start_tokens = torch.randint(0, vocab_size, (batch_size, start_len))

        # Benchmark WITHOUT cache
        time_no_cache, tps_no_cache = benchmark_generation(
            model, start_tokens, max_length, use_cache=False, num_runs=3
        )

        # Benchmark WITH cache
        time_with_cache, tps_with_cache = benchmark_generation(
            model, start_tokens, max_length, use_cache=True, num_runs=3
        )

        # Calculate speedup
        speedup = time_no_cache / time_with_cache if time_with_cache > 0 else 0

        # Add to table
        results_table.add_row(
            str(max_length),
            str(tokens_to_generate),
            f"{tps_no_cache:.1f} tok/s ({time_no_cache:.3f}s)",
            f"{tps_with_cache:.1f} tok/s ({time_with_cache:.3f}s)",
            f"[bold]{speedup:.1f}x[/bold]"
        )

    console.print()
    console.print(results_table)
    console.print()

    # Summary
    console.print(Panel(
        "[bold green]Key Takeaways:[/bold green]\n\n"
        "1. Speedup increases with sequence length (O(n²) → O(n))\n"
        "2. Short sequences: 2-5x faster\n"
        "3. Long sequences: 20-50x faster\n"
        "4. KV-cache is ALWAYS faster for generation\n\n"
        "[bold yellow]Recommendation:[/bold yellow] Always use cache for generation (use_cache=True)",
        style="green",
        expand=False
    ))
    console.print()

    # Complexity explanation
    console.print(Panel(
        "[bold]Time Complexity Explanation:[/bold]\n\n"
        "[red]Without cache:[/red] O(n²)\n"
        "  - Recompute K, V for all tokens at each step\n"
        "  - Step 1: process 1 token\n"
        "  - Step 2: process 2 tokens\n"
        "  - Step n: process n tokens\n"
        "  - Total: 1 + 2 + ... + n = n(n+1)/2 ≈ O(n²)\n\n"
        "[green]With cache:[/green] O(n)\n"
        "  - Compute K, V only for new token\n"
        "  - Reuse cached K, V from previous tokens\n"
        "  - Each step processes only 1 new token\n"
        "  - Total: n steps × 1 token = O(n)",
        title="Why KV-Cache is Faster",
        border_style="blue"
    ))


if __name__ == "__main__":
    main()
