#!/usr/bin/env python3
"""
Transformer CLI - Main entry point for transformer operations.

This CLI provides both an interactive mode and traditional command-line interface
for all transformer functionality:
- Training models
- Generating text
- Evaluating model performance
- Comparing checkpoints
- Demonstrating sampling strategies
- Interpretability analysis

Usage:
    # Interactive mode (recommended for beginners)
    uv run python main.py

    # Direct command-line mode (for advanced users)
    uv run python main.py train [OPTIONS]
    uv run python main.py generate CHECKPOINT [OPTIONS]
    uv run python main.py evaluate [OPTIONS]
    uv run python main.py compare [OPTIONS]
    uv run python main.py demo-sampling
    uv run python main.py interpret [OPTIONS]
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Import from commands
from commands.train import train
from commands.download_shards import download_shards
from commands.generate import main as generate_main
from commands.evaluate_perplexity import (
    evaluate_checkpoint,
    compare_checkpoints,
)
from commands.sampling_comparison import (
    demonstrate_sampling_strategies,
    demonstrate_with_model,
)
from commands import interpret
from src.interactive import interactive_main


def create_parser():
    """Create the main argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Transformer CLI - Train, generate, and evaluate transformer models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # ============================================================================
    # TRAIN subcommand
    # ============================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train a decoder-only transformer model",
        description="Train a transformer model on text data with configurable hyperparameters",
    )
    train_parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode with diagnostic prints for NaN detection",
    )
    train_parser.add_argument(
        "--mps",
        action="store_true",
        help="Use MPS (Apple Silicon GPU) - EXPERIMENTAL, has known NaN issues",
    )
    train_parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick training mode: smaller model (4 layers, d_model=128) and fewer tokens (10M/epoch)",
    )
    train_parser.add_argument(
        "--medium",
        action="store_true",
        help="Medium training mode: balanced quality and speed (4 layers, d_model=256, 50M tokens/epoch × 15 epochs). "
             "Epoch 1: ~2h (builds cache), Epochs 2+: ~30-60min (cached)",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    train_parser.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile() optimization (for educational comparison). "
             "Default: compilation enabled for 20-40%% speedup on AMD/NVIDIA GPUs",
    )

    # ============================================================================
    # DOWNLOAD subcommand
    # ============================================================================
    download_parser = subparsers.add_parser(
        "download",
        help="Pre-download training data shards for offline training",
        description="Download and cache all FineWeb shards needed for training. "
                    "This allows you to train later without internet access.",
    )
    download_parser.add_argument(
        "--quick",
        action="store_true",
        help="Download shards for quick mode (10M tokens, ~1 GB)",
    )
    download_parser.add_argument(
        "--medium",
        action="store_true",
        help="Download shards for medium mode (50M tokens, ~5 GB)",
    )
    download_parser.add_argument(
        "--encoding",
        type=str,
        default="cl100k_base",
        choices=["p50k_base", "cl100k_base"],
        help="Tokenizer encoding to use (default: cl100k_base)",
    )

    # ============================================================================
    # GENERATE subcommand
    # ============================================================================
    generate_parser = subparsers.add_parser(
        "generate",
        help="Generate text using a trained model",
        description="Generate text from a trained transformer model with various sampling strategies",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode
  python main.py generate checkpoints/model_epoch_10.pt --preset balanced

  # Single prompt
  python main.py generate checkpoints/model_epoch_10.pt \\
      --prompt "Once upon a time" --preset creative

  # Custom parameters
  python main.py generate checkpoints/model_epoch_10.pt \\
      --prompt "The" --temperature 0.8 --top-k 60
        """,
    )
    generate_parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",
        default=None,
        help="Path to model checkpoint file",
    )
    generate_parser.add_argument(
        "--prompt", type=str, default=None, help="Text prompt (if not provided, enters interactive mode)"
    )
    generate_parser.add_argument(
        "--preset",
        type=str,
        choices=["greedy", "precise", "balanced", "creative", "very-creative"],
        default="balanced",
        help="Generation preset (default: balanced)",
    )
    generate_parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )
    generate_parser.add_argument(
        "--temperature", type=float, default=None, help="Temperature (overrides preset)"
    )
    generate_parser.add_argument(
        "--top-k", type=int, default=None, help="Top-k parameter (overrides preset)"
    )
    generate_parser.add_argument(
        "--top-p", type=float, default=None, help="Top-p parameter (overrides preset)"
    )
    generate_parser.add_argument(
        "--method",
        type=str,
        choices=["greedy", "top_k", "top_p", "top_k_top_p"],
        default=None,
        help="Sampling method (overrides preset)",
    )
    generate_parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use (default: auto-detect)",
    )
    generate_parser.add_argument(
        "--list-presets", action="store_true", help="List all available presets and exit"
    )

    # ============================================================================
    # EVALUATE subcommand
    # ============================================================================
    evaluate_parser = subparsers.add_parser(
        "evaluate",
        help="Evaluate a model checkpoint using perplexity",
        description="Evaluate model performance on a text dataset",
    )
    evaluate_parser.add_argument(
        "--checkpoint", type=str, help="Path to specific checkpoint to evaluate"
    )
    evaluate_parser.add_argument(
        "--text-file",
        type=str,
        default="Singular.txt",
        help="Text file to evaluate on (default: Singular.txt)",
    )
    evaluate_parser.add_argument(
        "--seq-length", type=int, default=128, help="Sequence length (default: 128)"
    )
    evaluate_parser.add_argument(
        "--batch-size", type=int, default=8, help="Batch size (default: 8)"
    )
    evaluate_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)",
    )

    # ============================================================================
    # COMPARE subcommand
    # ============================================================================
    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare all model checkpoints",
        description="Compare all checkpoints in a directory to find the best model",
    )
    compare_parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (default: checkpoints)",
    )
    compare_parser.add_argument(
        "--text-file",
        type=str,
        default="Singular.txt",
        help="Text file to evaluate on (default: Singular.txt)",
    )
    compare_parser.add_argument(
        "--seq-length", type=int, default=128, help="Sequence length (default: 128)"
    )
    compare_parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run on (default: cpu)",
    )

    # ============================================================================
    # DEMO-SAMPLING subcommand
    # ============================================================================
    demo_parser = subparsers.add_parser(
        "demo-sampling",
        help="Demonstrate different sampling strategies",
        description="Educational demonstration of how different sampling methods work",
    )
    demo_parser.add_argument(
        "--with-model",
        action="store_true",
        help="Also demonstrate with an actual trained model (if available)",
    )

    # ============================================================================
    # INTERPRET subcommand
    # ============================================================================
    interpret_parser = subparsers.add_parser(
        "interpret",
        help="Interpretability tools for understanding model internals",
        description="Analyze what your transformer has learned using mechanistic interpretability",
    )
    interpret.setup_parser(interpret_parser)

    return parser


def main():
    """Main entry point for the transformer CLI."""
    parser = create_parser()
    args = parser.parse_args()

    # Launch interactive mode if no command specified
    if args.command is None:
        interactive_main()
        return

    # ============================================================================
    # Route to appropriate handler
    # ============================================================================

    if args.command == "train":
        print("=" * 80)
        print("TRAINING MODE")
        print("=" * 80)
        print()
        train(debug=args.debug, use_mps=args.mps, quick=args.quick, medium=args.medium, resume=args.resume, compile=not args.no_compile)

    elif args.command == "download":
        download_shards(quick=args.quick, medium=args.medium, encoding=args.encoding)
        # Explicitly exit to ensure background threads from datasets library are cleaned up
        sys.exit(0)

    elif args.command == "generate":
        print("=" * 80)
        print("GENERATION MODE")
        print("=" * 80)
        print()
        # Call generate main with sys.argv modified to match its expectations
        # The generate script expects: script_name checkpoint [options]
        original_argv = sys.argv
        try:
            # Build new argv for generate script
            new_argv = ["generate"]
            if args.checkpoint:
                new_argv.append(args.checkpoint)
            if args.prompt:
                new_argv.extend(["--prompt", args.prompt])
            if args.preset:
                new_argv.extend(["--preset", args.preset])
            if args.max_length:
                new_argv.extend(["--max-length", str(args.max_length)])
            if args.temperature is not None:
                new_argv.extend(["--temperature", str(args.temperature)])
            if args.top_k is not None:
                new_argv.extend(["--top-k", str(args.top_k)])
            if args.top_p is not None:
                new_argv.extend(["--top-p", str(args.top_p)])
            if args.method:
                new_argv.extend(["--method", args.method])
            if args.device:
                new_argv.extend(["--device", args.device])
            if args.list_presets:
                new_argv.append("--list-presets")

            sys.argv = new_argv
            generate_main()
        finally:
            sys.argv = original_argv

    elif args.command == "evaluate":
        print("=" * 80)
        print("EVALUATION MODE")
        print("=" * 80)
        print()

        if args.checkpoint:
            # Evaluate specific checkpoint
            evaluate_checkpoint(
                args.checkpoint,
                args.text_file,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
                device=args.device,
            )
        else:
            # Find and evaluate latest checkpoint
            checkpoint_dir = Path(args.checkpoint_dir) if hasattr(args, 'checkpoint_dir') else Path("checkpoints")
            checkpoint_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))

            if not checkpoint_files:
                print(f"No checkpoints found in {checkpoint_dir}")
                print("Please train a model first using: python main.py train")
                sys.exit(1)

            latest_checkpoint = checkpoint_files[-1]
            print(f"No checkpoint specified, using latest: {latest_checkpoint.name}")
            print()

            evaluate_checkpoint(
                str(latest_checkpoint),
                args.text_file,
                seq_length=args.seq_length,
                batch_size=args.batch_size,
                device=args.device,
            )

    elif args.command == "compare":
        print("=" * 80)
        print("COMPARISON MODE")
        print("=" * 80)
        print()
        compare_checkpoints(
            args.checkpoint_dir,
            args.text_file,
            seq_length=args.seq_length,
            device=args.device,
        )

    elif args.command == "demo-sampling":
        print("=" * 80)
        print("SAMPLING DEMONSTRATION MODE")
        print("=" * 80)
        print()
        demonstrate_sampling_strategies()
        if args.with_model:
            print("\n" * 2)
            demonstrate_with_model()

    elif args.command == "interpret":
        print("=" * 80)
        print("INTERPRETABILITY MODE")
        print("=" * 80)
        print()
        interpret.main(args)


if __name__ == "__main__":
    main()
