"""
Text generation script for transformer model.

This script demonstrates how to use the trained transformer for autoregressive
text generation with various sampling strategies.

Generation Presets:
-------------------
Different tasks require different sampling strategies:

1. greedy: Deterministic, always picks highest probability
   - Use for: Debugging, reproducibility, most confident predictions
   - Behavior: Repetitive but consistent

2. precise: Low temperature, narrow nucleus (k=40, p=0.85, temp=0.7)
   - Use for: Factual text, code generation, precise answers
   - Behavior: Focused, coherent, safe

3. balanced: Medium settings (k=50, p=0.9, temp=1.0)
   - Use for: General purpose text generation
   - Behavior: Good quality-diversity tradeoff

4. creative: Higher temperature, wider nucleus (k=100, p=0.95, temp=1.2)
   - Use for: Creative writing, storytelling, varied output
   - Behavior: More diverse and interesting

5. very-creative: High temperature, very wide nucleus (k=200, p=0.98, temp=1.5)
   - Use for: Experimental generation, maximum diversity
   - Behavior: Unexpected, sometimes incoherent

Usage Examples:
---------------
    # Interactive mode with balanced preset
    uv run python main.py generate checkpoints/model_epoch_10.pt --preset balanced

    # Single prompt with creative preset
    uv run python main.py generate checkpoints/model_epoch_10.pt \
        --prompt "Once upon a time" --preset creative --max-length 100

    # Greedy decoding (deterministic)
    uv run python main.py generate checkpoints/model_epoch_10.pt \
        --prompt "The quick brown" --preset greedy

    # Custom parameters (overrides preset)
    uv run python main.py generate checkpoints/model_epoch_10.pt \
        --preset balanced --temperature 0.8 --top-k 60
"""

import torch
import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.dataset import TextDataset


# Generation presets for different use cases
GENERATION_PRESETS = {
    "greedy": {
        "method": "greedy",
        "description": "Deterministic, always picks highest probability token",
        "params": {}
    },
    "precise": {
        "method": "top_k_top_p",
        "description": "Focused and deterministic. Best for factual text and code generation",
        "params": {"k": 40, "p": 0.85, "temperature": 0.7}
    },
    "balanced": {
        "method": "top_k_top_p",
        "description": "Balanced quality and diversity. Good default for most tasks",
        "params": {"k": 50, "p": 0.9, "temperature": 1.0}
    },
    "creative": {
        "method": "top_k_top_p",
        "description": "More diverse and creative. Good for storytelling",
        "params": {"k": 100, "p": 0.95, "temperature": 1.2}
    },
    "very-creative": {
        "method": "top_k_top_p",
        "description": "Very diverse and experimental. Maximum creativity",
        "params": {"k": 200, "p": 0.98, "temperature": 1.5}
    }
}


def load_model(checkpoint_path, device):
    """
    Load a trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded DecoderOnlyTransformer
        config: Model configuration dict
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']
    print(f"  Model config: {config['num_layers']} layers, "
          f"{config['d_model']} d_model, {config['num_heads']} heads")

    # Create model with saved configuration
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout']
    )

    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    print(f"  Checkpoint from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.4f}")
    print()

    return model, config


def generate_text(model, dataset, prompt, max_length, sampling_method, sampling_params, device):
    """
    Generate text from a prompt using the specified sampling method.

    Args:
        model: Trained DecoderOnlyTransformer
        dataset: TextDataset (for tokenization)
        prompt: Input text to continue from
        max_length: Maximum length to generate
        sampling_method: "greedy", "top_k", "top_p", or "top_k_top_p"
        sampling_params: Dict of parameters for sampling method
        device: Device to run on

    Returns:
        generated_text: Generated text as string
    """
    # Tokenize prompt
    prompt_tokens = dataset.tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Generating {max_length} tokens using {sampling_method} sampling...")
    print()

    # Generate
    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            method=sampling_method,
            **sampling_params
        )

    # Decode
    generated_text = dataset.decode(output_ids[0])
    return generated_text


def interactive_mode(model, dataset, preset, device):
    """
    Interactive generation mode - keep prompting user for input.

    Args:
        model: Trained model
        dataset: TextDataset for tokenization
        preset: Generation preset configuration
        device: Device to run on
    """
    print("=" * 80)
    print("INTERACTIVE GENERATION MODE")
    print("=" * 80)
    print()
    print(f"Preset: {preset['method']} - {GENERATION_PRESETS[preset['name']]['description']}")
    print(f"Parameters: {preset['params']}")
    print()
    print("Enter your prompt and press Enter to generate.")
    print("Type 'quit' or 'exit' to stop.")
    print("Type 'help' to see available commands.")
    print()

    while True:
        try:
            prompt = input(">>> ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break

            if prompt.lower() == 'help':
                print("\nCommands:")
                print("  quit/exit - Exit interactive mode")
                print("  help - Show this help message")
                print("\nCurrent settings:")
                print(f"  Preset: {preset['name']}")
                print(f"  Method: {preset['method']}")
                print(f"  Parameters: {preset['params']}")
                print()
                continue

            if not prompt.strip():
                continue

            # Generate
            generated = generate_text(
                model, dataset, prompt,
                max_length=preset['max_length'],
                sampling_method=preset['method'],
                sampling_params=preset['params'],
                device=device
            )

            print()
            print("Generated:")
            print("-" * 80)
            print(generated)
            print("-" * 80)
            print()

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type 'quit' to exit or continue with a new prompt.")
            print()
        except Exception as e:
            print(f"\nError during generation: {e}")
            print()


def main():
    parser = argparse.ArgumentParser(
        description="Generate text using a trained transformer model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive mode with balanced preset
  uv run python main.py generate checkpoints/model_epoch_10.pt --preset balanced

  # Single prompt with creative preset
  uv run python main.py generate checkpoints/model_epoch_10.pt \\
      --prompt "Once upon a time" --preset creative

  # Custom parameters
  uv run python main.py generate checkpoints/model_epoch_10.pt \\
      --prompt "The" --temperature 0.8 --top-k 60 --top-p 0.92
        """
    )

    parser.add_argument(
        "checkpoint",
        type=str,
        nargs="?",  # Make optional
        default=None,
        help="Path to model checkpoint file"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Text prompt to generate from (if not provided, enters interactive mode)"
    )

    parser.add_argument(
        "--preset",
        type=str,
        choices=list(GENERATION_PRESETS.keys()),
        default="balanced",
        help="Generation preset (default: balanced)"
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=100,
        help="Maximum number of tokens to generate (default: 100)"
    )

    # Advanced parameters (override preset)
    parser.add_argument(
        "--temperature",
        type=float,
        default=None,
        help="Temperature for sampling (overrides preset)"
    )

    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="Top-k parameter (overrides preset)"
    )

    parser.add_argument(
        "--top-p",
        type=float,
        default=None,
        help="Top-p parameter (overrides preset)"
    )

    parser.add_argument(
        "--method",
        type=str,
        choices=["greedy", "top_k", "top_p", "top_k_top_p"],
        default=None,
        help="Sampling method (overrides preset)"
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use (default: auto-detect)"
    )

    parser.add_argument(
        "--list-presets",
        action="store_true",
        help="List all available presets and exit"
    )

    args = parser.parse_args()

    # List presets if requested
    if args.list_presets:
        print("\nAvailable generation presets:\n")
        for name, preset in GENERATION_PRESETS.items():
            print(f"  {name:15s} - {preset['description']}")
            if preset['params']:
                print(f"                  Parameters: {preset['params']}")
            print()
        return

    # Check checkpoint is provided
    if args.checkpoint is None:
        parser.error("the following arguments are required: checkpoint")

    # Check checkpoint exists
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint file not found: {checkpoint_path}")
        print("\nAvailable checkpoints:")
        checkpoints_dir = Path("checkpoints")
        if checkpoints_dir.exists():
            for cp in sorted(checkpoints_dir.glob("*.pt")):
                print(f"  {cp}")
        else:
            print("  No checkpoints directory found")
        sys.exit(1)

    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

    print()
    print("=" * 80)
    print("TRANSFORMER TEXT GENERATION")
    print("=" * 80)
    print()
    print(f"Device: {device}")
    print()

    # Load model
    model, config = load_model(checkpoint_path, device)

    # Initialize dataset for tokenization
    # Note: We don't need the actual text, just the tokenizer
    # Using a dummy file path - only vocab_size matters for generation
    print("Initializing tokenizer...")
    dataset = TextDataset.__new__(TextDataset)
    import tiktoken
    dataset.tokenizer = tiktoken.get_encoding("p50k_base")
    dataset.vocab_size = config['vocab_size']
    print(f"  Vocabulary size: {dataset.vocab_size:,}")
    print()

    # Setup generation parameters
    preset_config = GENERATION_PRESETS[args.preset]
    generation_config = {
        "name": args.preset,
        "method": args.method if args.method else preset_config["method"],
        "params": preset_config["params"].copy(),
        "max_length": args.max_length
    }

    # Override with custom parameters if provided
    if args.temperature is not None:
        generation_config["params"]["temperature"] = args.temperature
    if args.top_k is not None:
        generation_config["params"]["k"] = args.top_k
    if args.top_p is not None:
        generation_config["params"]["p"] = args.top_p

    # Remove unused parameters based on method
    if generation_config["method"] == "greedy":
        generation_config["params"] = {}
    elif generation_config["method"] == "top_k":
        generation_config["params"].pop("p", None)
    elif generation_config["method"] == "top_p":
        generation_config["params"].pop("k", None)

    # Interactive or single prompt mode
    if args.prompt is None:
        # Interactive mode
        interactive_mode(model, dataset, generation_config, device)
    else:
        # Single generation
        print(f"Preset: {args.preset} - {preset_config['description']}")
        print(f"Method: {generation_config['method']}")
        print(f"Parameters: {generation_config['params']}")
        print()
        print(f"Prompt: '{args.prompt}'")
        print()

        generated = generate_text(
            model, dataset, args.prompt,
            max_length=args.max_length,
            sampling_method=generation_config["method"],
            sampling_params=generation_config["params"],
            device=device
        )

        print("Generated text:")
        print("=" * 80)
        print(generated)
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
