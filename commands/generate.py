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
from src.transformer.device_utils import init_device, get_autocast_context
import tiktoken

# Import encoding detection from train.py
sys.path.append(str(Path(__file__).parent))
from train import detect_encoding_from_checkpoint


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
        detected_encoding: Detected encoding from checkpoint
    """
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    config = checkpoint['config']

    # Detect encoding
    detected_encoding = detect_encoding_from_checkpoint(checkpoint)

    print(f"  Model config: {config['num_layers']} layers, "
          f"{config['d_model']} d_model, {config['num_heads']} heads")
    print(f"  Encoding: {detected_encoding}")

    # Strip torch.compile() prefix if present (_orig_mod.)
    # Checkpoints saved from compiled models have this prefix on all keys
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        print("  Detected torch.compile() checkpoint, stripping prefix...")
        state_dict = {k.replace('_orig_mod.', '', 1): v for k, v in state_dict.items()}

    # Infer max_seq_len from positional encoding shape if not in config
    max_seq_len = config.get('max_seq_len')
    if max_seq_len is None:
        pos_embedding_shape = state_dict['pos_encoding.pos_embedding.weight'].shape
        max_seq_len = pos_embedding_shape[0]
        print(f"  Inferred max_seq_len from checkpoint: {max_seq_len}")

    # Create model with saved configuration
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
        max_seq_len=max_seq_len
    )

    # Load trained weights
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Display checkpoint info (loss may not always be saved)
    epoch_info = f"  Checkpoint from epoch {checkpoint['epoch']}"
    if 'loss' in checkpoint:
        epoch_info += f", loss: {checkpoint['loss']:.4f}"
    print(epoch_info)
    print()

    return model, config, detected_encoding


def generate_text(model, tokenizer, prompt, max_length, sampling_method, sampling_params, device, autocast_ctx):
    """
    Generate text from a prompt using the specified sampling method.

    Args:
        model: Trained DecoderOnlyTransformer
        tokenizer: tiktoken tokenizer (for tokenization)
        prompt: Input text to continue from
        max_length: Maximum length to generate
        sampling_method: "greedy", "top_k", "top_p", or "top_k_top_p"
        sampling_params: Dict of parameters for sampling method
        device: Device to run on
        autocast_ctx: Autocast context for mixed precision (CUDA only)

    Returns:
        generated_text: Generated text as string
    """
    # Tokenize prompt
    prompt_tokens = tokenizer.encode(prompt)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    print(f"Prompt tokens: {len(prompt_tokens)}")
    print(f"Generating {max_length} tokens using {sampling_method} sampling...")
    print()

    # Generate with autocast (mixed precision on CUDA, no-op on MPS/CPU)
    with torch.no_grad(), autocast_ctx:
        output_ids = model.generate(
            input_ids,
            max_length=max_length,
            method=sampling_method,
            **sampling_params
        )

    # Decode
    generated_text = tokenizer.decode(output_ids[0].tolist())
    return generated_text


def interactive_mode(model, tokenizer, preset, device, autocast_ctx):
    """
    Interactive generation mode - keep prompting user for input.

    Args:
        model: Trained model
        tokenizer: tiktoken tokenizer for tokenization
        preset: Generation preset configuration
        device: Device to run on
        autocast_ctx: Autocast context for mixed precision
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
                model, tokenizer, prompt,
                max_length=preset['max_length'],
                sampling_method=preset['method'],
                sampling_params=preset['params'],
                device=device,
                autocast_ctx=autocast_ctx
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

    # Setup device with proper initialization
    device, device_name = init_device(args.device, seed=42)
    autocast_ctx = get_autocast_context(device.type)

    print()
    print("=" * 80)
    print("TRANSFORMER TEXT GENERATION")
    print("=" * 80)
    print()
    print(f"Device: {device_name}")
    print()

    # Load model
    model, config, detected_encoding = load_model(checkpoint_path, device)

    # Initialize tokenizer (always use cl100k_base)
    encoding = "cl100k_base"
    print("Initializing tokenizer...")
    print(f"  Using encoding: {encoding}")
    tokenizer = tiktoken.get_encoding(encoding)
    vocab_size = config['vocab_size']
    print(f"  Vocabulary size: {vocab_size:,}")
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
        interactive_mode(model, tokenizer, generation_config, device, autocast_ctx)
    else:
        # Single generation
        print(f"Preset: {args.preset} - {preset_config['description']}")
        print(f"Method: {generation_config['method']}")
        print(f"Parameters: {generation_config['params']}")
        print()
        print(f"Prompt: '{args.prompt}'")
        print()

        generated = generate_text(
            model, tokenizer, args.prompt,
            max_length=args.max_length,
            sampling_method=generation_config["method"],
            sampling_params=generation_config["params"],
            device=device,
            autocast_ctx=autocast_ctx
        )

        print("Generated text:")
        print("=" * 80)
        print(generated)
        print("=" * 80)
        print()


if __name__ == "__main__":
    main()
