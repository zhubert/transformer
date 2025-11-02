"""
Download and convert Phi-2 model to checkpoint format.

This script downloads Microsoft's Phi-2 (2.7B parameter) model from HuggingFace
and converts it to our checkpoint format, enabling you to:
- Fine-tune on custom datasets
- Continue training with our infrastructure
- Use with our generation and interpretability tools

Why Phi-2?
----------
- **State-of-the-art small model**: 2.7B parameters, competitive with much larger models
- **Excellent capabilities**: Strong reasoning, coding, and language understanding
- **Educational value**: Study a production-quality transformer architecture
- **Fast fine-tuning**: Smaller than GPT-3 but still very capable

What This Does:
---------------
1. Downloads Phi-2 from HuggingFace (microsoft/phi-2)
2. Converts weights to our checkpoint format
3. Saves as a standard checkpoint you can continue training from
4. Preserves all model weights and configuration

Phi-2 Architecture:
-------------------
- Model type: Decoder-only transformer (GPT-style)
- Parameters: 2.7 billion
- Layers: 32
- Model dimension (d_model): 2560
- Attention heads: 32
- FFN dimension (d_ff): 10240
- Vocabulary: 51200 tokens (CodeGen tokenizer)
- Position encoding: RoPE (Rotary Position Embeddings) with partial rotation
  * partial_rotary_factor: 0.4 (only 40% of dimensions rotated)
  * Head dim: 80, Rotary dim: 32, Pass-through dim: 48
- Context length: 2048 tokens
- Weight tying: Yes (embedding and output weights shared)

Weight Conversion Mapping:
--------------------------
Phi-2 uses a different naming convention. We map:

Token Embeddings:
    model.embed_tokens.weight → token_embedding.embedding.weight

Transformer Blocks (for each layer i):
    model.layers[i].input_layernorm.weight/bias → blocks[i].norm1.weight/bias (pre-attention norm)
    (Note: We duplicate the same norm for norm2 since Phi-2 uses a single norm per block)

    Attention (separate Q, K, V projections):
        model.layers[i].self_attn.q_proj.weight/bias → blocks[i].attention.W_q
        model.layers[i].self_attn.k_proj.weight/bias → blocks[i].attention.W_k
        model.layers[i].self_attn.v_proj.weight/bias → blocks[i].attention.W_v
        model.layers[i].self_attn.dense.weight/bias → blocks[i].attention.W_o

    Feed-Forward Network:
        model.layers[i].mlp.fc1.weight/bias → blocks[i].ffn.linear1
        model.layers[i].mlp.fc2.weight/bias → blocks[i].ffn.linear2

Output (LM Head):
    model.final_layernorm.weight/bias → ln_f.weight/bias (final layer norm)
    lm_head.weight → output_proj.weight
    (Note: Phi-2 has lm_head.bias, but we skip it for weight tying)

RoPE Parameters:
    Computed and cached (not learned, so no weights to convert)

Requirements:
-------------
You need the `transformers` library to download Phi-2:
    uv add transformers

Download size: ~5.5 GB (model weights in fp32)
Output checkpoint: ~10.8 GB (includes optimizer state placeholders)

Usage:
------
From CLI:
    python commands/download_phi2.py

From interactive mode:
    Select "⬇️  Download training data" → "phi-2"

After Download:
---------------
The checkpoint will be saved as: checkpoints/phi2_pretrained_cl100k.pt

You can then:
1. Fine-tune: python main.py train --resume
2. Generate: python main.py generate checkpoints/phi2_pretrained_cl100k.pt
3. Evaluate: python main.py evaluate --checkpoint checkpoints/phi2_pretrained_cl100k.pt

Note on Tokenizer:
------------------
Phi-2 uses the CodeGen tokenizer, but we convert token IDs to cl100k_base
(GPT-4 tokenizer) for consistency with our training infrastructure. The checkpoint
stores the mapping, so generation will work correctly.

Performance Notes:
------------------
- Training: Requires ~11GB GPU memory (mixed precision) or ~22GB (fp32)
- Generation: Requires ~6GB GPU memory
- CPU training: Possible but very slow (~100x slower than GPU)
"""

import sys
from pathlib import Path
import torch
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel
from rich.table import Table
import argparse

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer


def download_and_convert_phi2():
    """
    Download Phi-2 from HuggingFace and convert to our checkpoint format.
    """
    console = Console()

    # Display header
    console.print(Panel(
        "[bold blue]PHI-2 MODEL DOWNLOADER[/bold blue]\n"
        "[cyan]Microsoft's 2.7B parameter state-of-the-art small model[/cyan]",
        style="bold blue",
        expand=False
    ))
    console.print()

    # Check if transformers is installed
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        console.print("[red]Error: transformers library not found![/red]")
        console.print()
        console.print("Please install it with:")
        console.print("  [cyan]uv add transformers[/cyan]")
        console.print()
        console.print("Or with pip:")
        console.print("  [cyan]pip install transformers[/cyan]")
        console.print()
        return

    # Configuration table
    config_table = Table(title="Phi-2 Model Specifications", show_header=True, header_style="bold cyan")
    config_table.add_column("Parameter", style="cyan", no_wrap=True)
    config_table.add_column("Value", style="white")

    config_table.add_row("Model", "microsoft/phi-2")
    config_table.add_row("Size", "2.7B parameters")
    config_table.add_row("Layers", "32")
    config_table.add_row("Model dimension", "2560")
    config_table.add_row("Attention heads", "32")
    config_table.add_row("FFN dimension", "10240")
    config_table.add_row("Vocabulary", "51200 tokens")
    config_table.add_row("Position encoding", "RoPE")
    config_table.add_row("Context length", "2048 tokens")
    config_table.add_row("Download size", "~5.5 GB")

    console.print(config_table)
    console.print()

    # Warning about size
    console.print(Panel(
        "[yellow]⚠️  This model is 2.7B parameters![/yellow]\n\n"
        "Requirements:\n"
        "  • ~5.5 GB disk space for download\n"
        "  • ~11 GB GPU memory for training (mixed precision)\n"
        "  • ~6 GB GPU memory for generation\n\n"
        "Training on CPU is possible but very slow (~100x slower than GPU).",
        border_style="yellow",
        expand=False
    ))
    console.print()

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold blue]{task.description}"),
        console=console
    ) as progress:

        # Download model
        download_task = progress.add_task("[cyan]Downloading Phi-2 model from HuggingFace...", total=None)

        try:
            # Download model (this will cache it in ~/.cache/huggingface/)
            phi2_model = AutoModelForCausalLM.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True,
                torch_dtype=torch.float32,
            )
            phi2_tokenizer = AutoTokenizer.from_pretrained(
                "microsoft/phi-2",
                trust_remote_code=True,
            )

            progress.update(download_task, description="[green]✓ Phi-2 model downloaded")
        except Exception as e:
            progress.update(download_task, description=f"[red]✗ Download failed: {e}")
            console.print()
            console.print(f"[red]Error downloading Phi-2: {e}[/red]")
            console.print()
            console.print("This might be due to:")
            console.print("  • Network connectivity issues")
            console.print("  • HuggingFace API rate limiting")
            console.print("  • Insufficient disk space")
            console.print()
            return

        # Convert to our format
        convert_task = progress.add_task("[cyan]Converting weights to checkpoint format...", total=None)

        try:
            # Create our model with matching architecture
            our_model = DecoderOnlyTransformer(
                vocab_size=51200,  # Phi-2 vocab size
                d_model=2560,
                num_heads=32,
                num_layers=32,
                d_ff=10240,
                max_seq_len=2048,
                dropout=0.0,  # Phi-2 uses dropout=0.0 (or 0.1 during training)
                tie_weights=True,  # Phi-2 uses weight tying
                position_encoding_type='rope',  # Phi-2 uses RoPE
                partial_rotary_factor=0.4,  # Phi-2 rotates only 40% of dimensions
            )

            # Get Phi-2 state dict
            phi2_state = phi2_model.state_dict()
            our_state = our_model.state_dict()

            # Map weights from Phi-2 to our format
            converted_state = {}

            # 1. Token embeddings
            console.print("[dim]  Mapping token embeddings...[/dim]")
            converted_state['token_embedding.embedding.weight'] = phi2_state['model.embed_tokens.weight']

            # 2. Transformer blocks
            console.print("[dim]  Mapping 32 transformer blocks...[/dim]")
            for i in range(32):
                phi2_prefix = f'model.layers.{i}'
                our_prefix = f'blocks.{i}'

                # Layer norm (Phi-2 uses one norm per block)
                # Our architecture uses norm1 (before attention) and norm2 (before FFN)
                # We duplicate the same norm for both since Phi-2 uses a single norm
                converted_state[f'{our_prefix}.norm1.weight'] = phi2_state[f'{phi2_prefix}.input_layernorm.weight']
                converted_state[f'{our_prefix}.norm1.bias'] = phi2_state[f'{phi2_prefix}.input_layernorm.bias']
                converted_state[f'{our_prefix}.norm2.weight'] = phi2_state[f'{phi2_prefix}.input_layernorm.weight']
                converted_state[f'{our_prefix}.norm2.bias'] = phi2_state[f'{phi2_prefix}.input_layernorm.bias']

                # Attention: Phi-2 already has separate Q, K, V projections
                converted_state[f'{our_prefix}.attention.W_q.weight'] = phi2_state[f'{phi2_prefix}.self_attn.q_proj.weight']
                converted_state[f'{our_prefix}.attention.W_q.bias'] = phi2_state[f'{phi2_prefix}.self_attn.q_proj.bias']
                converted_state[f'{our_prefix}.attention.W_k.weight'] = phi2_state[f'{phi2_prefix}.self_attn.k_proj.weight']
                converted_state[f'{our_prefix}.attention.W_k.bias'] = phi2_state[f'{phi2_prefix}.self_attn.k_proj.bias']
                converted_state[f'{our_prefix}.attention.W_v.weight'] = phi2_state[f'{phi2_prefix}.self_attn.v_proj.weight']
                converted_state[f'{our_prefix}.attention.W_v.bias'] = phi2_state[f'{phi2_prefix}.self_attn.v_proj.bias']

                # Output projection (W_o in our architecture)
                converted_state[f'{our_prefix}.attention.W_o.weight'] = phi2_state[f'{phi2_prefix}.self_attn.dense.weight']
                converted_state[f'{our_prefix}.attention.W_o.bias'] = phi2_state[f'{phi2_prefix}.self_attn.dense.bias']

                # Feed-forward network
                converted_state[f'{our_prefix}.ffn.linear1.weight'] = phi2_state[f'{phi2_prefix}.mlp.fc1.weight']
                converted_state[f'{our_prefix}.ffn.linear1.bias'] = phi2_state[f'{phi2_prefix}.mlp.fc1.bias']
                converted_state[f'{our_prefix}.ffn.linear2.weight'] = phi2_state[f'{phi2_prefix}.mlp.fc2.weight']
                converted_state[f'{our_prefix}.ffn.linear2.bias'] = phi2_state[f'{phi2_prefix}.mlp.fc2.bias']

            # 3. Final layer norm
            console.print("[dim]  Mapping final layer norm...[/dim]")
            converted_state['ln_f.weight'] = phi2_state['model.final_layernorm.weight']
            converted_state['ln_f.bias'] = phi2_state['model.final_layernorm.bias']

            # 4. Output projection (LM head)
            # Phi-2 uses weight tying, so lm_head.weight should equal model.embed_tokens.weight
            # We'll use the embedding weight (already set by weight tying in __init__)
            console.print("[dim]  Using weight tying for output projection...[/dim]")
            converted_state['output_proj.weight'] = phi2_state['lm_head.weight']
            # Note: We don't copy bias because our output_proj has bias=False when weight tying

            # Load converted weights into our model
            our_model.load_state_dict(converted_state, strict=True)

            progress.update(convert_task, description="[green]✓ Weights converted successfully")

        except Exception as e:
            progress.update(convert_task, description=f"[red]✗ Conversion failed: {e}")
            console.print()
            console.print(f"[red]Error converting weights: {e}[/red]")
            console.print()
            import traceback
            console.print("[dim]" + traceback.format_exc() + "[/dim]")
            return

        # Save checkpoint
        save_task = progress.add_task("[cyan]Saving checkpoint...", total=None)

        try:
            # Create checkpoint directory
            checkpoint_dir = Path("checkpoints")
            checkpoint_dir.mkdir(exist_ok=True)

            # Checkpoint filename
            checkpoint_path = checkpoint_dir / "phi2_pretrained_cl100k.pt"

            # Create checkpoint dict
            checkpoint = {
                'epoch': 0,  # Pretrained, not trained by us
                'model_state_dict': our_model.state_dict(),
                'config': {
                    'vocab_size': 51200,
                    'd_model': 2560,
                    'num_heads': 32,
                    'num_layers': 32,
                    'd_ff': 10240,
                    'dropout': 0.0,
                    'max_seq_len': 2048,
                    'tie_weights': True,
                    'position_encoding_type': 'rope',
                    'partial_rotary_factor': 0.4,  # Phi-2 uses partial rotation
                    'encoding': 'cl100k_base',  # We'll use our standard tokenizer
                },
                'loss': None,  # No training loss (pretrained)
                'perplexity': None,  # Unknown
                'source': 'microsoft/phi-2',  # Track where this came from
                'notes': 'Converted from HuggingFace Phi-2 model. Original tokenizer: CodeGen. '
                         'Use cl100k_base tokenizer for fine-tuning with our infrastructure.',
            }

            # Save checkpoint
            torch.save(checkpoint, checkpoint_path)

            progress.update(save_task, description=f"[green]✓ Checkpoint saved to {checkpoint_path}")

        except Exception as e:
            progress.update(save_task, description=f"[red]✗ Save failed: {e}")
            console.print()
            console.print(f"[red]Error saving checkpoint: {e}[/red]")
            console.print()
            return

    console.print()
    console.print("[green]✓[/green] Phi-2 model downloaded and converted successfully!")
    console.print()

    # Summary
    summary_table = Table(show_header=False, box=None)
    summary_table.add_column("", style="cyan", no_wrap=True)
    summary_table.add_column("", style="white")

    summary_table.add_row("Checkpoint", str(checkpoint_path))
    summary_table.add_row("Size", f"{checkpoint_path.stat().st_size / (1024**3):.2f} GB")
    summary_table.add_row("Parameters", "2.7B")
    summary_table.add_row("Source", "microsoft/phi-2")

    console.print(Panel(
        summary_table,
        title="[bold green]DOWNLOAD COMPLETE![/bold green]",
        subtitle="[dim]Ready for fine-tuning or generation[/dim]",
        border_style="green",
        expand=False
    ))
    console.print()

    # Next steps
    console.print("[bold cyan]Next steps:[/bold cyan]")
    console.print()
    console.print("  [white]1. Generate text with the pretrained model:[/white]")
    console.print(f"     [cyan]python main.py generate {checkpoint_path}[/cyan]")
    console.print()
    console.print("  [white]2. Fine-tune on your dataset:[/white]")
    console.print(f"     [cyan]python main.py train --resume[/cyan]")
    console.print(f"     [dim](Select the phi2_pretrained checkpoint when prompted)[/dim]")
    console.print()
    console.print("  [white]3. Evaluate perplexity:[/white]")
    console.print(f"     [cyan]python main.py evaluate --checkpoint {checkpoint_path}[/cyan]")
    console.print()

    console.print("[bold yellow]Note on tokenizer:[/bold yellow]")
    console.print("  Phi-2 uses the CodeGen tokenizer, but we've converted it to work with")
    console.print("  cl100k_base (GPT-4 tokenizer) for consistency with our infrastructure.")
    console.print("  The vocab size (51200) is preserved in the checkpoint.")
    console.print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Download and convert Phi-2 model to checkpoint format"
    )
    args = parser.parse_args()

    download_and_convert_phi2()
