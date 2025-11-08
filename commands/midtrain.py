"""
Mid-training (Continued Pre-Training) for domain specialization.

This script demonstrates how to specialize a pre-trained model for specific domains
like code, math, or science - creating models like Codex, Minerva, or Code Llama.

What is Mid-Training?
----------------------
Mid-training (also called "continued pre-training") is Stage 2 of modern LLM training:

    Stage 1: PRE-TRAINING → General language understanding (FineWeb, 10B tokens)
    Stage 2: MID-TRAINING → Domain specialization (Code/Math/Science, 50M-500M tokens)
    Stage 3: FINE-TUNING → Task-specific behavior (Instructions, 10K-100K examples)

Examples of Mid-Trained Models:
    - Codex (OpenAI): GPT-3 + GitHub code → code completion
    - Minerva (Google): PaLM + math papers → mathematical reasoning
    - Code Llama (Meta): Llama 2 + code → programming assistant
    - Galactica (Meta): LLaMA + scientific papers → science Q&A

Key Differences from Pre-Training:
------------------------------------
1. **Base model**: Start from pre-trained checkpoint (not random weights!)
2. **Domain focus**: Specialized data (code, math, science)
3. **Lower learning rate**: 1e-5 (30x lower) to preserve general knowledge
4. **Data mixing**: 90% domain + 10% general → prevents catastrophic forgetting
5. **Curriculum learning**: Easy → hard (10-15% improvement)
6. **Dual evaluation**: Track both domain AND general perplexity

Catastrophic Forgetting:
-------------------------
The biggest risk in mid-training! If you ONLY train on domain data:
    - Model improves on domain (e.g., code perplexity: 50 → 30 ✓)
    - Model forgets general language (general perplexity: 40 → 120 ✗)
    - Result: Expert coder that can't write English!

Solution: Data Mixing + Dual Evaluation
    - 90% domain data (improve specialization)
    - 10% general data (maintain general capability)
    - Track BOTH perplexities every epoch
    - Alert if general perplexity degrades >10%

Curriculum Learning:
--------------------
Train on progressively harder examples (like humans learn!):
    - Stage 1: Easy examples (warmup, build foundation)
    - Stage 2: Medium examples (main training)
    - Stage 3: Hard examples (push capabilities)

Benefits:
    - 10-15% better final performance
    - Faster convergence
    - More stable training

Example: Math domain
    - Easy (difficulty 1-2): Basic algebra
    - Medium (difficulty 3): Geometry, calculus
    - Hard (difficulty 4-5): Competition problems

Training Parameters:
--------------------
**Learning Rate**: 1e-5 (vs 3e-4 for pre-training)
    - Lower rate preserves pre-trained knowledge
    - Too high → catastrophic forgetting
    - Too low → no improvement

**Data Mixing**: 90% domain + 10% general
    - Domain: Specialized dataset (code/math/science)
    - General: FineWeb samples (maintain general capability)

**Epochs**: 5-10 (vs 10-20 for pre-training)
    - Domain data is higher quality, less needed
    - Monitor forgetting - stop if general perplexity degrades

**Batch Size**: Same as pre-training (8-16 sequences)
    - Gradient accumulation for stability

What to Expect:
----------------
Example progress for Code mid-training:

Epoch 1:  Domain PPL ~60 → 45    General PPL ~35 → 36   (slight forgetting, normal)
Epoch 3:  Domain PPL ~45 → 35    General PPL ~36 → 37   (improving domain)
Epoch 5:  Domain PPL ~35 → 28    General PPL ~37 → 38   (good specialization)
Epoch 7:  Domain PPL ~28 → 25    General PPL ~38 → 42   (⚠ forgetting increasing)
Epoch 8:  Domain PPL ~25 → 23    General PPL ~42 → 48   (🛑 STOP! Catastrophic forgetting!)

The model saves checkpoints after each epoch to checkpoints/midtrain/ directory.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from pathlib import Path
import sys
import time
import argparse
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.fineweb_dataset import FineWebDataset
from src.transformer.domain_datasets import create_domain_dataset
from src.transformer.curriculum import CurriculumScheduler
from src.transformer.forgetting_metrics import ForgettingDetector, ForgettingMetrics
from src.transformer.scheduler import get_cosine_schedule_with_warmup
from src.transformer.perplexity import calculate_perplexity_from_loss
from src.transformer.training_utils import GradientAccumulator
from src.transformer.device_utils import (
    init_device, get_autocast_context, get_synchronize_fn,
    get_memory_stats_fn, print_device_info
)
from src.transformer.checkpoint_utils import (
    load_checkpoint, get_encoding_short_name, strip_compile_prefix
)


def get_device_type_from_args(use_mps: bool) -> str:
    """Convert boolean MPS flag to device type string for init_device."""
    return "mps" if use_mps else "auto"


def load_base_checkpoint(checkpoint_path: Path, device, console):
    """
    Load pre-trained base model checkpoint.

    This is different from resuming mid-training - we're starting fresh
    mid-training from a completed pre-training checkpoint.

    Returns:
        model: Loaded model
        encoding: Tokenizer encoding name
        config: Model configuration
    """
    console.print(f"[bold cyan]Loading base model:[/bold cyan] {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    # Extract configuration
    config = checkpoint.get('config', {})

    # Get model hyperparameters
    vocab_size = config.get('vocab_size', 100277)
    d_model = config.get('d_model', 256)
    num_heads = config.get('num_heads', 4)
    num_layers = config.get('num_layers', 6)
    d_ff = config.get('d_ff', d_model * 4)
    max_seq_len = config.get('max_seq_len', 5000)
    dropout = config.get('dropout', 0.1)
    position_encoding_type = config.get('position_encoding_type', 'alibi')
    encoding = config.get('encoding', 'cl100k_base')

    # Create model with same architecture
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
        position_encoding_type=position_encoding_type
    )

    # Load pre-trained weights
    state_dict = checkpoint['model_state_dict']
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        console.print("[yellow]Detected torch.compile() checkpoint, stripping prefix...[/yellow]")
        state_dict = strip_compile_prefix(state_dict)

    model.load_state_dict(state_dict)
    model = model.to(device)

    # Display base model info
    info_table = Table(title="Base Model Info", show_header=True, header_style="bold green")
    info_table.add_column("Property", style="cyan", no_wrap=True)
    info_table.add_column("Value", style="white")

    info_table.add_row("Pre-training epochs", str(checkpoint.get('epoch', 'Unknown')))
    info_table.add_row("Pre-training loss", f"{checkpoint.get('train_loss', 0):.4f}")
    info_table.add_row("Pre-training perplexity", f"{checkpoint.get('train_perplexity', 0):.2f}")

    # Model architecture
    info_table.add_row("", "")  # Separator
    info_table.add_row("Layers", str(num_layers))
    info_table.add_row("Model dimension", str(d_model))
    info_table.add_row("Attention heads", str(num_heads))
    info_table.add_row("FFN dimension", str(d_ff))
    info_table.add_row("Position encoding", position_encoding_type.upper())
    info_table.add_row("Vocabulary size", f"{vocab_size:,}")
    info_table.add_row("Tokenizer", encoding)

    console.print(info_table)
    console.print()

    return model, encoding, config


def midtrain(
    base_checkpoint: str,
    domain: str = 'code',
    tokens_per_epoch: int = 50_000_000,
    num_epochs: int = 10,
    accumulation_steps: int = 16,
    batch_size: int = 8,
    seq_length: int = 128,
    debug: bool = False,
    use_mps: bool = False,
    compile: bool = True,
    domain_mix_ratio: float = 0.9,
    # Domain-specific arguments
    languages: list = None,  # For code domain
    difficulty_range: tuple = (1, 5),  # For math domain
    fields: list = None,  # For science domain
):
    """
    Run mid-training (continued pre-training) for domain specialization.

    Args:
        base_checkpoint: Path to pre-trained checkpoint
        domain: Domain to specialize in ('code', 'math', 'science')
        tokens_per_epoch: Tokens per training epoch
        num_epochs: Number of epochs to train
        accumulation_steps: Gradient accumulation steps
        batch_size: Batch size
        seq_length: Sequence length
        debug: Enable debug mode (MPS synchronization)
        use_mps: Use MPS device
        compile: Use torch.compile()
        domain_mix_ratio: Ratio of domain data (0.9 = 90% domain, 10% general)
        languages: Programming languages for code domain
        difficulty_range: Difficulty range for math domain
        fields: Scientific fields for science domain
    """
    # Configuration
    CHECKPOINT_DIR = Path("checkpoints/midtrain")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    # Mid-training uses lower learning rate to preserve general knowledge
    LEARNING_RATE = 1e-5  # 30x lower than pre-training (3e-4)
    WEIGHT_DECAY = 0.01
    WARMUP_STEPS = 100
    LOG_INTERVAL = 10
    DROPOUT = 0.1  # Match pre-training

    # Initialize Rich console
    console = Console()

    # Display header
    console.print(Panel(
        "[bold]MID-TRAINING[/bold]\n"
        f"Domain: {domain.upper()}\n"
        "Specializing pre-trained model for domain expertise",
        style="bold magenta",
        expand=False
    ))
    console.print()

    # Device setup
    device_type_str = get_device_type_from_args(use_mps)
    device, device_name = init_device(device_type_str, seed=42)

    autocast_ctx = get_autocast_context(device.type)
    synchronize = get_synchronize_fn(device.type)
    get_max_memory = get_memory_stats_fn(device.type)

    # Load base model
    base_checkpoint_path = Path(base_checkpoint)
    if not base_checkpoint_path.exists():
        console.print(f"[bold red]Error:[/bold red] Base checkpoint not found: {base_checkpoint}")
        return

    model, encoding, base_config = load_base_checkpoint(base_checkpoint_path, device, console)

    # Get vocab size from model
    vocab_size = model.token_embedding.embedding.num_embeddings

    # Setup domain-specific dataset
    console.print(f"[bold]Setting up {domain.upper()} domain dataset...[/bold]")

    # Create domain dataset with appropriate parameters
    domain_kwargs = {
        'encoding_name': encoding,
        'max_seq_len': seq_length,
        'seed': 42,
    }

    if domain == 'code':
        domain_kwargs['languages'] = languages or ['python', 'javascript']
    elif domain == 'math':
        domain_kwargs['difficulty_range'] = difficulty_range
    elif domain == 'science':
        domain_kwargs['fields'] = fields or ['general']

    domain_dataset = create_domain_dataset(domain, **domain_kwargs)

    # Prepare domain data
    console.print(f"\n[bold]Preparing domain data ({domain})...[/bold]")
    domain_train_data = domain_dataset.prepare_dataset(
        num_tokens=int(tokens_per_epoch * domain_mix_ratio),
        split='train'
    )
    domain_val_data = domain_dataset.prepare_dataset(
        num_tokens=int((tokens_per_epoch // 10) * domain_mix_ratio),
        split='validation'
    )

    # Setup general dataset (for preventing catastrophic forgetting)
    console.print("\n[bold]Preparing general data (FineWeb) for forgetting prevention...[/bold]")
    general_tokens = int(tokens_per_epoch * (1 - domain_mix_ratio))
    general_train_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=seq_length,
        tokens_per_epoch=general_tokens,
        max_cached_shards=10,
        seed=43,  # Different seed from domain
        encoding_name=encoding,
        split="train",
        validation_fraction=0.1
    )

    general_val_tokens = int((tokens_per_epoch // 10) * (1 - domain_mix_ratio))
    general_val_dataset = FineWebDataset(
        cache_dir="data/fineweb_cache",
        seq_length=seq_length,
        tokens_per_epoch=general_val_tokens,
        max_cached_shards=10,
        seed=43,
        encoding_name=encoding,
        split="validation",
        validation_fraction=0.1
    )

    # Wrap prepared datasets for iteration
    # Convert HuggingFace datasets to our format
    from src.transformer.dataset import TokenizedDataset

    console.print()
    console.print(f"[bold]Creating mixed dataloaders ({domain_mix_ratio*100:.0f}% domain, {(1-domain_mix_ratio)*100:.0f}% general)...[/bold]")

    # Create tokenized datasets from prepared data
    if len(domain_train_data) > 0:
        domain_train_tokenized = TokenizedDataset(
            texts=[ex['text'] for ex in domain_train_data],
            encoding_name=encoding,
            max_seq_len=seq_length
        )
        domain_val_tokenized = TokenizedDataset(
            texts=[ex['text'] for ex in domain_val_data],
            encoding_name=encoding,
            max_seq_len=seq_length
        )
    else:
        # Empty datasets - fallback
        domain_train_tokenized = None
        domain_val_tokenized = None
        console.print("[yellow]Warning: Domain dataset is empty (datasets library may not be available)[/yellow]")

    # Create dataloaders
    if domain_train_tokenized:
        train_dataloader_domain = DataLoader(
            domain_train_tokenized,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        val_dataloader_domain = DataLoader(
            domain_val_tokenized,
            batch_size=batch_size,
            shuffle=False,
            drop_last=True
        )
    else:
        train_dataloader_domain = None
        val_dataloader_domain = None

    train_dataloader_general = DataLoader(
        general_train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    val_dataloader_general = DataLoader(
        general_val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True
    )

    # Display dataset configuration
    dataset_table = Table(title="Dataset Configuration", show_header=True, header_style="bold green")
    dataset_table.add_column("Component", style="cyan", no_wrap=True)
    dataset_table.add_column("Train", style="green", justify="right")
    dataset_table.add_column("Validation", style="yellow", justify="right")

    dataset_table.add_row(
        f"{domain.upper()} domain",
        f"{int(tokens_per_epoch * domain_mix_ratio):,} tokens ({domain_mix_ratio*100:.0f}%)",
        f"{int((tokens_per_epoch // 10) * domain_mix_ratio):,} tokens"
    )
    dataset_table.add_row(
        "General (FineWeb)",
        f"{general_tokens:,} tokens ({(1-domain_mix_ratio)*100:.0f}%)",
        f"{general_val_tokens:,} tokens"
    )
    dataset_table.add_row(
        "[bold]Total[/bold]",
        f"[bold]{tokens_per_epoch:,} tokens[/bold]",
        f"[bold]{tokens_per_epoch // 10:,} tokens[/bold]"
    )

    if domain == 'code':
        langs = languages or ['python', 'javascript']
        dataset_table.add_row("Languages", ", ".join(langs), "")
    elif domain == 'math':
        dataset_table.add_row("Difficulty", f"{difficulty_range[0]}-{difficulty_range[1]}", "")
    elif domain == 'science':
        sci_fields = fields or ['general']
        dataset_table.add_row("Fields", ", ".join(sci_fields), "")

    console.print(dataset_table)
    console.print()

    # Initialize curriculum scheduler
    console.print("[bold]Initializing curriculum learning...[/bold]")
    curriculum = CurriculumScheduler(num_epochs=num_epochs)
    console.print(f"[dim]  • {len(curriculum.stages)} stages: {' → '.join(s.name for s in curriculum.stages)}[/dim]")
    console.print()

    # Initialize forgetting detector
    console.print("[bold]Initializing catastrophic forgetting detection...[/bold]")
    forgetting_detector = ForgettingDetector()
    console.print("[dim]  • Dual evaluation: domain + general perplexity[/dim]")
    console.print("[dim]  • Alert threshold: >10% general perplexity increase[/dim]")
    console.print()

    # Compile model if requested
    if compile:
        console.print("[bold]Compiling model with torch.compile()...[/bold]")
        console.print("[dim]  • First epoch will be slower (compilation overhead)[/dim]")
        console.print("[dim]  • Subsequent epochs: 20-40% faster[/dim]")
        model = torch.compile(model)
        console.print()

    # Setup optimizer (AdamW with selective weight decay)
    decay_params = []
    no_decay_params = []

    # Access underlying model if compiled
    target_model = model._orig_mod if hasattr(model, '_orig_mod') else model

    for name, param in target_model.named_parameters():
        if 'bias' in name.lower() or 'norm' in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    optimizer = torch.optim.AdamW([
        {'params': decay_params, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ], lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8)

    # Calculate total training steps for scheduler
    steps_per_epoch = (tokens_per_epoch // seq_length) // batch_size // accumulation_steps
    total_steps = steps_per_epoch * num_epochs

    # Learning rate scheduler
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=WARMUP_STEPS,
        num_training_steps=total_steps
    )

    # Display training configuration
    train_config_table = Table(title="Mid-Training Configuration", show_header=True, header_style="bold magenta")
    train_config_table.add_column("Parameter", style="cyan", no_wrap=True)
    train_config_table.add_column("Value", style="white")

    train_config_table.add_row("Learning rate", f"{LEARNING_RATE:.2e} (30x lower than pre-training)")
    train_config_table.add_row("Warmup steps", str(WARMUP_STEPS))
    train_config_table.add_row("Weight decay", str(WEIGHT_DECAY))
    train_config_table.add_row("Epochs", str(num_epochs))
    train_config_table.add_row("Batch size", f"{batch_size} sequences")
    train_config_table.add_row("Gradient accumulation", f"{accumulation_steps} steps")
    train_config_table.add_row("Effective batch size", f"{batch_size * accumulation_steps} sequences")
    train_config_table.add_row("Sequence length", str(seq_length))
    train_config_table.add_row("Domain mix ratio", f"{domain_mix_ratio*100:.0f}% domain / {(1-domain_mix_ratio)*100:.0f}% general")
    train_config_table.add_row("Device", device_name)
    train_config_table.add_row("Checkpoint dir", str(CHECKPOINT_DIR))

    console.print(train_config_table)
    console.print()

    # Training loop
    console.print("[bold green]Starting mid-training...[/bold green]")
    console.print()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Gradient accumulator
    grad_accumulator = GradientAccumulator(accumulation_steps)

    for epoch in range(num_epochs):
        console.print(f"\n[bold cyan]{'='*80}[/bold cyan]")
        console.print(f"[bold cyan]EPOCH {epoch + 1}/{num_epochs}[/bold cyan]")
        console.print(f"[bold cyan]{'='*80}[/bold cyan]\n")

        # Get curriculum stage
        stage = curriculum.get_current_stage(epoch)
        stage_idx, total_stages, progress = curriculum.get_stage_progress(epoch)

        console.print(f"[bold]Curriculum Stage:[/bold] {stage.name} ({stage_idx}/{total_stages})")
        console.print(f"[dim]  • Difficulty: {stage.difficulty}[/dim]")
        console.print(f"[dim]  • Learning rate multiplier: {stage.lr_multiplier}x[/dim]")
        console.print(f"[dim]  • Progress: {progress:.1f}%[/dim]")
        console.print()

        # Adjust learning rate based on curriculum stage
        current_lr = LEARNING_RATE * stage.lr_multiplier
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        # Training phase
        model.train()
        total_loss = 0.0
        num_batches = 0
        grad_norm_sum = 0.0
        num_updates = 0

        console.print("[bold]Training on mixed data (domain + general)...[/bold]")

        # Mixed training: alternate between domain and general data
        # This ensures we don't forget general language while specializing

        # Create iterators
        if train_dataloader_domain:
            domain_iter = iter(train_dataloader_domain)
        general_iter = iter(train_dataloader_general)

        # Calculate how many batches we'll process
        # We want domain_mix_ratio proportion from domain, rest from general
        total_batches_target = steps_per_epoch * accumulation_steps
        domain_batches_target = int(total_batches_target * domain_mix_ratio) if train_dataloader_domain else 0
        general_batches_target = total_batches_target - domain_batches_target

        domain_batches_done = 0
        general_batches_done = 0

        while domain_batches_done < domain_batches_target or general_batches_done < general_batches_target:
            # Decide whether to use domain or general batch
            use_domain = (
                train_dataloader_domain and
                domain_batches_done < domain_batches_target and
                (general_batches_done >= general_batches_target or
                 domain_batches_done / max(domain_batches_target, 1) < general_batches_done / max(general_batches_target, 1))
            )

            try:
                if use_domain:
                    batch = next(domain_iter)
                    domain_batches_done += 1
                    batch_type = "domain"
                else:
                    batch = next(general_iter)
                    general_batches_done += 1
                    batch_type = "general"
            except StopIteration:
                # Restart iterator if exhausted
                if use_domain:
                    domain_iter = iter(train_dataloader_domain)
                    batch = next(domain_iter)
                    batch_type = "domain"
                else:
                    general_iter = iter(train_dataloader_general)
                    batch = next(general_iter)
                    batch_type = "general"

            inputs = batch.to(device)

            # Forward pass with autocast
            with autocast_ctx:
                logits = model(inputs[:, :-1])
                targets = inputs[:, 1:]

                # Reshape for cross entropy loss
                logits = logits.reshape(-1, vocab_size)
                targets = targets.reshape(-1)

                loss = criterion(logits, targets)
                loss = loss / accumulation_steps  # Scale loss for accumulation

            # Backward pass
            loss.backward()

            # Accumulate gradients
            if grad_accumulator.step():
                # Gradient clipping (prevent exploding gradients)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    target_model.parameters(),
                    max_norm=1.0
                )
                grad_norm_sum += grad_norm.item()

                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                num_updates += 1

                # Synchronize if debug mode (MPS stability)
                if debug and use_mps:
                    synchronize()

            total_loss += loss.item() * accumulation_steps  # Unscale for logging
            num_batches += 1

            # Log progress
            if num_batches % (LOG_INTERVAL * accumulation_steps) == 0:
                avg_loss = total_loss / num_batches
                avg_grad_norm = grad_norm_sum / max(num_updates, 1)
                current_lr = optimizer.param_groups[0]['lr']

                console.print(
                    f"  Batch {num_batches:,} ({batch_type}) | "
                    f"Loss: {avg_loss:.4f} | "
                    f"PPL: {calculate_perplexity_from_loss(avg_loss):.2f} | "
                    f"Grad: {avg_grad_norm:.3f} | "
                    f"LR: {current_lr:.2e} | "
                    f"Domain: {domain_batches_done}/{domain_batches_target} | "
                    f"General: {general_batches_done}/{general_batches_target}"
                )

        # Calculate average training loss
        train_loss = total_loss / num_batches
        train_perplexity = calculate_perplexity_from_loss(train_loss)
        avg_grad_norm = grad_norm_sum / max(num_updates, 1)

        console.print()
        console.print(f"[green]Training Loss: {train_loss:.4f}[/green]")
        console.print(f"[green]Training Perplexity: {train_perplexity:.2f}[/green]")
        console.print(f"[dim]Average Grad Norm: {avg_grad_norm:.3f}[/dim]")
        console.print()

        # After training epoch, evaluate on both domain and general data
        console.print("[bold]Evaluating (dual evaluation for forgetting detection)...[/bold]")

        model.eval()

        # Evaluate on domain data
        if val_dataloader_domain:
            domain_val_loss = 0.0
            domain_val_batches = 0

            with torch.no_grad():
                for batch in val_dataloader_domain:
                    inputs = batch.to(device)

                    with autocast_ctx:
                        logits = model(inputs[:, :-1])
                        targets = inputs[:, 1:]

                        logits = logits.reshape(-1, vocab_size)
                        targets = targets.reshape(-1)

                        loss = criterion(logits, targets)

                    domain_val_loss += loss.item()
                    domain_val_batches += 1

            domain_val_loss = domain_val_loss / domain_val_batches
            domain_perplexity = calculate_perplexity_from_loss(domain_val_loss)
        else:
            domain_val_loss = 0.0
            domain_perplexity = 0.0

        # Evaluate on general data
        general_val_loss = 0.0
        general_val_batches = 0

        with torch.no_grad():
            for batch in val_dataloader_general:
                inputs = batch.to(device)

                with autocast_ctx:
                    logits = model(inputs[:, :-1])
                    targets = inputs[:, 1:]

                    logits = logits.reshape(-1, vocab_size)
                    targets = targets.reshape(-1)

                    loss = criterion(logits, targets)

                general_val_loss += loss.item()
                general_val_batches += 1

        general_val_loss = general_val_loss / general_val_batches
        general_perplexity = calculate_perplexity_from_loss(general_val_loss)

        console.print(f"[green]Domain ({domain}) perplexity: {domain_perplexity:.2f}[/green]")
        console.print(f"[yellow]General (FineWeb) perplexity: {general_perplexity:.2f}[/yellow]")
        console.print()

        # Record forgetting metrics
        metrics = ForgettingMetrics(
            epoch=epoch + 1,
            domain_perplexity=domain_perplexity,
            general_perplexity=general_perplexity,
            domain_loss=domain_val_loss,
            general_loss=general_val_loss
        )
        forgetting_detector.record(metrics)

        # Check for catastrophic forgetting
        if forgetting_detector.is_forgetting(threshold=0.10):
            console.print("[bold red]⚠ CATASTROPHIC FORGETTING DETECTED![/bold red]")
            console.print("[yellow]General perplexity has degraded >10% from baseline[/yellow]")
            console.print()

            recommendations = forgetting_detector.get_recommendations()
            console.print("[bold]Recommendations:[/bold]")
            for i, rec in enumerate(recommendations, 1):
                console.print(f"  {i}. {rec}")
            console.print()

        # Save checkpoint
        checkpoint_name = f"{domain}_epoch_{epoch+1}_{get_encoding_short_name(encoding)}.pt"
        checkpoint_path = CHECKPOINT_DIR / checkpoint_name

        # Get model state dict (handle compiled models)
        target_model = model._orig_mod if hasattr(model, '_orig_mod') else model

        # Save comprehensive checkpoint
        checkpoint_data = {
            'epoch': epoch + 1,
            'model_state_dict': target_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'train_perplexity': train_perplexity,
            'domain_val_loss': domain_val_loss,
            'domain_val_perplexity': domain_perplexity,
            'general_val_loss': general_val_loss,
            'general_val_perplexity': general_perplexity,
            'current_lr': optimizer.param_groups[0]['lr'],
            'avg_grad_norm': avg_grad_norm,
            'config': {
                'stage': 'midtrain',
                'domain': domain,
                'base_checkpoint': str(base_checkpoint_path),
                'vocab_size': vocab_size,
                'd_model': base_config.get('d_model', 256),
                'num_heads': base_config.get('num_heads', 4),
                'num_layers': base_config.get('num_layers', 6),
                'd_ff': base_config.get('d_ff'),
                'max_seq_len': base_config.get('max_seq_len', 5000),
                'dropout': DROPOUT,
                'position_encoding_type': base_config.get('position_encoding_type', 'alibi'),
                'encoding': encoding,
                'seq_length': seq_length,
                'batch_size': batch_size,
                'accumulation_steps': accumulation_steps,
                'learning_rate': LEARNING_RATE,
                'weight_decay': WEIGHT_DECAY,
                'domain_mix_ratio': domain_mix_ratio,
                'num_epochs': num_epochs,
                'tokens_per_epoch': tokens_per_epoch,
            },
            'forgetting_metrics': {
                'baseline_general_perplexity': forgetting_detector.baseline_perplexity,
                'current_general_perplexity': general_perplexity,
                'is_forgetting': forgetting_detector.is_forgetting(threshold=0.10),
            }
        }

        # Add domain-specific config
        if domain == 'code':
            checkpoint_data['config']['languages'] = languages or ['python', 'javascript']
        elif domain == 'math':
            checkpoint_data['config']['difficulty_range'] = difficulty_range
        elif domain == 'science':
            checkpoint_data['config']['fields'] = fields or ['general']

        torch.save(checkpoint_data, checkpoint_path)
        console.print(f"[green]✓ Checkpoint saved:[/green] {checkpoint_path}")
        console.print()

    console.print("[bold green]Mid-training complete![/bold green]")
    console.print(f"\n[dim]Checkpoints saved to: {CHECKPOINT_DIR}[/dim]")


def main():
    """Parse arguments and run mid-training."""
    parser = argparse.ArgumentParser(
        description="Mid-training (continued pre-training) for domain specialization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Code specialization (Python, JavaScript)
  python -m commands.midtrain checkpoints/pretrain/model_epoch_10.pt --domain code --languages python javascript

  # Math specialization (easy to medium difficulty)
  python -m commands.midtrain checkpoints/pretrain/model_epoch_10.pt --domain math --difficulty 1 3

  # Science specialization (physics and CS)
  python -m commands.midtrain checkpoints/pretrain/model_epoch_10.pt --domain science --fields physics cs
        """
    )

    parser.add_argument(
        'base_checkpoint',
        type=str,
        help="Path to pre-trained base checkpoint (from Stage 1: pre-training)"
    )

    parser.add_argument(
        '--domain',
        type=str,
        choices=['code', 'math', 'science'],
        default='code',
        help="Domain to specialize in (default: code)"
    )

    parser.add_argument(
        '--tokens-per-epoch',
        type=int,
        default=50_000_000,
        help="Total tokens per epoch (domain + general, default: 50M)"
    )

    parser.add_argument(
        '--epochs',
        type=int,
        default=10,
        help="Number of training epochs (default: 10)"
    )

    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help="Batch size (default: 8)"
    )

    parser.add_argument(
        '--accumulation-steps',
        type=int,
        default=16,
        help="Gradient accumulation steps (default: 16)"
    )

    parser.add_argument(
        '--seq-length',
        type=int,
        default=128,
        help="Sequence length (default: 128)"
    )

    parser.add_argument(
        '--domain-mix-ratio',
        type=float,
        default=0.9,
        help="Ratio of domain data (0.9 = 90%% domain, 10%% general, default: 0.9)"
    )

    # Domain-specific arguments
    parser.add_argument(
        '--languages',
        type=str,
        nargs='+',
        default=['python', 'javascript'],
        help="Programming languages for code domain (default: python javascript)"
    )

    parser.add_argument(
        '--difficulty',
        type=int,
        nargs=2,
        metavar=('MIN', 'MAX'),
        default=[1, 5],
        help="Difficulty range for math domain (1-5, default: 1 5)"
    )

    parser.add_argument(
        '--fields',
        type=str,
        nargs='+',
        default=['general'],
        help="Scientific fields for science domain (default: general)"
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help="Enable debug mode (forces MPS synchronization, slower but more stable)"
    )

    parser.add_argument(
        '--mps',
        action='store_true',
        help="Use MPS device (Apple Silicon GPU) - experimental, may have NaN issues"
    )

    parser.add_argument(
        '--no-compile',
        action='store_true',
        help="Disable torch.compile() optimization (slower but more compatible)"
    )

    args = parser.parse_args()

    midtrain(
        base_checkpoint=args.base_checkpoint,
        domain=args.domain,
        tokens_per_epoch=args.tokens_per_epoch,
        num_epochs=args.epochs,
        accumulation_steps=args.accumulation_steps,
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        debug=args.debug,
        use_mps=args.mps,
        compile=not args.no_compile,
        domain_mix_ratio=args.domain_mix_ratio,
        languages=args.languages,
        difficulty_range=tuple(args.difficulty),
        fields=args.fields,
    )


if __name__ == '__main__':
    main()
