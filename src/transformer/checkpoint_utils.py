"""
Checkpoint loading and saving utilities.

This module provides utilities for working with model checkpoints, consolidating
logic that was previously duplicated across multiple command files.

Why Separate Checkpoint Utilities?
-----------------------------------
Before this module, checkpoint loading logic was duplicated in 8 different files:
- train.py, generate.py, evaluate_perplexity.py, interpret.py, etc.

This caused several problems:
1. **Maintenance burden**: Bug fixes needed to be applied in 8 places
2. **Inconsistency**: Different files handled edge cases differently
3. **Code duplication**: ~120 lines of nearly identical code
4. **Discoverability**: No single source of truth for "how to load a checkpoint"

By centralizing checkpoint utilities here:
- ✅ Single source of truth for checkpoint handling
- ✅ Easier to maintain and test
- ✅ Consistent behavior across all commands
- ✅ Better discoverability for new developers

Common Checkpoint Issues Handled
---------------------------------
1. **torch.compile() prefix**: Compiled models add '_orig_mod.' prefix to all keys
2. **Missing max_seq_len**: Older checkpoints don't have max_seq_len in config
3. **Encoding detection**: Need backward compatibility with old checkpoints
4. **Device handling**: Loading on different devices (CPU, CUDA, MPS)

Educational Philosophy
----------------------
This module follows the transformer project's educational principles:
- Comprehensive documentation explaining the "why" not just the "what"
- Clear function names that describe intent
- Explicit handling of edge cases with comments
- Examples showing common usage patterns
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Dict, Any, Tuple, Optional


def detect_encoding(checkpoint: Dict[str, Any]) -> str:
    """
    Detect tokenizer encoding from checkpoint.

    Handles backward compatibility with older checkpoints that don't have
    encoding information stored.

    Why is this needed?
    -------------------
    Older checkpoints (before encoding was added to config) don't store the
    encoding name. We need to detect it or provide a sensible default.

    Args:
        checkpoint: Checkpoint dictionary loaded from torch.load()

    Returns:
        encoding: Encoding name (e.g., 'cl100k_base', 'o200k_base')

    Example:
        >>> checkpoint = torch.load('model.pt')
        >>> encoding = detect_encoding(checkpoint)
        >>> print(encoding)  # 'cl100k_base'
    """
    # Modern checkpoints store encoding in config
    if 'encoding' in checkpoint.get('config', {}):
        return checkpoint['config']['encoding']

    # Fallback: Always use cl100k_base for backward compatibility
    # This was the default encoding before we started storing it
    return 'cl100k_base'


def get_encoding_short_name(encoding: str) -> str:
    """
    Convert full encoding name to short version for filenames.

    Why abbreviate?
    ---------------
    Checkpoint filenames get long with full encoding names:
        model_epoch_10_cl100k_base.pt  (too long)
        model_epoch_10_cl100k.pt       (better!)

    Args:
        encoding: Full encoding name (e.g., 'cl100k_base')

    Returns:
        short_name: Abbreviated encoding name (e.g., 'cl100k')

    Example:
        >>> get_encoding_short_name('cl100k_base')
        'cl100k'
        >>> get_encoding_short_name('o200k_base')
        'o200k'
        >>> get_encoding_short_name('custom_encoding')
        'custom_encoding'
    """
    # Special case for cl100k_base (most common)
    if encoding == 'cl100k_base':
        return 'cl100k'

    # General case: remove '_base' suffix if present
    return encoding.replace('_base', '')


def strip_compile_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove torch.compile() prefix from state dict keys.

    What is the compile prefix?
    ----------------------------
    When you save a checkpoint from a compiled model:
        model = torch.compile(model)
        torch.save({'model_state_dict': model.state_dict()}, 'checkpoint.pt')

    All keys in the state dict get prefixed with '_orig_mod.':
        '_orig_mod.token_embedding.embedding.weight'
        '_orig_mod.pos_encoding.pos_embedding.weight'
        etc.

    This prefix must be stripped before loading into a non-compiled model.

    Why does this happen?
    ---------------------
    torch.compile() wraps your model in a compiled graph. The wrapper stores
    the original model as an attribute called '_orig_mod', hence the prefix.

    Args:
        state_dict: Model state dict from checkpoint['model_state_dict']

    Returns:
        cleaned_state_dict: State dict with prefix removed (if present)

    Example:
        >>> state_dict = {'_orig_mod.layer.weight': tensor([...])}
        >>> cleaned = strip_compile_prefix(state_dict)
        >>> print(cleaned.keys())
        dict_keys(['layer.weight'])
    """
    # Check if any keys have the compile prefix
    has_prefix = any(k.startswith('_orig_mod.') for k in state_dict.keys())

    if not has_prefix:
        # No prefix to strip, return as-is
        return state_dict

    # Strip '_orig_mod.' from all keys (only the first occurrence)
    # Using replace(..., 1) ensures we only replace the prefix, not any
    # '_orig_mod.' that might appear elsewhere in the key name
    cleaned_state_dict = {
        k.replace('_orig_mod.', '', 1): v
        for k, v in state_dict.items()
    }

    return cleaned_state_dict


def infer_max_seq_len(state_dict: Dict[str, torch.Tensor]) -> int:
    """
    Infer max_seq_len from positional embedding shape.

    Why is this needed?
    -------------------
    Older checkpoints don't store max_seq_len in the config dict. But we can
    infer it from the shape of the positional embedding weight matrix (for learned
    position embeddings) or use a sensible default (for ALiBi/RoPE).

    Position Encoding Types:
    - Learned: Has 'pos_encoding.pos_embedding.weight' with shape (max_seq_len, d_model)
    - ALiBi/RoPE: No learnable position parameters, use default max_seq_len=5000

    Args:
        state_dict: Model state dict (after stripping compile prefix)

    Returns:
        max_seq_len: Maximum sequence length the model supports

    Raises:
        KeyError: If positional embedding not found AND no ALiBi/RoPE buffers detected

    Example:
        >>> # Learned position embeddings
        >>> state_dict = {'pos_encoding.pos_embedding.weight': torch.randn(256, 512)}
        >>> max_seq_len = infer_max_seq_len(state_dict)
        >>> print(max_seq_len)  # 256

        >>> # ALiBi (no pos_embedding)
        >>> state_dict = {'alibi.slopes': torch.randn(8), 'token_embedding.embedding.weight': ...}
        >>> max_seq_len = infer_max_seq_len(state_dict)
        >>> print(max_seq_len)  # 5000 (default)
    """
    pos_embedding_key = 'pos_encoding.pos_embedding.weight'

    # Check if this is a learned position encoding checkpoint
    if pos_embedding_key in state_dict:
        # Shape is (max_seq_len, d_model), we want the first dimension
        pos_embedding_shape = state_dict[pos_embedding_key].shape
        max_seq_len = pos_embedding_shape[0]
        return max_seq_len

    # Check if this is ALiBi or RoPE (no learnable position parameters)
    # ALiBi has 'alibi.slopes' buffer
    # RoPE has 'rope.inv_freq' buffer
    # Note: These buffers are registered with persistent=False, so they may not be saved
    has_alibi = any(k.startswith('alibi.') for k in state_dict.keys())
    has_rope = any(k.startswith('rope.') for k in state_dict.keys())

    if has_alibi or has_rope:
        # ALiBi and RoPE don't have max_seq_len in state dict
        # Use default value of 5000 (standard for this implementation)
        return 5000

    # Backward compatibility: Old checkpoints used ALiBi (default) but buffers weren't saved
    # If we can't find learned embeddings OR ALiBi/RoPE buffers, assume ALiBi/RoPE
    # This handles checkpoints created before we added position_encoding_type to config
    # and where ALiBi/RoPE buffers weren't saved (persistent=False)
    return 5000


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device = torch.device('cpu'),
    load_training_state: bool = False,
    model: Optional[nn.Module] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Load a model checkpoint with all necessary preprocessing.

    This is the main checkpoint loading function that handles all common edge cases:
    - torch.compile() prefix stripping
    - Missing max_seq_len inference
    - Encoding detection
    - Optional training state loading (optimizer, scheduler)

    Usage Patterns
    --------------

    Pattern 1: Load for inference (generate, evaluate, interpret)
    ```python
    result = load_checkpoint('model.pt', device='cuda')
    model = result['model']
    config = result['config']
    encoding = result['encoding']
    ```

    Pattern 2: Resume training
    ```python
    result = load_checkpoint(
        'model.pt',
        device='cuda',
        load_training_state=True,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler
    )
    start_epoch = result['epoch'] + 1
    ```

    What gets loaded?
    -----------------
    Always loaded:
    - Model architecture and weights
    - Config (with max_seq_len inferred if missing)
    - Encoding detection
    - Epoch number

    Optionally loaded (if load_training_state=True):
    - Optimizer state
    - Scheduler state

    Args:
        checkpoint_path: Path to checkpoint file (.pt)
        device: Device to load model on ('cpu', 'cuda', 'mps')
        load_training_state: If True, load optimizer and scheduler state
        model: Model to load training state into (required if load_training_state=True)
        optimizer: Optimizer to load state into (optional)
        scheduler: Scheduler to load state into (optional)
        verbose: If True, print loading information

    Returns:
        result: Dictionary containing:
            - 'model': Loaded DecoderOnlyTransformer model
            - 'config': Model configuration dict
            - 'encoding': Detected encoding name
            - 'epoch': Epoch number from checkpoint
            - 'checkpoint': Full checkpoint dict
            - 'loss': Training loss (if available)
            - 'perplexity': Training perplexity (if available)

    Raises:
        FileNotFoundError: If checkpoint file doesn't exist
        RuntimeError: If load_training_state=True but model/optimizer/scheduler not provided

    Example:
        >>> # Load for generation
        >>> result = load_checkpoint('checkpoints/model_epoch_10.pt', device='cuda')
        >>> model = result['model']
        >>> model.eval()
        >>> # Generate text...

        >>> # Resume training
        >>> result = load_checkpoint(
        ...     'checkpoints/model_epoch_10.pt',
        ...     device='cuda',
        ...     load_training_state=True,
        ...     model=model,
        ...     optimizer=optimizer,
        ...     scheduler=scheduler
        ... )
        >>> start_epoch = result['epoch'] + 1
    """
    # Import here to avoid circular dependency
    from .model import DecoderOnlyTransformer

    # Validate inputs
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if load_training_state and model is None:
        raise RuntimeError(
            "load_training_state=True requires model to be provided. "
            "Did you forget to pass model=your_model?"
        )

    if verbose:
        print(f"Loading checkpoint: {checkpoint_path}")

    # Load checkpoint from disk
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract configuration
    config = checkpoint['config'].copy()  # Copy to avoid modifying checkpoint

    # Extract and clean state dict
    state_dict = checkpoint['model_state_dict']

    # Handle torch.compile() prefix
    if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
        if verbose:
            print("  Detected torch.compile() checkpoint, stripping prefix...")
        state_dict = strip_compile_prefix(state_dict)

    # Handle missing max_seq_len (backward compatibility)
    if 'max_seq_len' not in config:
        max_seq_len = infer_max_seq_len(state_dict)
        config['max_seq_len'] = max_seq_len
        if verbose:
            print(f"  Inferred max_seq_len from checkpoint: {max_seq_len}")

    # Detect encoding
    encoding = detect_encoding(checkpoint)

    # Detect position encoding type from state dict
    # Check config first (newer checkpoints), then infer from state dict
    if 'position_encoding_type' in config:
        position_encoding_type = config['position_encoding_type']
    else:
        # Infer from state dict for backward compatibility
        has_pos_embedding = 'pos_encoding.pos_embedding.weight' in state_dict
        has_alibi = any(k.startswith('alibi.') for k in state_dict.keys())
        has_rope = any(k.startswith('rope.') for k in state_dict.keys())

        if has_pos_embedding:
            # Learned embeddings are always saved in state dict
            position_encoding_type = 'learned'
        elif has_alibi:
            # ALiBi buffers found (though usually persistent=False)
            position_encoding_type = 'alibi'
        elif has_rope:
            # RoPE buffers found (though usually persistent=False)
            position_encoding_type = 'rope'
        else:
            # No position encoding found in state dict
            # Default to ALiBi (the current default, and most common for old checkpoints)
            # ALiBi/RoPE buffers are registered with persistent=False, so they're not saved
            position_encoding_type = 'alibi'

        if verbose:
            print(f"  Inferred position encoding type: {position_encoding_type}")

    # Create model if not provided (inference mode)
    if model is None:
        model = DecoderOnlyTransformer(
            vocab_size=config['vocab_size'],
            d_model=config['d_model'],
            num_heads=config['num_heads'],
            num_layers=config['num_layers'],
            d_ff=config['d_ff'],
            dropout=config['dropout'],
            max_seq_len=config.get('max_seq_len', 5000),
            tie_weights=config.get('tie_weights', True),
            position_encoding_type=position_encoding_type,
        )

    # Check if model is compiled (wrapped in OptimizedModule)
    is_compiled = hasattr(model, '_orig_mod')
    target_model = model._orig_mod if is_compiled else model

    # Load model weights into underlying model if compiled
    target_model.load_state_dict(state_dict)
    if is_compiled and verbose:
        print("  Loaded into underlying model (compiled model detected)")

    model = model.to(device)

    # Set to eval mode by default (can be changed to train() by caller)
    if not load_training_state:
        model.eval()

    # Load training state if requested
    if load_training_state:
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            if verbose:
                print("  Loaded optimizer state")

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if verbose:
                print("  Loaded scheduler state")

    # Print checkpoint info
    if verbose:
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Encoding: {encoding}")
        if 'loss' in checkpoint and checkpoint['loss'] is not None:
            print(f"  Training loss: {checkpoint['loss']:.4f}")
        if 'perplexity' in checkpoint and checkpoint['perplexity'] is not None:
            print(f"  Training perplexity: {checkpoint['perplexity']:.2f}")
        print()

    # Return comprehensive result
    result = {
        'model': model,
        'config': config,
        'encoding': encoding,
        'epoch': checkpoint['epoch'],
        'checkpoint': checkpoint,
    }

    # Add optional fields if present
    if 'loss' in checkpoint:
        result['loss'] = checkpoint['loss']
    if 'perplexity' in checkpoint:
        result['perplexity'] = checkpoint['perplexity']

    return result
