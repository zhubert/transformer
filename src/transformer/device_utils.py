"""
Device utilities for PyTorch training.

Provides helpers for device detection, initialization, and context management
across CUDA, MPS (Apple Silicon), and CPU devices.

Based on patterns from nanochat project for robust multi-device support.
"""

import torch
from contextlib import nullcontext
from typing import Tuple


def autodetect_device() -> Tuple[torch.device, str]:
    """
    Automatically detect the best available device.

    Preference order: CUDA/ROCm > MPS > CPU

    Note: AMD GPUs with ROCm use torch.cuda API (HIP compatibility layer),
    so they appear as CUDA devices. We detect the actual GPU vendor for
    more informative messages.

    Returns:
        device: torch.device object
        device_name: Human-readable device description

    Examples:
        >>> device, name = autodetect_device()
        >>> print(f"Using {name}")
        Using CUDA (NVIDIA GPU)
        # or on AMD: Using CUDA (AMD GPU via ROCm)
    """
    if torch.cuda.is_available():
        # Detect if it's AMD (ROCm) or NVIDIA (CUDA)
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'amd' in gpu_name or 'radeon' in gpu_name:
            return torch.device("cuda"), "CUDA (AMD GPU via ROCm)"
        else:
            return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    elif torch.backends.mps.is_available():
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    else:
        return torch.device("cpu"), "CPU"


def get_device(device_type: str = None) -> Tuple[torch.device, str]:
    """
    Get a specific device or autodetect.

    Args:
        device_type: "cuda", "mps", "cpu", or None for autodetect

    Returns:
        device: torch.device object
        device_name: Human-readable device description

    Raises:
        RuntimeError: If requested device is not available

    Examples:
        >>> device, name = get_device("mps")
        >>> device, name = get_device()  # Autodetect
    """
    if device_type is None or device_type == "":
        return autodetect_device()

    device_type = device_type.lower()

    if device_type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available. Install CUDA-enabled PyTorch.")
        # Detect GPU vendor for informative message
        gpu_name = torch.cuda.get_device_name(0).lower()
        if 'amd' in gpu_name or 'radeon' in gpu_name:
            return torch.device("cuda"), "CUDA (AMD GPU via ROCm)"
        else:
            return torch.device("cuda"), "CUDA (NVIDIA GPU)"
    elif device_type == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS requested but not available. Requires macOS 12.3+ and Apple Silicon.")
        return torch.device("mps"), "MPS (Apple Silicon GPU)"
    elif device_type == "cpu":
        return torch.device("cpu"), "CPU"
    else:
        raise ValueError(f"Invalid device type: {device_type}. Must be 'cuda', 'mps', 'cpu', or None.")


def init_device(device_type: str = None, seed: int = 42) -> Tuple[torch.device, str]:
    """
    Initialize device with proper settings and seeding.

    This function:
    1. Selects the device (auto or manual)
    2. Sets random seeds for reproducibility
    3. Applies device-specific optimizations (e.g., TF32 for CUDA)

    Args:
        device_type: "cuda", "mps", "cpu", or None for autodetect
        seed: Random seed for reproducibility

    Returns:
        device: torch.device object
        device_name: Human-readable device description

    Examples:
        >>> device, name = init_device("mps", seed=42)
        >>> print(f"Initialized {name}")
        Initialized MPS (Apple Silicon GPU)
    """
    device, device_name = get_device(device_type)

    # Set random seeds for reproducibility
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
        # Enable TF32 for better performance on Ampere+ GPUs (A100, RTX 30xx/40xx)
        # This trades tiny precision loss for ~3x speedup on matmuls
        torch.set_float32_matmul_precision("high")
    # Note: MPS doesn't need explicit seeding beyond torch.manual_seed()

    return device, device_name


def get_autocast_context(device_type: str):
    """
    Get the appropriate autocast context for mixed precision training.

    IMPORTANT: Autocast is ONLY used for CUDA. MPS and CPU use nullcontext.

    Why?
    - CUDA: Supports bfloat16 mixed precision for ~2x speedup
    - MPS: No proper mixed precision support yet (PyTorch limitation)
    - CPU: No benefit from mixed precision

    Args:
        device_type: "cuda", "mps", or "cpu" (from device.type)

    Returns:
        Context manager for autocast (or nullcontext for MPS/CPU)

    Examples:
        >>> device = torch.device("cuda")
        >>> autocast_ctx = get_autocast_context(device.type)
        >>> with autocast_ctx:
        ...     output = model(input)  # Runs in mixed precision on CUDA

    Usage in training:
        >>> device, _ = init_device("cuda")
        >>> autocast_ctx = get_autocast_context(device.type)
        >>> for batch in dataloader:
        ...     with autocast_ctx:
        ...         logits = model(inputs)
        ...         loss = criterion(logits, targets)
        ...     loss.backward()
    """
    if device_type == "cuda":
        return torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
    else:
        # MPS and CPU: use nullcontext (no-op context manager)
        return nullcontext()


def get_synchronize_fn(device_type: str):
    """
    Get device-specific synchronization function for accurate timing.

    Synchronization ensures all GPU operations complete before continuing.
    This is critical for accurate performance measurements.

    Args:
        device_type: "cuda", "mps", or "cpu"

    Returns:
        Function to synchronize device (no-op for CPU/MPS)

    Examples:
        >>> device = torch.device("cuda")
        >>> sync = get_synchronize_fn(device.type)
        >>> sync()  # Wait for all CUDA ops to finish
        >>> start = time.time()
        >>> # ... run operations ...
        >>> sync()  # Ensure complete before measuring
        >>> elapsed = time.time() - start
    """
    if device_type == "cuda":
        return torch.cuda.synchronize
    else:
        # MPS and CPU don't need explicit synchronization
        return lambda: None


def get_memory_stats_fn(device_type: str):
    """
    Get device-specific memory statistics function.

    Args:
        device_type: "cuda", "mps", or "cpu"

    Returns:
        Function that returns peak memory allocated in bytes

    Examples:
        >>> device = torch.device("cuda")
        >>> get_memory = get_memory_stats_fn(device.type)
        >>> # ... train model ...
        >>> peak_memory_mb = get_memory() / (1024 * 1024)
        >>> print(f"Peak GPU memory: {peak_memory_mb:.1f} MB")
    """
    if device_type == "cuda":
        return torch.cuda.max_memory_allocated
    else:
        # MPS and CPU: no built-in memory tracking in PyTorch
        return lambda: 0


def print_device_info(device: torch.device):
    """
    Print detailed information about the device.

    Args:
        device: torch.device object

    Examples:
        >>> device, _ = init_device()
        >>> print_device_info(device)
        Device: CUDA (NVIDIA GPU)
        CUDA Device: NVIDIA GeForce RTX 3090
        CUDA Version: 11.8
    """
    print(f"Device: {device}")

    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(0)
        print(f"  GPU Device: {device_name}")

        # Detect vendor
        if 'amd' in device_name.lower() or 'radeon' in device_name.lower():
            print(f"  Backend: ROCm (AMD)")
            print(f"  ROCm Version: {torch.version.hip if hasattr(torch.version, 'hip') else 'Unknown'}")
        else:
            print(f"  Backend: CUDA (NVIDIA)")
            print(f"  CUDA Version: {torch.version.cuda}")

        total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  Total Memory: {total_memory:.1f} GB")
    elif device.type == "mps":
        print(f"  MPS Backend: Available")
        print(f"  Note: MPS does not support mixed precision autocast")
    else:
        print(f"  CPU Backend: Available")

    print(f"  PyTorch Version: {torch.__version__}")
