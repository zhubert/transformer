"""
Base dataset classes for transformer training.

This module provides abstract base classes that define the common interface
for all datasets used in the transformer training pipeline.

Educational Purpose:
===================
In production LLM training, you'll work with many different datasets:
- Pre-training: General web text (FineWeb, C4, RedPajama)
- Mid-training: Domain-specific (code, math, science)
- Fine-tuning: Task-specific (instructions, dialogues)

Rather than reimplementing common functionality in each dataset, we use
inheritance to share common code. This is the DRY (Don't Repeat Yourself)
principle in action.

Common Functionality:
====================
All datasets need:
1. Tokenization (convert text → token IDs)
2. Sequence length handling (truncate/pad to max_seq_len)
3. Batching (group sequences for efficient training)

By putting this in a base class, we ensure:
- Consistency: All datasets work the same way
- Maintainability: Fix bugs once, benefit everywhere
- Clarity: Dataset implementations focus on data loading, not boilerplate
"""

from abc import ABC
from torch.utils.data import IterableDataset
import tiktoken
from typing import Iterator, Tuple


class BaseDataset(ABC, IterableDataset):
    """
    Abstract base class for all transformer datasets.

    This class provides common initialization for tokenization and sequence
    handling. Subclasses must implement __iter__() to yield training data.

    Why ABC (Abstract Base Class)?
    ==============================
    We use Python's ABC module to make this an "interface" - a class that
    defines what methods subclasses MUST implement. This is common in large
    codebases to enforce consistency.

    Why IterableDataset?
    ====================
    PyTorch provides two dataset types:
    - Dataset: Random access (dataset[i]), needs known length
    - IterableDataset: Sequential access (for batch in dataset), no length needed

    For LLM training, we use IterableDataset because:
    - Data is often streamed (can't fit in memory)
    - We don't need random access (just iterate epoch by epoch)
    - More memory efficient for large datasets

    Inheritance Pattern:
    ===================
        BaseDataset (this class)
             ↓
        DomainDataset (adds domain-specific utilities)
             ↓
        CodeDataset, MathDataset, ScienceDataset

    Each level adds more specific functionality.
    """

    def __init__(
        self,
        encoding_name: str = 'cl100k_base',
        max_seq_len: int = 128,
    ):
        """
        Initialize common dataset parameters.

        Args:
            encoding_name: Tokenizer encoding to use
                - 'cl100k_base': ~100K vocab (GPT-4, default)
                - 'gpt2': ~50K vocab (GPT-2/GPT-3)
                - 'r50k_base': ~50K vocab (older models)
            max_seq_len: Maximum sequence length for training
                - Longer = more context but slower and more memory
                - Typical values: 128 (fast), 512 (standard), 2048 (large)

        What is tiktoken?
        =================
        tiktoken is OpenAI's fast BPE (Byte-Pair Encoding) tokenizer.
        It converts text into tokens (subword units):

        Example:
            "Hello world!" → [9906, 1917, 0]  (3 tokens)
            "transformer" → [4246, 261, 261]   (3 tokens, splits "trans-form-er")

        Why BPE?
        ========
        - Balances vocabulary size and text coverage
        - Handles rare words by breaking into subwords
        - No unknown tokens (can encode any Unicode text)
        - Used by GPT-2, GPT-3, GPT-4, LLaMA, etc.
        """
        super().__init__()
        self.encoding_name = encoding_name
        self.max_seq_len = max_seq_len
        self.encoding = tiktoken.get_encoding(encoding_name)

    def __iter__(self) -> Iterator[Tuple]:
        """
        Iterate over dataset batches.

        Subclasses must implement this method to yield training data.
        The exact format depends on the dataset type, but typically:
        - (input_ids, target_ids) for language modeling
        - (input_ids,) for some streaming datasets

        Raises:
            NotImplementedError: Subclasses must implement this method
        """
        raise NotImplementedError("Subclasses must implement __iter__()")
