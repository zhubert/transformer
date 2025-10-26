"""
Text dataset for transformer training.

This module handles loading text data, tokenization, and creating training batches
for autoregressive language model training.

What is Tokenization?
---------------------
Tokenization is the process of converting text into numbers that the model can process.

    Text:   "The cat sat"
    Tokens: [464, 8415, 7731]  ← Numbers representing pieces of text

Why can't we just use characters or words?
- Characters: Too many sequences, models learn slowly
- Words: Vocabulary too large (millions of words), rare words are problems
- Subwords (BPE): Perfect balance! ✓

What is BPE (Byte Pair Encoding)?
----------------------------------
BPE is a subword tokenization algorithm used by GPT models. It breaks text into
common subword units rather than full words or individual characters.

Examples of BPE tokenization:
    "playing"     → ["play", "ing"]     ← Common word parts
    "unbelievable" → ["un", "believ", "able"]  ← Prefix + root + suffix
    "cat"         → ["cat"]             ← Common words stay whole
    "supercalifragilisticexpialidocious" → ["super", "cal", "if", "rag", ...]

Why BPE?
- Handles unknown words by breaking into known pieces
- Smaller vocabulary than word-level (~50k vs millions)
- More efficient than character-level
- This is what GPT-2, GPT-3, GPT-4 all use!

We use tiktoken's "cl100k_base" encoding (same as GPT-3.5/GPT-4).

What is Autoregressive Training?
---------------------------------
Autoregressive means predicting one token at a time, using all previous tokens.

Think of it like predicting the next word in a sentence:
    Given: "The cat sat on the"
    Predict: "mat"

For training, we create many examples from one sequence:

    Input sequence:  [The, cat, sat, on, the, mat]

    Training examples created:
    Input:  [The]           → Target: cat
    Input:  [The, cat]      → Target: sat
    Input:  [The, cat, sat] → Target: on
    Input:  [The, cat, sat, on] → Target: the
    Input:  [The, cat, sat, on, the] → Target: mat

We get MULTIPLE training examples from one sequence! Very efficient.

Why Shift Targets by 1?
------------------------
This is the key to autoregressive training!

    Input tokens:  [The, cat, sat, on, the, mat]
    Target tokens: [cat, sat, on, the, mat, <end>]
                    ↑
                    Shifted left by 1 position!

At each position, the model predicts the NEXT token:
    Position 0: Input="The"    → Predict "cat"  (target[0])
    Position 1: Input="cat"    → Predict "sat"  (target[1])
    Position 2: Input="sat"    → Predict "on"   (target[2])
    ...

The causal mask ensures position i can only see positions 0 to i (not future tokens).

How This Dataset Works:
------------------------
1. Load text file (e.g., a novel, Shakespeare, Wikipedia)
2. Tokenize entire text using tiktoken BPE
3. Split into chunks of fixed sequence length
4. Create (input, target) pairs where target = input shifted by 1
5. Batch multiple sequences together for efficient training

Example:
    Text: "The cat sat on the mat. The dog ran."
        ↓ Tokenize
    Tokens: [464, 8415, 7731, 389, 279, 5634, 13, 578, 5679, 10837]
        ↓ Create sequence (length=5)
    Input:  [464, 8415, 7731, 389, 279]  "The cat sat on the"
    Target: [8415, 7731, 389, 279, 5634] "cat sat on the mat"
                                          ↑ each is next token

The model learns to predict what comes next at every position!
"""

import torch
from torch.utils.data import Dataset
import tiktoken
from pathlib import Path


class TextDataset(Dataset):
    """
    Dataset for autoregressive language model training.

    Loads a text file, tokenizes it using BPE, and creates sequences for training.
    Each sequence is paired with its target (shifted by 1 position).
    """

    def __init__(self, text_file_path, seq_length=128, encoding_name="p50k_base"):
        """
        Initialize text dataset.

        Args:
            text_file_path: Path to text file for training
            seq_length: Length of each training sequence (default 128)
                       Longer = more context but slower training
                       Shorter = less context but faster training
            encoding_name: tiktoken encoding to use (default "p50k_base")
                          "p50k_base" is used by GPT-3 and Codex (~50k vocab)
                          "cl100k_base" is used by GPT-3.5/GPT-4 (~100k vocab, larger)
        """
        self.seq_length = seq_length

        # Load tiktoken BPE tokenizer (same as GPT models use)
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Read the text file
        text_path = Path(text_file_path)
        if not text_path.exists():
            raise FileNotFoundError(f"Text file not found: {text_file_path}")

        with open(text_path, 'r', encoding='utf-8') as f:
            text = f.read()

        print(f"Loaded text file: {text_file_path}")
        print(f"Text length: {len(text):,} characters")

        # Tokenize the entire text using BPE
        # This converts the text into a list of token IDs
        self.tokens = self.tokenizer.encode(text)
        print(f"Tokenized into {len(self.tokens):,} tokens")
        print(f"Vocabulary size: {self.tokenizer.n_vocab:,} tokens")

        # Calculate how many sequences we can create
        # We need seq_length + 1 tokens for each sequence (input + target)
        self.num_sequences = (len(self.tokens) - 1) // seq_length
        print(f"Created {self.num_sequences:,} training sequences of length {seq_length}")

    def __len__(self):
        """Return the number of training sequences."""
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get a training example.

        Returns:
            input_seq: Token IDs for input sequence (length: seq_length)
            target_seq: Token IDs for target sequence (length: seq_length)
                       Same as input_seq but shifted left by 1 position

        Example:
            If tokens are [10, 20, 30, 40, 50] and seq_length=4:
            input_seq:  [10, 20, 30, 40]
            target_seq: [20, 30, 40, 50]
                         ↑  Each position predicts next token
        """
        # Get starting position for this sequence
        start_idx = idx * self.seq_length

        # Extract input and target sequences
        # Input: tokens[start : start + seq_length]
        # Target: tokens[start + 1 : start + seq_length + 1]
        #         (shifted by 1 to the right)
        input_seq = self.tokens[start_idx : start_idx + self.seq_length]
        target_seq = self.tokens[start_idx + 1 : start_idx + self.seq_length + 1]

        # Convert to PyTorch tensors
        input_tensor = torch.tensor(input_seq, dtype=torch.long)
        target_tensor = torch.tensor(target_seq, dtype=torch.long)

        return input_tensor, target_tensor

    def decode(self, token_ids):
        """
        Convert token IDs back to text.

        Useful for seeing what the model generates!

        Args:
            token_ids: List or tensor of token IDs

        Returns:
            text: Decoded text string
        """
        # Convert tensor to list if needed
        if torch.is_tensor(token_ids):
            token_ids = token_ids.tolist()

        return self.tokenizer.decode(token_ids)

    @property
    def vocab_size(self):
        """Return the vocabulary size of the tokenizer."""
        return self.tokenizer.n_vocab
