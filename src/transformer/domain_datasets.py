#!/usr/bin/env python3
"""
Domain-specific datasets for mid-training (continued pre-training).

Mid-training specializes a pre-trained base model for specific domains like code,
math, or science. This is where models like Codex (from GPT-3) or AlphaCode emerge.

Key Differences from Pre-training:
1. **Curated data**: Quality matters more than quantity
2. **Domain focus**: Specialized content (e.g., Python code, math proofs)
3. **Smaller scale**: Millions-billions of tokens, not tens of billions
4. **Data mixing**: Blend domain data with general data to prevent forgetting

Educational Purpose:
This module demonstrates how production LLMs gain domain expertise while
maintaining general capabilities. The same approach is used by:
- Codex (OpenAI) - GPT-3 specialized on code
- Minerva (Google) - PaLM specialized on math/science
- Galactica (Meta) - LLaMA specialized on scientific literature
- Code Llama - Llama 2 specialized on code

References:
- "Training Compute-Optimal Large Language Models" (Chinchilla, 2022)
- "Solving Quantitative Reasoning Problems with Language Models" (Minerva, 2022)
- "Code Llama: Open Foundation Models for Code" (Meta, 2023)
"""

from pathlib import Path
from typing import Iterator, Optional, Dict, Any, Literal
import json
import random

import torch
from datasets import load_dataset, Dataset
import tiktoken

from .dataset import BaseDataset


class DomainDataset(BaseDataset):
    """
    Base class for domain-specific datasets used in mid-training.

    Mid-training uses the same next-token prediction objective as pre-training,
    but with focused, high-quality domain data. The key is balancing:
    - Domain expertise (learning specialized patterns)
    - General capability retention (not forgetting general language)

    This is achieved through:
    1. Lower learning rate (1e-5 vs 3e-4 for pre-training)
    2. Data mixing (90% domain, 10% general)
    3. Dual evaluation (track both domain and general perplexity)
    """

    def __init__(
        self,
        domain: str,
        encoding_name: str = 'cl100k_base',
        max_seq_len: int = 128,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize domain-specific dataset.

        Args:
            domain: Domain name (for logging/identification)
            encoding_name: Tokenizer encoding (default: cl100k_base ~100K vocab)
            max_seq_len: Maximum sequence length for training
            cache_dir: Directory to cache processed data
            seed: Random seed for reproducibility
        """
        super().__init__(encoding_name=encoding_name, max_seq_len=max_seq_len)
        self.domain = domain
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / '.cache' / 'transformer' / 'domain'
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.rng = random.Random(seed)

    def get_domain_info(self) -> Dict[str, Any]:
        """
        Get information about this domain dataset.

        Returns:
            Dictionary with domain metadata (name, description, source, etc.)
        """
        raise NotImplementedError("Subclasses must implement get_domain_info()")


class CodeDataset(DomainDataset):
    """
    Code domain dataset for mid-training.

    Specializes the model for programming tasks by training on:
    - High-quality code repositories (Python, JavaScript, etc.)
    - Documentation and docstrings
    - Stack Overflow Q&A

    Why Code Mid-Training Works:
    ============================
    Code has distinct patterns from natural language:
    - Syntactic structure (indentation, brackets, keywords)
    - Semantic relationships (function definitions, imports)
    - Common idioms (list comprehensions, lambda functions)

    By focusing on these patterns, the model becomes better at:
    - Code completion (GitHub Copilot-style)
    - Bug fixing and refactoring
    - Understanding API usage
    - Translating between programming languages

    Dataset Details:
    ================
    We use the "codeparrot/github-code-clean" dataset:
    - 115M code files from GitHub (deduplicated, filtered)
    - Multiple languages: Python, JavaScript, Java, Go, etc.
    - Filtered for quality (no auto-generated code, proper syntax)
    - ~50GB of clean source code

    Data Quality Matters:
    =====================
    Mid-training on 10M tokens of excellent code beats 100M tokens
    of random GitHub files. Quality indicators:
    - Proper documentation
    - Clear variable names
    - Consistent style
    - No syntax errors

    Example Usage:
    ==============
    >>> dataset = CodeDataset(languages=['python', 'javascript'])
    >>> for batch in dataset.get_dataloader(batch_size=8):
    ...     # Each batch contains code sequences
    ...     # Model learns programming patterns
    ...     loss = model(batch)
    """

    def __init__(
        self,
        languages: list[str] = ['python'],
        encoding_name: str = 'cl100k_base',
        max_seq_len: int = 128,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize code dataset.

        Args:
            languages: Programming languages to include (e.g., ['python', 'javascript'])
            encoding_name: Tokenizer encoding
            max_seq_len: Maximum sequence length
            cache_dir: Cache directory
            seed: Random seed
        """
        super().__init__(
            domain='code',
            encoding_name=encoding_name,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            seed=seed,
        )
        self.languages = languages

    def get_domain_info(self) -> Dict[str, Any]:
        """Get code domain metadata."""
        return {
            'domain': 'code',
            'description': 'Programming code (Python, JavaScript, etc.)',
            'languages': self.languages,
            'source': 'codeparrot/github-code-clean (HuggingFace)',
            'quality_filters': [
                'Deduplicated files',
                'Syntax validation',
                'No auto-generated code',
                'Minimum length requirements',
            ],
            'use_cases': [
                'Code completion',
                'Bug fixing',
                'Code translation',
                'API usage understanding',
            ],
        }

    def prepare_dataset(
        self,
        num_tokens: int = 50_000_000,
        split: Literal['train', 'validation'] = 'train',
    ) -> Dataset:
        """
        Prepare code dataset for mid-training.

        Why This Dataset Structure Works:
        ==================================
        1. **Language filtering**: Focus on languages relevant to your use case
           - Python for data science, web backends
           - JavaScript for web frontends
           - Multiple languages for translation tasks

        2. **Deduplication**: Remove identical/near-identical code
           - Prevents memorization of specific files
           - Encourages learning general patterns

        3. **Quality filtering**: Only well-written code
           - Proper variable names (not 'x', 'y', 'temp')
           - Documentation and comments
           - Consistent formatting

        Args:
            num_tokens: Target number of tokens to sample
            split: 'train' or 'validation'

        Returns:
            HuggingFace Dataset ready for tokenization
        """
        print(f"\n{'='*80}")
        print(f"PREPARING CODE DATASET")
        print(f"{'='*80}")
        print(f"Languages: {', '.join(self.languages)}")
        print(f"Target tokens: {num_tokens:,}")
        print(f"Split: {split}")
        print()

        # Note: This is a placeholder implementation
        # In production, you would:
        # 1. Load from HuggingFace: load_dataset('codeparrot/github-code-clean')
        # 2. Filter by language: dataset.filter(lambda x: x['language'] in languages)
        # 3. Sample to reach target token count
        # 4. Cache the processed dataset

        print("[WARNING] Full code dataset integration coming soon!")
        print("For now, returning placeholder that demonstrates the concept.")
        print()
        print("To implement:")
        print("1. Install datasets: pip install datasets")
        print("2. Load dataset: load_dataset('codeparrot/github-code-clean')")
        print("3. Filter and sample based on num_tokens target")
        print()

        # Return empty dataset as placeholder
        return Dataset.from_dict({'text': []})


class MathDataset(DomainDataset):
    """
    Math domain dataset for mid-training.

    Specializes the model for mathematical reasoning by training on:
    - Math problem-solution pairs
    - Theorem proofs and derivations
    - Mathematical textbooks
    - Scientific papers with equations

    Why Math Mid-Training Works:
    =============================
    Mathematical reasoning requires distinct capabilities:
    - Symbol manipulation (x² + 2x + 1 = (x+1)²)
    - Step-by-step derivations
    - Formal logic and proof techniques
    - Understanding mathematical notation

    By focusing on these patterns, the model becomes better at:
    - Solving word problems
    - Step-by-step explanations
    - Mathematical proofs
    - Equation solving and simplification

    Dataset Details:
    ================
    We use the "hendrycks/MATH" dataset:
    - 12,500 challenging competition math problems
    - Multiple difficulty levels (1-5)
    - Step-by-step solutions included
    - Topics: Algebra, Geometry, Number Theory, etc.

    Also includes:
    - arXiv papers (mathematical sections)
    - Mathematics StackExchange Q&A
    - Textbook excerpts (when available)

    Curriculum Learning Opportunity:
    =================================
    Math problems have natural difficulty levels. We can use curriculum learning:
    1. Start with difficulty level 1 (basic algebra)
    2. Progress to level 5 (competition-level problems)

    This mirrors how humans learn math: start simple, build complexity.
    Research shows 10-15% improvement vs. random ordering!

    Example Usage:
    ==============
    >>> dataset = MathDataset(difficulty_range=(1, 3))
    >>> # Start with easier problems
    >>> for batch in dataset.get_dataloader(batch_size=8):
    ...     loss = model(batch)
    >>>
    >>> # Later, increase difficulty
    >>> dataset.set_difficulty_range((3, 5))
    """

    def __init__(
        self,
        difficulty_range: tuple[int, int] = (1, 5),
        encoding_name: str = 'cl100k_base',
        max_seq_len: int = 128,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize math dataset.

        Args:
            difficulty_range: (min, max) difficulty levels (1-5)
            encoding_name: Tokenizer encoding
            max_seq_len: Maximum sequence length
            cache_dir: Cache directory
            seed: Random seed
        """
        super().__init__(
            domain='math',
            encoding_name=encoding_name,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            seed=seed,
        )
        self.difficulty_range = difficulty_range

    def get_domain_info(self) -> Dict[str, Any]:
        """Get math domain metadata."""
        return {
            'domain': 'math',
            'description': 'Mathematical reasoning and problem-solving',
            'difficulty_range': self.difficulty_range,
            'source': 'hendrycks/MATH, arXiv papers (HuggingFace)',
            'topics': [
                'Algebra',
                'Geometry',
                'Number Theory',
                'Calculus',
                'Probability',
            ],
            'use_cases': [
                'Word problem solving',
                'Step-by-step explanations',
                'Mathematical proofs',
                'Equation manipulation',
            ],
        }

    def prepare_dataset(
        self,
        num_tokens: int = 50_000_000,
        split: Literal['train', 'validation'] = 'train',
    ) -> Dataset:
        """
        Prepare math dataset for mid-training.

        Args:
            num_tokens: Target number of tokens to sample
            split: 'train' or 'validation'

        Returns:
            HuggingFace Dataset ready for tokenization
        """
        print(f"\n{'='*80}")
        print(f"PREPARING MATH DATASET")
        print(f"{'='*80}")
        print(f"Difficulty range: {self.difficulty_range[0]}-{self.difficulty_range[1]}")
        print(f"Target tokens: {num_tokens:,}")
        print(f"Split: {split}")
        print()

        print("[WARNING] Full math dataset integration coming soon!")
        print("For now, returning placeholder.")
        print()

        return Dataset.from_dict({'text': []})


class ScienceDataset(DomainDataset):
    """
    Science domain dataset for mid-training.

    Specializes the model for scientific reasoning by training on:
    - Scientific papers (arXiv, PubMed)
    - Textbooks and encyclopedias
    - Scientific Q&A (e.g., Physics/Chemistry StackExchange)

    Why Science Mid-Training Works:
    ================================
    Scientific text has unique characteristics:
    - Technical vocabulary (photosynthesis, thermodynamics)
    - Structured reasoning (hypothesis → evidence → conclusion)
    - Quantitative analysis (measurements, statistics)
    - Citation and reference patterns

    By focusing on these patterns, the model becomes better at:
    - Explaining scientific concepts
    - Understanding research papers
    - Generating hypotheses
    - Reasoning about experimental results

    Dataset Details:
    ================
    We use multiple scientific sources:
    - arXiv papers (physics, CS, biology)
    - PubMed abstracts (biomedical research)
    - ScienceDirect articles (multidisciplinary)
    - Scientific textbooks (when available)

    Example Usage:
    ==============
    >>> dataset = ScienceDataset(fields=['physics', 'biology'])
    >>> for batch in dataset.get_dataloader(batch_size=8):
    ...     loss = model(batch)
    """

    def __init__(
        self,
        fields: list[str] = ['general'],
        encoding_name: str = 'cl100k_base',
        max_seq_len: int = 128,
        cache_dir: Optional[str] = None,
        seed: int = 42,
    ):
        """
        Initialize science dataset.

        Args:
            fields: Scientific fields (e.g., ['physics', 'biology'])
            encoding_name: Tokenizer encoding
            max_seq_len: Maximum sequence length
            cache_dir: Cache directory
            seed: Random seed
        """
        super().__init__(
            domain='science',
            encoding_name=encoding_name,
            max_seq_len=max_seq_len,
            cache_dir=cache_dir,
            seed=seed,
        )
        self.fields = fields

    def get_domain_info(self) -> Dict[str, Any]:
        """Get science domain metadata."""
        return {
            'domain': 'science',
            'description': 'Scientific literature and reasoning',
            'fields': self.fields,
            'source': 'arXiv, PubMed, scientific papers',
            'topics': [
                'Physics',
                'Biology',
                'Chemistry',
                'Computer Science',
                'Medicine',
            ],
            'use_cases': [
                'Scientific explanation',
                'Research paper understanding',
                'Hypothesis generation',
                'Experimental reasoning',
            ],
        }

    def prepare_dataset(
        self,
        num_tokens: int = 50_000_000,
        split: Literal['train', 'validation'] = 'train',
    ) -> Dataset:
        """
        Prepare science dataset for mid-training.

        Args:
            num_tokens: Target number of tokens to sample
            split: 'train' or 'validation'

        Returns:
            HuggingFace Dataset ready for tokenization
        """
        print(f"\n{'='*80}")
        print(f"PREPARING SCIENCE DATASET")
        print(f"{'='*80}")
        print(f"Fields: {', '.join(self.fields)}")
        print(f"Target tokens: {num_tokens:,}")
        print(f"Split: {split}")
        print()

        print("[WARNING] Full science dataset integration coming soon!")
        print("For now, returning placeholder.")
        print()

        return Dataset.from_dict({'text': []})


def create_domain_dataset(
    domain: str,
    **kwargs
) -> DomainDataset:
    """
    Factory function to create domain-specific datasets.

    Educational Note:
    =================
    This factory pattern makes it easy to experiment with different domains.
    Want to specialize your model for medical text? Just create MedicalDataset!

    The key insight: Same architecture, same training loop, different data.
    That's the power of mid-training.

    Args:
        domain: Domain name ('code', 'math', 'science')
        **kwargs: Domain-specific arguments

    Returns:
        Appropriate DomainDataset instance

    Example:
        >>> # Create code dataset
        >>> code_ds = create_domain_dataset('code', languages=['python'])
        >>>
        >>> # Create math dataset with curriculum learning
        >>> math_ds = create_domain_dataset('math', difficulty_range=(1, 3))
    """
    domain_classes = {
        'code': CodeDataset,
        'math': MathDataset,
        'science': ScienceDataset,
    }

    if domain not in domain_classes:
        raise ValueError(
            f"Unknown domain: {domain}. "
            f"Available domains: {', '.join(domain_classes.keys())}"
        )

    return domain_classes[domain](**kwargs)
