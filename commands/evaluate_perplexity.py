"""
Evaluate transformer model using perplexity metric.

This script demonstrates how to:
1. Load a trained model checkpoint
2. Evaluate perplexity on a dataset
3. Compare multiple checkpoints
4. Understand what the perplexity values mean

What is Model Evaluation?
-------------------------
After training, we want to know: "How good is our model?"

Perplexity is the standard metric for this in language modeling.
It tells us how "confused" the model is when predicting text.

Why Evaluate Separately from Training?
---------------------------------------
1. Test on held-out data the model hasn't seen (test set)
2. Compare different model checkpoints objectively
3. Detect overfitting (train perplexity << test perplexity)
4. Make decisions about which model to deploy

Use Cases:
----------
1. Final evaluation: "Our model achieves perplexity of 28 on the test set"
2. Model selection: "Epoch 15 checkpoint has best validation perplexity"
3. Debugging: "Why is test perplexity so much higher than training?"
4. Research: "Method A achieves 25 perplexity vs. Method B's 30"
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
import sys
import argparse
import tiktoken

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from src.transformer.perplexity import evaluate_perplexity, calculate_perplexity_from_loss

# Import encoding detection from train.py
sys.path.append(str(Path(__file__).parent))
from train import detect_encoding_from_checkpoint


class SimpleTextDataset(Dataset):
    """
    Simple dataset for loading text from a file for evaluation.

    This is a minimal dataset implementation for evaluation purposes only.
    For training, use FineWebDataset instead.
    """

    def __init__(self, text_file, seq_length=128, encoding_name="p50k_base"):
        """
        Load text from a file and create training sequences.

        Args:
            text_file: Path to text file
            seq_length: Length of each sequence
            encoding_name: tiktoken encoding to use
        """
        self.seq_length = seq_length
        self.tokenizer = tiktoken.get_encoding(encoding_name)

        # Read text file
        with open(text_file, 'r', encoding='utf-8') as f:
            text = f.read()

        # Tokenize entire text
        self.tokens = self.tokenizer.encode(text)

        # Calculate number of complete sequences
        self.num_sequences = (len(self.tokens) - 1) // seq_length

        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Sequences: {self.num_sequences:,}")
        print(f"  Vocabulary size: {self.tokenizer.n_vocab:,}")

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Get a single training pair (input, target)."""
        start_idx = idx * self.seq_length
        end_idx = start_idx + self.seq_length

        input_seq = self.tokens[start_idx:end_idx]
        target_seq = self.tokens[start_idx + 1:end_idx + 1]

        return (
            torch.tensor(input_seq, dtype=torch.long),
            torch.tensor(target_seq, dtype=torch.long)
        )

    @property
    def vocab_size(self):
        return self.tokenizer.n_vocab


def load_checkpoint(checkpoint_path, device='cpu'):
    """
    Load a model checkpoint.

    Args:
        checkpoint_path: Path to the checkpoint file
        device: Device to load model on

    Returns:
        model: Loaded model
        checkpoint: Full checkpoint dict (contains config, loss, etc.)
    """
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract configuration
    config = checkpoint['config']

    # Create model with saved configuration
    model = DecoderOnlyTransformer(
        vocab_size=config['vocab_size'],
        d_model=config['d_model'],
        num_heads=config['num_heads'],
        num_layers=config['num_layers'],
        d_ff=config['d_ff'],
        dropout=config['dropout'],
    )

    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    # Detect encoding
    detected_encoding = detect_encoding_from_checkpoint(checkpoint)

    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Encoding: {detected_encoding}")
    print(f"  Training loss: {checkpoint['loss']:.4f}")
    if 'perplexity' in checkpoint:
        print(f"  Training perplexity: {checkpoint['perplexity']:.2f}")
    print()

    return model, checkpoint, detected_encoding


def evaluate_checkpoint(checkpoint_path, text_file, seq_length=128, batch_size=8, device='cpu', encoding='p50k_base'):
    """
    Evaluate a single checkpoint on a dataset.

    This is useful for final evaluation on a test set.

    Args:
        checkpoint_path: Path to model checkpoint
        text_file: Path to text file for evaluation
        seq_length: Sequence length for evaluation
        batch_size: Batch size for evaluation
        device: Device to run on
        encoding: Tokenizer encoding to use

    Returns:
        perplexity: Perplexity on the dataset
        loss: Cross-entropy loss on the dataset
    """
    # Load model
    model, checkpoint, detected_encoding = load_checkpoint(checkpoint_path, device=device)

    # Check for encoding mismatch
    if detected_encoding != encoding:
        print(f"  ERROR: Checkpoint was trained with {detected_encoding}")
        print(f"         but you're trying to evaluate with {encoding}")
        print()
        print(f"  Please use the same encoding as the checkpoint:")
        print(f"    uv run python commands/evaluate_perplexity.py --encoding {detected_encoding}")
        print()
        sys.exit(1)

    # Load dataset
    print(f"Loading evaluation dataset: {text_file}")
    dataset = SimpleTextDataset(text_file, seq_length=seq_length, encoding_name=encoding)
    print()

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for evaluation
        drop_last=False  # Include all data
    )

    print(f"Evaluation configuration:")
    print(f"  Sequences: {len(dataset):,}")
    print(f"  Batches: {len(dataloader):,}")
    print(f"  Device: {device}")
    print()

    # Evaluate
    print("Evaluating...")
    perplexity, loss = evaluate_perplexity(model, dataloader, device=device)

    print("=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Loss: {loss:.4f}")
    print(f"Perplexity: {perplexity:.2f}")
    print()

    # Interpret results
    print("What does this mean?")
    print("-" * 80)
    if perplexity < 20:
        print(f"EXCELLENT! Perplexity {perplexity:.2f} is GPT-2 level performance.")
        print("The model has learned the language patterns very well.")
    elif perplexity < 50:
        print(f"GOOD! Perplexity {perplexity:.2f} shows the model has learned patterns.")
        print("There's room for improvement, but it's a solid model.")
    elif perplexity < 100:
        print(f"DECENT. Perplexity {perplexity:.2f} means the model has some understanding.")
        print("Consider training longer or tuning hyperparameters.")
    else:
        print(f"NEEDS WORK. Perplexity {perplexity:.2f} is quite high.")
        print("The model is still quite confused. More training needed.")

    print()
    print(f"Interpretation: On average, the model is as confused as if it had to")
    print(f"choose uniformly from ~{int(perplexity)} words at each step.")
    print()

    # Compare to training
    if 'perplexity' in checkpoint:
        train_ppl = checkpoint['perplexity']
        print("Comparison to training perplexity:")
        print("-" * 80)
        print(f"  Training perplexity: {train_ppl:.2f}")
        print(f"  Evaluation perplexity: {perplexity:.2f}")
        print(f"  Difference: {perplexity - train_ppl:+.2f}")
        print()

        if perplexity > train_ppl * 1.5:
            print("  WARNING: Evaluation perplexity is much higher than training!")
            print("  This suggests overfitting. The model memorized training data")
            print("  but doesn't generalize well to new text.")
        elif perplexity > train_ppl * 1.1:
            print("  Evaluation perplexity is slightly higher than training.")
            print("  This is normal - models perform better on training data.")
        else:
            print("  Evaluation and training perplexity are similar - good!")
            print("  The model generalizes well to new text.")
        print()

    return perplexity, loss


def compare_checkpoints(checkpoint_dir, text_file, seq_length=128, device='cpu', encoding='p50k_base'):
    """
    Compare all checkpoints in a directory to find the best one.

    This is useful for model selection: which epoch checkpoint is best?

    Args:
        checkpoint_dir: Directory containing checkpoints
        text_file: Text file to evaluate on
        seq_length: Sequence length
        device: Device to run on
        encoding: Tokenizer encoding to use
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))

    if not checkpoint_files:
        print(f"No checkpoints found in {checkpoint_dir}")
        return

    print("=" * 80)
    print(f"COMPARING {len(checkpoint_files)} CHECKPOINTS")
    print("=" * 80)
    print()

    # Load dataset once
    dataset = SimpleTextDataset(text_file, seq_length=seq_length, encoding_name=encoding)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, drop_last=False)

    results = []

    for checkpoint_path in checkpoint_files:
        print(f"Evaluating: {checkpoint_path.name}")
        print("-" * 80)

        # Load model
        model, checkpoint, detected_encoding = load_checkpoint(checkpoint_path, device=device)

        # Check for encoding mismatch
        if detected_encoding != encoding:
            print(f"  ERROR: This checkpoint was trained with {detected_encoding}")
            print(f"         but you're trying to evaluate with {encoding}")
            print()
            print(f"  Please use the same encoding as the checkpoints:")
            print(f"    uv run python commands/evaluate_perplexity.py --compare --encoding {detected_encoding}")
            print()
            sys.exit(1)

        # Evaluate
        perplexity, loss = evaluate_perplexity(model, dataloader, device=device)

        results.append({
            'checkpoint': checkpoint_path.name,
            'epoch': checkpoint['epoch'],
            'train_loss': checkpoint['loss'],
            'train_ppl': checkpoint.get('perplexity', None),
            'eval_loss': loss,
            'eval_ppl': perplexity,
        })

        print(f"  Evaluation perplexity: {perplexity:.2f}")
        print()

    # Summary
    print("=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)
    print()

    # Sort by evaluation perplexity (lower is better)
    results.sort(key=lambda x: x['eval_ppl'])

    print(f"{'Checkpoint':<25} {'Epoch':<8} {'Train PPL':<12} {'Eval PPL':<12} {'Status'}")
    print("-" * 80)

    for i, result in enumerate(results):
        epoch = result['epoch']
        train_ppl = result['train_ppl']
        eval_ppl = result['eval_ppl']

        # Format training perplexity
        train_ppl_str = f"{train_ppl:.2f}" if train_ppl is not None else "N/A"

        # Status indicator
        if i == 0:
            status = "★ BEST"
        elif eval_ppl < results[0]['eval_ppl'] * 1.05:
            status = "Very good"
        else:
            status = ""

        print(f"{result['checkpoint']:<25} {epoch:<8} {train_ppl_str:<12} {eval_ppl:<12.2f} {status}")

    print()
    print(f"RECOMMENDATION: Use {results[0]['checkpoint']} (lowest perplexity)")
    print()

    # Analyze training progression
    print("Training progression analysis:")
    print("-" * 80)

    # Re-sort by epoch
    results.sort(key=lambda x: x['epoch'])

    print("How did perplexity change during training?")
    for i, result in enumerate(results):
        if i > 0:
            prev_ppl = results[i-1]['eval_ppl']
            curr_ppl = result['eval_ppl']
            change = curr_ppl - prev_ppl
            change_pct = (change / prev_ppl) * 100

            arrow = "↓" if change < 0 else "↑"
            print(f"  Epoch {results[i-1]['epoch']} → {result['epoch']}: "
                  f"{prev_ppl:.2f} → {curr_ppl:.2f} "
                  f"({arrow} {abs(change):.2f}, {change_pct:+.1f}%)")

    print()


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate transformer model using perplexity"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to specific checkpoint to evaluate"
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints",
        help="Directory containing checkpoints (for comparison mode)"
    )
    parser.add_argument(
        "--text-file",
        type=str,
        default="Singular.txt",
        help="Text file to evaluate on"
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=128,
        help="Sequence length for evaluation"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare all checkpoints in checkpoint-dir"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to run evaluation on"
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="p50k_base",
        choices=["p50k_base", "cl100k_base"],
        help="Tokenizer encoding to use (default: p50k_base, ~50K vocab; cl100k_base: ~100K vocab)"
    )

    args = parser.parse_args()

    # Detect device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    if args.compare:
        # Compare all checkpoints
        compare_checkpoints(
            args.checkpoint_dir,
            args.text_file,
            seq_length=args.seq_length,
            device=device,
            encoding=args.encoding
        )
    elif args.checkpoint:
        # Evaluate single checkpoint
        evaluate_checkpoint(
            args.checkpoint,
            args.text_file,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            device=device,
            encoding=args.encoding
        )
    else:
        # Default: find latest checkpoint
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_files = sorted(checkpoint_dir.glob("model_epoch_*.pt"))

        if not checkpoint_files:
            print(f"No checkpoints found in {checkpoint_dir}")
            print("Please train a model first using: uv run python main.py train")
            return

        latest_checkpoint = checkpoint_files[-1]
        print(f"No checkpoint specified, using latest: {latest_checkpoint.name}")
        print()

        evaluate_checkpoint(
            latest_checkpoint,
            args.text_file,
            seq_length=args.seq_length,
            batch_size=args.batch_size,
            device=device,
            encoding=args.encoding
        )


if __name__ == "__main__":
    main()
