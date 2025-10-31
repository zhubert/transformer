"""
Test checkpoint loading logic.

This ensures that checkpoints can be loaded correctly regardless of:
1. Whether max_seq_len is in the config
2. Whether the checkpoint was saved from a compiled model
3. Different model configurations

These tests are critical because checkpoint loading issues can prevent
model evaluation and deployment.
"""

import pytest
import torch
import tempfile
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.model import DecoderOnlyTransformer
from commands.evaluate_perplexity import load_checkpoint


def create_test_checkpoint(
    vocab_size=1000,
    d_model=64,
    num_heads=4,
    num_layers=2,
    d_ff=256,
    max_seq_len=128,
    dropout=0.1,
    include_max_seq_len_in_config=True,
    use_compile_prefix=False
):
    """
    Create a test checkpoint with specified configuration.

    Args:
        vocab_size: Vocabulary size
        d_model: Model dimension
        num_heads: Number of attention heads
        num_layers: Number of transformer layers
        d_ff: Feed-forward dimension
        max_seq_len: Maximum sequence length
        dropout: Dropout rate
        include_max_seq_len_in_config: Whether to include max_seq_len in config
        use_compile_prefix: Whether to add torch.compile() prefix to state dict keys

    Returns:
        Path to temporary checkpoint file
    """
    # Create model
    model = DecoderOnlyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        dropout=dropout,
    )

    # Get state dict
    state_dict = model.state_dict()

    # Optionally add torch.compile() prefix
    if use_compile_prefix:
        state_dict = {f'_orig_mod.{k}': v for k, v in state_dict.items()}

    # Create config dict
    config = {
        'vocab_size': vocab_size,
        'd_model': d_model,
        'num_heads': num_heads,
        'num_layers': num_layers,
        'd_ff': d_ff,
        'dropout': dropout,
        'encoding': 'cl100k_base',
    }

    # Optionally include max_seq_len
    if include_max_seq_len_in_config:
        config['max_seq_len'] = max_seq_len

    # Create checkpoint dict
    checkpoint = {
        'model_state_dict': state_dict,
        'config': config,
        'epoch': 1,
        'loss': 2.5,
        'perplexity': 12.18,
    }

    # Save to temporary file
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.pt')
    torch.save(checkpoint, temp_file.name)
    temp_file.close()

    return temp_file.name


class TestCheckpointLoading:
    """Test suite for checkpoint loading functionality."""

    def test_load_checkpoint_with_max_seq_len_in_config(self):
        """Test loading checkpoint when max_seq_len is in config."""
        checkpoint_path = create_test_checkpoint(
            max_seq_len=256,
            include_max_seq_len_in_config=True
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Verify model was created with correct max_seq_len
            assert model.pos_encoding.pos_embedding.weight.shape[0] == 256

            # Verify model is in eval mode
            assert not model.training

            # Verify encoding detection
            assert encoding == 'cl100k_base'

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_without_max_seq_len_in_config(self):
        """
        Test loading checkpoint when max_seq_len is NOT in config.

        This is the critical test case that was failing before the fix.
        The loader should infer max_seq_len from the positional embedding shape.
        """
        checkpoint_path = create_test_checkpoint(
            max_seq_len=256,
            include_max_seq_len_in_config=False  # Simulate old checkpoint
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Verify model was created with correct max_seq_len (inferred)
            assert model.pos_encoding.pos_embedding.weight.shape[0] == 256

            # Verify model is in eval mode
            assert not model.training

            # Verify encoding detection
            assert encoding == 'cl100k_base'

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_with_compile_prefix(self):
        """
        Test loading checkpoint saved from torch.compile() model.

        Compiled models have '_orig_mod.' prefix on all state dict keys.
        The loader should strip this prefix before loading.
        """
        checkpoint_path = create_test_checkpoint(
            max_seq_len=128,
            use_compile_prefix=True
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Verify model loaded correctly despite compile prefix
            assert model.pos_encoding.pos_embedding.weight.shape[0] == 128

            # Verify model is in eval mode
            assert not model.training

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_with_compile_prefix_and_no_max_seq_len(self):
        """
        Test the worst-case scenario: compiled checkpoint without max_seq_len.

        This combines both edge cases:
        1. torch.compile() prefix needs stripping
        2. max_seq_len needs inference from positional embedding
        """
        checkpoint_path = create_test_checkpoint(
            max_seq_len=512,
            include_max_seq_len_in_config=False,
            use_compile_prefix=True
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Verify model loaded correctly
            assert model.pos_encoding.pos_embedding.weight.shape[0] == 512

            # Verify model is in eval mode
            assert not model.training

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_load_checkpoint_different_configs(self):
        """Test loading checkpoints with various configurations."""
        configs = [
            {'max_seq_len': 64, 'd_model': 32, 'num_heads': 2},
            {'max_seq_len': 128, 'd_model': 64, 'num_heads': 4},
            {'max_seq_len': 256, 'd_model': 128, 'num_heads': 8},
            {'max_seq_len': 1024, 'd_model': 256, 'num_heads': 8},
        ]

        for config in configs:
            checkpoint_path = create_test_checkpoint(
                max_seq_len=config['max_seq_len'],
                d_model=config['d_model'],
                num_heads=config['num_heads'],
                include_max_seq_len_in_config=False  # Force inference
            )

            try:
                model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

                # Verify correct configuration
                assert model.pos_encoding.pos_embedding.weight.shape[0] == config['max_seq_len']
                assert model.d_model == config['d_model']
                assert model.blocks[0].attention.num_heads == config['num_heads']

            finally:
                # Clean up
                Path(checkpoint_path).unlink()

    def test_loaded_model_forward_pass(self):
        """Test that loaded model can perform forward pass."""
        checkpoint_path = create_test_checkpoint(
            vocab_size=1000,
            max_seq_len=128,
            include_max_seq_len_in_config=False
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Create dummy input
            batch_size = 2
            seq_len = 10
            input_ids = torch.randint(0, 1000, (batch_size, seq_len))

            # Forward pass should work
            with torch.no_grad():
                result = model(input_ids)

            # Handle both single logits and tuple return
            if isinstance(result, tuple):
                logits = result[0]
            else:
                logits = result

            # Verify output shape
            assert logits.shape == (batch_size, seq_len, 1000)

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_loaded_model_generation(self):
        """Test that loaded model can generate text."""
        checkpoint_path = create_test_checkpoint(
            vocab_size=1000,
            max_seq_len=128,
            include_max_seq_len_in_config=False
        )

        try:
            model, checkpoint, encoding = load_checkpoint(checkpoint_path, device='cpu')

            # Create initial prompt
            prompt = torch.tensor([[1, 2, 3]], dtype=torch.long)

            # Generate tokens
            with torch.no_grad():
                generated = model.generate(
                    prompt,
                    max_length=13,  # 3 prompt + 10 new tokens
                    sampling_strategy='greedy'
                )

            # Verify generation worked
            assert generated.shape[0] == 1  # batch size
            assert generated.shape[1] == 13  # 3 prompt + 10 new tokens

        finally:
            # Clean up
            Path(checkpoint_path).unlink()

    def test_load_into_compiled_model(self):
        """
        Test loading checkpoint into a compiled model.

        This tests the bug fix where loading a checkpoint into a torch.compile()
        model would fail due to _orig_mod. prefix mismatch.

        Scenario:
        1. Create a checkpoint (with or without compile prefix)
        2. Create and compile a model
        3. Load checkpoint into the compiled model
        4. Verify loading succeeds
        """
        # Test both cases: checkpoint with and without compile prefix
        for use_prefix in [False, True]:
            checkpoint_path = create_test_checkpoint(
                vocab_size=1000,
                d_model=64,
                num_heads=4,
                num_layers=2,
                max_seq_len=128,
                use_compile_prefix=use_prefix
            )

            try:
                # Load checkpoint using the checkpoint_utils function
                from src.transformer.checkpoint_utils import load_checkpoint as util_load_checkpoint

                # Create and compile a model
                model = DecoderOnlyTransformer(
                    vocab_size=1000,
                    d_model=64,
                    num_heads=4,
                    num_layers=2,
                    d_ff=256,
                    max_seq_len=128,
                    dropout=0.1,
                )
                compiled_model = torch.compile(model, backend="inductor", mode="default")

                # Load checkpoint into compiled model
                result = util_load_checkpoint(
                    checkpoint_path,
                    device='cpu',
                    load_training_state=False,
                    model=compiled_model,
                    verbose=False
                )

                # Verify model loaded successfully
                loaded_model = result['model']
                assert hasattr(loaded_model, '_orig_mod')  # Still compiled

                # Test forward pass works
                batch_size = 2
                seq_len = 10
                input_ids = torch.randint(0, 1000, (batch_size, seq_len))
                with torch.no_grad():
                    output = loaded_model(input_ids)
                    if isinstance(output, tuple):
                        output = output[0]
                    assert output.shape == (batch_size, seq_len, 1000)

            finally:
                # Clean up
                Path(checkpoint_path).unlink()


if __name__ == '__main__':
    # Run tests with pytest
    pytest.main([__file__, '-v'])
