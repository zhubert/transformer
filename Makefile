.PHONY: help install test test-cov clean train train-medium train-quick resume resume-medium resume-quick generate evaluate compare demo-sampling

# Default target
help:
	@echo "Available commands:"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install project dependencies with uv"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make clean         - Remove cache files, checkpoints, and build artifacts"
	@echo ""
	@echo "Transformer Operations:"
	@echo "  make train         - Train a transformer model (100M tokens/epoch × 20, 6 layers)"
	@echo "  make train-medium  - Medium training (50M tokens/epoch × 15, 4 layers, ~2h epoch 1, ~30-60min epochs 2+)"
	@echo "  make train-quick   - Quick training (10M tokens/epoch × 10, 4 layers, ~40min epoch 1, ~15-25min epochs 2+)"
	@echo "  make resume        - Resume training from latest checkpoint"
	@echo "  make resume-medium - Resume medium training from latest checkpoint"
	@echo "  make resume-quick  - Resume quick training from latest checkpoint"
	@echo "  make generate      - Generate text (interactive mode)"
	@echo "  make evaluate      - Evaluate latest checkpoint"
	@echo "  make compare       - Compare all checkpoints"
	@echo "  make demo-sampling - Demonstrate sampling strategies"
	@echo ""
	@echo "For more options, use: uv run python main.py <command> --help"

# Install dependencies
install:
	uv sync

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Clean up cache files
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf build dist *.egg-info
	rm -rf data/fineweb_cache
	rm -rf checkpoints
	rm -rf checkpoints_medium
	rm -rf checkpoints_quick

# Transformer operations via main.py CLI
train:
	uv run python main.py train

train-medium:
	uv run python main.py train --medium

train-quick:
	uv run python commands/train.py --quick

resume:
	uv run python main.py train --resume

resume-medium:
	uv run python main.py train --medium --resume

resume-quick:
	uv run python commands/train.py --quick --resume

generate:
	@echo "Starting interactive generation mode..."
	@echo "Tip: Use 'make generate-prompt CHECKPOINT=path/to/checkpoint.pt PROMPT=\"your text\"' for single prompts"
	uv run python main.py generate

# Generate with custom prompt (use: make generate-prompt CHECKPOINT=checkpoints/model_epoch_10.pt PROMPT="Once upon a time")
generate-prompt:
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "Error: CHECKPOINT not specified. Usage: make generate-prompt CHECKPOINT=path/to/checkpoint.pt PROMPT=\"your text\""; \
		exit 1; \
	fi
	@if [ -z "$(PROMPT)" ]; then \
		echo "Error: PROMPT not specified. Usage: make generate-prompt CHECKPOINT=path/to/checkpoint.pt PROMPT=\"your text\""; \
		exit 1; \
	fi
	uv run python main.py generate $(CHECKPOINT) --prompt "$(PROMPT)"

evaluate:
	uv run python main.py evaluate

compare:
	uv run python main.py compare

demo-sampling:
	uv run python main.py demo-sampling
