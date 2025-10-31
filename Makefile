.PHONY: help install install-rocm test test-cov clean

# Default target
help:
	@echo "Transformer - Educational Implementation"
	@echo ""
	@echo "Setup:"
	@echo "  make install       - Install project dependencies (NVIDIA CUDA or CPU)"
	@echo "  make install-rocm  - Install with AMD ROCm support (Linux only)"
	@echo ""
	@echo "Development:"
	@echo "  make test          - Run all tests"
	@echo "  make test-cov      - Run tests with coverage report"
	@echo "  make clean         - Remove cache files, checkpoints, and build artifacts"
	@echo ""
	@echo "Using the Transformer:"
	@echo "  uv run python main.py     - Launch interactive CLI (recommended!)"
	@echo "  uv run python main.py --help - Show advanced command-line options"

# Install dependencies (default: NVIDIA CUDA or CPU)
install:
	uv sync

# Install with AMD ROCm support (Linux only)
install-rocm:
	@echo "Installing with AMD ROCm support..."
	uv sync --extra rocm

# Run tests
test:
	uv run pytest tests/ -v

# Run tests with coverage
test-cov:
	uv run pytest tests/ -v --cov=src --cov-report=term-missing --cov-report=html

# Clean up cache files (preserves data/fineweb_cache shards)
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.coverage" -delete
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf build dist *.egg-info
	rm -rf checkpoints
	rm -rf checkpoints_medium
	rm -rf checkpoints_quick
