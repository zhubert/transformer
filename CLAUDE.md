# Transformer Implementation - Educational Project

## Project Goal
Implement a decoder-only transformer from scratch for educational purposes, building on our series about learning LLMs.

## What is a Transformer?
A transformer is a neural network architecture introduced in "Attention is All You Need" (Vaswani et al., 2017). It revolutionized NLP and is the foundation of modern LLMs like GPT, BERT, and Claude.

### Key Components
1. **Self-Attention Mechanism** - Allows the model to weigh the importance of different words when processing each word
2. **Multi-Head Attention** - Runs multiple attention mechanisms in parallel, each learning different relationships
3. **Position Encodings** - Adds positional information since transformers process all tokens in parallel
4. **Feed-Forward Networks** - Applied to each position independently
5. **Layer Normalization & Residual Connections** - Helps with training stability
6. **Decoder-Only Structure** - We'll implement this (like GPT), simpler than encoder-decoder

## Implementation Approach
- Build from scratch to understand internals
- Use PyTorch for:
  - Automatic differentiation (autograd)
  - GPU acceleration
  - Tensor operations
  - BUT avoid pre-built transformer modules
- Use `uv` for Python environment management

## Planned Components
1. Positional encoding (sinusoidal)
2. Scaled dot-product attention
3. Multi-head attention with causal masking
4. Feed-forward network (MLP)
5. Transformer block (attention + FFN + residuals + LayerNorm)
6. Full decoder-only transformer model
7. Training script with toy dataset
8. Text generation (inference) capability

## Current Status
- Git repository initialized
- Planning phase
