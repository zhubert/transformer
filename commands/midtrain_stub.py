#!/usr/bin/env python3
"""
Mid-Training (Continued Pre-Training) - Domain Specialization

This command demonstrates mid-training: taking a pre-trained base model and
specializing it for a specific domain (code, math, science) while preventing
catastrophic forgetting.

STATUS: Educational demonstration ready. Full dataset integration coming soon.

What is Mid-Training?
---------------------
Mid-training (also called "continued pre-training") is where pre-trained models
become domain experts:

- Codex: GPT-3 → specialized on code
- Minerva: PaLM → specialized on math/science  
- Code Llama: Llama 2 → specialized on code
- Galactica: LLaMA → specialized on scientific literature

Same Architecture, Different Data:
- Uses the SAME transformer from pre-training
- Trains on domain-specific curated data
- Uses LOWER learning rate to preserve general capability
- Mixes domain data with general data to prevent forgetting

Key Concepts Demonstrated:
--------------------------
1. **Curriculum Learning**: Start easy, progressively harder
2. **Catastrophic Forgetting Detection**: Monitor general capability loss  
3. **Dual Evaluation**: Track both domain AND general perplexity
4. **Data Mixing**: Blend domain (90%) + general (10%) data
5. **Lower Learning Rate**: 1e-5 vs 3e-4 (30x lower than pre-training)

Educational Purpose:
====================
This is a DEMONSTRATION of mid-training concepts using the infrastructure
we've built:
- src/transformer/domain_datasets.py
- src/transformer/curriculum.py
- src/transformer/forgetting_metrics.py

The command shows HOW mid-training works. Full dataset integration with
HuggingFace datasets is the next step for production use.

Usage:
======
From interactive CLI:
    python main.py
    → Select "Start mid-training"
    → Choose domain (code/math/science)
    → Select base model from pre-training

From command line:
    python -m commands.midtrain_stub \\
        --base-checkpoint checkpoints/model_epoch_10_fineweb.pt \\
        --domain code \\
        --epochs 10

For implementation details, see:
    - docs/src/content/docs/pipeline.mdx (concepts)
    - CLAUDE.md (architecture decisions)
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.transformer.curriculum import create_math_curriculum, create_code_curriculum
from src.transformer.forgetting_metrics import ForgettingDetector, ForgettingMetrics
from src.transformer.domain_datasets import create_domain_dataset


def demonstrate_midtraining_concepts():
    """
    Demonstrate mid-training concepts with our infrastructure.
   
    This shows how the pieces fit together:
    1. Curriculum learning scheduler
    2. Catastrophic forgetting detection
    3. Domain datasets
    """
    print("=" * 80)
    print("MID-TRAINING CONCEPTS DEMONSTRATION")
    print("=" * 80)
    print()
    
    # 1. Curriculum Learning
    print("1. CURRICULUM LEARNING")
    print("-" * 80)
    curriculum = create_math_curriculum(total_epochs=9, difficulty_levels=3)
    curriculum.print_schedule()
    
    # 2. Catastrophic Forgetting Detection
    print("\n2. CATASTROPHIC FORGETTING DETECTION")
    print("-" * 80)
    detector = ForgettingDetector(baseline_general_perplexity=25.0)
    
    # Simulate some epochs
    for epoch in range(5):
        metrics = ForgettingMetrics(
            epoch=epoch,
            train_loss=3.0 - (epoch * 0.1),
            domain_perplexity=22.0 - (epoch * 0.5),
            general_perplexity=25.0 + (epoch * 0.3),
        )
        detector.record(metrics)
    
    detector.print_summary()
    
    # 3. Domain Dataset Info
    print("\n3. DOMAIN DATASETS")
    print("-" * 80)
    code_ds = create_domain_dataset('code', languages=['python'])
    info = code_ds.get_domain_info()
    
    print(f"Domain: {info['domain']}")
    print(f"Description: {info['description']}")
    print(f"Source: {info['source']}")
    print(f"Use cases:")
    for use_case in info['use_cases']:
        print(f"  - {use_case}")
    
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("This demonstration shows the infrastructure is ready.")
    print("To enable full mid-training:")
    print()
    print("1. Integrate HuggingFace datasets in domain_datasets.py")
    print("   - Remove placeholder returns")
    print("   - Add actual dataset loading (codeparrot/github-code, etc.)")
    print()
    print("2. Create full training loop in commands/midtrain.py")  
    print("   - Load base checkpoint")
    print("   - Mix domain + general data")
    print("   - Apply curriculum if enabled")
    print("   - Track dual metrics")
    print("   - Detect forgetting")
    print()
    print("3. Update interactive CLI to call full command")
    print("   - src/interactive.py midtrain_menu()")
    print()
    print("=" * 80)


if __name__ == "__main__":
    demonstrate_midtraining_concepts()
