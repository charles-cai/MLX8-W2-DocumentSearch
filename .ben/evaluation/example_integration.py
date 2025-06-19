#!/usr/bin/env python3
"""
Example: How to Integrate Evaluation into Training

This shows how to add evaluation to your existing training loop.
"""

import torch
import sys
import os

# Add paths
sys.path.append('../training')
sys.path.append('.')

from eval_integration import evaluate_trained_model

def example_training_with_eval():
    """Example showing how to add evaluation to training loop."""
    
    print("📚 Example: Training with Evaluation Integration")
    print("=" * 60)
    
    # This is pseudo-code showing the integration points
    print("""
# In your training script, add these imports:
import sys
sys.path.append('../evaluation')
from eval_integration import evaluate_trained_model

# Then in your training loop:
def train_model():
    for epoch in range(num_epochs):
        
        # 1. Training phase (your existing code)
        model.train()
        for batch in dataloader:
            # ... your training code ...
            pass
        
        # 2. Evaluation phase (NEW!)
        if epoch % eval_frequency == 0:  # e.g., every epoch
            print(f"🔍 Evaluating after epoch {epoch + 1}...")
            
            eval_results = evaluate_trained_model(
                model=model,
                device=device,
                epoch=epoch + 1,
                data_path="./data",
                max_queries=1000  # Adjust for speed vs accuracy
            )
            
            # 3. Track best model (NEW!)
            current_mrr = eval_results.get('MRR', 0)
            if current_mrr > best_mrr:
                best_mrr = current_mrr
                print(f"🏆 New best model! MRR: {best_mrr:.4f}")
                
                # Save best model
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'eval_results': eval_results,
                    # ... other checkpoint data ...
                }, f"checkpoints/best_model_epoch{epoch+1}.pt")
            
            # 4. Early stopping (OPTIONAL)
            if epochs_without_improvement >= patience:
                print("🛑 Early stopping triggered")
                break
""")

def example_quick_evaluation():
    """Example of quick evaluation during development."""
    
    print("\n⚡ Quick Evaluation Example")
    print("=" * 40)
    
    print("""
# For rapid testing during development:
from eval_integration import quick_eval

# Quick eval with just 100 queries (very fast)
results = quick_eval("checkpoints/my_model.pt", max_queries=100)
print(f"Quick MRR: {results['MRR']:.4f}")
""")

def example_model_comparison():
    """Example of comparing multiple models."""
    
    print("\n📊 Model Comparison Example") 
    print("=" * 40)
    
    print("""
# Compare different checkpoints:
from eval_integration import compare_checkpoints

checkpoints = [
    "checkpoints/model_epoch1.pt",
    "checkpoints/model_epoch2.pt", 
    "checkpoints/model_epoch3.pt"
]

# This will print a nice comparison table
results = compare_checkpoints(checkpoints, max_queries=500)

# Output will look like:
# Checkpoint                     MRR      MRR@10   Recall@10  NDCG@10   
# model_epoch1                   0.2134   0.2134   0.4567     0.3201    
# model_epoch2                   0.2456   0.2456   0.4890     0.3456    
# model_epoch3                   0.2398   0.2398   0.4823     0.3389    
""")

def example_standalone_evaluation():
    """Example of evaluating a finished model."""
    
    print("\n🎯 Standalone Evaluation Example")
    print("=" * 40)
    
    print("""
# Evaluate a trained model from command line:

# Basic evaluation
uv run evaluation/ms_marco_evaluator.py checkpoints/my_model.pt

# Custom evaluation
uv run evaluation/ms_marco_evaluator.py checkpoints/my_model.pt \\
    --split validation \\
    --max-queries 2000 \\
    --batch-size 64 \\
    --output evaluation/my_results.json

# Quick evaluation  
uv run evaluation/eval_integration.py checkpoints/my_model.pt --quick
""")

def main():
    """Run all examples."""
    example_training_with_eval()
    example_quick_evaluation()
    example_model_comparison()
    example_standalone_evaluation()
    
    print("\n" + "=" * 60)
    print("✅ Integration Examples Complete!")
    print("\nNext Steps:")
    print("1. Add evaluation to your training loop")
    print("2. Use quick_eval() for rapid development")
    print("3. Compare models with compare_checkpoints()")
    print("4. Run full evaluation on your best model")
    print("\n📖 See evaluation/README.md for detailed usage!")

if __name__ == "__main__":
    main() 