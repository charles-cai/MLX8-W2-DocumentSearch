#!/usr/bin/env python3
"""
Evaluation Integration for Two-Tower Training

This script provides easy integration of evaluation into the training loop.
It can be called after each epoch to evaluate the model performance.
"""

import torch
import sys
import os
from typing import Dict, Optional

# Add paths for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, '..', 'training'))
sys.path.append(current_dir)

from ms_marco_evaluator import MSMarcoEvaluator, load_model_for_evaluation
from two_tower_model import TwoTowerModel

def evaluate_model_checkpoint(checkpoint_path: str, 
                            data_path: str = "./data",
                            split: str = "validation",
                            max_queries: int = 1000,
                            batch_size: int = 64,
                            save_results: bool = True) -> Dict:
    """
    Evaluate a model checkpoint and return metrics.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        data_path: Path to data directory
        split: Dataset split to evaluate on
        max_queries: Maximum number of queries to evaluate (for speed)
        batch_size: Batch size for evaluation
        save_results: Whether to save evaluation results
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"🔍 Evaluating checkpoint: {checkpoint_path}")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = load_model_for_evaluation(checkpoint_path, device)
    
    # Create evaluator
    evaluator = MSMarcoEvaluator(
        model=model, 
        device=device, 
        data_path=data_path, 
        batch_size=batch_size
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        split=split,
        max_queries=max_queries
    )
    
    # Save results if requested
    if save_results:
        # Create results filename based on checkpoint
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        results_path = f"evaluation/results_{checkpoint_name}_eval.json"
        
        evaluator.save_evaluation_results(
            results, 
            results_path,
            additional_info={
                'checkpoint_path': checkpoint_path,
                'evaluation_config': {
                    'split': split,
                    'max_queries': max_queries,
                    'batch_size': batch_size
                }
            }
        )
    
    return results

def evaluate_trained_model(model: TwoTowerModel, 
                          device: torch.device,
                          epoch: int,
                          data_path: str = "./data",
                          max_queries: int = 1000) -> Dict:
    """
    Evaluate a model during training (without saving/loading checkpoint).
    
    Args:
        model: The trained model
        device: Device the model is on
        epoch: Current epoch number
        data_path: Path to data directory
        max_queries: Maximum number of queries to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"🔍 Evaluating model at epoch {epoch}")
    
    # Create evaluator
    evaluator = MSMarcoEvaluator(
        model=model, 
        device=device, 
        data_path=data_path, 
        batch_size=64
    )
    
    # Run evaluation
    results = evaluator.evaluate(
        split="validation",
        max_queries=max_queries
    )
    
    # Add epoch info
    results['epoch'] = epoch
    
    return results

def quick_eval(checkpoint_path: str, max_queries: int = 100) -> Dict:
    """
    Quick evaluation with minimal queries for fast feedback.
    
    Args:
        checkpoint_path: Path to model checkpoint
        max_queries: Number of queries to evaluate (small for speed)
        
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"⚡ Quick evaluation of {os.path.basename(checkpoint_path)}")
    
    return evaluate_model_checkpoint(
        checkpoint_path=checkpoint_path,
        max_queries=max_queries,
        save_results=False
    )

def compare_checkpoints(checkpoint_paths: list, max_queries: int = 500) -> Dict:
    """
    Compare multiple checkpoints and return a comparison.
    
    Args:
        checkpoint_paths: List of checkpoint paths to compare
        max_queries: Number of queries to evaluate on each
        
    Returns:
        Dictionary with comparison results
    """
    print(f"📊 Comparing {len(checkpoint_paths)} checkpoints")
    
    results = {}
    
    for checkpoint_path in checkpoint_paths:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        print(f"\n🔍 Evaluating {checkpoint_name}...")
        
        try:
            metrics = evaluate_model_checkpoint(
                checkpoint_path=checkpoint_path,
                max_queries=max_queries,
                save_results=False
            )
            results[checkpoint_name] = metrics
        except Exception as e:
            print(f"❌ Error evaluating {checkpoint_name}: {e}")
            results[checkpoint_name] = None
    
    # Print comparison
    print("\n📊 Checkpoint Comparison:")
    print("=" * 80)
    print(f"{'Checkpoint':<30} {'MRR':<8} {'MRR@10':<8} {'Recall@10':<10} {'NDCG@10':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        if metrics:
            mrr = metrics.get('MRR', 0)
            mrr10 = metrics.get('MRR@10', 0)
            recall10 = metrics.get('Recall@10', 0)
            ndcg10 = metrics.get('NDCG@10', 0)
            print(f"{name:<30} {mrr:<8.4f} {mrr10:<8.4f} {recall10:<10.4f} {ndcg10:<10.4f}")
        else:
            print(f"{name:<30} {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")
    
    return results

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluation Integration for Two-Tower Model")
    parser.add_argument("checkpoint", help="Path to model checkpoint")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick evaluation with 100 queries")
    parser.add_argument("--max-queries", type=int, default=1000,
                       help="Maximum number of queries to evaluate")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save evaluation results")
    
    args = parser.parse_args()
    
    if args.quick:
        results = quick_eval(args.checkpoint)
    else:
        results = evaluate_model_checkpoint(
            checkpoint_path=args.checkpoint,
            max_queries=args.max_queries,
            save_results=not args.no_save
        )
    
    print(f"\n🎯 Key Results:")
    print(f"   MRR: {results.get('MRR', 0):.4f}")
    print(f"   MRR@10: {results.get('MRR@10', 0):.4f}")
    print(f"   Recall@10: {results.get('Recall@10', 0):.4f}")
    print(f"   NDCG@10: {results.get('NDCG@10', 0):.4f}") 