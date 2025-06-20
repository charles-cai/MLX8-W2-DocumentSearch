#!/usr/bin/env python3
"""
Evaluate All Checkpoints Script

This script evaluates all available checkpoints and generates proper training history files
with evaluation metrics. This fixes the issue where find_best_checkpoint.py shows zero MRR.

Usage:
    uv run evaluation/evaluate_all_checkpoints.py [--quick] [--max-queries 100] [--checkpoints-dir ./checkpoints]
"""

import os
import json
import glob
import argparse
from pathlib import Path
from typing import Dict, List
from datetime import datetime
import sys

# Add evaluation path to sys.path
sys.path.append('./evaluation')
from eval_integration import evaluate_model_checkpoint

def find_all_checkpoints(checkpoints_dir: str = "./checkpoints") -> List[str]:
    """Find all model checkpoint files (excluding word2vec)."""
    checkpoints = []
    
    for pt_file in glob.glob(os.path.join(checkpoints_dir, "*.pt")):
        if "word2vec" not in pt_file.lower():
            checkpoints.append(pt_file)
    
    return sorted(checkpoints)

def evaluate_checkpoint(checkpoint_path: str, quick: bool = False, max_queries: int = 100) -> Dict:
    """Evaluate a single checkpoint and return metrics."""
    print(f"\n🔍 Evaluating: {os.path.basename(checkpoint_path)}")
    
    try:
        if quick:
            results = evaluate_model_checkpoint(
                checkpoint_path=checkpoint_path,
                max_queries=max_queries,
                save_results=False
            )
        else:
            results = evaluate_model_checkpoint(
                checkpoint_path=checkpoint_path,
                max_queries=max_queries,
                save_results=True
            )
        
        print(f"✅ Evaluation completed - MRR: {results.get('MRR', 0):.4f}")
        return results
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return {"MRR": 0.0, "MRR@10": 0.0, "Recall@10": 0.0, "NDCG@10": 0.0, "MAP": 0.0, "error": str(e)}

def create_training_history(checkpoint_path: str, eval_results: Dict) -> Dict:
    """Create a training history dictionary from evaluation results."""
    checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # Extract metadata from filename
    sweep_name = None
    timestamp = None
    
    if "_best_" in checkpoint_name:
        parts = checkpoint_name.split("_best_")
        if len(parts) > 1:
            sweep_part = parts[1]
            if "_" in sweep_part:
                sweep_name, timestamp = sweep_part.rsplit("_", 1)
    
    # Create training history structure
    training_history = {
        "checkpoint_path": checkpoint_path,
        "checkpoint_name": checkpoint_name,
        "sweep_name": sweep_name,
        "timestamp": timestamp,
        "evaluation_timestamp": datetime.now().isoformat(),
        "best_metrics": eval_results,
        "best_epoch": 1,  # Assume best epoch for now
        "evaluation_stats": [eval_results],
        "training_stats": [
            {
                "epoch": 1,
                "eval_mrr": eval_results.get("MRR", 0),
                "eval_mrr@10": eval_results.get("MRR@10", 0),
                "eval_recall@10": eval_results.get("Recall@10", 0),
                "eval_ndcg@10": eval_results.get("NDCG@10", 0),
                "eval_map": eval_results.get("MAP", 0)
            }
        ],
        "config": {
            "evaluation_max_queries": eval_results.get("num_queries_evaluated", 0),
            "evaluation_type": "post_training_evaluation"
        }
    }
    
    return training_history

def save_training_history(training_history: Dict, checkpoints_dir: str = "./checkpoints"):
    """Save training history to JSON file."""
    checkpoint_name = training_history["checkpoint_name"]
    history_filename = f"training_history_{checkpoint_name}.json"
    history_path = os.path.join(checkpoints_dir, history_filename)
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"💾 Training history saved: {history_filename}")
    return history_path

def main():
    parser = argparse.ArgumentParser(description="Evaluate all checkpoints and generate training history")
    parser.add_argument("--quick", action="store_true", 
                       help="Quick evaluation with fewer queries")
    parser.add_argument("--max-queries", type=int, default=100,
                       help="Maximum number of queries for evaluation")
    parser.add_argument("--checkpoints-dir", default="./checkpoints",
                       help="Directory containing checkpoints")
    parser.add_argument("--skip-existing", action="store_true",
                       help="Skip checkpoints that already have training history")
    
    args = parser.parse_args()
    
    print("🚀 Evaluating All Checkpoints")
    print("=" * 60)
    print(f"Checkpoints directory: {args.checkpoints_dir}")
    print(f"Quick mode: {args.quick}")
    print(f"Max queries: {args.max_queries}")
    print(f"Skip existing: {args.skip_existing}")
    
    # Find all checkpoints
    checkpoints = find_all_checkpoints(args.checkpoints_dir)
    
    if not checkpoints:
        print("❌ No checkpoints found!")
        return
    
    print(f"\n📦 Found {len(checkpoints)} checkpoints:")
    for checkpoint in checkpoints:
        print(f"   - {os.path.basename(checkpoint)}")
    
    # Evaluate each checkpoint
    results = {}
    
    for checkpoint_path in checkpoints:
        checkpoint_name = os.path.basename(checkpoint_path).replace('.pt', '')
        history_filename = f"training_history_{checkpoint_name}.json"
        history_path = os.path.join(args.checkpoints_dir, history_filename)
        
        # Skip if history already exists and skip_existing is True
        if args.skip_existing and os.path.exists(history_path):
            print(f"⏭️  Skipping {checkpoint_name} (history exists)")
            continue
        
        # Evaluate checkpoint
        eval_results = evaluate_checkpoint(
            checkpoint_path=checkpoint_path,
            quick=args.quick,
            max_queries=args.max_queries
        )
        
        # Create and save training history
        training_history = create_training_history(checkpoint_path, eval_results)
        save_training_history(training_history, args.checkpoints_dir)
        
        results[checkpoint_name] = eval_results
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Evaluation Summary")
    print("=" * 60)
    
    # Sort by MRR
    sorted_results = sorted(results.items(), key=lambda x: x[1].get('MRR', 0), reverse=True)
    
    print(f"{'Checkpoint':<40} {'MRR':<8} {'MRR@10':<8} {'Recall@10':<10}")
    print("-" * 70)
    
    for checkpoint_name, metrics in sorted_results:
        mrr = metrics.get('MRR', 0)
        mrr10 = metrics.get('MRR@10', 0)
        recall10 = metrics.get('Recall@10', 0)
        print(f"{checkpoint_name:<40} {mrr:<8.4f} {mrr10:<8.4f} {recall10:<10.4f}")
    
    # Show best checkpoint
    if sorted_results:
        best_name, best_metrics = sorted_results[0]
        print(f"\n🏆 Best Checkpoint: {best_name}")
        print(f"   MRR: {best_metrics.get('MRR', 0):.4f}")
        print(f"   MRR@10: {best_metrics.get('MRR@10', 0):.4f}")
        print(f"   Recall@10: {best_metrics.get('Recall@10', 0):.4f}")
        print(f"   NDCG@10: {best_metrics.get('NDCG@10', 0):.4f}")
    
    print(f"\n✅ Evaluation completed for {len(results)} checkpoints")
    print("💡 Now run 'uv run find_best_checkpoint.py' to see the updated rankings!")

if __name__ == "__main__":
    main() 