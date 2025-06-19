#!/usr/bin/env python3
"""
Two-Tower Training with Integrated Evaluation

This script extends the standard training to include evaluation after each epoch.
It saves the best model based on validation metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm
import argparse
from typing import Dict, List
import json
from datetime import datetime
import sys

# Import training components
from train_two_tower_from_triplets import PreprocessedMSMarcoDataset, train_from_triplets
from two_tower_model import TwoTowerModel, ContrastiveLoss, TwoTowerCollator

# Import evaluation components
sys.path.append('./evaluation')
from eval_integration import evaluate_trained_model

def train_with_evaluation(triplets_file: str, 
                         epochs: int = 3,
                         batch_size: int = 32,
                         learning_rate: float = 1e-4,
                         num_workers: int = 0,
                         checkpoint_dir: str = "./checkpoints",
                         eval_every_epoch: bool = True,
                         eval_max_queries: int = 1000,
                         save_best_only: bool = True,
                         patience: int = 3) -> Dict:
    """
    Train two-tower model with evaluation after each epoch.
    
    Args:
        triplets_file: Path to pre-generated triplets
        epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        num_workers: Number of DataLoader workers
        checkpoint_dir: Directory to save checkpoints
        eval_every_epoch: Whether to evaluate after each epoch
        eval_max_queries: Max queries for evaluation (for speed)
        save_best_only: Only save checkpoints that improve validation metrics
        patience: Early stopping patience (epochs without improvement)
        
    Returns:
        Dictionary with training and evaluation history
    """
    
    print("🚀 Two-Tower Training with Integrated Evaluation")
    print("=" * 70)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load pre-trained embeddings
    print("📦 Loading pre-trained embeddings...")
    embeddings_path = os.path.join(checkpoint_dir, "msmarco_word2vec.pt")
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: Embeddings file not found at: {embeddings_path}")
        raise FileNotFoundError(f"Pre-trained embeddings not found: {embeddings_path}")
    
    checkpoint = torch.load(embeddings_path, map_location='cpu')
    embedding_matrix = checkpoint["embedding_matrix"]
    word_to_index = checkpoint["word_to_index"]
    index_to_word = checkpoint["index_to_word"]
    
    print(f"✅ Loaded embeddings: {embedding_matrix.shape}")
    
    # Create model
    model = TwoTowerModel(
        embedding_matrix=embedding_matrix,
        word_to_index=word_to_index,
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    print(f"🏗️  Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Load dataset
    dataset = PreprocessedMSMarcoDataset(triplets_file)
    
    # Create DataLoader
    if num_workers > 0:
        collator = TwoTowerCollator(model)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=collator,
            pin_memory=True if device.type == 'cuda' else False
        )
    else:
        from two_tower_model import collate_fn
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: collate_fn(batch, model)
        )
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(dataloader) * epochs
    )
    
    print(f"🎯 Training configuration:")
    print(f"   Epochs: {epochs}")
    print(f"   Batch size: {batch_size}")
    print(f"   Learning rate: {learning_rate}")
    print(f"   Evaluation every epoch: {eval_every_epoch}")
    print(f"   Evaluation max queries: {eval_max_queries}")
    print(f"   Save best only: {save_best_only}")
    
    # Training history
    training_history = {
        'training_stats': [],
        'evaluation_stats': [],
        'best_metrics': {},
        'best_epoch': 0,
        'config': {
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'eval_max_queries': eval_max_queries
        }
    }
    
    best_mrr = 0.0
    epochs_without_improvement = 0
    
    print("=" * 70)
    print("🎯 Starting Training with Evaluation")
    print("=" * 70)
    
    for epoch in range(epochs):
        print(f"\n📚 Epoch {epoch + 1}/{epochs}")
        
        # Training phase
        model.train()
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Training Epoch {epoch+1}")):
            # Move to device
            query_ids = batch['query_ids'].to(device)
            pos_doc_ids = batch['pos_doc_ids'].to(device)
            neg_doc_ids = batch['neg_doc_ids'].to(device)
            
            # Forward pass
            optimizer.zero_grad()
            query_emb, pos_doc_emb, neg_doc_emb = model(query_ids, pos_doc_ids, neg_doc_ids)
            
            # Compute loss
            loss, accuracy = criterion(query_emb, pos_doc_emb, neg_doc_emb)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            
            # Track metrics
            epoch_loss += loss.item()
            epoch_accuracy += accuracy.item()
            num_batches += 1
            
            # Log progress
            if (batch_idx + 1) % 200 == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_accuracy / num_batches
                lr = optimizer.param_groups[0]['lr']
                
                print(f"  Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {lr:.2e}")
        
        # Epoch training summary
        final_loss = epoch_loss / num_batches
        final_accuracy = epoch_accuracy / num_batches
        
        epoch_stats = {
            'epoch': epoch + 1,
            'train_loss': final_loss,
            'train_accuracy': final_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        
        print(f"📊 Epoch {epoch + 1} Training Summary:")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Accuracy: {final_accuracy:.4f}")
        
        # Evaluation phase
        if eval_every_epoch:
            print(f"\n🔍 Evaluating after epoch {epoch + 1}...")
            
            try:
                eval_results = evaluate_trained_model(
                    model=model,
                    device=device,
                    epoch=epoch + 1,
                    data_path="./data",
                    max_queries=eval_max_queries
                )
                
                # Add evaluation results to epoch stats
                epoch_stats.update({
                    'eval_mrr': eval_results.get('MRR', 0),
                    'eval_mrr@10': eval_results.get('MRR@10', 0),
                    'eval_recall@10': eval_results.get('Recall@10', 0),
                    'eval_ndcg@10': eval_results.get('NDCG@10', 0),
                    'eval_map': eval_results.get('MAP', 0)
                })
                
                training_history['evaluation_stats'].append(eval_results)
                
                print(f"📈 Validation Results:")
                print(f"   MRR: {eval_results.get('MRR', 0):.4f}")
                print(f"   MRR@10: {eval_results.get('MRR@10', 0):.4f}")
                print(f"   Recall@10: {eval_results.get('Recall@10', 0):.4f}")
                print(f"   NDCG@10: {eval_results.get('NDCG@10', 0):.4f}")
                
                # Check if this is the best model
                current_mrr = eval_results.get('MRR', 0)
                if current_mrr > best_mrr:
                    best_mrr = current_mrr
                    training_history['best_metrics'] = eval_results
                    training_history['best_epoch'] = epoch + 1
                    epochs_without_improvement = 0
                    
                    print(f"🏆 New best model! MRR improved to {best_mrr:.4f}")
                    
                    # Save best model
                    if save_best_only or epoch == epochs - 1:  # Always save last epoch
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        model_filename = f"two_tower_best_epoch{epoch+1}_{timestamp}.pt"
                        model_save_path = os.path.join(checkpoint_dir, model_filename)
                        
                        torch.save({
                            'model_state_dict': model.state_dict(),
                            'embedding_matrix': embedding_matrix,
                            'word_to_index': word_to_index,
                            'index_to_word': index_to_word,
                            'model_config': {
                                'hidden_dim': 256,
                                'num_layers': 2,
                                'dropout': 0.1
                            },
                            'training_stats': training_history,
                            'epoch': epoch + 1,
                            'eval_results': eval_results
                        }, model_save_path)
                        
                        print(f"💾 Best model saved to: {model_save_path}")
                else:
                    epochs_without_improvement += 1
                    print(f"📉 No improvement. Best MRR: {best_mrr:.4f} (epoch {training_history['best_epoch']})")
                    
            except Exception as e:
                print(f"⚠️  Evaluation failed: {e}")
                epoch_stats.update({
                    'eval_error': str(e)
                })
        
        training_history['training_stats'].append(epoch_stats)
        
        # Early stopping check
        if patience > 0 and epochs_without_improvement >= patience:
            print(f"\n🛑 Early stopping triggered after {patience} epochs without improvement")
            break
    
    print("=" * 70)
    print("✅ Training with Evaluation completed!")
    
    if training_history['best_metrics']:
        print(f"🏆 Best Results (Epoch {training_history['best_epoch']}):")
        best = training_history['best_metrics']
        print(f"   MRR: {best.get('MRR', 0):.4f}")
        print(f"   MRR@10: {best.get('MRR@10', 0):.4f}")
        print(f"   Recall@10: {best.get('Recall@10', 0):.4f}")
        print(f"   NDCG@10: {best.get('NDCG@10', 0):.4f}")
    
    # Save training history
    history_file = os.path.join(checkpoint_dir, f"training_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"📋 Training history saved to: {history_file}")
    
    return training_history

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train two-tower model with evaluation")
    parser.add_argument("triplets_file", 
                       help="Path to pre-generated triplets file (.pkl)")
    parser.add_argument("--epochs", type=int, default=3,
                       help="Number of training epochs (default: 3)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of DataLoader workers (default: 0)")
    parser.add_argument("--checkpoint-dir", "-c", default="./checkpoints",
                       help="Directory to save model checkpoints (default: ./checkpoints)")
    parser.add_argument("--no-eval", action="store_true",
                       help="Skip evaluation after each epoch")
    parser.add_argument("--eval-queries", type=int, default=1000,
                       help="Max queries for evaluation (default: 1000)")
    parser.add_argument("--save-all", action="store_true",
                       help="Save all epochs, not just the best")
    parser.add_argument("--patience", type=int, default=3,
                       help="Early stopping patience (default: 3, 0 to disable)")
    
    args = parser.parse_args()
    
    # Validate triplets file
    if not os.path.exists(args.triplets_file):
        print(f"❌ Error: Triplets file not found: {args.triplets_file}")
        return
    
    # Train model with evaluation
    history = train_with_evaluation(
        triplets_file=args.triplets_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        checkpoint_dir=args.checkpoint_dir,
        eval_every_epoch=not args.no_eval,
        eval_max_queries=args.eval_queries,
        save_best_only=not args.save_all,
        patience=args.patience
    )
    
    print(f"\n🎉 Training completed successfully!")
    if history['best_metrics']:
        print(f"   Best MRR: {history['best_metrics'].get('MRR', 0):.4f} (Epoch {history['best_epoch']})")

if __name__ == "__main__":
    main() 