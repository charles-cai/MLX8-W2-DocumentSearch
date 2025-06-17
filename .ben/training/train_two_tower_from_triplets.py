#!/usr/bin/env python3
"""
Train Two-Tower Model from Pre-generated Triplets

This script loads pre-generated MS MARCO triplets and trains the two-tower model.
This approach separates data preprocessing from training for better efficiency.
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

# Import model components
from two_tower_model import TwoTowerModel, ContrastiveLoss, TwoTowerCollator

class PreprocessedMSMarcoDataset(Dataset):
    """
    Dataset that loads pre-generated triplets from disk.
    """
    
    def __init__(self, triplets_file: str):
        self.triplets_file = triplets_file
        
        print(f"📦 Loading pre-generated triplets from: {triplets_file}")
        
        # Load triplets
        with open(triplets_file, 'rb') as f:
            data = pickle.load(f)
        
        self.triplets = data['triplets']
        self.metadata = data.get('metadata', {})
        
        print(f"✅ Loaded {len(self.triplets):,} training triplets")
        print(f"   Generated at: {self.metadata.get('generated_at', 'Unknown')}")
        print(f"   Original max samples: {self.metadata.get('max_samples', 'No limit')}")
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        triplet = self.triplets[idx]
        return {
            'query': triplet['query_text'],
            'positive_doc': triplet['pos_passage_text'],
            'negative_doc': triplet['neg_passage_text'],
            'query_id': triplet['query_id'],
            'pos_passage_id': triplet['pos_passage_id'],
            'neg_passage_id': triplet['neg_passage_id']
        }

def train_from_triplets(triplets_file: str, 
                       epochs: int = 1,
                       batch_size: int = 32,
                       learning_rate: float = 1e-4,
                       num_workers: int = 0,  # Default to 0 to avoid multiprocessing issues
                       save_model: bool = True,
                       checkpoint_dir: str = "./checkpoints"):
    """Train two-tower model from pre-generated triplets."""
    
    print("🚀 Two-Tower Training from Pre-generated Triplets")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load pre-trained embeddings
    print("📦 Loading pre-trained embeddings...")
    embeddings_path = os.path.join(checkpoint_dir, "msmarco_word2vec.pt")
    
    if not os.path.exists(embeddings_path):
        print(f"❌ Error: Embeddings file not found at: {embeddings_path}")
        print("💡 Make sure to run the CBOW training script first:")
        print(f"   uv run text_embeddings/train_text_embeddings_cbow_msmarco.py -c {checkpoint_dir}")
        raise FileNotFoundError(f"Pre-trained embeddings not found: {embeddings_path}")
    
    checkpoint = torch.load(embeddings_path, map_location='cpu')
    embedding_matrix = checkpoint["embedding_matrix"]
    word_to_index = checkpoint["word_to_index"]
    index_to_word = checkpoint["index_to_word"]
    
    print(f"✅ Loaded embeddings: {embedding_matrix.shape}")
    print(f"   Vocabulary size: {len(word_to_index):,}")
    
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
        print(f"⚡ Using multiprocessing with {num_workers} workers")
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
        print("🔧 Using single-process DataLoader (safer)")
        # Use lambda with num_workers=0 to avoid multiprocessing
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
    print(f"   Total batches per epoch: {len(dataloader)}")
    print(f"   Total training steps: {len(dataloader) * epochs}")
    
    # Training loop
    model.train()
    training_stats = []
    
    print("=" * 60)
    print("🎯 Starting Training")
    print("=" * 60)
    
    for epoch in range(epochs):
        epoch_loss = 0
        epoch_accuracy = 0
        num_batches = 0
        
        print(f"\n📚 Epoch {epoch + 1}/{epochs}")
        
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
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
            if (batch_idx + 1) % 100 == 0:
                avg_loss = epoch_loss / num_batches
                avg_acc = epoch_accuracy / num_batches
                lr = optimizer.param_groups[0]['lr']
                
                print(f"  Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                      f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {lr:.2e}")
        
        # Epoch summary
        final_loss = epoch_loss / num_batches
        final_accuracy = epoch_accuracy / num_batches
        
        epoch_stats = {
            'epoch': epoch + 1,
            'loss': final_loss,
            'accuracy': final_accuracy,
            'learning_rate': optimizer.param_groups[0]['lr']
        }
        training_stats.append(epoch_stats)
        
        print(f"📊 Epoch {epoch + 1} Summary:")
        print(f"   Loss: {final_loss:.4f}")
        print(f"   Accuracy: {final_accuracy:.4f}")
        print(f"   Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
    
    print("=" * 60)
    print("✅ Training completed!")
    
    # Final metrics
    final_stats = training_stats[-1]
    print(f"🏆 Final Results:")
    print(f"   Final Loss: {final_stats['loss']:.4f}")
    print(f"   Final Accuracy: {final_stats['accuracy']:.4f}")
    
    # Save model if requested
    if save_model:
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_filename = f"two_tower_model_{timestamp}.pt"
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
            'training_stats': training_stats,
            'training_config': {
                'epochs': epochs,
                'batch_size': batch_size,
                'learning_rate': learning_rate,
                'triplets_file': triplets_file,
                'num_samples': len(dataset)
            }
        }, model_save_path)
        
        print(f"💾 Model saved to: {model_save_path}")
        
        # Save training log
        log_file = model_save_path.replace('.pt', '_training_log.json')
        with open(log_file, 'w') as f:
            json.dump({
                'training_stats': training_stats,
                'final_metrics': final_stats,
                'config': {
                    'epochs': epochs,
                    'batch_size': batch_size,
                    'learning_rate': learning_rate,
                    'device': str(device),
                    'num_workers': num_workers
                }
            }, f, indent=2)
        
        print(f"📋 Training log saved to: {log_file}")
    
    # Quick test
    print("\n🔍 Quick model test...")
    test_model_quick(model, device)
    
    return model, training_stats

def test_model_quick(model, device):
    """Quick test of the trained model."""
    model.eval()
    
    test_queries = [
        "machine learning algorithms",
        "deep neural networks", 
        "information retrieval"
    ]
    
    test_docs = [
        "Machine learning algorithms learn patterns from data automatically.",
        "Deep neural networks use multiple layers for complex learning.",
        "Information retrieval helps find relevant documents efficiently.",
        "Cooking recipes include ingredients like flour and eggs."  # Negative
    ]
    
    with torch.no_grad():
        print("📊 Query-Document Similarities:")
        
        for query in test_queries:
            query_ids = model.tokenize(query).unsqueeze(0).to(device)
            query_emb = model.encode_query(query_ids)
            
            print(f"\nQuery: '{query}'")
            similarities = []
            
            for doc in test_docs:
                doc_ids = model.tokenize(doc).unsqueeze(0).to(device)
                doc_emb = model.encode_document(doc_ids)
                sim = torch.cosine_similarity(query_emb, doc_emb).item()
                similarities.append((sim, doc))
            
            # Sort and show top matches
            similarities.sort(reverse=True)
            for i, (sim, doc) in enumerate(similarities[:2]):
                print(f"  {i+1}. [{sim:.3f}] {doc[:50]}...")

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Train two-tower model from pre-generated triplets")
    parser.add_argument("triplets_file", 
                       help="Path to pre-generated triplets file (.pkl)")
    parser.add_argument("--epochs", type=int, default=1,
                       help="Number of training epochs (default: 1)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                       help="Learning rate (default: 1e-4)")
    parser.add_argument("--num-workers", type=int, default=0,
                       help="Number of DataLoader workers (default: 0 for safety)")
    parser.add_argument("--no-save", action="store_true",
                       help="Don't save the trained model")
    parser.add_argument("--checkpoint-dir", "-c", default="./checkpoints",
                       help="Directory to save model checkpoints (default: ./checkpoints)")
    
    args = parser.parse_args()
    
    # Validate triplets file
    if not os.path.exists(args.triplets_file):
        print(f"❌ Error: Triplets file not found: {args.triplets_file}")
        return
    
    # Train model
    model, stats = train_from_triplets(
        triplets_file=args.triplets_file,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_workers=args.num_workers,
        save_model=not args.no_save,
        checkpoint_dir=args.checkpoint_dir
    )
    
    print(f"\n🎉 Training completed successfully!")
    print(f"   Trained on: {args.triplets_file}")
    print(f"   Final accuracy: {stats[-1]['accuracy']:.4f}")

if __name__ == "__main__":
    main() 