#!/usr/bin/env python3
"""
Two-Tower Architecture for MS MARCO Document Ranking

This implements a dual encoder model where:
- Query Tower: Encodes queries into dense vectors
- Document Tower: Encodes documents into dense vectors
- Training: Uses contrastive learning with positive/negative document pairs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
import pickle
import os
from tqdm import tqdm
import random
from typing import Dict, List, Tuple
import json
import re
from gensim.utils import simple_preprocess

def clean_text(text):
    """Clean and preprocess text (same as CBOW training)."""
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

class TwoTowerModel(nn.Module):
    """
    Two-Tower architecture with separate RNN encoders for queries and documents.
    """
    
    def __init__(self, embedding_matrix: torch.Tensor, word_to_index: Dict, 
                 hidden_dim: int = 256, num_layers: int = 2, dropout: float = 0.1):
        super(TwoTowerModel, self).__init__()
        
        self.word_to_index = word_to_index
        self.vocab_size = len(word_to_index)
        self.embedding_dim = embedding_matrix.size(1)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared embedding layer (frozen pre-trained embeddings)
        self.embedding = nn.Embedding.from_pretrained(embedding_matrix, freeze=True)
        
        # Query Tower - RNN + Projection
        self.query_rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.query_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Document Tower - RNN + Projection  
        self.document_rnn = nn.RNN(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True,
            batch_first=True
        )
        self.document_projection = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize RNN weights
        for rnn in [self.query_rnn, self.document_rnn]:
            for name, param in rnn.named_parameters():
                if 'weight_ih' in name:
                    nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
        
        # Initialize projection layers
        for module in [self.query_projection, self.document_projection]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)
    
    def tokenize(self, text: str, max_len: int = 128) -> torch.Tensor:
        """
        Tokenize text using the same preprocessing as CBOW training.
        
        Args:
            text: Input text
            max_len: Maximum sequence length
            
        Returns:
            Token IDs tensor
        """
        # Handle None or empty text
        if not text or not text.strip():
            # Return a tensor with at least one non-zero token (unknown token)
            token_ids = [self.word_to_index.get("<unk>", 1)]  # Use 1 instead of 0 for unknown
            token_ids += [0] * (max_len - len(token_ids))
            return torch.tensor(token_ids, dtype=torch.long)
        
        # Use the SAME preprocessing as CBOW training
        cleaned_text = clean_text(text)
        
        if not cleaned_text:
            # If cleaning results in empty text, use unknown token
            tokens = ["<unk>"]
        else:
            # Use Gensim's simple_preprocess (same as CBOW training)
            tokens = simple_preprocess(cleaned_text)
        
        # If no tokens after preprocessing, add unknown token
        if not tokens:
            tokens = ["<unk>"]
        
        token_ids = [self.word_to_index.get(tok, self.word_to_index.get("<unk>", 1)) 
                    for tok in tokens]
        
        # Ensure we have at least one non-zero token
        if all(tid == 0 for tid in token_ids):
            token_ids[0] = self.word_to_index.get("<unk>", 1)
        
        # Truncate or pad
        token_ids = token_ids[:max_len]
        token_ids += [0] * (max_len - len(token_ids))
        
        return torch.tensor(token_ids, dtype=torch.long)
    
    def encode_text_with_rnn(self, token_ids: torch.Tensor, rnn_layer: nn.RNN) -> torch.Tensor:
        """
        Encode token IDs to embeddings using RNN.
        
        Args:
            token_ids: Token IDs tensor [batch_size, seq_len]
            rnn_layer: RNN layer to use for encoding
            
        Returns:
            Text embeddings [batch_size, hidden_dim * 2] (bidirectional)
        """
        # Get embeddings
        embeddings = self.embedding(token_ids)  # [batch_size, seq_len, embedding_dim]
        
        # Calculate actual sequence lengths (for packing)
        seq_lengths = (token_ids != 0).sum(dim=1).cpu()  # [batch_size]
        
        # Handle empty sequences - ensure minimum length of 1
        seq_lengths = torch.clamp(seq_lengths, min=1)
        
        # Pack padded sequences for efficient RNN processing
        packed_embeddings = nn.utils.rnn.pack_padded_sequence(
            embeddings, seq_lengths, batch_first=True, enforce_sorted=False
        )
        
        # Pass through RNN
        packed_output, hidden = rnn_layer(packed_embeddings)
        
        # Unpack the sequences
        rnn_output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True
        )
        
        # Use the last hidden state from both directions
        # hidden: [num_layers * num_directions, batch_size, hidden_dim]
        # We want the last layer's hidden states from both directions
        final_hidden = hidden[-2:, :, :]  # [2, batch_size, hidden_dim] (forward + backward)
        final_hidden = final_hidden.transpose(0, 1).contiguous()  # [batch_size, 2, hidden_dim]
        final_hidden = final_hidden.view(final_hidden.size(0), -1)  # [batch_size, hidden_dim * 2]
        
        return final_hidden
    
    def encode_query(self, query_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode queries through the query tower RNN.
        
        Args:
            query_ids: Query token IDs [batch_size, seq_len]
            
        Returns:
            Query embeddings [batch_size, hidden_dim]
        """
        # Encode with query RNN
        rnn_output = self.encode_text_with_rnn(query_ids, self.query_rnn)
        
        # Project to final dimension
        query_emb = self.query_projection(rnn_output)
        
        return F.normalize(query_emb, p=2, dim=1)  # L2 normalize
    
    def encode_document(self, doc_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode documents through the document tower RNN.
        
        Args:
            doc_ids: Document token IDs [batch_size, seq_len]
            
        Returns:
            Document embeddings [batch_size, hidden_dim]
        """
        # Encode with document RNN
        rnn_output = self.encode_text_with_rnn(doc_ids, self.document_rnn)
        
        # Project to final dimension
        doc_emb = self.document_projection(rnn_output)
        
        return F.normalize(doc_emb, p=2, dim=1)  # L2 normalize
    
    def forward(self, query_ids: torch.Tensor, pos_doc_ids: torch.Tensor, 
                neg_doc_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            query_ids: Query token IDs [batch_size, seq_len]
            pos_doc_ids: Positive document token IDs [batch_size, seq_len]
            neg_doc_ids: Negative document token IDs [batch_size, seq_len]
            
        Returns:
            Query embeddings, positive doc embeddings, negative doc embeddings
        """
        query_emb = self.encode_query(query_ids)
        pos_doc_emb = self.encode_document(pos_doc_ids)
        neg_doc_emb = self.encode_document(neg_doc_ids)
        
        return query_emb, pos_doc_emb, neg_doc_emb

class MSMarcoDataset(Dataset):
    """
    MS MARCO dataset for two-tower training.
    """
    
    def __init__(self, data_path: str = "./data", max_len: int = 128, 
                 max_samples: int = None):
        self.data_path = data_path
        self.max_len = max_len
        
        # Load cached data or download
        self.queries, self.passages, self.qrels = self._load_data()
        
        # Create training triplets (query, positive_doc, negative_doc)
        self.triplets = self._create_triplets(max_samples)
        
        print(f"📊 Dataset loaded: {len(self.triplets)} training triplets")
    
    def _load_data(self):
        """Load MS MARCO data from cache or download."""
        cache_file = os.path.join(self.data_path, "msmarco_train_data.pkl")
        
        if os.path.exists(cache_file):
            print("📦 Loading cached MS MARCO training data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['queries'], data['passages'], data['qrels']
        
        print("🔄 Loading MS MARCO dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
        
        # Process queries and passages
        queries = {}
        passages = {}
        qrels = {}  # query_id -> [relevant_passage_ids]
        
        print("🔄 Processing dataset...")
        
        # Debug: Check first item structure
        first_item = next(iter(dataset))
        print(f"📋 Dataset structure preview:")
        print(f"   Keys: {list(first_item.keys())}")
        print(f"   Query: {first_item['query']}")
        print(f"   Passages type: {type(first_item['passages'])}")
        if isinstance(first_item['passages'], dict):
            print(f"   Passages keys: {list(first_item['passages'].keys())}")
        elif isinstance(first_item['passages'], list) and len(first_item['passages']) > 0:
            print(f"   First passage type: {type(first_item['passages'][0])}")
            if isinstance(first_item['passages'][0], dict):
                print(f"   First passage keys: {list(first_item['passages'][0].keys())}")
        
        for item in tqdm(dataset):
            query_id = str(item['query_id'])
            query_text = item['query']
            
            queries[query_id] = query_text
            qrels[query_id] = []
            
            # Process passages - MS MARCO v1.1 format
            passages_data = item['passages']
            if isinstance(passages_data, dict) and 'passage_text' in passages_data:
                # Handle dict format with lists
                passage_texts = passages_data['passage_text']
                is_selected_list = passages_data['is_selected']
                
                for i, (passage_text, is_selected) in enumerate(zip(passage_texts, is_selected_list)):
                    passage_id = f"{query_id}_{i}"
                    passages[passage_id] = passage_text
                    
                    # If passage is relevant (is_selected = 1)
                    if is_selected == 1:
                        qrels[query_id].append(passage_id)
        
        # Cache the processed data
        os.makedirs(self.data_path, exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'queries': queries,
                'passages': passages,
                'qrels': qrels
            }, f)
        
        print(f"✅ Processed {len(queries)} queries and {len(passages)} passages")
        # qrels (query relevance judgments) maps query IDs to lists of relevant passage IDs,
        # indicating which passages were marked as relevant for each query
        return queries, passages, qrels
    
    def _create_triplets(self, max_samples: int = None):
        """Create training triplets (query, positive_doc, negative_doc)."""
        triplets = []
        
        # Get all passage IDs for negative sampling
        all_passage_ids = list(self.passages.keys())
        
        print("🔄 Creating training triplets...")
        for query_id, relevant_passage_ids in tqdm(self.qrels.items()):
            if not relevant_passage_ids:  # Skip queries with no relevant passages
                continue
            
            # For each query, create multiple triplets
            for pos_passage_id in relevant_passage_ids:
                # Sample negative passages (not relevant to this query)
                neg_candidates = [pid for pid in all_passage_ids 
                                if pid not in relevant_passage_ids]
                
                if neg_candidates:
                    neg_passage_id = random.choice(neg_candidates)
                    triplets.append((query_id, pos_passage_id, neg_passage_id))
        
        # Shuffle and limit if specified
        random.shuffle(triplets)
        if max_samples and len(triplets) > max_samples:
            triplets = triplets[:max_samples]
        
        return triplets
    
    def __len__(self):
        return len(self.triplets)
    
    def __getitem__(self, idx):
        query_id, pos_passage_id, neg_passage_id = self.triplets[idx]
        
        return {
            'query': self.queries[query_id],
            'positive_doc': self.passages[pos_passage_id],
            'negative_doc': self.passages[neg_passage_id],
            'query_id': query_id,
            'pos_passage_id': pos_passage_id,
            'neg_passage_id': neg_passage_id
        }

class TwoTowerCollator:
    """
    Collate function class for DataLoader that can be pickled for multiprocessing.
    """
    
    def __init__(self, model):
        self.model = model
    
    def __call__(self, batch):
        """
        Collate function for DataLoader.
        """
        queries = [item['query'] for item in batch]
        pos_docs = [item['positive_doc'] for item in batch]
        neg_docs = [item['negative_doc'] for item in batch]
        
        # Tokenize
        query_ids = torch.stack([self.model.tokenize(q) for q in queries])
        pos_doc_ids = torch.stack([self.model.tokenize(d) for d in pos_docs])
        neg_doc_ids = torch.stack([self.model.tokenize(d) for d in neg_docs])
        
        return {
            'query_ids': query_ids,
            'pos_doc_ids': pos_doc_ids,
            'neg_doc_ids': neg_doc_ids,
            'queries': queries,
            'pos_docs': pos_docs,
            'neg_docs': neg_docs
        }

def collate_fn(batch, model):
    """
    Legacy collate function for DataLoader (kept for compatibility).
    """
    queries = [item['query'] for item in batch]
    pos_docs = [item['positive_doc'] for item in batch]
    neg_docs = [item['negative_doc'] for item in batch]
    
    # Tokenize
    query_ids = torch.stack([model.tokenize(q) for q in queries])
    pos_doc_ids = torch.stack([model.tokenize(d) for d in pos_docs])
    neg_doc_ids = torch.stack([model.tokenize(d) for d in neg_docs])
    
    return {
        'query_ids': query_ids,
        'pos_doc_ids': pos_doc_ids,
        'neg_doc_ids': neg_doc_ids,
        'queries': queries,
        'pos_docs': pos_docs,
        'neg_docs': neg_docs
    }

class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for two-tower training using InfoNCE approach.
    """
    
    def __init__(self, temperature: float = 0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
    
    def forward(self, query_emb: torch.Tensor, pos_doc_emb: torch.Tensor, 
                neg_doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            query_emb: Query embeddings [batch_size, hidden_dim]
            pos_doc_emb: Positive document embeddings [batch_size, hidden_dim]
            neg_doc_emb: Negative document embeddings [batch_size, hidden_dim]
            
        Returns:
            Contrastive loss
        """
        # Compute similarities
        pos_sim = torch.sum(query_emb * pos_doc_emb, dim=1) / self.temperature
        neg_sim = torch.sum(query_emb * neg_doc_emb, dim=1) / self.temperature
        
        # Contrastive loss: maximize positive similarity, minimize negative similarity
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # [batch_size, 2]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        
        loss = F.cross_entropy(logits, labels)
        
        # Compute accuracy
        predictions = torch.argmax(logits, dim=1)
        accuracy = (predictions == labels).float().mean()
        
        return loss, accuracy

class MarginContrastiveLoss(nn.Module):
    """
    Traditional margin-based contrastive loss for two-tower training.
    
    Loss = max(0, margin - pos_sim + neg_sim)
    This encourages pos_sim > neg_sim + margin
    """
    
    def __init__(self, margin: float = 0.2, temperature: float = 1.0):
        super(MarginContrastiveLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, query_emb: torch.Tensor, pos_doc_emb: torch.Tensor, 
                neg_doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute margin-based contrastive loss.
        
        Args:
            query_emb: Query embeddings [batch_size, hidden_dim]
            pos_doc_emb: Positive document embeddings [batch_size, hidden_dim]
            neg_doc_emb: Negative document embeddings [batch_size, hidden_dim]
            
        Returns:
            Margin contrastive loss and accuracy
        """
        # Compute similarities (cosine similarity since embeddings are L2 normalized)
        pos_sim = torch.sum(query_emb * pos_doc_emb, dim=1) / self.temperature
        neg_sim = torch.sum(query_emb * neg_doc_emb, dim=1) / self.temperature
        
        # Margin loss: max(0, margin - (pos_sim - neg_sim))
        # Encourages pos_sim to be at least margin higher than neg_sim
        loss = torch.clamp(self.margin - (pos_sim - neg_sim), min=0.0)
        loss = loss.mean()
        
        # Compute accuracy (positive similarity should be higher than negative)
        correct = (pos_sim > neg_sim).float()
        accuracy = correct.mean()
        
        return loss, accuracy

class TripletLoss(nn.Module):
    """
    Triplet loss for two-tower training.
    
    Loss = max(0, margin - pos_sim + neg_sim)
    Same as MarginContrastiveLoss but with different naming convention.
    """
    
    def __init__(self, margin: float = 0.2, temperature: float = 1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin
        self.temperature = temperature
    
    def forward(self, query_emb: torch.Tensor, pos_doc_emb: torch.Tensor, 
                neg_doc_emb: torch.Tensor) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            query_emb: Query embeddings (anchor) [batch_size, hidden_dim]
            pos_doc_emb: Positive document embeddings [batch_size, hidden_dim]
            neg_doc_emb: Negative document embeddings [batch_size, hidden_dim]
            
        Returns:
            Triplet loss and accuracy
        """
        # Compute distances (using negative cosine similarity as distance)
        pos_dist = -torch.sum(query_emb * pos_doc_emb, dim=1) / self.temperature
        neg_dist = -torch.sum(query_emb * neg_doc_emb, dim=1) / self.temperature
        
        # Triplet loss: max(0, pos_dist - neg_dist + margin)
        # Encourages pos_dist < neg_dist - margin (positive closer than negative)
        loss = torch.clamp(pos_dist - neg_dist + self.margin, min=0.0)
        loss = loss.mean()
        
        # Compute accuracy (positive distance should be smaller than negative)
        correct = (pos_dist < neg_dist).float()
        accuracy = correct.mean()
        
        return loss, accuracy

def train_two_tower_model():
    """Main training function."""
    print("🚀 Starting Two-Tower Model Training")
    print("=" * 50)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"🔧 Using device: {device}")
    
    # Load pre-trained embeddings
    print("📦 Loading pre-trained embeddings...")
    checkpoint = torch.load("msmarco_word2vec.pt", map_location='cpu')
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
    
    # Create dataset
    dataset = MSMarcoDataset(max_samples=10000)  # Limit for faster training
    
    # Create DataLoader with class-based collator for multiprocessing support
    collator = TwoTowerCollator(model)
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        num_workers=2,
        collate_fn=collator
    )
    
    # Loss and optimizer
    criterion = ContrastiveLoss(temperature=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(dataloader))
    
    # Training loop
    model.train()
    total_loss = 0
    total_accuracy = 0
    num_batches = 0
    
    print(f"🎯 Training on {len(dataset)} samples, {len(dataloader)} batches")
    print("=" * 50)
    
    for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training")):
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
        total_loss += loss.item()
        total_accuracy += accuracy.item()
        num_batches += 1
        
        # Log progress
        if (batch_idx + 1) % 100 == 0:
            avg_loss = total_loss / num_batches
            avg_acc = total_accuracy / num_batches
            lr = optimizer.param_groups[0]['lr']
            
            print(f"Batch {batch_idx + 1:4d}/{len(dataloader)} | "
                  f"Loss: {avg_loss:.4f} | Acc: {avg_acc:.4f} | LR: {lr:.2e}")
    
    # Final metrics
    final_loss = total_loss / num_batches
    final_accuracy = total_accuracy / num_batches
    
    print("=" * 50)
    print(f"✅ Training completed!")
    print(f"   Final Loss: {final_loss:.4f}")
    print(f"   Final Accuracy: {final_accuracy:.4f}")
    
    # Save model
    model_save_path = "two_tower_model.pt"
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
        'training_stats': {
            'final_loss': final_loss,
            'final_accuracy': final_accuracy,
            'num_samples': len(dataset)
        }
    }, model_save_path)
    
    print(f"💾 Model saved to: {model_save_path}")
    
    # Test the model with sample queries
    print("\n🔍 Testing model with sample queries...")
    test_model(model, device)

def test_model(model, device):
    """Test the trained model with sample queries."""
    model.eval()
    
    # Sample test data
    test_queries = [
        "machine learning algorithms",
        "deep neural networks",
        "information retrieval",
        "natural language processing"
    ]
    
    test_docs = [
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Deep learning uses neural networks with multiple layers to learn complex patterns.",
        "Information retrieval systems help users find relevant documents from large collections.",
        "Natural language processing enables computers to understand and process human language.",
        "Cooking recipes often include ingredients like flour, eggs, and sugar."  # Negative example
    ]
    
    with torch.no_grad():
        # Encode queries
        query_embeddings = []
        for query in test_queries:
            query_ids = model.tokenize(query).unsqueeze(0).to(device)
            query_emb = model.encode_query(query_ids)
            query_embeddings.append(query_emb)
        
        # Encode documents
        doc_embeddings = []
        for doc in test_docs:
            doc_ids = model.tokenize(doc).unsqueeze(0).to(device)
            doc_emb = model.encode_document(doc_ids)
            doc_embeddings.append(doc_emb)
        
        # Compute similarities
        print("\n📊 Query-Document Similarities:")
        print("-" * 60)
        
        for i, query in enumerate(test_queries):
            print(f"\nQuery: '{query}'")
            similarities = []
            
            for j, doc in enumerate(test_docs):
                sim = torch.cosine_similarity(query_embeddings[i], doc_embeddings[j]).item()
                similarities.append((sim, doc))
            
            # Sort by similarity
            similarities.sort(reverse=True)
            
            for k, (sim, doc) in enumerate(similarities[:3]):
                print(f"  {k+1}. [{sim:.3f}] {doc[:60]}...")

if __name__ == "__main__":
    train_two_tower_model() 