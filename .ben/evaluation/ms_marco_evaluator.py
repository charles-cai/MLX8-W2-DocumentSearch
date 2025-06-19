#!/usr/bin/env python3
"""
MS MARCO Evaluator for Two-Tower Model

This module provides evaluation functionality for the two-tower model using
MS MARCO dev/validation dataset. It computes standard IR metrics like MRR@10,
Recall@k, and NDCG@k.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from datasets import load_dataset
import pickle
import os
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime
import sys

# Add parent directories to path for imports
sys.path.append('../training')
from two_tower_model import TwoTowerModel

class MSMarcoEvalDataset(Dataset):
    """
    MS MARCO evaluation dataset for computing retrieval metrics.
    """
    
    def __init__(self, data_path: str = "./data", split: str = "validation", 
                 max_samples: int = None, cache_eval_data: bool = True):
        self.data_path = data_path
        self.split = split
        self.max_samples = max_samples
        
        print(f"🔍 Loading MS MARCO {split} dataset for evaluation...")
        
        # Load evaluation data
        self.queries, self.passages, self.qrels = self._load_eval_data(cache_eval_data)
        
        # Create query list for evaluation
        self.query_ids = list(self.queries.keys())
        if max_samples and len(self.query_ids) > max_samples:
            self.query_ids = self.query_ids[:max_samples]
        
        print(f"📊 Evaluation dataset loaded:")
        print(f"   Queries: {len(self.query_ids):,}")
        print(f"   Total passages: {len(self.passages):,}")
        print(f"   Queries with relevance judgments: {len([q for q in self.query_ids if q in self.qrels]):,}")
    
    def _load_eval_data(self, cache_data: bool = True):
        """Load MS MARCO evaluation data."""
        cache_file = os.path.join(self.data_path, f"msmarco_{self.split}_eval.pkl")
        
        if cache_data and os.path.exists(cache_file):
            print("📦 Loading cached evaluation data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['queries'], data['passages'], data['qrels']
        
        print(f"🔄 Loading MS MARCO {self.split} dataset from Hugging Face...")
        
        # Load dataset
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split=self.split)
        
        queries = {}
        passages = {}
        qrels = {}
        
        print("🔄 Processing evaluation dataset...")
        for item in tqdm(dataset, desc=f"Processing {self.split}"):
            query_id = str(item['query_id'])
            query_text = item['query']
            
            queries[query_id] = query_text
            qrels[query_id] = []
            
            # Process passages
            passages_data = item['passages']
            if isinstance(passages_data, dict) and 'passage_text' in passages_data:
                passage_texts = passages_data['passage_text']
                is_selected_list = passages_data['is_selected']
                
                for i, (passage_text, is_selected) in enumerate(zip(passage_texts, is_selected_list)):
                    passage_id = f"{query_id}_{i}"
                    passages[passage_id] = passage_text
                    
                    if is_selected == 1:
                        qrels[query_id].append(passage_id)
        
        # Cache the data
        if cache_data:
            os.makedirs(self.data_path, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump({
                    'queries': queries,
                    'passages': passages,
                    'qrels': qrels
                }, f)
            print(f"💾 Cached evaluation data to: {cache_file}")
        
        return queries, passages, qrels
    
    def __len__(self):
        return len(self.query_ids)
    
    def __getitem__(self, idx):
        query_id = self.query_ids[idx]
        return {
            'query_id': query_id,
            'query_text': self.queries[query_id],
            'relevant_passage_ids': self.qrels.get(query_id, [])
        }

class MSMarcoEvaluator:
    """
    Comprehensive evaluator for MS MARCO two-tower model.
    """
    
    def __init__(self, model: TwoTowerModel, device: torch.device, 
                 data_path: str = "./data", batch_size: int = 64):
        self.model = model
        self.device = device
        self.data_path = data_path
        self.batch_size = batch_size
        
        print(f"🎯 MS MARCO Evaluator initialized")
        print(f"   Device: {device}")
        print(f"   Batch size: {batch_size}")
    
    def evaluate(self, split: str = "validation", max_queries: int = None, 
                 k_values: List[int] = [1, 5, 10, 20], 
                 compute_full_ranking: bool = False) -> Dict:
        """
        Evaluate the model on MS MARCO dataset.
        
        Args:
            split: Dataset split to evaluate on ('validation' or 'test')
            max_queries: Maximum number of queries to evaluate (None for all)
            k_values: List of k values for Recall@k and NDCG@k
            compute_full_ranking: Whether to compute full ranking metrics
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"🚀 Starting MS MARCO Evaluation")
        print("=" * 60)
        
        # Load evaluation dataset
        eval_dataset = MSMarcoEvalDataset(
            data_path=self.data_path, 
            split=split, 
            max_samples=max_queries
        )
        
        # Create DataLoader for efficient batch processing
        eval_dataloader = DataLoader(
            eval_dataset, 
            batch_size=1,  # Process one query at a time for ranking
            shuffle=False, 
            num_workers=0
        )
        
        # Initialize metrics
        metrics = {
            'mrr': [],
            'mrr@10': [],
            'recall@k': {k: [] for k in k_values},
            'ndcg@k': {k: [] for k in k_values},
            'map': [],
            'num_queries_evaluated': 0,
            'num_queries_with_relevance': 0
        }
        
        self.model.eval()
        
        print(f"🔍 Evaluating on {len(eval_dataset)} queries...")
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(eval_dataloader, desc="Evaluating")):
                # Handle batch data safely
                try:
                    query_id = batch['query_id'][0] if isinstance(batch['query_id'], list) else batch['query_id'].item()
                    query_text = batch['query_text'][0] if isinstance(batch['query_text'], list) else batch['query_text']
                    
                    # Handle empty relevant_passage_ids list
                    relevant_passage_ids_raw = batch['relevant_passage_ids']
                    if isinstance(relevant_passage_ids_raw, list) and len(relevant_passage_ids_raw) > 0:
                        relevant_passage_ids = relevant_passage_ids_raw[0]
                    elif isinstance(relevant_passage_ids_raw, list) and len(relevant_passage_ids_raw) == 0:
                        relevant_passage_ids = []  # Empty list for queries with no relevant passages
                    else:
                        relevant_passage_ids = relevant_passage_ids_raw
                        
                except (IndexError, TypeError) as e:
                    print(f"⚠️  Error processing batch {batch_idx}: {e}")
                    print(f"   Batch keys: {batch.keys()}")
                    print(f"   Query ID type: {type(batch['query_id'])}")
                    print(f"   Query text type: {type(batch['query_text'])}")
                    print(f"   Relevant IDs type: {type(batch['relevant_passage_ids'])}")
                    continue
                
                # Skip queries without relevant passages
                if not relevant_passage_ids:
                    continue
                
                # Get all passages for this query (candidates for ranking)
                candidate_passages = self._get_candidate_passages(query_id, eval_dataset)
                
                if not candidate_passages:
                    continue
                
                # Compute rankings
                rankings = self._rank_passages(query_text, candidate_passages)
                
                # Compute metrics for this query
                query_metrics = self._compute_query_metrics(
                    rankings, relevant_passage_ids, k_values
                )
                
                # Accumulate metrics
                if query_metrics['mrr'] > 0:  # Only count queries with relevant results found
                    metrics['mrr'].append(query_metrics['mrr'])
                    metrics['mrr@10'].append(query_metrics['mrr@10'])
                    metrics['map'].append(query_metrics['map'])
                    
                    for k in k_values:
                        metrics['recall@k'][k].append(query_metrics[f'recall@{k}'])
                        metrics['ndcg@k'][k].append(query_metrics[f'ndcg@{k}'])
                    
                    metrics['num_queries_with_relevance'] += 1
                
                metrics['num_queries_evaluated'] += 1
                
                # Progress update
                if (batch_idx + 1) % 100 == 0:
                    current_mrr = np.mean(metrics['mrr']) if metrics['mrr'] else 0
                    print(f"  Processed {batch_idx + 1} queries | Current MRR: {current_mrr:.4f}")
        
        # Compute final metrics
        final_metrics = self._compute_final_metrics(metrics)
        
        print("=" * 60)
        print("📊 Evaluation Results:")
        print("=" * 60)
        
        self._print_metrics(final_metrics)
        
        return final_metrics
    
    def _get_candidate_passages(self, query_id: str, eval_dataset: MSMarcoEvalDataset) -> List[Tuple[str, str]]:
        """Get candidate passages for ranking for a given query."""
        # For MS MARCO, we typically rank against all passages for the query
        # In practice, you might want to use a subset or pre-filtered candidates
        
        candidates = []
        query_prefix = f"{query_id}_"
        
        for passage_id, passage_text in eval_dataset.passages.items():
            if passage_id.startswith(query_prefix):
                candidates.append((passage_id, passage_text))
        
        return candidates
    
    def _rank_passages(self, query_text: str, candidate_passages: List[Tuple[str, str]]) -> List[Tuple[str, float]]:
        """Rank passages for a given query."""
        if not candidate_passages:
            return []
        
        # Encode query
        query_ids = self.model.tokenize(query_text).unsqueeze(0).to(self.device)
        query_emb = self.model.encode_query(query_ids)
        
        # Encode all candidate passages
        passage_embeddings = []
        passage_ids = []
        
        # Process passages in batches for efficiency
        for i in range(0, len(candidate_passages), self.batch_size):
            batch_passages = candidate_passages[i:i + self.batch_size]
            
            # Tokenize batch
            batch_tokens = torch.stack([
                self.model.tokenize(passage_text) 
                for _, passage_text in batch_passages
            ]).to(self.device)
            
            # Encode batch
            batch_emb = self.model.encode_document(batch_tokens)
            passage_embeddings.append(batch_emb)
            passage_ids.extend([pid for pid, _ in batch_passages])
        
        # Concatenate all embeddings
        if passage_embeddings:
            all_passage_emb = torch.cat(passage_embeddings, dim=0)
            
            # Compute similarities
            similarities = torch.cosine_similarity(
                query_emb.expand(all_passage_emb.size(0), -1), 
                all_passage_emb
            )
            
            # Create rankings (passage_id, similarity_score)
            rankings = list(zip(passage_ids, similarities.cpu().numpy()))
            
            # Sort by similarity (descending)
            rankings.sort(key=lambda x: x[1], reverse=True)
            
            return rankings
        
        return []
    
    def _compute_query_metrics(self, rankings: List[Tuple[str, float]], 
                              relevant_passage_ids: List[str], k_values: List[int]) -> Dict:
        """Compute metrics for a single query."""
        if not rankings or not relevant_passage_ids:
            return {f'recall@{k}': 0.0 for k in k_values} | \
                   {f'ndcg@{k}': 0.0 for k in k_values} | \
                   {'mrr': 0.0, 'mrr@10': 0.0, 'map': 0.0}
        
        # Create relevance list
        ranked_passage_ids = [pid for pid, _ in rankings]
        relevant_set = set(relevant_passage_ids)
        
        metrics = {}
        
        # MRR (Mean Reciprocal Rank)
        mrr = 0.0
        mrr_10 = 0.0
        for i, passage_id in enumerate(ranked_passage_ids):
            if passage_id in relevant_set:
                mrr = 1.0 / (i + 1)
                if i < 10:
                    mrr_10 = 1.0 / (i + 1)
                break
        
        metrics['mrr'] = mrr
        metrics['mrr@10'] = mrr_10
        
        # Recall@k and NDCG@k
        for k in k_values:
            top_k = ranked_passage_ids[:k]
            relevant_in_top_k = [pid for pid in top_k if pid in relevant_set]
            
            # Recall@k
            recall_k = len(relevant_in_top_k) / len(relevant_set) if relevant_set else 0.0
            metrics[f'recall@{k}'] = recall_k
            
            # NDCG@k
            dcg = sum(1.0 / np.log2(i + 2) for i, pid in enumerate(top_k) if pid in relevant_set)
            idcg = sum(1.0 / np.log2(i + 2) for i in range(min(k, len(relevant_set))))
            ndcg_k = dcg / idcg if idcg > 0 else 0.0
            metrics[f'ndcg@{k}'] = ndcg_k
        
        # MAP (Mean Average Precision)
        precision_at_relevant = []
        num_relevant_found = 0
        for i, passage_id in enumerate(ranked_passage_ids):
            if passage_id in relevant_set:
                num_relevant_found += 1
                precision_at_i = num_relevant_found / (i + 1)
                precision_at_relevant.append(precision_at_i)
        
        map_score = np.mean(precision_at_relevant) if precision_at_relevant else 0.0
        metrics['map'] = map_score
        
        return metrics
    
    def _compute_final_metrics(self, metrics: Dict) -> Dict:
        """Compute final aggregated metrics."""
        final_metrics = {
            'num_queries_evaluated': metrics['num_queries_evaluated'],
            'num_queries_with_relevance': metrics['num_queries_with_relevance']
        }
        
        if metrics['mrr']:
            final_metrics['MRR'] = np.mean(metrics['mrr'])
            final_metrics['MRR@10'] = np.mean(metrics['mrr@10'])
            final_metrics['MAP'] = np.mean(metrics['map'])
            
            for k in metrics['recall@k'].keys():
                final_metrics[f'Recall@{k}'] = np.mean(metrics['recall@k'][k])
                final_metrics[f'NDCG@{k}'] = np.mean(metrics['ndcg@k'][k])
        else:
            final_metrics.update({
                'MRR': 0.0, 'MRR@10': 0.0, 'MAP': 0.0
            })
            for k in [1, 5, 10, 20]:
                final_metrics[f'Recall@{k}'] = 0.0
                final_metrics[f'NDCG@{k}'] = 0.0
        
        return final_metrics
    
    def _print_metrics(self, metrics: Dict):
        """Print evaluation metrics in a formatted way."""
        print(f"Queries evaluated: {metrics['num_queries_evaluated']}")
        print(f"Queries with relevance: {metrics['num_queries_with_relevance']}")
        print()
        
        print("🎯 Ranking Metrics:")
        print(f"   MRR:      {metrics['MRR']:.4f}")
        print(f"   MRR@10:   {metrics['MRR@10']:.4f}")
        print(f"   MAP:      {metrics['MAP']:.4f}")
        print()
        
        print("📊 Recall@k:")
        for k in [1, 5, 10, 20]:
            if f'Recall@{k}' in metrics:
                print(f"   Recall@{k:2d}: {metrics[f'Recall@{k}']:.4f}")
        print()
        
        print("📈 NDCG@k:")
        for k in [1, 5, 10, 20]:
            if f'NDCG@{k}' in metrics:
                print(f"   NDCG@{k:2d}:   {metrics[f'NDCG@{k}']:.4f}")
    
    def save_evaluation_results(self, metrics: Dict, save_path: str, 
                               additional_info: Dict = None):
        """Save evaluation results to file."""
        results = {
            'metrics': metrics,
            'evaluation_timestamp': datetime.now().isoformat(),
            'model_info': {
                'hidden_dim': getattr(self.model, 'hidden_dim', 'unknown'),
                'num_layers': getattr(self.model, 'num_layers', 'unknown'),
                'vocab_size': getattr(self.model, 'vocab_size', 'unknown')
            }
        }
        
        if additional_info:
            results.update(additional_info)
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Evaluation results saved to: {save_path}")

def load_model_for_evaluation(checkpoint_path: str, device: torch.device) -> TwoTowerModel:
    """Load a trained model for evaluation."""
    print(f"📦 Loading model from: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract model components
    embedding_matrix = checkpoint['embedding_matrix']
    word_to_index = checkpoint['word_to_index']
    model_config = checkpoint.get('model_config', {})
    
    # Create model
    model = TwoTowerModel(
        embedding_matrix=embedding_matrix,
        word_to_index=word_to_index,
        hidden_dim=model_config.get('hidden_dim', 256),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.1)
    )
    
    # Load trained weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"✅ Model loaded successfully")
    print(f"   Vocabulary size: {len(word_to_index):,}")
    print(f"   Hidden dim: {model_config.get('hidden_dim', 256)}")
    print(f"   Num layers: {model_config.get('num_layers', 2)}")
    
    return model

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate MS MARCO Two-Tower Model")
    parser.add_argument("model_path", help="Path to trained model checkpoint")
    parser.add_argument("--split", default="validation", choices=["validation", "test"],
                       help="Dataset split to evaluate on")
    parser.add_argument("--max-queries", type=int, default=None,
                       help="Maximum number of queries to evaluate")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="Batch size for evaluation")
    parser.add_argument("--output", default=None,
                       help="Path to save evaluation results")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load model
    model = load_model_for_evaluation(args.model_path, device)
    
    # Create evaluator
    evaluator = MSMarcoEvaluator(model, device, batch_size=args.batch_size)
    
    # Run evaluation
    results = evaluator.evaluate(
        split=args.split,
        max_queries=args.max_queries
    )
    
    # Save results if requested
    if args.output:
        evaluator.save_evaluation_results(results, args.output) 