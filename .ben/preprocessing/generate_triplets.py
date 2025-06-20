#!/usr/bin/env python3
"""
Generate MS MARCO Training Triplets

This script processes the MS MARCO dataset and generates training triplets
(query, positive_doc, negative_doc) for two-tower model training.
The triplets are saved to disk for reuse in training.
"""

import torch
import pickle
import os
import random
from datasets import load_dataset
from tqdm import tqdm
from typing import Dict, List, Tuple
import argparse
import json
from datetime import datetime

class MSMarcoTripletGenerator:
    """
    Generate and save MS MARCO training triplets.
    """
    
    def __init__(self, data_path: str = "./data", max_samples: int = None):
        self.data_path = data_path
        self.max_samples = max_samples
        
        # Ensure data directory exists
        os.makedirs(self.data_path, exist_ok=True)
        
        print(f"🚀 MS MARCO Triplet Generator")
        print(f"   Data path: {self.data_path}")
        print(f"   Max samples: {max_samples if max_samples else 'No limit'}")
    
    def load_raw_data(self):
        """Load raw MS MARCO data from cache or download."""
        # Decide whether to use fast mode based on max_samples
        if self.max_samples and self.max_samples <= 100000:
            print(f"🚀 Using fast mode for {self.max_samples:,} samples (early stopping enabled)")
            return self._load_raw_data_limited()
        else:
            print("📦 Using full mode (processing all data)")
            return self._load_raw_data_full()
    
    def _load_raw_data_full(self):
        """Load full dataset for large samples or no limit."""
        cache_file = os.path.join(self.data_path, "msmarco_raw_data.pkl")
        
        if os.path.exists(cache_file):
            print("📦 Loading cached MS MARCO raw data...")
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
        
        for item in tqdm(dataset, desc="Processing MS MARCO"):
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
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'queries': queries,
                'passages': passages,
                'qrels': qrels
            }, f)
        
        print(f"✅ Processed {len(queries)} queries and {len(passages)} passages")
        print(f"   Cached to: {cache_file}")
        
        return queries, passages, qrels
    
    def _load_raw_data_limited(self):
        """Load limited data for fast triplet generation."""
        cache_file = os.path.join(self.data_path, f"msmarco_raw_data_limited_{self.max_samples}.pkl")
        
        if os.path.exists(cache_file):
            print("📦 Loading cached limited MS MARCO data...")
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
            return data['queries'], data['passages'], data['qrels']
        
        print(f"🔄 Loading limited MS MARCO dataset (targeting {self.max_samples:,} triplets)...")
        
        # Load dataset
        dataset = load_dataset("microsoft/ms_marco", "v1.1", split="train")
        
        # Process queries and passages with early stopping
        queries = {}
        passages = {}
        qrels = {}
        triplet_count = 0
        
        # Estimate: each query produces ~1-3 triplets on average
        # So we need roughly max_samples / 2 queries to be safe
        target_queries = min(self.max_samples * 2, len(dataset))
        
        print(f"🎯 Processing up to {target_queries:,} queries (estimated for {self.max_samples:,} triplets)")
        
        for i, item in enumerate(tqdm(dataset, desc="Processing MS MARCO (limited)", total=target_queries)):
            if i >= target_queries:
                break
                
            query_id = str(item['query_id'])
            query_text = item['query']
            
            queries[query_id] = query_text
            qrels[query_id] = []
            
            # Process passages
            passages_data = item['passages']
            if isinstance(passages_data, dict) and 'passage_text' in passages_data:
                passage_texts = passages_data['passage_text']
                is_selected_list = passages_data['is_selected']
                
                for j, (passage_text, is_selected) in enumerate(zip(passage_texts, is_selected_list)):
                    passage_id = f"{query_id}_{j}"
                    passages[passage_id] = passage_text
                    
                    if is_selected == 1:
                        qrels[query_id].append(passage_id)
                        triplet_count += 1
            
            # Early stopping if we have enough potential triplets
            if triplet_count >= self.max_samples * 1.5:  # 50% buffer
                print(f"🛑 Early stopping: Found {triplet_count:,} potential triplets (target: {self.max_samples:,})")
                break
        
        # Cache the limited data
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'queries': queries,
                'passages': passages,
                'qrels': qrels,
                'max_samples': self.max_samples,
                'actual_queries_processed': len(queries)
            }, f)
        
        print(f"✅ Processed {len(queries):,} queries and {len(passages):,} passages (limited mode)")
        print(f"   Potential triplets: ~{triplet_count:,}")
        print(f"   Cached to: {cache_file}")
        
        return queries, passages, qrels
    
    def generate_triplets(self, queries: Dict, passages: Dict, qrels: Dict):
        """Generate training triplets from the data."""
        triplets = []
        
        # Get all passage IDs for negative sampling
        all_passage_ids = list(passages.keys())
        print(f"📊 Total passages available for negative sampling: {len(all_passage_ids):,}")
        
        print("🔄 Creating training triplets...")
        
        # If we have max_samples, show target
        if self.max_samples:
            print(f"🎯 Target: {self.max_samples:,} triplets")
        
        for query_id, relevant_passage_ids in tqdm(qrels.items(), desc="Generating triplets"):
            if not relevant_passage_ids:  # Skip queries with no relevant passages
                continue
            
            # Early stopping check (for efficiency when max_samples is set)
            if self.max_samples and len(triplets) >= self.max_samples:
                print(f"🛑 Reached target of {self.max_samples:,} triplets, stopping early")
                break
            
            # For each query, create multiple triplets
            for pos_passage_id in relevant_passage_ids:
                # Early stopping check within inner loop too
                if self.max_samples and len(triplets) >= self.max_samples:
                    break
                    
                # Sample negative passages (not relevant to this query)
                neg_candidates = [pid for pid in all_passage_ids 
                                if pid not in relevant_passage_ids]
                
                if neg_candidates:
                    neg_passage_id = random.choice(neg_candidates)
                    
                    # Store triplet with actual text content
                    triplet = {
                        'query_id': query_id,
                        'query_text': queries[query_id],
                        'pos_passage_id': pos_passage_id,
                        'pos_passage_text': passages[pos_passage_id],
                        'neg_passage_id': neg_passage_id,
                        'neg_passage_text': passages[neg_passage_id]
                    }
                    
                    triplets.append(triplet)
        
        # Shuffle triplets
        random.shuffle(triplets)
        
        print(f"✅ Generated {len(triplets):,} training triplets")
        return triplets
    
    def save_triplets(self, triplets: List[Dict], suffix: str = ""):
        """Save triplets to disk."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create filename
        if suffix:
            filename = f"msmarco_triplets_{suffix}_{timestamp}.pkl"
        else:
            filename = f"msmarco_triplets_{timestamp}.pkl"
        
        filepath = os.path.join(self.data_path, filename)
        
        # Save triplets
        print(f"💾 Saving triplets to: {filepath}")
        with open(filepath, 'wb') as f:
            pickle.dump({
                'triplets': triplets,
                'metadata': {
                    'num_triplets': len(triplets),
                    'generated_at': timestamp,
                    'max_samples': self.max_samples,
                    'random_seed': random.getstate()
                }
            }, f)
        
        # Also save metadata as JSON for easy inspection
        metadata_file = filepath.replace('.pkl', '_metadata.json')
        with open(metadata_file, 'w') as f:
            json.dump({
                'num_triplets': len(triplets),
                'generated_at': timestamp,
                'max_samples': self.max_samples,
                'filepath': filepath,
                'sample_triplet': triplets[0] if triplets else None
            }, f, indent=2)
        
        print(f"📋 Metadata saved to: {metadata_file}")
        print(f"✅ Total file size: {os.path.getsize(filepath) / (1024*1024):.1f} MB")
        
        return filepath
    
    def generate_and_save(self):
        """Main pipeline: load data, generate triplets, save them."""
        print("=" * 60)
        print("🚀 Starting MS MARCO Triplet Generation Pipeline")
        print("=" * 60)
        
        # Step 1: Load raw data
        queries, passages, qrels = self.load_raw_data()
        
        # Step 2: Generate triplets
        triplets = self.generate_triplets(queries, passages, qrels)
        
        # Step 3: Save triplets
        suffix = f"{len(triplets)//1000}k" if len(triplets) >= 1000 else str(len(triplets))
        filepath = self.save_triplets(triplets, suffix=suffix)
        
        print("=" * 60)
        print("✅ Triplet generation completed successfully!")
        print(f"   Generated: {len(triplets):,} triplets")
        print(f"   Saved to: {filepath}")
        print("=" * 60)
        
        return filepath

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description="Generate MS MARCO training triplets")
    parser.add_argument("--data-path", default="./data", 
                       help="Path to data directory (default: ./data - current directory)")
    parser.add_argument("--max-samples", type=int, default=None,
                       help="Maximum number of triplets to generate (default: no limit)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility (default: 42)")
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"🎲 Random seed set to: {args.seed}")
    
    # Create generator and run
    generator = MSMarcoTripletGenerator(
        data_path=args.data_path,
        max_samples=args.max_samples
    )
    
    filepath = generator.generate_and_save()
    
    print(f"\n🎉 Ready for training! Use this file:")
    print(f"   {filepath}")

if __name__ == "__main__":
    main() 