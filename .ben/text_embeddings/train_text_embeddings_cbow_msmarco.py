#!/usr/bin/env python3
"""
CBOW Implementation with MS Marco Dataset using Gensim Word2Vec
"""

import os
import numpy as np
import torch
import pickle
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from tqdm import tqdm
import re
from collections import defaultdict
from datasets import load_dataset





def download_msmarco_data(cache_dir="./data"):
    """Download MS Marco dataset from Hugging Face with caching."""
    print("Loading MS Marco dataset from Hugging Face...")
    
    # Create cache directory if it doesn't exist
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load dataset with caching enabled
    dataset = load_dataset(
        "microsoft/ms_marco", 
        "v1.1",
        cache_dir=cache_dir  # This enables Hugging Face's built-in caching
    )
    
    print(f"✅ Dataset loaded (cached in: {cache_dir})")
    return dataset

def clean_text(text):
    """Clean and preprocess text."""
    # Remove special characters and normalize whitespace
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def load_msmarco_texts(dataset, cache_file="./data/processed_texts.pkl", max_samples=50000):
    """Load and preprocess MS Marco texts from Hugging Face dataset with caching."""
    
    # Check if processed texts are already cached
    if os.path.exists(cache_file):
        print(f"📦 Loading cached processed texts from {cache_file}...")
        try:
            with open(cache_file, 'rb') as f:
                texts = pickle.load(f)
            print(f"✅ Loaded {len(texts)} cached text documents")
            return texts
        except Exception as e:
            print(f"⚠️  Error loading cache: {e}. Reprocessing...")
    
    print("Processing MS Marco dataset...")
    
    texts = []
    
    # Process training split
    train_data = dataset['train']
    print(f"Processing {min(len(train_data), max_samples)} samples from training set...")
    
    for i, example in enumerate(tqdm(train_data, desc="Loading MS Marco data")):
        if i >= max_samples:
            break
            
        # Process query
        query_text = example['query']
        if query_text:
            cleaned_query = clean_text(query_text)
            if cleaned_query:
                texts.append(simple_preprocess(cleaned_query))
        
        # Process passages
        passages = example['passages']
        if passages and 'passage_text' in passages:
            for passage_text in passages['passage_text']:
                if passage_text:
                    cleaned_passage = clean_text(passage_text)
                    if cleaned_passage:
                        texts.append(simple_preprocess(cleaned_passage))
    
    print(f"Loaded {len(texts)} text documents")
    
    # Cache the processed texts
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(texts, f)
        print(f"💾 Cached processed texts to {cache_file}")
    except Exception as e:
        print(f"⚠️  Warning: Could not cache texts: {e}")
    
    return texts

def clear_cache(cache_dir="./data"):
    """Clear all cached data."""
    import shutil
    
    if os.path.exists(cache_dir):
        print(f"🗑️  Clearing cache directory: {cache_dir}")
        shutil.rmtree(cache_dir)
        print("✅ Cache cleared!")
    else:
        print("📁 No cache directory found.")

def train_cbow_model(texts):
    """Train CBOW model using gensim Word2Vec."""
    print("Training CBOW model using gensim...")
    
    # CBOW parameters
    model = Word2Vec(
        sentences=texts,
        vector_size=300,  # Embedding dimension
        window=5,         # Context window size
        min_count=5,      # Minimum word frequency
        workers=4,        # Number of worker threads
        sg=0,             # Use CBOW (sg=0), Skip-gram would be sg=1
        epochs=10,        # Number of training epochs
        alpha=0.025,      # Learning rate
        min_alpha=0.0001, # Minimum learning rate
        sample=1e-3,      # Subsampling threshold
        negative=5,       # Negative sampling
        seed=42           # For reproducibility
    )
    
    print(f"Vocabulary size: {len(model.wv.key_to_index)}")
    print(f"Vector size: {model.wv.vector_size}")
    
    return model

def extract_embeddings_and_mappings(model):
    """Extract embedding matrix and word mappings from trained model."""
    print("Extracting embeddings and mappings...")
    
    # Get vocabulary
    vocab = model.wv.key_to_index
    
    # Create word-to-index and index-to-word mappings
    word_to_index = {word: idx for word, idx in vocab.items()}
    index_to_word = {idx: word for word, idx in vocab.items()}
    
    # Get embedding matrix
    embedding_matrix = model.wv.vectors
    
    print(f"Embedding matrix shape: {embedding_matrix.shape}")
    print(f"Vocabulary size: {len(word_to_index)}")
    
    return embedding_matrix, word_to_index, index_to_word

def save_embeddings(embedding_matrix, word_to_index, index_to_word, filename="msmarco_word2vec.pt"):
    """Save embeddings and mappings using torch.save."""
    print(f"Saving embeddings to {filename}...")
    
    # Convert numpy array to torch tensor
    embedding_weights = torch.from_numpy(embedding_matrix).float()
    
    # Save using torch.save
    torch.save({
        "embedding_matrix": embedding_weights,
        "word_to_index": word_to_index,
        "index_to_word": index_to_word
    }, filename)
    
    print(f"Successfully saved embeddings to {filename}")

def main():
    """Main function to run the complete pipeline."""
    print("Starting CBOW implementation with MS Marco dataset")
    
    # Step 1: Download MS Marco data from Hugging Face (with caching)
    dataset = download_msmarco_data(cache_dir="./data")
    
    # Step 2: Load and preprocess texts (with caching)
    texts = load_msmarco_texts(dataset, cache_file="./data/processed_texts.pkl", max_samples=50000)
    
    if not texts:
        print("No texts loaded. Please check the dataset.")
        return
    
    # Step 3: Train CBOW model
    model = train_cbow_model(texts)
    
    # Step 4: Extract embeddings and mappings
    embedding_matrix, word_to_index, index_to_word = extract_embeddings_and_mappings(model)
    
    # Step 5: Save using torch.save
    save_embeddings(embedding_matrix, word_to_index, index_to_word)
    
    # Display some statistics
    print("\n=== Model Statistics ===")
    print(f"Vocabulary size: {len(word_to_index)}")
    print(f"Embedding dimension: {embedding_matrix.shape[1]}")
    print(f"Total parameters: {embedding_matrix.shape[0] * embedding_matrix.shape[1]:,}")
    
    # Show some example words and their embeddings
    print("\n=== Sample Words ===")
    sample_words = list(word_to_index.keys())[:10]
    for word in sample_words:
        idx = word_to_index[word]
        print(f"'{word}' -> index {idx}")
    
    print("\nCBOW model training and saving completed successfully!")

if __name__ == "__main__":
    main() 