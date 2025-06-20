#!/usr/bin/env python3
"""
Cache Documents from SQLite to Redis Vector Store

This script efficiently caches document embeddings from SQLite to Redis,
avoiding the need to re-load large pickle files.

Advantages over pickle-based caching:
- No need to re-parse large pickle files
- Fast indexed queries from SQLite
- Can cache documents in batches with better control
- Can resume interrupted caching sessions
- Can selectively cache by source, length, or other criteria
"""

import os
import sys
import argparse
import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from tqdm import tqdm
import json
from datetime import datetime

# Add paths for imports - make it work from any directory
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../training'))
sys.path.append(script_dir)

from redis_vector_store import RedisVectorStore, load_model_from_checkpoint
from document_store import DocumentStore
from two_tower_model import TwoTowerModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cache_from_sqlite_to_redis(redis_config: Dict,
                              checkpoint_path: str,
                              sqlite_db_path: str = "./documents.db",
                              limit: int = None,
                              minimal: bool = True,
                              device: str = "cpu"):
    """
    Cache documents from SQLite to Redis efficiently.
    
    Args:
        redis_config: Redis connection configuration
        checkpoint_path: Path to trained model checkpoint
        sqlite_db_path: Path to SQLite database
        limit: Maximum number of documents to cache
        minimal: Whether to store only embeddings (no text) in Redis
        device: Device for model inference
    """
    logger.info("🚀 Starting SQLite to Redis caching...")
    logger.info("=" * 60)
    
    # Load model
    logger.info("📦 Loading trained model...")
    model = load_model_from_checkpoint(checkpoint_path, device)
    
    # Connect to SQLite
    logger.info("💾 Connecting to SQLite document store...")
    doc_store = DocumentStore(sqlite_db_path)
    
    # Connect to Redis
    logger.info("🔴 Connecting to Redis...")
    vector_store = RedisVectorStore(**redis_config)
    
    # Get documents from SQLite
    logger.info("📂 Loading documents from SQLite...")
    cursor = doc_store.conn.cursor()
    
    if limit:
        cursor.execute('SELECT doc_id, doc_text, source, metadata FROM documents LIMIT ?', (limit,))
    else:
        cursor.execute('SELECT doc_id, doc_text, source, metadata FROM documents')
    
    sqlite_docs = cursor.fetchall()
    logger.info(f"📊 Found {len(sqlite_docs)} documents in SQLite")
    
    # Filter out already cached documents using batch operations with freshness check
    logger.info("🔍 Checking which documents are already cached and up-to-date...")
    uncached_docs = []
    cached_count = 0
    stale_count = 0
    
    # Batch check for better performance
    batch_size = 1000
    for i in tqdm(range(0, len(sqlite_docs), batch_size), desc="Checking cache"):
        batch = sqlite_docs[i:i + batch_size]
        
        # Prepare batch of Redis keys
        redis_keys = [f"doc:{vector_store.index_name}:{doc[0]}" for doc in batch]
        
        # Batch check for existence AND get timestamps
        pipe = vector_store.redis_client.pipeline()
        for key in redis_keys:
            pipe.hget(key, "timestamp")  # Get cached timestamp (None if doesn't exist)
        cached_timestamps = pipe.execute()
        
        # Process results
        for j, (doc_id, doc_text, source, metadata_str) in enumerate(batch):
            cached_timestamp = cached_timestamps[j]
            
            if cached_timestamp is None:
                # Document not cached at all
                try:
                    metadata = json.loads(metadata_str) if metadata_str else {}
                except:
                    metadata = {}
                uncached_docs.append((doc_id, doc_text, source, metadata))
            else:
                # Document exists in cache - check if it's fresh
                try:
                    # Get SQLite document timestamp from the current batch
                    sqlite_created_at = None
                    for sqlite_doc in sqlite_docs:
                        if sqlite_doc[0] == doc_id:
                            # Get full document data to access created_at
                            doc_data = doc_store.get_document(doc_id)
                            if doc_data:
                                sqlite_created_at = doc_data.get('created_at')
                            break
                    
                    if sqlite_created_at:
                        cached_timestamp_str = cached_timestamp.decode('utf-8') if isinstance(cached_timestamp, bytes) else cached_timestamp
                        
                        # Compare timestamps (SQLite format: ISO datetime string)
                        sqlite_time = datetime.fromisoformat(sqlite_created_at)
                        cached_time = datetime.fromisoformat(cached_timestamp_str)
                        
                        if sqlite_time > cached_time:
                            # SQLite document is newer - needs re-caching
                            try:
                                metadata = json.loads(metadata_str) if metadata_str else {}
                            except:
                                metadata = {}
                            uncached_docs.append((doc_id, doc_text, source, metadata))
                            stale_count += 1
                        else:
                            # Cache is up-to-date
                            cached_count += 1
                    else:
                        # Can't determine SQLite timestamp - assume cache is valid
                        cached_count += 1
                        
                except Exception as e:
                    # Error comparing timestamps - assume cache is valid
                    logger.debug(f"Failed to compare timestamps for {doc_id}: {e}")
                    cached_count += 1
    
    logger.info(f"📋 Cache status:")
    logger.info(f"   Up-to-date: {cached_count:,}")
    logger.info(f"   Stale (needs update): {stale_count:,}")
    logger.info(f"   Not cached: {len(uncached_docs) - stale_count:,}")
    logger.info(f"   Total to process: {len(uncached_docs):,}")
    
    if not uncached_docs:
        logger.info("✅ All documents already cached!")
        return {"success": 0, "failed": 0}
    
    # Encode and cache documents
    logger.info("🧠 Encoding documents with two-tower model...")
    encoded_docs = []
    batch_size = 32
    
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(uncached_docs), batch_size), desc="Encoding"):
            batch = uncached_docs[i:i + batch_size]
            
            try:
                # Prepare batch
                doc_ids = []
                doc_texts = []
                metadatas = []
                
                for doc_id, doc_text, source, metadata in batch:
                    doc_ids.append(doc_id)
                    doc_texts.append(doc_text)
                    
                    # Enhanced metadata
                    enhanced_metadata = {
                        **metadata,
                        'source': source,
                        'cached_at': datetime.now().isoformat(),
                        'cached_from': 'sqlite'
                    }
                    metadatas.append(enhanced_metadata)
                
                # Tokenize and encode
                doc_tokens = torch.stack([model.tokenize(text) for text in doc_texts]).to(device)
                embeddings = model.encode_document(doc_tokens)
                embeddings_np = embeddings.cpu().numpy()
                
                # Prepare for Redis
                for j, (doc_id, doc_text, metadata) in enumerate(zip(doc_ids, doc_texts, metadatas)):
                    if minimal:
                        # Store only embeddings (no document text)
                        encoded_docs.append((doc_id, "", embeddings_np[j], metadata))
                    else:
                        # Store embeddings + document text
                        encoded_docs.append((doc_id, doc_text, embeddings_np[j], metadata))
                
            except Exception as e:
                logger.error(f"❌ Failed to encode batch starting at {i}: {e}")
    
    # Cache to Redis
    logger.info("💾 Caching to Redis...")
    stats = vector_store.add_documents_batch(encoded_docs, batch_size=100)
    
    # Final statistics
    redis_stats = vector_store.get_stats()
    sqlite_stats = doc_store.get_stats()
    
    logger.info("=" * 60)
    logger.info("✅ SQLite to Redis caching completed!")
    logger.info(f"📈 Results:")
    logger.info(f"   Documents processed: {len(uncached_docs):,}")
    logger.info(f"   Successfully cached: {stats['success']:,}")
    logger.info(f"   Failed: {stats['failed']:,}")
    logger.info(f"   Total in Redis: {redis_stats['document_count']:,}")
    logger.info(f"   Total in SQLite: {sqlite_stats['total_documents']:,}")
    logger.info(f"   Coverage: {redis_stats['document_count']/sqlite_stats['total_documents']:.1%}")
    logger.info(f"   Redis memory: {redis_stats['redis_info']['used_memory']}")
    
    doc_store.close()
    return stats

def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(
        description="""
🚀 Cache Documents from SQLite to Redis Vector Store

This script efficiently caches document embeddings from SQLite to Redis,
avoiding the need to re-load large pickle files.

Advantages:
- Fast indexed queries from SQLite
- Resume interrupted caching sessions  
- Selective caching by criteria
- Minimal Redis storage (embeddings only)

Examples:
  # Cache first 1000 documents (minimal mode)
  python cache_from_sqlite.py -c ../checkpoints/model.pt --limit 1000
  
  # Cache all documents (minimal mode)
  python cache_from_sqlite.py -c ../checkpoints/model.pt
  
  # Cache with full document text (not recommended)
  python cache_from_sqlite.py -c ../checkpoints/model.pt --full
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--checkpoint", "-c", required=True,
                       help="Path to trained two-tower model checkpoint")
    # Get script directory for default paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    parser.add_argument("--config", default=os.path.join(script_dir, "redis_config.json"),
                       help="Redis configuration file")
    parser.add_argument("--db", default=os.path.join(script_dir, "documents.db"),
                       help="SQLite database path")
    parser.add_argument("--limit", type=int,
                       help="Maximum number of documents to cache")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "mps"],
                       help="Device for model inference")
    parser.add_argument("--full", action="store_true",
                       help="Store full document text in Redis (not recommended)")
    
    # Parse arguments and show help if required params missing
    if len(sys.argv) == 1:
        parser.print_help()
        return
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.checkpoint):
        logger.error(f"❌ Checkpoint file not found: {args.checkpoint}")
        logger.info("💡 Use --help to see usage examples")
        return
    
    if not os.path.exists(args.db):
        logger.error(f"❌ SQLite database not found: {args.db}")
        return
    
    # Load Redis config
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        redis_config = {
            "redis_url": config["redis"]["url"],
            "index_name": config["vector_store"]["index_name"],
            "vector_dim": config["vector_store"]["vector_dim"]
        }
    except Exception as e:
        logger.error(f"❌ Failed to load Redis config: {e}")
        return
    
    # Run caching
    try:
        cache_from_sqlite_to_redis(
            redis_config=redis_config,
            checkpoint_path=args.checkpoint,
            sqlite_db_path=args.db,
            limit=args.limit,
            minimal=not args.full,  # Default to minimal unless --full specified
            device=args.device
        )
        
    except Exception as e:
        logger.error(f"❌ Caching failed: {e}")
        raise

if __name__ == "__main__":
    main() 