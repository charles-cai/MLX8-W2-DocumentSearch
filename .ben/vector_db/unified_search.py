#!/usr/bin/env python3
"""
Unified Search System

Combines Redis vector search with SQLite document retrieval for efficient
document search with minimal Redis memory usage.

Architecture:
- Redis: Stores only embeddings + doc_id for fast vector search
- SQLite: Stores full document text + metadata for retrieval
- Correlation: Both systems use the same doc_id as primary key
"""

import sys
import torch
import numpy as np
from typing import List, Dict, Optional, Tuple
import logging
from dataclasses import dataclass
from datetime import datetime

# Add paths for imports - make it work from any directory
import os
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(script_dir, '../training'))
sys.path.append(script_dir)

from redis_vector_store import RedisVectorStore, load_model_from_checkpoint
from document_store import DocumentStore
from two_tower_model import TwoTowerModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class UnifiedSearchResult:
    """Result from unified search combining vector similarity and document text."""
    doc_id: str
    doc_text: str
    similarity: float
    doc_length: int
    source: str
    metadata: Dict = None

class UnifiedSearchSystem:
    """
    Unified search system combining Redis vector search with SQLite document retrieval.
    
    This system demonstrates how document IDs correlate between:
    - Redis Vector Store: doc_id -> embedding (for fast similarity search)
    - SQLite Document Store: doc_id -> full document text (for retrieval)
    """
    
    def __init__(self, 
                 redis_config: Dict,
                 sqlite_db_path: str = "./documents.db",
                 model_checkpoint: str = None,
                 device: str = "cpu"):
        """
        Initialize unified search system.
        
        Args:
            redis_config: Redis connection configuration
            sqlite_db_path: Path to SQLite document database
            model_checkpoint: Path to trained two-tower model
            device: Device for model inference
        """
        self.device = device
        
        # Initialize components
        logger.info("🔗 Initializing unified search system...")
        
        # Load model
        if model_checkpoint:
            logger.info("📦 Loading two-tower model...")
            self.model = load_model_from_checkpoint(model_checkpoint, device)
        else:
            self.model = None
            logger.warning("⚠️ No model checkpoint provided - search queries won't work")
        
        # Connect to Redis vector store
        logger.info("🔴 Connecting to Redis vector store...")
        self.vector_store = RedisVectorStore(**redis_config)
        
        # Connect to SQLite document store
        logger.info("💾 Connecting to SQLite document store...")
        self.document_store = DocumentStore(sqlite_db_path)
        
        # Verify correlation
        self._verify_id_correlation()
        
        logger.info("✅ Unified search system ready!")
    
    def _verify_id_correlation(self):
        """Verify that document IDs correlate between Redis and SQLite."""
        try:
            # Get sample IDs from SQLite
            cursor = self.document_store.conn.cursor()
            cursor.execute('SELECT doc_id FROM documents LIMIT 5')
            sqlite_ids = [row[0] for row in cursor.fetchall()]
            
            # Check if these IDs would exist in Redis (when cached)
            redis_pattern = f"doc:{self.vector_store.index_name}:*"
            redis_keys = self.vector_store.redis_client.keys(redis_pattern)
            
            logger.info(f"📊 ID Correlation Check:")
            logger.info(f"   SQLite sample IDs: {sqlite_ids[:3]}...")
            logger.info(f"   Redis keys found: {len(redis_keys)}")
            
            if redis_keys:
                # Extract doc_id from Redis key format: "doc:index_name:doc_id"
                sample_redis_key = redis_keys[0].decode('utf-8')
                redis_doc_id = sample_redis_key.split(':')[-1]
                logger.info(f"   Redis sample ID: {redis_doc_id}")
                
                # Check if this ID exists in SQLite
                doc = self.document_store.get_document(redis_doc_id)
                if doc:
                    logger.info("✅ ID correlation verified - same doc_id in both systems")
                else:
                    logger.warning("⚠️ ID mismatch detected - may need to rebuild one of the stores")
            else:
                logger.info("ℹ️ No documents in Redis yet - correlation will be verified after caching")
            
        except Exception as e:
            logger.warning(f"⚠️ Could not verify ID correlation: {e}")
    
    def search_documents(self, 
                        query: str, 
                        k: int = 10,
                        include_text: bool = True) -> List[UnifiedSearchResult]:
        """
        Search for documents using vector similarity and retrieve full text.
        
        Args:
            query: Search query text
            k: Number of results to return
            include_text: Whether to retrieve full document text from SQLite
            
        Returns:
            List of UnifiedSearchResult objects
        """
        if not self.model:
            raise ValueError("Model not loaded - cannot encode query")
        
        logger.info(f"🔍 Searching for: '{query}' (top {k} results)")
        
        # Step 1: Encode query using two-tower model
        logger.debug("🔄 Encoding query...")
        with torch.no_grad():
            query_tokens = self.model.tokenize(query).unsqueeze(0).to(self.device)
            query_embedding = self.model.encode_query(query_tokens).cpu().numpy()[0]
        
        # Step 2: Vector search in Redis
        logger.debug("🔴 Searching Redis vector store...")
        vector_results = self.vector_store.search_similar_documents(query_embedding, k=k)
        
        if not vector_results:
            logger.info("📭 No results found in vector search")
            return []
        
        logger.info(f"📋 Found {len(vector_results)} vector matches")
        
        # Step 3: Retrieve document text from SQLite (if requested)
        if include_text:
            logger.debug("💾 Retrieving document text from SQLite...")
            doc_ids = [result.doc_id for result in vector_results]
            documents = self.document_store.get_documents_batch(doc_ids)
            
            logger.info(f"📄 Retrieved {len(documents)} documents from SQLite")
        else:
            documents = {}
        
        # Step 4: Combine results
        unified_results = []
        for vector_result in vector_results:
            doc_id = vector_result.doc_id
            
            if include_text and doc_id in documents:
                doc_data = documents[doc_id]
                unified_results.append(UnifiedSearchResult(
                    doc_id=doc_id,
                    doc_text=doc_data['doc_text'],
                    similarity=vector_result.similarity,
                    doc_length=doc_data['doc_length'],
                    source=doc_data['source'],
                    metadata=doc_data['metadata']
                ))
            else:
                # Vector result only (no text retrieval)
                unified_results.append(UnifiedSearchResult(
                    doc_id=doc_id,
                    doc_text=vector_result.doc_text if hasattr(vector_result, 'doc_text') else "",
                    similarity=vector_result.similarity,
                    doc_length=0,
                    source="unknown",
                    metadata=vector_result.metadata if hasattr(vector_result, 'metadata') else {}
                ))
        
        logger.info(f"✅ Unified search complete - {len(unified_results)} results")
        return unified_results
    
    def get_document_by_id(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a specific document by ID from SQLite.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        return self.document_store.get_document(doc_id)
    
    def get_vector_similarity(self, doc_id: str, query: str) -> Optional[float]:
        """
        Get vector similarity score for a specific document and query.
        
        Args:
            doc_id: Document identifier
            query: Query text
            
        Returns:
            Similarity score or None if not found
        """
        if not self.model:
            return None
        
        # Encode query
        with torch.no_grad():
            query_tokens = self.model.tokenize(query).unsqueeze(0).to(self.device)
            query_embedding = self.model.encode_query(query_tokens).cpu().numpy()[0]
        
        # Get document from Redis
        redis_key = f"doc:{self.vector_store.index_name}:{doc_id}"
        doc_data = self.vector_store.redis_client.hgetall(redis_key)
        
        if not doc_data:
            return None
        
        # Extract embedding
        embedding_bytes = doc_data.get(b'embedding')
        if not embedding_bytes:
            return None
        
        doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
        
        # Calculate cosine similarity
        similarity = np.dot(query_embedding, doc_embedding) / (
            np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding)
        )
        
        return float(similarity)
    
    def demonstrate_id_correlation(self, sample_size: int = 5):
        """
        Demonstrate how document IDs correlate between Redis and SQLite.
        
        Args:
            sample_size: Number of examples to show
        """
        logger.info("🔍 Demonstrating ID Correlation Between Systems")
        logger.info("=" * 60)
        
        # Get sample documents from SQLite
        cursor = self.document_store.conn.cursor()
        cursor.execute(f'SELECT doc_id, doc_text, source FROM documents LIMIT {sample_size}')
        sqlite_docs = cursor.fetchall()
        
        # Check corresponding Redis entries
        for i, (doc_id, doc_text, source) in enumerate(sqlite_docs, 1):
            logger.info(f"\n📄 Example {i}:")
            logger.info(f"   Document ID: {doc_id}")
            logger.info(f"   SQLite text: {doc_text[:80]}...")
            logger.info(f"   Source: {source}")
            
            # Check Redis
            redis_key = f"doc:{self.vector_store.index_name}:{doc_id}"
            redis_exists = self.vector_store.redis_client.exists(redis_key)
            
            if redis_exists:
                redis_data = self.vector_store.redis_client.hgetall(redis_key)
                redis_doc_id = redis_data.get(b'doc_id', b'').decode('utf-8')
                has_embedding = b'embedding' in redis_data
                
                logger.info(f"   ✅ Redis entry: EXISTS")
                logger.info(f"   ✅ Redis doc_id: {redis_doc_id}")
                logger.info(f"   ✅ Has embedding: {has_embedding}")
                
                if has_embedding:
                    embedding = np.frombuffer(redis_data[b'embedding'], dtype=np.float32)
                    logger.info(f"   ✅ Embedding shape: {embedding.shape}")
            else:
                logger.info(f"   ❌ Redis entry: NOT FOUND")
                logger.info(f"   ℹ️ Need to cache this document to Redis")
        
        logger.info("\n" + "=" * 60)
        logger.info("🎯 Key Insight: Both systems use the SAME doc_id as primary key!")
        logger.info("   • SQLite: doc_id -> full document text + metadata")
        logger.info("   • Redis: doc_id -> embedding vector + minimal metadata")
        logger.info("   • Search: Redis finds similar doc_ids → SQLite retrieves text")
    
    def get_system_stats(self) -> Dict:
        """Get statistics from both storage systems."""
        redis_stats = self.vector_store.get_stats()
        sqlite_stats = self.document_store.get_stats()
        
        return {
            'redis': redis_stats,
            'sqlite': sqlite_stats,
            'correlation': {
                'redis_docs': redis_stats.get('document_count', 0),
                'sqlite_docs': sqlite_stats.get('total_documents', 0),
                'coverage_ratio': (
                    redis_stats.get('document_count', 0) / 
                    max(sqlite_stats.get('total_documents', 1), 1)
                )
            }
        }
    
    def close(self):
        """Close database connections."""
        if hasattr(self, 'document_store'):
            self.document_store.close()

def main():
    """Example usage of unified search system."""
    import argparse
    import json
    
    parser = argparse.ArgumentParser(description="Unified Search System Demo")
    parser.add_argument("--config", default="./redis_config.json", help="Redis config file")
    parser.add_argument("--checkpoint", help="Model checkpoint path")
    parser.add_argument("--db", default="./documents.db", help="SQLite database path")
    parser.add_argument("--query", help="Search query")
    parser.add_argument("--demo", action="store_true", help="Show ID correlation demo")
    parser.add_argument("--stats", action="store_true", help="Show system statistics")
    
    args = parser.parse_args()
    
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
        logger.error(f"❌ Failed to load config: {e}")
        return
    
    # Initialize unified search
    try:
        search_system = UnifiedSearchSystem(
            redis_config=redis_config,
            sqlite_db_path=args.db,
            model_checkpoint=args.checkpoint
        )
        
        if args.demo:
            search_system.demonstrate_id_correlation()
        
        if args.stats:
            stats = search_system.get_system_stats()
            logger.info("📊 System Statistics:")
            logger.info(f"   Redis documents: {stats['redis'].get('document_count', 0):,}")
            logger.info(f"   SQLite documents: {stats['sqlite'].get('total_documents', 0):,}")
            logger.info(f"   Coverage ratio: {stats['correlation']['coverage_ratio']:.2%}")
        
        if args.query and args.checkpoint:
            results = search_system.search_documents(args.query, k=5)
            
            logger.info(f"\n🔍 Search Results for: '{args.query}'")
            logger.info("=" * 60)
            
            for i, result in enumerate(results, 1):
                logger.info(f"\n{i}. [{result.similarity:.3f}] {result.doc_id}")
                logger.info(f"   Source: {result.source}")
                logger.info(f"   Text: {result.doc_text[:150]}...")
        
        search_system.close()
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search system: {e}")
        raise

if __name__ == "__main__":
    main() 