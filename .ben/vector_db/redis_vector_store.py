#!/usr/bin/env python3
"""
Redis Vector Store for Document Search

This module provides functionality to:
1. Connect to Redis Cloud
2. Cache document embeddings as vectors
3. Perform vector similarity search
4. Manage vector indices

Requirements:
- Redis Stack or Redis Cloud with RediSearch module
- Pre-trained two-tower model for encoding documents
"""

import redis
import numpy as np
import torch
import json
import pickle
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from tqdm import tqdm
import logging
from datetime import datetime
import sys

# Import the two-tower model
sys.path.append('../training')
from two_tower_model import TwoTowerModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class VectorSearchResult:
    """Result from vector search."""
    doc_id: str
    doc_text: str
    similarity: float
    metadata: Dict[str, Any] = None

class RedisVectorStore:
    """
    Redis-based vector store for document embeddings.
    
    Uses Redis Stack's vector similarity search capabilities.
    """
    
    def __init__(self, 
                 redis_url: str = None,
                 redis_host: str = "localhost",
                 redis_port: int = 6379,
                 redis_password: str = None,
                 redis_db: int = 0,
                 index_name: str = "doc_embeddings",
                 vector_dim: int = 256):
        """
        Initialize Redis vector store.
        
        Args:
            redis_url: Full Redis URL (for Redis Cloud)
            redis_host: Redis host (if not using URL)
            redis_port: Redis port
            redis_password: Redis password
            redis_db: Redis database number
            index_name: Name for the vector index
            vector_dim: Dimension of embedding vectors
        """
        self.index_name = index_name
        self.vector_dim = vector_dim
        
        # Connect to Redis
        if redis_url:
            self.redis_client = redis.from_url(redis_url)
        else:
            self.redis_client = redis.Redis(
                host=redis_host,
                port=redis_port,
                password=redis_password,
                db=redis_db,
                decode_responses=False  # Keep as bytes for binary data
            )
        
        # Test connection
        try:
            self.redis_client.ping()
            logger.info("✅ Connected to Redis successfully")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
        
        # Initialize vector index
        self._ensure_vector_index()
    
    def _ensure_vector_index(self):
        """Create vector index if it doesn't exist."""
        try:
            # Check if index exists
            try:
                info = self.redis_client.execute_command("FT.INFO", self.index_name)
                logger.info(f"✅ Vector index '{self.index_name}' already exists")
                return
            except:
                # Index doesn't exist, create it
                pass
            
            # Create vector index
            index_cmd = [
                "FT.CREATE", self.index_name,
                "ON", "HASH",
                "PREFIX", "1", f"doc:{self.index_name}:",
                "SCHEMA",
                "doc_text", "TEXT",
                "doc_id", "TEXT",
                "passage_id", "TEXT", 
                "query_id", "TEXT",
                "metadata", "TEXT",
                "embedding", "VECTOR", "FLAT", "6",
                "TYPE", "FLOAT32",
                "DIM", str(self.vector_dim),
                "DISTANCE_METRIC", "COSINE"
            ]
            
            result = self.redis_client.execute_command(*index_cmd)
            logger.info(f"✅ Created vector index '{self.index_name}': {result}")
            
        except Exception as e:
            logger.error(f"❌ Failed to create vector index: {e}")
            raise
    
    def add_document(self, 
                     doc_id: str,
                     doc_text: str, 
                     embedding: np.ndarray,
                     metadata: Dict[str, Any] = None) -> bool:
        """
        Add a document with its embedding to the vector store.
        
        Args:
            doc_id: Unique document identifier
            doc_text: Document text content
            embedding: Document embedding vector
            metadata: Additional metadata (query_id, passage_id, etc.)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Normalize embedding
            if isinstance(embedding, torch.Tensor):
                embedding = embedding.cpu().numpy()
            
            embedding = embedding.astype(np.float32)
            
            # Create Redis key
            redis_key = f"doc:{self.index_name}:{doc_id}"
            
            # Prepare document data
            doc_data = {
                "doc_id": doc_id,
                "doc_text": doc_text,
                "embedding": embedding.tobytes(),
                "metadata": json.dumps(metadata or {}),
                "timestamp": datetime.now().isoformat()
            }
            
            # Add passage_id and query_id if available in metadata
            if metadata:
                if "passage_id" in metadata:
                    doc_data["passage_id"] = metadata["passage_id"]
                if "query_id" in metadata:
                    doc_data["query_id"] = metadata["query_id"]
            
            # Store in Redis
            result = self.redis_client.hset(redis_key, mapping=doc_data)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add document {doc_id}: {e}")
            return False
    
    def add_documents_batch(self, 
                           documents: List[Tuple[str, str, np.ndarray, Dict[str, Any]]],
                           batch_size: int = 100) -> Dict[str, int]:
        """
        Add multiple documents in batches.
        
        Args:
            documents: List of (doc_id, doc_text, embedding, metadata) tuples
            batch_size: Number of documents to process in each batch
            
        Returns:
            Dictionary with success/failure counts
        """
        logger.info(f"🔄 Adding {len(documents)} documents to vector store...")
        
        stats = {"success": 0, "failed": 0}
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents"):
            batch = documents[i:i + batch_size]
            
            # Use pipeline for better performance
            pipe = self.redis_client.pipeline()
            
            for doc_id, doc_text, embedding, metadata in batch:
                try:
                    # Normalize embedding
                    if isinstance(embedding, torch.Tensor):
                        embedding = embedding.cpu().numpy()
                    
                    embedding = embedding.astype(np.float32)
                    
                    # Create Redis key
                    redis_key = f"doc:{self.index_name}:{doc_id}"
                    
                    # Prepare document data
                    doc_data = {
                        "doc_id": doc_id,
                        "doc_text": doc_text,
                        "embedding": embedding.tobytes(),
                        "metadata": json.dumps(metadata or {}),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Add passage_id and query_id if available
                    if metadata:
                        if "passage_id" in metadata:
                            doc_data["passage_id"] = metadata["passage_id"]
                        if "query_id" in metadata:
                            doc_data["query_id"] = metadata["query_id"]
                    
                    # Add to pipeline
                    pipe.hset(redis_key, mapping=doc_data)
                    stats["success"] += 1
                    
                except Exception as e:
                    logger.error(f"❌ Failed to prepare document {doc_id}: {e}")
                    stats["failed"] += 1
            
            # Execute pipeline
            try:
                pipe.execute()
            except Exception as e:
                logger.error(f"❌ Failed to execute batch: {e}")
                stats["failed"] += len(batch)
                stats["success"] -= len(batch)
        
        logger.info(f"✅ Batch complete - Success: {stats['success']}, Failed: {stats['failed']}")
        return stats
    
    def search_similar_documents(self, 
                               query_embedding: np.ndarray,
                               k: int = 10,
                               filter_query: str = None) -> List[VectorSearchResult]:
        """
        Search for similar documents using vector similarity.
        
        Args:
            query_embedding: Query embedding vector
            k: Number of top results to return
            filter_query: Optional filter query (e.g., "@query_id:123")
            
        Returns:
            List of VectorSearchResult objects
        """
        try:
            # Normalize query embedding
            if isinstance(query_embedding, torch.Tensor):
                query_embedding = query_embedding.cpu().numpy()
            
            query_embedding = query_embedding.astype(np.float32)
            
            # First try Redis vector search (if supported)
            try:
                # Build search query
                if filter_query:
                    vector_query = f"({filter_query})=>[KNN {k} @embedding $query_vec AS vector_score]"
                else:
                    vector_query = f"*=>[KNN {k} @embedding $query_vec AS vector_score]"
                
                # Execute search
                search_cmd = [
                    "FT.SEARCH", self.index_name,
                    vector_query,
                    "PARAMS", "2", "query_vec", query_embedding.tobytes(),
                    "SORTBY", "vector_score",
                    "RETURN", "4", "doc_id", "doc_text", "metadata", "vector_score",
                    "DIALECT", "2"
                ]
                
                result = self.redis_client.execute_command(*search_cmd)
                
                # Parse results
                if result and len(result) >= 2:
                    results = []
                    
                    # Process results (skip count, then process pairs)
                    for i in range(1, len(result), 2):
                        if i + 1 >= len(result):
                            break
                        
                        doc_key = result[i]
                        doc_fields = result[i + 1]
                        
                        # Parse document fields
                        doc_data = {}
                        for j in range(0, len(doc_fields), 2):
                            if j + 1 < len(doc_fields):
                                field_name = doc_fields[j].decode('utf-8')
                                field_value = doc_fields[j + 1]
                                
                                if isinstance(field_value, bytes):
                                    field_value = field_value.decode('utf-8')
                                
                                doc_data[field_name] = field_value
                        
                        # Create result object
                        metadata = {}
                        if 'metadata' in doc_data:
                            try:
                                metadata = json.loads(doc_data['metadata'])
                            except:
                                pass
                        
                        similarity = 1.0 - float(doc_data.get('vector_score', 1.0))  # Convert distance to similarity
                        
                        results.append(VectorSearchResult(
                            doc_id=doc_data.get('doc_id', ''),
                            doc_text=doc_data.get('doc_text', ''),
                            similarity=similarity,
                            metadata=metadata
                        ))
                    
                    return results
                    
            except Exception as vector_search_error:
                logger.warning(f"⚠️ Vector search not supported, falling back to manual similarity: {vector_search_error}")
            
            # Fallback: Manual similarity calculation
            logger.info("🔄 Using manual similarity calculation (fallback mode)")
            
            # Get all document keys (or filtered set)
            if filter_query:
                # For now, get all keys and filter later
                # TODO: Implement proper filtering
                keys = self.redis_client.keys(f"doc:{self.index_name}:*")
            else:
                keys = self.redis_client.keys(f"doc:{self.index_name}:*")
            
            if not keys:
                return []
            
            # Calculate similarities for all documents
            similarities = []
            
            # Process in batches to avoid memory issues
            batch_size = min(1000, len(keys))  # Process up to 1000 docs at a time
            
            for i in range(0, len(keys), batch_size):
                batch_keys = keys[i:i + batch_size]
                
                # Use pipeline for efficiency
                pipe = self.redis_client.pipeline()
                for key in batch_keys:
                    pipe.hgetall(key)
                
                batch_docs = pipe.execute()
                
                # Calculate similarities for this batch
                for key, doc_data in zip(batch_keys, batch_docs):
                    if doc_data and b'embedding' in doc_data:
                        try:
                            # Extract embedding
                            embedding_bytes = doc_data[b'embedding']
                            doc_embedding = np.frombuffer(embedding_bytes, dtype=np.float32)
                            
                            # Calculate cosine similarity
                            dot_product = np.dot(query_embedding, doc_embedding)
                            norm_query = np.linalg.norm(query_embedding)
                            norm_doc = np.linalg.norm(doc_embedding)
                            
                            if norm_query > 0 and norm_doc > 0:
                                similarity = dot_product / (norm_query * norm_doc)
                                
                                doc_id = doc_data[b'doc_id'].decode() if b'doc_id' in doc_data else key.decode()
                                doc_text = doc_data[b'doc_text'].decode() if b'doc_text' in doc_data else ""
                                
                                metadata = {}
                                if b'metadata' in doc_data:
                                    try:
                                        metadata = json.loads(doc_data[b'metadata'].decode())
                                    except:
                                        pass
                                
                                similarities.append((similarity, doc_id, doc_text, metadata))
                                
                        except Exception as e:
                            logger.debug(f"Failed to process document {key}: {e}")
                            continue
            
            # Sort by similarity (descending) and take top k
            similarities.sort(key=lambda x: x[0], reverse=True)
            top_similarities = similarities[:k]
            
            # Convert to VectorSearchResult objects
            results = []
            for similarity, doc_id, doc_text, metadata in top_similarities:
                results.append(VectorSearchResult(
                    doc_id=doc_id,
                    doc_text=doc_text,
                    similarity=similarity,
                    metadata=metadata
                ))
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Search failed: {e}")
            return []
    
    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            redis_key = f"doc:{self.index_name}:{doc_id}"
            doc_data = self.redis_client.hgetall(redis_key)
            
            if not doc_data:
                return None
            
            # Convert bytes to strings
            result = {}
            for key, value in doc_data.items():
                key_str = key.decode('utf-8') if isinstance(key, bytes) else key
                
                if key_str == 'embedding':
                    # Convert bytes back to numpy array
                    result[key_str] = np.frombuffer(value, dtype=np.float32)
                elif key_str == 'metadata':
                    try:
                        result[key_str] = json.loads(value.decode('utf-8'))
                    except:
                        result[key_str] = {}
                else:
                    result[key_str] = value.decode('utf-8') if isinstance(value, bytes) else value
            
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {doc_id}: {e}")
            return None
    
    def delete_document(self, doc_id: str) -> bool:
        """
        Delete a document from the vector store.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            True if successful, False otherwise
        """
        try:
            redis_key = f"doc:{self.index_name}:{doc_id}"
            result = self.redis_client.delete(redis_key)
            return result > 0
            
        except Exception as e:
            logger.error(f"❌ Failed to delete document {doc_id}: {e}")
            return False
    
    def clear_all_documents(self) -> bool:
        """
        Clear all documents from the vector store.
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Find all document keys
            pattern = f"doc:{self.index_name}:*"
            keys = self.redis_client.keys(pattern)
            
            if keys:
                result = self.redis_client.delete(*keys)
                logger.info(f"✅ Deleted {result} documents")
                return True
            else:
                logger.info("ℹ️ No documents to delete")
                return True
                
        except Exception as e:
            logger.error(f"❌ Failed to clear documents: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector store.
        
        Returns:
            Dictionary with statistics
        """
        try:
            # Count documents
            pattern = f"doc:{self.index_name}:*"
            doc_count = len(self.redis_client.keys(pattern))
            
            # Get index info
            try:
                index_info = self.redis_client.execute_command("FT.INFO", self.index_name)
                # Parse index info (it's a list of key-value pairs)
                index_data = {}
                for i in range(0, len(index_info), 2):
                    if i + 1 < len(index_info):
                        key = index_info[i].decode('utf-8') if isinstance(index_info[i], bytes) else index_info[i]
                        value = index_info[i + 1]
                        if isinstance(value, bytes):
                            value = value.decode('utf-8')
                        index_data[key] = value
            except:
                index_data = {}
            
            return {
                "document_count": doc_count,
                "index_name": self.index_name,
                "vector_dimension": self.vector_dim,
                "index_info": index_data,
                "redis_info": {
                    "used_memory": self.redis_client.info().get('used_memory_human', 'N/A'),
                    "connected_clients": self.redis_client.info().get('connected_clients', 'N/A')
                }
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {"error": str(e)}

def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> TwoTowerModel:
    """
    Load a trained two-tower model from checkpoint.
    
    Args:
        checkpoint_path: Path to the model checkpoint
        device: Device to load the model on
        
    Returns:
        Loaded TwoTowerModel
    """
    logger.info(f"📦 Loading model from: {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model components
    embedding_matrix = checkpoint["embedding_matrix"]
    word_to_index = checkpoint["word_to_index"]
    model_config = checkpoint.get("model_config", {
        "hidden_dim": 256,
        "num_layers": 2,
        "dropout": 0.1
    })
    
    # Create model
    model = TwoTowerModel(
        embedding_matrix=embedding_matrix,
        word_to_index=word_to_index,
        hidden_dim=model_config["hidden_dim"],
        num_layers=model_config["num_layers"],
        dropout=model_config["dropout"]
    )
    
    # Load model state
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded successfully")
    return model

def test_redis_connection():
    """Test Redis connection with sample data."""
    logger.info("🔍 Testing Redis connection...")
    
    # Test with local Redis first
    try:
        store = RedisVectorStore(
            redis_host="localhost",
            redis_port=6379,
            index_name="test_index"
        )
        
        # Add a test document
        test_embedding = np.random.rand(256).astype(np.float32)
        success = store.add_document(
            doc_id="test_doc_1",
            doc_text="This is a test document for vector search.",
            embedding=test_embedding,
            metadata={"test": True}
        )
        
        if success:
            logger.info("✅ Redis connection test successful!")
            
            # Test search
            results = store.search_similar_documents(test_embedding, k=1)
            if results:
                logger.info(f"✅ Search test successful! Found: {results[0].doc_text}")
            
            # Clean up
            store.delete_document("test_doc_1")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Redis connection test failed: {e}")
        return False

if __name__ == "__main__":
    # Quick test
    test_redis_connection()