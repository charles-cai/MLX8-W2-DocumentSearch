#!/usr/bin/env python3
"""
Redis Vector Database Manager

A simple vector database implementation using Redis for storage and numpy for vector operations.
This provides basic vector storage and similarity search functionality.
"""

import redis
import numpy as np
import json
import pickle
from typing import List, Dict, Tuple, Optional
import logging
from datetime import datetime

class RedisVectorDB:
    """Simple vector database using Redis as storage backend."""
    
    def __init__(self, config_path: str = "redis_config.json"):
        """Initialize Redis vector database."""
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Connect to Redis
        self.redis_client = redis.Redis(
            host=self.config['redis']['host'],
            port=self.config['redis']['port'],
            db=self.config['redis']['db'],
            username=self.config['redis']['username'],
            password=self.config['redis']['password']
        )
        
        # Configuration
        self.index_name = self.config['vector_store']['index_name']
        self.vector_dim = self.config['vector_store']['vector_dim']
        self.batch_size = self.config['processing']['batch_size']
        
        # Test connection
        try:
            self.redis_client.ping()
            print(f"✅ Connected to Redis at {self.config['redis']['host']}:{self.config['redis']['port']}")
        except redis.ConnectionError as e:
            raise ConnectionError(f"Failed to connect to Redis: {e}")
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def create_index(self, overwrite: bool = False):
        """Create vector index (initialize metadata)."""
        index_key = f"{self.index_name}:meta"
        
        if self.redis_client.exists(index_key) and not overwrite:
            print(f"Index '{self.index_name}' already exists. Use overwrite=True to recreate.")
            return
        
        # Store index metadata
        metadata = {
            'index_name': self.index_name,
            'vector_dim': self.vector_dim,
            'created_at': datetime.now().isoformat(),
            'doc_count': 0
        }
        
        self.redis_client.hset(index_key, mapping={
            'metadata': json.dumps(metadata)
        })
        
        # Clear existing documents if overwriting
        if overwrite:
            self._clear_all_documents()
        
        print(f"✅ Created vector index '{self.index_name}' with dimension {self.vector_dim}")
    
    def add_document(self, doc_id: str, vector: np.ndarray, metadata: Dict = None):
        """Add a single document with its vector."""
        if vector.shape[0] != self.vector_dim:
            raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match index dimension {self.vector_dim}")
        
        # Store vector and metadata
        doc_key = f"{self.index_name}:doc:{doc_id}"
        
        doc_data = {
            'vector': pickle.dumps(vector.astype(np.float32)),
            'metadata': json.dumps(metadata or {}),
            'added_at': datetime.now().isoformat()
        }
        
        self.redis_client.hset(doc_key, mapping=doc_data)
        
        # Add to document list
        self.redis_client.sadd(f"{self.index_name}:docs", doc_id)
        
        # Update document count
        self._increment_doc_count()
    
    def add_documents_batch(self, documents: List[Tuple[str, np.ndarray, Dict]]):
        """Add multiple documents in batch."""
        print(f"Adding {len(documents)} documents to index '{self.index_name}'...")
        
        # Use pipeline for efficiency
        pipe = self.redis_client.pipeline()
        
        for doc_id, vector, metadata in documents:
            if vector.shape[0] != self.vector_dim:
                raise ValueError(f"Vector dimension {vector.shape[0]} doesn't match index dimension {self.vector_dim}")
            
            doc_key = f"{self.index_name}:doc:{doc_id}"
            doc_data = {
                'vector': pickle.dumps(vector.astype(np.float32)),
                'metadata': json.dumps(metadata or {}),
                'added_at': datetime.now().isoformat()
            }
            
            pipe.hset(doc_key, mapping=doc_data)
            pipe.sadd(f"{self.index_name}:docs", doc_id)
        
        # Execute pipeline
        pipe.execute()
        
        # Update document count
        self._set_doc_count(len(documents))
        
        print(f"✅ Added {len(documents)} documents to index")
    
    def search(self, query_vector: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, Dict]]:
        """Search for similar vectors."""
        if query_vector.shape[0] != self.vector_dim:
            raise ValueError(f"Query vector dimension {query_vector.shape[0]} doesn't match index dimension {self.vector_dim}")
        
        # Get all document IDs
        doc_ids = self.redis_client.smembers(f"{self.index_name}:docs")
        
        if not doc_ids:
            return []
        
        similarities = []
        
        # Calculate similarities for all documents
        for doc_id_bytes in doc_ids:
            doc_id = doc_id_bytes.decode('utf-8')
            doc_key = f"{self.index_name}:doc:{doc_id}"
            
            # Get document data
            doc_data = self.redis_client.hgetall(doc_key)
            if not doc_data:
                continue
            
            # Deserialize vector
            vector = pickle.loads(doc_data[b'vector'])
            metadata = json.loads(doc_data[b'metadata'].decode('utf-8'))
            
            # Calculate cosine similarity
            similarity = self._cosine_similarity(query_vector, vector)
            similarities.append((doc_id, similarity, metadata))
        
        # Sort by similarity (descending) and return top_k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]
    
    def get_document(self, doc_id: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Get a document by ID."""
        doc_key = f"{self.index_name}:doc:{doc_id}"
        doc_data = self.redis_client.hgetall(doc_key)
        
        if not doc_data:
            return None
        
        vector = pickle.loads(doc_data[b'vector'])
        metadata = json.loads(doc_data[b'metadata'].decode('utf-8'))
        
        return vector, metadata
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document."""
        doc_key = f"{self.index_name}:doc:{doc_id}"
        
        # Remove from documents set
        removed = self.redis_client.srem(f"{self.index_name}:docs", doc_id)
        
        # Delete document data
        self.redis_client.delete(doc_key)
        
        if removed:
            self._decrement_doc_count()
        
        return bool(removed)
    
    def get_stats(self) -> Dict:
        """Get index statistics."""
        index_key = f"{self.index_name}:meta"
        metadata_json = self.redis_client.hget(index_key, 'metadata')
        
        if not metadata_json:
            return {'error': 'Index not found'}
        
        metadata = json.loads(metadata_json.decode('utf-8'))
        
        # Get current document count
        doc_count = self.redis_client.scard(f"{self.index_name}:docs")
        metadata['current_doc_count'] = doc_count
        
        return metadata
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors."""
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    def _increment_doc_count(self):
        """Increment document count in metadata."""
        index_key = f"{self.index_name}:meta"
        metadata_json = self.redis_client.hget(index_key, 'metadata')
        
        if metadata_json:
            metadata = json.loads(metadata_json.decode('utf-8'))
            metadata['doc_count'] = metadata.get('doc_count', 0) + 1
            self.redis_client.hset(index_key, 'metadata', json.dumps(metadata))
    
    def _decrement_doc_count(self):
        """Decrement document count in metadata."""
        index_key = f"{self.index_name}:meta"
        metadata_json = self.redis_client.hget(index_key, 'metadata')
        
        if metadata_json:
            metadata = json.loads(metadata_json.decode('utf-8'))
            metadata['doc_count'] = max(0, metadata.get('doc_count', 0) - 1)
            self.redis_client.hset(index_key, 'metadata', json.dumps(metadata))
    
    def _set_doc_count(self, count: int):
        """Set document count in metadata."""
        index_key = f"{self.index_name}:meta"
        metadata_json = self.redis_client.hget(index_key, 'metadata')
        
        if metadata_json:
            metadata = json.loads(metadata_json.decode('utf-8'))
            metadata['doc_count'] = metadata.get('doc_count', 0) + count
            self.redis_client.hset(index_key, 'metadata', json.dumps(metadata))
    
    def _clear_all_documents(self):
        """Clear all documents from the index."""
        # Get all document IDs
        doc_ids = self.redis_client.smembers(f"{self.index_name}:docs")
        
        if doc_ids:
            # Delete all document keys
            pipe = self.redis_client.pipeline()
            for doc_id_bytes in doc_ids:
                doc_id = doc_id_bytes.decode('utf-8')
                doc_key = f"{self.index_name}:doc:{doc_id}"
                pipe.delete(doc_key)
            pipe.execute()
            
            # Clear document set
            self.redis_client.delete(f"{self.index_name}:docs")
        
        print(f"Cleared {len(doc_ids)} documents from index")


def test_vector_db():
    """Test the vector database functionality."""
    print("🧪 Testing Redis Vector Database...")
    
    # Initialize database
    db = RedisVectorDB()
    
    # Create index
    db.create_index(overwrite=True)
    
    # Test data
    test_vectors = [
        ("doc1", np.random.rand(256), {"title": "Document 1", "content": "This is document 1"}),
        ("doc2", np.random.rand(256), {"title": "Document 2", "content": "This is document 2"}),
        ("doc3", np.random.rand(256), {"title": "Document 3", "content": "This is document 3"}),
    ]
    
    # Add documents
    db.add_documents_batch(test_vectors)
    
    # Test search
    query_vector = np.random.rand(256)
    results = db.search(query_vector, top_k=2)
    
    print(f"Search results for random query:")
    for doc_id, similarity, metadata in results:
        print(f"  {doc_id}: {similarity:.4f} - {metadata['title']}")
    
    # Test individual document retrieval
    vector, metadata = db.get_document("doc1")
    print(f"Retrieved doc1: {metadata['title']}")
    
    # Get stats
    stats = db.get_stats()
    print(f"Index stats: {stats}")
    
    print("✅ Vector database test completed!")


if __name__ == "__main__":
    test_vector_db() 