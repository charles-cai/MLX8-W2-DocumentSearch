#!/usr/bin/env python3
"""
Document Store for MS MARCO Documents

Provides efficient storage and retrieval of document text content,
designed to work alongside Redis vector store for embeddings.

This keeps document text separate from embeddings to minimize Redis memory usage.
"""

import sqlite3
import pickle
import os
import json
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentStore:
    """
    SQLite-based document store for efficient text retrieval.
    
    Stores document text separately from embeddings to minimize
    Redis memory usage while maintaining fast retrieval.
    """
    
    def __init__(self, db_path: str = "./documents.db"):
        """
        Initialize document store.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row  # Enable dict-like access
        self._create_tables()
    
    def _create_tables(self):
        """Create necessary database tables."""
        cursor = self.conn.cursor()
        
        # Main documents table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                doc_text TEXT NOT NULL,
                source TEXT,
                doc_length INTEGER,
                created_at TEXT,
                metadata TEXT
            )
        ''')
        
        # Create index for fast lookups
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_source ON documents(source)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_length ON documents(doc_length)
        ''')
        
        self.conn.commit()
    
    def add_document(self, doc_id: str, doc_text: str, source: str = None, metadata: Dict = None) -> bool:
        """
        Add a single document to the store.
        
        Args:
            doc_id: Unique document identifier
            doc_text: Document text content
            source: Source of the document (e.g., 'raw_data', 'validation')
            metadata: Additional metadata as dictionary
            
        Returns:
            True if successful, False otherwise
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('''
                INSERT OR REPLACE INTO documents 
                (doc_id, doc_text, source, doc_length, created_at, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                doc_id,
                doc_text,
                source or 'unknown',
                len(doc_text),
                datetime.now().isoformat(),
                json.dumps(metadata or {})
            ))
            self.conn.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to add document {doc_id}: {e}")
            return False
    
    def add_documents_batch(self, documents: List[Tuple[str, str, str, Dict]], batch_size: int = 1000) -> Dict[str, int]:
        """
        Add multiple documents in batches.
        
        Args:
            documents: List of (doc_id, doc_text, source, metadata) tuples
            batch_size: Number of documents per batch
            
        Returns:
            Dictionary with success/failure counts
        """
        logger.info(f"📦 Adding {len(documents)} documents to document store...")
        
        stats = {"success": 0, "failed": 0}
        cursor = self.conn.cursor()
        
        for i in tqdm(range(0, len(documents), batch_size), desc="Adding documents"):
            batch = documents[i:i + batch_size]
            
            try:
                batch_data = []
                for doc_id, doc_text, source, metadata in batch:
                    batch_data.append((
                        doc_id,
                        doc_text,
                        source or 'unknown',
                        len(doc_text),
                        datetime.now().isoformat(),
                        json.dumps(metadata or {})
                    ))
                
                cursor.executemany('''
                    INSERT OR REPLACE INTO documents 
                    (doc_id, doc_text, source, doc_length, created_at, metadata)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', batch_data)
                
                self.conn.commit()
                stats["success"] += len(batch)
                
            except Exception as e:
                logger.error(f"❌ Failed to add batch starting at {i}: {e}")
                stats["failed"] += len(batch)
        
        logger.info(f"✅ Batch complete - Success: {stats['success']}, Failed: {stats['failed']}")
        return stats
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document by ID.
        
        Args:
            doc_id: Document identifier
            
        Returns:
            Document data or None if not found
        """
        try:
            cursor = self.conn.cursor()
            cursor.execute('SELECT * FROM documents WHERE doc_id = ?', (doc_id,))
            row = cursor.fetchone()
            
            if row:
                result = dict(row)
                # Parse metadata JSON
                try:
                    result['metadata'] = json.loads(result['metadata'])
                except:
                    result['metadata'] = {}
                return result
            return None
            
        except Exception as e:
            logger.error(f"❌ Failed to get document {doc_id}: {e}")
            return None
    
    def get_documents_batch(self, doc_ids: List[str]) -> Dict[str, Dict]:
        """
        Retrieve multiple documents by IDs.
        
        Args:
            doc_ids: List of document identifiers
            
        Returns:
            Dictionary mapping doc_id to document data
        """
        if not doc_ids:
            return {}
        
        try:
            cursor = self.conn.cursor()
            placeholders = ','.join('?' for _ in doc_ids)
            cursor.execute(f'SELECT * FROM documents WHERE doc_id IN ({placeholders})', doc_ids)
            rows = cursor.fetchall()
            
            results = {}
            for row in rows:
                doc_data = dict(row)
                try:
                    doc_data['metadata'] = json.loads(doc_data['metadata'])
                except:
                    doc_data['metadata'] = {}
                results[doc_data['doc_id']] = doc_data
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to get documents batch: {e}")
            return {}
    
    def search_documents(self, query: str = None, source: str = None, limit: int = 100) -> List[Dict]:
        """
        Search documents with optional filters.
        
        Args:
            query: Text to search in document content (simple LIKE search)
            source: Filter by document source
            limit: Maximum number of results
            
        Returns:
            List of matching documents
        """
        try:
            cursor = self.conn.cursor()
            sql = 'SELECT * FROM documents WHERE 1=1'
            params = []
            
            if query:
                sql += ' AND doc_text LIKE ?'
                params.append(f'%{query}%')
            
            if source:
                sql += ' AND source = ?'
                params.append(source)
            
            sql += f' ORDER BY doc_length DESC LIMIT {limit}'
            
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            
            results = []
            for row in rows:
                doc_data = dict(row)
                try:
                    doc_data['metadata'] = json.loads(doc_data['metadata'])
                except:
                    doc_data['metadata'] = {}
                results.append(doc_data)
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Failed to search documents: {e}")
            return []
    
    def get_stats(self) -> Dict:
        """Get statistics about the document store."""
        try:
            cursor = self.conn.cursor()
            
            # Total documents
            cursor.execute('SELECT COUNT(*) as total FROM documents')
            total = cursor.fetchone()['total']
            
            # By source
            cursor.execute('SELECT source, COUNT(*) as count FROM documents GROUP BY source')
            by_source = {row['source']: row['count'] for row in cursor.fetchall()}
            
            # Average document length
            cursor.execute('SELECT AVG(doc_length) as avg_length FROM documents')
            avg_length = cursor.fetchone()['avg_length']
            
            # Database file size
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            
            return {
                'total_documents': total,
                'by_source': by_source,
                'average_length': round(avg_length, 2) if avg_length else 0,
                'database_size_mb': round(db_size / (1024 * 1024), 2),
                'database_path': self.db_path
            }
            
        except Exception as e:
            logger.error(f"❌ Failed to get stats: {e}")
            return {'error': str(e)}
    
    def clear_all(self) -> bool:
        """Clear all documents from the store."""
        try:
            cursor = self.conn.cursor()
            cursor.execute('DELETE FROM documents')
            self.conn.commit()
            logger.info("✅ Cleared all documents from store")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to clear documents: {e}")
            return False
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()

def create_document_store_from_pickle(data_path: str, db_path: str = "./documents.db") -> DocumentStore:
    """
    Create a document store from existing pickle files.
    
    Args:
        data_path: Path to MS MARCO data directory
        db_path: Path for SQLite database
        
    Returns:
        Initialized DocumentStore
    """
    logger.info("🏗️ Creating document store from pickle files...")
    
    # Initialize store
    doc_store = DocumentStore(db_path)
    
    # Clear existing data
    doc_store.clear_all()
    
    all_documents = []
    
    # Load from raw data files
    raw_data_files = [
        "msmarco_raw_data.pkl",
        "msmarco_raw_data_limited_50000.pkl", 
        "msmarco_raw_data_limited_5000.pkl",
        "msmarco_raw_data_limited_2000.pkl",
        "msmarco_raw_data_limited_1000.pkl"
    ]
    
    for filename in raw_data_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            logger.info(f"📂 Loading from raw data: {filename}")
            
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            if 'passages' in data:
                passages = data['passages']
                for doc_id, doc_text in passages.items():
                    all_documents.append((
                        doc_id,
                        doc_text,
                        'raw_data',
                        {'source_file': filename}
                    ))
                logger.info(f"✅ Added {len(passages)} documents from raw data")
                break  # Use the largest available file
    
    # Load from validation data
    validation_files = ["msmarco_validation_eval.pkl"]
    
    for filename in validation_files:
        filepath = os.path.join(data_path, filename)
        if os.path.exists(filepath):
            logger.info(f"📂 Loading from validation: {filename}")
            
            with open(filepath, 'rb') as f:
                val_data = pickle.load(f)
            
            if 'passages' in val_data:
                passages = val_data['passages']
                existing_ids = {doc[0] for doc in all_documents}
                
                for doc_id, doc_text in passages.items():
                    if doc_id not in existing_ids:
                        all_documents.append((
                            doc_id,
                            doc_text,
                            'validation_eval',
                            {'source_file': filename}
                        ))
                
                new_count = len([d for d in all_documents if d[2] == 'validation_eval'])
                logger.info(f"✅ Added {new_count} new documents from validation")
    
    # Add documents to store
    if all_documents:
        doc_store.add_documents_batch(all_documents)
        
        # Show final stats
        stats = doc_store.get_stats()
        logger.info("📊 Document store created successfully!")
        logger.info(f"   Total documents: {stats['total_documents']:,}")
        logger.info(f"   Database size: {stats['database_size_mb']} MB")
        logger.info(f"   By source: {stats['by_source']}")
    else:
        logger.warning("⚠️ No documents found to add to store")
    
    return doc_store

if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="Create document store from MS MARCO data")
    parser.add_argument("--data", default="../data", help="Path to MS MARCO data directory")
    parser.add_argument("--db", default="./documents.db", help="Path for SQLite database")
    parser.add_argument("--test", action="store_true", help="Run test queries")
    
    args = parser.parse_args()
    
    # Create document store
    doc_store = create_document_store_from_pickle(args.data, args.db)
    
    if args.test:
        # Test retrieval
        logger.info("\n🔍 Testing document retrieval...")
        stats = doc_store.get_stats()
        
        if stats['total_documents'] > 0:
            # Test single document retrieval
            cursor = doc_store.conn.cursor()
            cursor.execute('SELECT doc_id FROM documents LIMIT 1')
            test_id = cursor.fetchone()['doc_id']
            
            doc = doc_store.get_document(test_id)
            if doc:
                logger.info(f"✅ Retrieved document {test_id}: {doc['doc_text'][:100]}...")
            
            # Test batch retrieval
            cursor.execute('SELECT doc_id FROM documents LIMIT 5')
            test_ids = [row['doc_id'] for row in cursor.fetchall()]
            
            docs = doc_store.get_documents_batch(test_ids)
            logger.info(f"✅ Batch retrieved {len(docs)} documents")
        
        # Show stats
        logger.info(f"\n📊 Final stats: {stats}")
    
    doc_store.close() 