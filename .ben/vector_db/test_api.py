#!/usr/bin/env python3
"""
Test Script for Search API

This script demonstrates how to interact with the search API server.
"""

import requests
import json
import time
from typing import Dict, List

class SearchAPIClient:
    """Simple client for the Search API."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip('/')
    
    def health_check(self) -> Dict:
        """Check API health."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict:
        """Get system statistics."""
        response = requests.get(f"{self.base_url}/stats")
        response.raise_for_status()
        return response.json()
    
    def search_get(self, query: str, k: int = 10, include_text: bool = True) -> Dict:
        """Search using GET request."""
        params = {
            "q": query,
            "k": k,
            "include_text": include_text
        }
        response = requests.get(f"{self.base_url}/search", params=params)
        response.raise_for_status()
        return response.json()
    
    def search_post(self, query: str, k: int = 10, include_text: bool = True) -> Dict:
        """Search using POST request."""
        data = {
            "query": query,
            "k": k,
            "include_text": include_text
        }
        response = requests.post(f"{self.base_url}/search", json=data)
        response.raise_for_status()
        return response.json()
    
    def get_document(self, doc_id: str) -> Dict:
        """Get a specific document."""
        response = requests.get(f"{self.base_url}/document/{doc_id}")
        response.raise_for_status()
        return response.json()
    
    def get_similarity(self, doc_id: str, query: str) -> Dict:
        """Get similarity between document and query."""
        params = {"query": query}
        response = requests.get(f"{self.base_url}/similarity/{doc_id}", params=params)
        response.raise_for_status()
        return response.json()

def test_api():
    """Test the search API functionality."""
    client = SearchAPIClient()
    
    print("🧪 Testing Search API")
    print("=" * 50)
    
    try:
        # Test health check
        print("\n1. Health Check:")
        health = client.health_check()
        print(f"   Status: {health['status']}")
        print(f"   Redis: {'✅' if health['redis_connected'] else '❌'}")
        print(f"   SQLite: {'✅' if health['sqlite_connected'] else '❌'}")
        print(f"   Model: {'✅' if health['model_loaded'] else '❌'}")
        print(f"   Uptime: {health['uptime_seconds']:.1f}s")
        
        # Test system stats
        print("\n2. System Statistics:")
        stats = client.get_stats()
        print(f"   Redis documents: {stats['redis_documents']:,}")
        print(f"   SQLite documents: {stats['sqlite_documents']:,}")
        print(f"   Coverage: {stats['coverage_ratio']:.1%}")
        print(f"   Redis memory: {stats['redis_memory']}")
        
        # Test search (GET)
        print("\n3. Search Test (GET):")
        query = "cooking an egg"
        results = client.search_get(query, k=3)
        print(f"   Query: '{results['query']}'")
        print(f"   Results: {results['total_results']}")
        print(f"   Search time: {results['search_time_ms']:.1f}ms")
        
        for i, result in enumerate(results['results'][:2], 1):
            print(f"   {i}. [{result['similarity']:.3f}] {result['doc_id']}")
            print(f"      Text: {result['doc_text'][:80]}...")
        
        # Test search (POST)
        print("\n4. Search Test (POST):")
        query = "machine learning algorithms"
        results = client.search_post(query, k=2)
        print(f"   Query: '{results['query']}'")
        print(f"   Results: {results['total_results']}")
        print(f"   Search time: {results['search_time_ms']:.1f}ms")
        
        # Test document retrieval
        if results['results']:
            print("\n5. Document Retrieval:")
            doc_id = results['results'][0]['doc_id']
            doc = client.get_document(doc_id)
            print(f"   Document ID: {doc['doc_id']}")
            print(f"   Length: {doc['doc_length']} chars")
            print(f"   Source: {doc['source']}")
            print(f"   Text: {doc['doc_text'][:100]}...")
            
            # Test similarity calculation
            print("\n6. Similarity Calculation:")
            similarity = client.get_similarity(doc_id, "artificial intelligence")
            print(f"   Document: {similarity['doc_id']}")
            print(f"   Query: '{similarity['query']}'")
            print(f"   Similarity: {similarity['similarity']:.3f}")
        
        print("\n✅ All tests passed!")
        
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API server. Make sure it's running!")
        print("   Start with: uv run vector_db/search_api.py --checkpoint <path> --config <config>")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP Error: {e}")
        print(f"   Response: {e.response.text}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

def demo_search_queries():
    """Demo different search queries."""
    client = SearchAPIClient()
    
    queries = [
        "cooking recipes",
        "machine learning",
        "weather forecast",
        "travel destinations",
        "health and fitness"
    ]
    
    print("\n🔍 Search Query Demo")
    print("=" * 50)
    
    for query in queries:
        try:
            results = client.search_get(query, k=2)
            print(f"\nQuery: '{query}'")
            print(f"Time: {results['search_time_ms']:.1f}ms")
            
            for i, result in enumerate(results['results'], 1):
                print(f"  {i}. [{result['similarity']:.3f}] {result['doc_id']}")
                print(f"     {result['doc_text'][:60]}...")
        
        except Exception as e:
            print(f"❌ Query '{query}' failed: {e}")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test the Search API")
    parser.add_argument("--demo", action="store_true", help="Run search query demo")
    parser.add_argument("--url", default="http://localhost:8000", help="API base URL")
    
    args = parser.parse_args()
    
    if args.demo:
        demo_search_queries()
    else:
        test_api() 