#!/usr/bin/env python3
"""
Example Usage of Redis Vector Store

This script shows how to use the Redis vector store with Redis Cloud
for caching and searching document embeddings.
"""

import os
import sys

# Add paths for imports
sys.path.append('../training')
sys.path.append('.')

def example_redis_cloud_config():
    """
    Example Redis Cloud configuration.
    
    Update these values with your Redis Cloud details from the screenshot:
    - Get your Redis URL from the Redis Cloud Console
    - It should look like: redis://default:password@host:port
    """
    
    # REPLACE THESE WITH YOUR REDIS CLOUD DETAILS
    redis_config = {
        # Option 1: Use Redis URL (recommended for Redis Cloud)
        "redis_url": "redis://default:YOUR_PASSWORD@YOUR_HOST:YOUR_PORT",
        
        # Option 2: Or use individual parameters
        # "redis_host": "your-redis-host.redis.cloud",
        # "redis_port": 12345,
        # "redis_password": "your-password",
        # "redis_db": 0,
        
        "index_name": "msmarco_docs",
        "vector_dim": 256
    }
    
    return redis_config

def quick_test_with_small_dataset():
    """
    Quick test with a small dataset to verify everything works.
    """
    print("🚀 Quick Test with Small Dataset")
    print("=" * 50)
    
    # Configuration
    redis_config = example_redis_cloud_config()
    checkpoint_path = "../checkpoints/two_tower_best_epoch2_20250619_094155.pt"
    data_path = "../data"
    
    # Test command
    test_command = f"""
cd .ben/vector_db

python cache_documents.py \\
    --checkpoint {checkpoint_path} \\
    --data {data_path} \\
    --redis-url "{redis_config['redis_url']}" \\
    --index-name {redis_config['index_name']} \\
    --max-docs 100 \\
    --clear \\
    --device cpu
"""
    
    print("📋 Command to run:")
    print(test_command)
    
    print("\n💡 Steps to get your Redis Cloud URL:")
    print("1. Go to your Redis Cloud Console")
    print("2. Click on your database (database-MC366FOJ)")
    print("3. Go to 'Configuration' tab")
    print("4. Look for 'Public endpoint' or 'Connect' button")
    print("5. Copy the Redis URL (should include password)")
    print("6. Update the redis_config in this file")
    
    return test_command

def example_search_only():
    """
    Example of searching cached documents (after they're already cached).
    """
    print("🔍 Search Only Example")
    print("=" * 50)
    
    redis_config = example_redis_cloud_config()
    checkpoint_path = "../checkpoints/two_tower_best_epoch2_20250619_094155.pt"
    
    search_command = f"""
cd .ben/vector_db

python cache_documents.py \\
    --checkpoint {checkpoint_path} \\
    --redis-url "{redis_config['redis_url']}" \\
    --index-name {redis_config['index_name']} \\
    --test-only \\
    --device cpu
"""
    
    print("📋 Command to run:")
    print(search_command)
    
    return search_command

def production_caching_example():
    """
    Example for caching larger dataset in production.
    """
    print("🏭 Production Caching Example")
    print("=" * 50)
    
    redis_config = example_redis_cloud_config()
    checkpoint_path = "../checkpoints/two_tower_best_epoch2_20250619_094155.pt"
    data_path = "../data"
    
    prod_command = f"""
cd .ben/vector_db

python cache_documents.py \\
    --checkpoint {checkpoint_path} \\
    --data {data_path} \\
    --redis-url "{redis_config['redis_url']}" \\
    --index-name {redis_config['index_name']} \\
    --batch-size 200 \\
    --encoding-batch-size 64 \\
    --device cpu \\
    --no-test
"""
    
    print("📋 Command to run:")
    print(prod_command)
    
    print("\n⚠️ This will cache ALL documents - may take time and Redis memory!")
    
    return prod_command

def main():
    """Main function to show examples."""
    print("🎯 Redis Vector Store Examples")
    print("=" * 60)
    
    print("\n1️⃣ QUICK TEST (recommended first step):")
    quick_test_with_small_dataset()
    
    print("\n\n2️⃣ SEARCH ONLY (after documents are cached):")
    example_search_only()
    
    print("\n\n3️⃣ PRODUCTION CACHING (full dataset):")
    production_caching_example()
    
    print("\n" + "=" * 60)
    print("📚 Next Steps:")
    print("1. Update Redis URL in example_redis_cloud_config()")
    print("2. Run the quick test command")
    print("3. If successful, run production caching")
    print("4. Use search-only for testing queries")

if __name__ == "__main__":
    main() 