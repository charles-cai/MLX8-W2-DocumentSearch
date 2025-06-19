# Redis Vector Store for Document Search

This system provides fast vector similarity search for MS MARCO documents using Redis Cloud and your trained two-tower model.

## 🏗️ Architecture

```
MS MARCO Documents → Two-Tower Model → Document Embeddings → Redis Vector Store → Search API
```

**Components:**
- `redis_vector_store.py`: Core Redis vector store implementation
- `cache_documents.py`: Script to cache documents with embeddings
- `example_usage.py`: Usage examples and configuration templates

## 🚀 Quick Start

### 1. Get Your Redis Cloud Connection

From your Redis Cloud Console (screenshot):
1. Click on your database: `database-MC366FOJ`
2. Go to **Configuration** tab
3. Find **Public endpoint** or click **Connect**
4. Copy the Redis URL (format: `redis://default:password@host:port`)

### 2. Update Configuration

```python
# In example_usage.py, update:
redis_config = {
    "redis_url": "redis://default:YOUR_ACTUAL_PASSWORD@YOUR_ACTUAL_HOST:PORT",
    "index_name": "msmarco_docs",
    "vector_dim": 256
}
```

### 3. Quick Test (Recommended First Step)

```bash
cd .ben/vector_db

python cache_documents.py \
    --checkpoint ../checkpoints/two_tower_best_epoch2_20250619_094155.pt \
    --data ../data \
    --redis-url "redis://default:YOUR_PASSWORD@YOUR_HOST:PORT" \
    --index-name msmarco_docs \
    --max-docs 100 \
    --clear \
    --device cpu
```

This will:
- ✅ Load 100 documents (small test)
- ✅ Encode them with your trained model
- ✅ Cache embeddings in Redis
- ✅ Run test queries
- ✅ Clear existing data first

### 4. Production Caching (Full Dataset)

```bash
python cache_documents.py \
    --checkpoint ../checkpoints/two_tower_best_epoch2_20250619_094155.pt \
    --data ../data \
    --redis-url "redis://default:YOUR_PASSWORD@YOUR_HOST:PORT" \
    --index-name msmarco_docs \
    --batch-size 200 \
    --encoding-batch-size 64 \
    --device cpu \
    --no-test
```

## 📊 What Gets Cached

For each document, the system stores:

```json
{
    "doc_id": "unique_document_id",
    "doc_text": "Document content...",
    "embedding": [0.1, 0.2, 0.3, ...],  // 256-dim vector
    "metadata": {
        "source": "positive/negative", 
        "cached_at": "2024-12-01T12:00:00",
        "doc_length": 150
    }
}
```

## 🔍 Search Usage

### Search Cached Documents

```bash
python cache_documents.py \
    --checkpoint ../checkpoints/two_tower_best_epoch2_20250619_094155.pt \
    --redis-url "redis://default:YOUR_PASSWORD@YOUR_HOST:PORT" \
    --index-name msmarco_docs \
    --test-only \
    --device cpu
```

### Programmatic Search

```python
from redis_vector_store import RedisVectorStore, load_model_from_checkpoint
import torch

# Load model and connect to Redis
model = load_model_from_checkpoint("path/to/checkpoint.pt")
store = RedisVectorStore(redis_url="your_redis_url")

# Search for similar documents
query = "machine learning algorithms"
with torch.no_grad():
    query_tokens = model.tokenize(query).unsqueeze(0)
    query_embedding = model.encode_query(query_tokens).cpu().numpy()[0]

results = store.search_similar_documents(query_embedding, k=10)

for result in results:
    print(f"[{result.similarity:.3f}] {result.doc_text[:100]}...")
```

## 📈 Performance & Scaling

### Memory Usage
- **Each document**: ~1KB (256 floats + text + metadata)
- **1K documents**: ~1MB
- **10K documents**: ~10MB  
- **100K documents**: ~100MB

### Search Speed
- **Vector search**: Sub-millisecond with Redis
- **Index size**: O(n) documents
- **Query complexity**: O(log n) with proper indexing

### Batch Processing
```bash
# Optimize for your Redis Cloud instance
--batch-size 200          # Redis batch operations
--encoding-batch-size 64  # Model inference batch
```

## 🛠️ Configuration Options

### Cache Documents Script

For detailed help with examples and performance tips:
```bash
python cache_documents.py --help
```

**Quick Reference:**
```bash
python cache_documents.py [OPTIONS]

Required:
  --checkpoint PATH        Path to trained model checkpoint

Redis Connection:
  --redis-url URL         Redis URL (recommended for Redis Cloud)
  --redis-host HOST       Redis host (alternative)
  --redis-port PORT       Redis port
  --redis-password PASS   Redis password
  --index-name NAME       Vector index name (default: msmarco_docs)

Processing:
  --data PATH             MS MARCO data directory (default: ../data)
  --batch-size N          Redis batch size (default: 100)
  --encoding-batch-size N Model batch size (default: 32)
  --max-docs N            Limit documents for testing
  --device DEVICE         cpu/cuda/mps (default: cpu)

Actions:
  --clear                 Clear existing documents first
  --test-only            Only run test queries (don't cache)
  --no-test              Don't run test queries after caching
```

💡 **Pro Tip**: The `--help` flag provides comprehensive guidance including:
- Usage examples for different scenarios
- Performance optimization tips
- Expected performance metrics
- Configuration priority explanations

## 🔧 Troubleshooting

### Common Issues

**1. Redis Connection Failed**
```
❌ Failed to connect to Redis: Connection refused
```
- ✅ Check your Redis URL format
- ✅ Verify Redis Cloud database is running (57% ready → 100%)
- ✅ Check firewall/network connectivity

**2. Index Creation Failed**
```
❌ Failed to create vector index: Module not loaded
```
- ✅ Ensure your Redis Cloud has **RediSearch** module enabled
- ✅ Redis Cloud should include this by default

**3. Model Loading Failed**
```
❌ Checkpoint file not found
```
- ✅ Check checkpoint path: `ls .ben/checkpoints/`
- ✅ Use absolute path if needed

**4. Out of Memory**
```
❌ Redis out of memory
```
- ✅ Use `--max-docs` to limit document count
- ✅ Increase Redis Cloud memory limit
- ✅ Use smaller batch sizes

### Performance Tuning

**For Large Datasets (>10K docs):**
```bash
--batch-size 500 \
--encoding-batch-size 128 \
--device cuda  # if available
```

**For Small Memory:**
```bash
--batch-size 50 \
--encoding-batch-size 16 \
--max-docs 1000
```

## 📊 Monitoring

### Get Statistics
```python
store = RedisVectorStore(redis_url="your_url")
stats = store.get_stats()

print(f"Documents: {stats['document_count']}")
print(f"Memory: {stats['redis_info']['used_memory']}")
```

### Clear All Data
```python
store.clear_all_documents()  # ⚠️ Deletes everything!
```

## 🎯 Next Steps

1. **Test with small dataset** (100 docs)
2. **Verify search quality** with your queries
3. **Scale to full dataset** if satisfied
4. **Integrate with your application**
5. **Monitor Redis memory usage**

## 🔗 Integration

This vector store is designed to work with:
- ✅ Your trained two-tower models
- ✅ MS MARCO evaluation pipeline  
- ✅ Real-time search applications
- ✅ Batch document processing
- ✅ A/B testing different models

The cached embeddings can be used for fast inference without re-encoding documents every time! 