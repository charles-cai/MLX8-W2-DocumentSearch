# Document Search API 🔍

A FastAPI-based REST API server that provides fast document search using vector similarity and full-text retrieval.

## Features

- **Fast Vector Search**: Uses Redis with native vector search capabilities
- **Full Document Retrieval**: SQLite for complete document text and metadata
- **RESTful API**: Both GET and POST endpoints for flexibility
- **Interactive Documentation**: Auto-generated Swagger UI and ReDoc
- **Health Monitoring**: System health and statistics endpoints
- **Similarity Scoring**: Calculate similarity between queries and specific documents

## Quick Start

### 1. Start the API Server

```bash
# From .ben directory
uv run vector_db/search_api.py \
  --checkpoint checkpoints/two_tower_best_true-sweep-1_20250620_084640.pt \
  --config vector_db/redis_config.json \
  --db data/documents.db
```

### 2. Access the API

- **API Base**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 3. Test the API

```bash
# Run comprehensive tests
uv run vector_db/test_api.py

# Run search demo
uv run vector_db/test_api.py --demo
```

## API Endpoints

### 🏠 Root
- **GET** `/` - Welcome page with endpoint overview

### 🩺 Health & Stats
- **GET** `/health` - System health check
- **GET** `/stats` - System statistics (Redis/SQLite document counts, memory usage)

### 🔍 Search
- **GET** `/search?q=query&k=10&include_text=true` - Search with query parameters
- **POST** `/search` - Search with JSON body

### 📄 Documents
- **GET** `/document/{doc_id}` - Get specific document by ID
- **GET** `/similarity/{doc_id}?query=text` - Calculate similarity score

## Example Usage

### Search with cURL

```bash
# GET request
curl "http://localhost:8000/search?q=cooking%20recipes&k=5"

# POST request
curl -X POST "http://localhost:8000/search" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "k": 3, "include_text": true}'
```

### Search with Python

```python
import requests

# Search for documents
response = requests.get("http://localhost:8000/search", params={
    "q": "cooking an egg",
    "k": 5
})

results = response.json()
print(f"Found {results['total_results']} results in {results['search_time_ms']:.1f}ms")

for result in results['results']:
    print(f"[{result['similarity']:.3f}] {result['doc_id']}")
    print(f"  {result['doc_text'][:100]}...")
```

### Get Document Details

```python
# Get specific document
doc_response = requests.get("http://localhost:8000/document/12345_1")
document = doc_response.json()

print(f"Document: {document['doc_id']}")
print(f"Length: {document['doc_length']} characters")
print(f"Source: {document['source']}")
```

## API Response Format

### Search Response
```json
{
  "query": "cooking recipes",
  "results": [
    {
      "doc_id": "12345_1",
      "doc_text": "How to cook the perfect egg...",
      "similarity": 0.924,
      "doc_length": 1250,
      "source": "raw_data",
      "metadata": {}
    }
  ],
  "total_results": 5,
  "search_time_ms": 15.2,
  "timestamp": "2025-01-20T12:00:00"
}
```

### Health Response
```json
{
  "status": "healthy",
  "redis_connected": true,
  "sqlite_connected": true,
  "model_loaded": true,
  "uptime_seconds": 3600.5
}
```

## Configuration

### Command Line Options

```bash
uv run vector_db/search_api.py \
  --checkpoint PATH        # Required: Path to model checkpoint
  --config PATH           # Redis config file (default: ./redis_config.json)
  --db PATH              # SQLite database (default: ./documents.db)
  --host HOST            # Host to bind (default: 0.0.0.0)
  --port PORT            # Port to bind (default: 8000)
  --reload               # Enable auto-reload for development
```

### Redis Configuration

The API uses the same Redis configuration as the unified search system:

```json
{
  "redis": {
    "url": "redis://localhost:6379",
    "host": "localhost",
    "port": 6379,
    "username": null,
    "password": null,
    "db": 0
  },
  "vector_store": {
    "index_name": "msmarco_docs_local",
    "vector_dim": 128
  }
}
```

## Performance

- **Vector Search**: Native Redis vector search with COSINE similarity
- **Response Times**: Typically 10-50ms for searches with cached documents
- **Scalability**: Handles concurrent requests efficiently
- **Memory Usage**: Minimal - only embeddings stored in Redis

## Development

### Running in Development Mode

```bash
# Enable auto-reload for development
uv run vector_db/search_api.py \
  --checkpoint checkpoints/model.pt \
  --config vector_db/redis_config.json \
  --reload
```

### Testing

```bash
# Test basic functionality
uv run vector_db/test_api.py

# Test with different queries
uv run vector_db/test_api.py --demo

# Test against different server
uv run vector_db/test_api.py --url http://localhost:8080
```

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Redis         │    │   SQLite        │
│   Server        │    │   Vector Store  │    │   Document Store│
│                 │    │                 │    │                 │
│ • REST API      │───▶│ • Embeddings    │    │ • Full Text     │
│ • Validation    │    │ • Vector Search │    │ • Metadata      │
│ • Documentation│    │ • Fast Lookup   │    │ • Structured    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │   Two-Tower     │
                    │   Model         │
                    │                 │
                    │ • Query Encoder │
                    │ • Doc Encoder   │
                    │ • Similarity    │
                    └─────────────────┘
```

## Error Handling

The API provides clear error responses:

- **400 Bad Request**: Invalid query parameters or JSON
- **404 Not Found**: Document not found
- **500 Internal Server Error**: System errors with details
- **503 Service Unavailable**: System not initialized or components down

## Security Considerations

- **Input Validation**: Query length and parameter validation
- **Rate Limiting**: Consider adding rate limiting for production
- **Authentication**: Add authentication for production deployments
- **CORS**: Configure CORS headers if serving web clients

---

🎯 **Ready to search!** Start the server and visit http://localhost:8000/docs for interactive API exploration. 