#!/usr/bin/env python3
"""
Search API Server

A FastAPI-based REST API server that provides search endpoints using the unified search system.
Combines Redis vector search with SQLite document retrieval.

Usage:
    uv run vector_db/search_api.py --checkpoint path/to/model.pt --config vector_db/redis_config.json
    
    Then visit: http://localhost:8000/docs for interactive API documentation
"""

import sys
import os
import json
import argparse
from typing import List, Dict, Optional
from datetime import datetime
import logging

# FastAPI imports
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add paths for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(script_dir)

from unified_search import UnifiedSearchSystem, UnifiedSearchResult

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global search system instance
search_system: Optional[UnifiedSearchSystem] = None
config_args = None  # Store command line arguments

# Pydantic models for API
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text", min_length=1, max_length=500)
    k: int = Field(default=10, description="Number of results to return", ge=1, le=100)
    include_text: bool = Field(default=True, description="Whether to include full document text")

class SearchResultResponse(BaseModel):
    doc_id: str
    doc_text: str
    similarity: float
    doc_length: int
    source: str
    metadata: Dict = {}

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResultResponse]
    total_results: int
    search_time_ms: float
    timestamp: str

class SystemStatsResponse(BaseModel):
    redis_documents: int
    sqlite_documents: int
    coverage_ratio: float
    redis_memory: str
    system_status: str

class HealthResponse(BaseModel):
    status: str
    redis_connected: bool
    sqlite_connected: bool
    model_loaded: bool
    uptime_seconds: float

# FastAPI app
app = FastAPI(
    title="Document Search API",
    description="Fast document search using vector similarity and full-text retrieval",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Startup time for uptime calculation
startup_time = datetime.now()

@app.on_event("startup")
async def startup_event():
    """Initialize search system on startup."""
    global search_system
    
    if config_args is None:
        logger.error("❌ Configuration not set")
        return
    
    logger.info("🚀 Initializing search system for API...")
    
    try:
        # Load Redis config
        with open(config_args.config, 'r') as f:
            config = json.load(f)
        
        redis_config = {
            "redis_url": config["redis"]["url"],
            "index_name": config["vector_store"]["index_name"],
            "vector_dim": config["vector_store"]["vector_dim"]
        }
        
        # Initialize search system
        search_system = UnifiedSearchSystem(
            redis_config=redis_config,
            sqlite_db_path=config_args.db,
            model_checkpoint=config_args.checkpoint
        )
        
        logger.info("✅ Search system initialized successfully!")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize search system: {e}")
        # Don't raise here, let the server start but endpoints will return 503

@app.get("/", response_class=HTMLResponse)
async def root():
    """Root endpoint with basic information."""
    return """
    <html>
        <head>
            <title>Document Search API</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { color: #2c3e50; }
                .endpoint { background: #f8f9fa; padding: 10px; margin: 10px 0; border-radius: 5px; }
                .method { color: #27ae60; font-weight: bold; }
            </style>
        </head>
        <body>
            <h1 class="header">🔍 Document Search API</h1>
            <p>Fast document search using vector similarity and full-text retrieval</p>
            
            <h2>Available Endpoints:</h2>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/health</strong> - System health check
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/stats</strong> - System statistics
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/search</strong> - Search documents (query parameter)
            </div>
            <div class="endpoint">
                <span class="method">POST</span> <strong>/search</strong> - Search documents (JSON body)
            </div>
            <div class="endpoint">
                <span class="method">GET</span> <strong>/document/{doc_id}</strong> - Get specific document
            </div>
            
            <h2>Interactive Documentation:</h2>
            <p>
                <a href="/docs">📚 Swagger UI</a> | 
                <a href="/redoc">📖 ReDoc</a>
            </p>
        </body>
    </html>
    """

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    uptime = (datetime.now() - startup_time).total_seconds()
    
    # Check system components
    redis_connected = True
    sqlite_connected = True
    model_loaded = search_system.model is not None
    
    try:
        search_system.vector_store.redis_client.ping()
    except:
        redis_connected = False
    
    try:
        search_system.document_store.conn.execute("SELECT 1")
    except:
        sqlite_connected = False
    
    status = "healthy" if (redis_connected and sqlite_connected and model_loaded) else "degraded"
    
    return HealthResponse(
        status=status,
        redis_connected=redis_connected,
        sqlite_connected=sqlite_connected,
        model_loaded=model_loaded,
        uptime_seconds=uptime
    )

@app.get("/stats", response_model=SystemStatsResponse)
async def get_system_stats():
    """Get system statistics."""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        stats = search_system.get_system_stats()
        
        return SystemStatsResponse(
            redis_documents=stats['redis'].get('document_count', 0),
            sqlite_documents=stats['sqlite'].get('total_documents', 0),
            coverage_ratio=stats['correlation']['coverage_ratio'],
            redis_memory=stats['redis'].get('redis_info', {}).get('used_memory', 'N/A'),
            system_status="operational"
        )
    except Exception as e:
        logger.error(f"Failed to get stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get system stats: {str(e)}")

@app.get("/search", response_model=SearchResponse)
async def search_get(
    q: str = Query(..., description="Search query", min_length=1, max_length=500),
    k: int = Query(default=10, description="Number of results", ge=1, le=100),
    include_text: bool = Query(default=True, description="Include full document text")
):
    """Search documents using GET request with query parameters."""
    return await search_documents(SearchRequest(query=q, k=k, include_text=include_text))

@app.post("/search", response_model=SearchResponse)
async def search_post(request: SearchRequest):
    """Search documents using POST request with JSON body."""
    return await search_documents(request)

async def search_documents(request: SearchRequest) -> SearchResponse:
    """Internal search function."""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    if not search_system.model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    start_time = datetime.now()
    
    try:
        results = search_system.search_documents(
            query=request.query,
            k=request.k,
            include_text=request.include_text
        )
        
        search_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # Convert results to response format
        result_responses = [
            SearchResultResponse(
                doc_id=result.doc_id,
                doc_text=result.doc_text,
                similarity=result.similarity,
                doc_length=result.doc_length,
                source=result.source,
                metadata=result.metadata or {}
            )
            for result in results
        ]
        
        return SearchResponse(
            query=request.query,
            results=result_responses,
            total_results=len(result_responses),
            search_time_ms=search_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")

@app.get("/document/{doc_id}")
async def get_document(doc_id: str):
    """Get a specific document by ID."""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        document = search_system.get_document_by_id(doc_id)
        
        if not document:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
        
        return {
            "doc_id": doc_id,
            "doc_text": document['doc_text'],
            "doc_length": document['doc_length'],
            "source": document['source'],
            "metadata": document.get('metadata', {}),
            "retrieved_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get document {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve document: {str(e)}")

@app.get("/similarity/{doc_id}")
async def get_similarity(
    doc_id: str,
    query: str = Query(..., description="Query to compare against document")
):
    """Get similarity score between a query and specific document."""
    if not search_system:
        raise HTTPException(status_code=503, detail="Search system not initialized")
    
    try:
        similarity = search_system.get_vector_similarity(doc_id, query)
        
        if similarity is None:
            raise HTTPException(status_code=404, detail=f"Document {doc_id} not found in vector store")
        
        return {
            "doc_id": doc_id,
            "query": query,
            "similarity": similarity,
            "calculated_at": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to calculate similarity for {doc_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to calculate similarity: {str(e)}")

def main():
    global config_args
    
    parser = argparse.ArgumentParser(description="Document Search API Server")
    parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")
    parser.add_argument("--config", default="./redis_config.json", help="Redis config file")
    parser.add_argument("--db", default="./documents.db", help="SQLite database path")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    
    config_args = parser.parse_args()
    
    # Start server (initialization happens in startup event)
    logger.info(f"🌐 Starting API server on {config_args.host}:{config_args.port}")
    logger.info(f"📚 API documentation: http://{config_args.host}:{config_args.port}/docs")
    
    uvicorn.run(
        app,  # Pass app directly instead of module string
        host=config_args.host,
        port=config_args.port,
        reload=config_args.reload,
        log_level="info"
    )

if __name__ == "__main__":
    main() 