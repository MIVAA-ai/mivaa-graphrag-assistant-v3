# api_server.py - FastAPI Wrapper for GraphRAG Document AI Platform

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
import asyncio
import uuid
import json
import time
from datetime import datetime
import logging
import os
import tempfile
from pathlib import Path

# Import your existing GraphRAG components
from GraphRAG_Document_AI_Platform import (
    load_config,
    load_qa_engine,
    get_correction_llm,
    get_enhanced_ocr_pipeline,
    init_neo4j_exporter,
    get_embedding_model,
    get_chroma_collection,
    get_requests_session,
    get_nlp_pipeline
)
from src.utils.processing_pipeline import start_ingestion_job_async, is_job_running
from src.utils.audit_db_manager import get_job_details, get_recent_jobs
import src.utils.audit_db_manager as audit_db

# Initialize FastAPI app
app = FastAPI(
    title="GraphRAG Document AI API",
    description="AI-powered document analysis and knowledge extraction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for web integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

# Global variables for cached resources
qa_engine = None
config = None
enhanced_ocr_pipeline = None
neo4j_exporter = None
embedding_model = None
chroma_collection = None
requests_session = None
nlp_pipeline = None


# Pydantic models for API requests/responses
class QuestionRequest(BaseModel):
    question: str = Field(..., description="Question to ask about the documents")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    max_results: Optional[int] = Field(5, description="Maximum number of results to return")


class QuestionResponse(BaseModel):
    answer: str = Field(..., description="Generated answer")
    confidence: Optional[float] = Field(None, description="Confidence score")
    sources: Optional[List[Dict]] = Field(None, description="Source documents")
    cypher_query: Optional[str] = Field(None, description="Generated Cypher query")
    linked_entities: Optional[Dict[str, str]] = Field(None, description="Linked entities")
    processing_time: float = Field(..., description="Processing time in seconds")
    generation_approach: Optional[str] = Field(None, description="Approach used for generation")


class DocumentUploadResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier for tracking")
    status: str = Field(..., description="Job status")
    message: str = Field(..., description="Status message")
    files_count: int = Field(..., description="Number of files submitted")


class JobStatusResponse(BaseModel):
    job_id: str = Field(..., description="Job identifier")
    status: str = Field(..., description="Current job status")
    progress: Dict[str, Any] = Field(..., description="Progress information")
    files_processed: int = Field(..., description="Number of files processed")
    total_files: int = Field(..., description="Total number of files")
    start_time: Optional[str] = Field(None, description="Job start time")
    end_time: Optional[str] = Field(None, description="Job end time")


class SystemStatusResponse(BaseModel):
    status: str = Field(..., description="Overall system status")
    components: Dict[str, str] = Field(..., description="Component status")
    knowledge_base_stats: Dict[str, Any] = Field(..., description="Knowledge base statistics")
    timestamp: str = Field(..., description="Status check timestamp")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    max_results: int = Field(10, description="Maximum results to return")
    filters: Optional[Dict[str, Any]] = Field(None, description="Search filters")
    search_type: str = Field("semantic", description="Type of search: semantic, exact, or hybrid")


class SearchResponse(BaseModel):
    query: str = Field(..., description="Original query")
    results: List[Dict[str, Any]] = Field(..., description="Search results")
    total_results: int = Field(..., description="Total number of results")
    processing_time: float = Field(..., description="Search processing time")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Error message")
    details: Optional[str] = Field(None, description="Error details")
    timestamp: str = Field(..., description="Error timestamp")


# Authentication functions
async def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Verify API key - implement your authentication logic here"""
    # TODO: Implement proper API key validation
    # For now, we'll use a simple check - replace with your auth system
    valid_api_keys = os.getenv("VALID_API_KEYS", "").split(",")

    if not valid_api_keys or credentials.credentials not in valid_api_keys:
        # For development, allow if no API keys are configured
        if not os.getenv("VALID_API_KEYS"):
            return {"user_id": "dev_user", "permissions": ["read", "write"]}
        raise HTTPException(
            status_code=401,
            detail="Invalid API key"
        )

    return {"user_id": "authenticated_user", "permissions": ["read", "write"]}


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize GraphRAG components on startup"""
    global qa_engine, config, enhanced_ocr_pipeline, neo4j_exporter
    global embedding_model, chroma_collection, requests_session, nlp_pipeline

    try:
        logging.info("Initializing GraphRAG API components...")

        # Load configuration
        config = load_config()
        if not config or not config.get('_CONFIG_VALID'):
            raise Exception("Invalid configuration")

        # Initialize audit database
        audit_db.initialize_database()

        # Initialize all components
        correction_llm = get_correction_llm(config)
        qa_engine = load_qa_engine(config, correction_llm)
        enhanced_ocr_pipeline = get_enhanced_ocr_pipeline(config)
        neo4j_exporter = init_neo4j_exporter(
            config.get('NEO4J_URI'),
            config.get('NEO4J_USER'),
            config.get('NEO4J_PASSWORD')
        )
        embedding_model = get_embedding_model(config.get('EMBEDDING_MODEL'))
        chroma_collection = get_chroma_collection(
            config.get('CHROMA_PERSIST_PATH'),
            config.get('COLLECTION_NAME'),
            config.get('EMBEDDING_MODEL')
        )
        requests_session = get_requests_session()
        nlp_pipeline = get_nlp_pipeline(config)

        logging.info("GraphRAG API initialized successfully")

    except Exception as e:
        logging.error(f"Failed to initialize GraphRAG API: {e}")
        # Don't raise here - let the app start but mark components as unavailable


# Dependency to check if system is ready
def get_qa_engine():
    if qa_engine is None or not qa_engine.is_ready():
        raise HTTPException(status_code=503, detail="GraphRAG system not ready")
    return qa_engine


def get_ocr_pipeline():
    if enhanced_ocr_pipeline is None:
        raise HTTPException(status_code=503, detail="OCR pipeline not available")
    return enhanced_ocr_pipeline


# Health check endpoint
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check"""
    try:
        components = {
            "api": "healthy",
            "qa_engine": "healthy" if qa_engine and qa_engine.is_ready() else "unhealthy",
            "ocr_pipeline": "healthy" if enhanced_ocr_pipeline else "unhealthy",
            "neo4j": "healthy" if neo4j_exporter else "unhealthy",
            "vector_db": "healthy" if chroma_collection else "unhealthy",
            "config": "healthy" if config and config.get('_CONFIG_VALID') else "unhealthy"
        }

        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"

        return {
            "status": overall_status,
            "components": components,
            "timestamp": datetime.now().isoformat(),
            "version": "1.0.0"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
        )


# System status endpoint
@app.get("/api/v1/system/status", response_model=SystemStatusResponse)
async def get_system_status(auth: dict = Depends(verify_api_key)):
    """Get detailed system status and statistics"""
    try:
        # Check component status
        components = {
            "qa_engine": "ready" if qa_engine and qa_engine.is_ready() else "not_ready",
            "ocr_pipeline": "ready" if enhanced_ocr_pipeline else "not_ready",
            "neo4j": "connected" if neo4j_exporter else "not_connected",
            "vector_db": "ready" if chroma_collection else "not_ready",
            "nlp_pipeline": "ready" if nlp_pipeline else "not_ready"
        }

        # Get knowledge base stats
        stats = {}
        if neo4j_exporter:
            try:
                stats = neo4j_exporter.get_graph_stats()
            except Exception as e:
                stats = {"error": f"Failed to get stats: {e}"}

        # Add vector database stats
        if chroma_collection:
            try:
                stats["vector_documents"] = chroma_collection.count()
            except Exception as e:
                stats["vector_error"] = str(e)

        return SystemStatusResponse(
            status="ready" if all(c in ["ready", "connected"] for c in components.values()) else "partial",
            components=components,
            knowledge_base_stats=stats,
            timestamp=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system status: {str(e)}")


# Question answering endpoint
@app.post("/api/v1/query", response_model=QuestionResponse)
async def ask_question(
        request: QuestionRequest,
        auth: dict = Depends(verify_api_key),
        engine: Any = Depends(get_qa_engine)
):
    """Ask a question about the processed documents"""
    try:
        start_time = time.time()

        # Process question with GraphRAG engine
        response_dict = engine.answer_question(request.question)

        processing_time = time.time() - start_time

        return QuestionResponse(
            answer=response_dict.get("answer", "No answer found"),
            confidence=response_dict.get("confidence"),
            sources=response_dict.get("sources", []),
            cypher_query=response_dict.get("cypher_query"),
            linked_entities=response_dict.get("linked_entities"),
            processing_time=processing_time,
            generation_approach=response_dict.get("generation_approach")
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing question: {str(e)}")


# Document upload endpoint
@app.post("/api/v1/documents/upload", response_model=DocumentUploadResponse)
async def upload_documents(
        background_tasks: BackgroundTasks,
        files: List[UploadFile] = File(...),
        use_cache: bool = True,
        save_ocr: bool = True,
        auth: dict = Depends(verify_api_key),
        ocr_pipeline: Any = Depends(get_ocr_pipeline)
):
    """Upload and process documents"""
    try:
        if not config:
            raise HTTPException(status_code=503, detail="System not configured")

        # Validate file types
        supported_types = ["application/pdf", "image/png", "image/jpeg", "text/plain"]
        for file in files:
            if file.content_type not in supported_types:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file.content_type}"
                )

        # Generate job ID
        job_id = str(uuid.uuid4())

        # Start background processing
        background_tasks.add_task(
            process_documents_background,
            job_id,
            files,
            use_cache,
            save_ocr
        )

        return DocumentUploadResponse(
            job_id=job_id,
            status="processing",
            message="Document processing started",
            files_count=len(files)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error uploading documents: {str(e)}")


# Background processing function
async def process_documents_background(
        job_id: str,
        files: List[UploadFile],
        use_cache: bool,
        save_ocr: bool
):
    """Process documents in the background"""
    try:
        # Save uploaded files temporarily
        temp_files = []
        for file in files:
            # Read file content
            content = await file.read()

            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
                temp_file.write(content)
                temp_files.append(temp_file.name)

        # Use existing processing pipeline
        actual_job_id = start_ingestion_job_async(
            uploaded_files=temp_files,
            config=config,
            use_cache=use_cache,
            enhanced_ocr_pipeline=enhanced_ocr_pipeline,
            neo4j_exporter=neo4j_exporter,
            embedding_model_resource=embedding_model,
            chroma_collection_resource=chroma_collection,
            requests_session_resource=requests_session,
            nlp_pipeline_resource=nlp_pipeline
        )

        logging.info(f"Background processing started for job {job_id} -> {actual_job_id}")

        # Cleanup temp files after processing
        for temp_file in temp_files:
            try:
                os.unlink(temp_file)
            except Exception as e:
                logging.warning(f"Failed to cleanup temp file {temp_file}: {e}")

    except Exception as e:
        logging.error(f"Background processing failed for job {job_id}: {e}")


# Job status endpoint
@app.get("/api/v1/jobs/{job_id}/status", response_model=JobStatusResponse)
async def get_job_status(job_id: str, auth: dict = Depends(verify_api_key)):
    """Get job processing status"""
    try:
        job_details = get_job_details(job_id)

        if not job_details:
            raise HTTPException(status_code=404, detail="Job not found")

        processed_files = job_details.get('processed_files', [])
        files_processed = len([f for f in processed_files if f['status'] != 'Processing'])

        # Calculate progress breakdown
        progress = {
            "completed": len([f for f in processed_files if f['status'] == 'Success']),
            "failed": len([f for f in processed_files if 'Failed' in f['status']]),
            "cached": len([f for f in processed_files if f['status'] == 'Cached']),
            "processing": len([f for f in processed_files if f['status'] == 'Processing'])
        }

        return JobStatusResponse(
            job_id=job_id,
            status=job_details.get('status', 'unknown'),
            progress=progress,
            files_processed=files_processed,
            total_files=job_details.get('total_files_in_job', 0),
            start_time=job_details.get('start_timestamp'),
            end_time=job_details.get('end_timestamp')
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting job status: {str(e)}")


# Recent jobs endpoint
@app.get("/api/v1/jobs/recent")
async def get_recent_jobs_endpoint(
        limit: int = 20,
        auth: dict = Depends(verify_api_key)
):
    """Get recent processing jobs"""
    try:
        jobs = get_recent_jobs(limit=limit)
        return {"jobs": jobs, "total": len(jobs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting recent jobs: {str(e)}")


# Search endpoint
@app.post("/api/v1/search", response_model=SearchResponse)
async def search_documents(
        request: SearchRequest,
        auth: dict = Depends(verify_api_key)
):
    """Search through processed documents"""
    try:
        start_time = time.time()

        if not chroma_collection:
            raise HTTPException(status_code=503, detail="Vector search not available")

        # Perform vector search
        results = chroma_collection.query(
            query_texts=[request.query],
            n_results=request.max_results,
            include=['documents', 'distances', 'metadatas'],
            where=request.filters
        )

        # Format results
        formatted_results = []
        if results and results.get('ids') and results['ids'][0]:
            for i, (doc, metadata, distance) in enumerate(zip(
                    results['documents'][0],
                    results['metadatas'][0],
                    results['distances'][0]
            )):
                formatted_results.append({
                    "rank": i + 1,
                    "content": doc,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,
                    "distance": distance
                })

        processing_time = time.time() - start_time

        return SearchResponse(
            query=request.query,
            results=formatted_results,
            total_results=len(formatted_results),
            processing_time=processing_time
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error searching documents: {str(e)}")


# Knowledge graph stats endpoint
@app.get("/api/v1/knowledge-graph/stats")
async def get_knowledge_graph_stats(auth: dict = Depends(verify_api_key)):
    """Get knowledge graph statistics"""
    try:
        if not neo4j_exporter:
            raise HTTPException(status_code=503, detail="Knowledge graph not available")

        stats = neo4j_exporter.get_graph_stats()
        return {"stats": stats, "timestamp": datetime.now().isoformat()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting graph stats: {str(e)}")


# Entity search endpoint
@app.get("/api/v1/entities/{entity_name}/relationships")
async def get_entity_relationships(
        entity_name: str,
        predicate_filter: Optional[str] = None,
        auth: dict = Depends(verify_api_key)
):
    """Get relationships for a specific entity"""
    try:
        if not neo4j_exporter:
            raise HTTPException(status_code=503, detail="Knowledge graph not available")

        relationships = neo4j_exporter.get_related_facts_with_context(
            entity_name, predicate_filter
        )

        return {
            "entity": entity_name,
            "relationships": relationships,
            "total": len(relationships),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting entity relationships: {str(e)}")


# WebSocket endpoint for real-time job updates
@app.websocket("/ws/jobs/{job_id}")
async def websocket_job_updates(websocket, job_id: str):
    """WebSocket endpoint for real-time job status updates"""
    await websocket.accept()
    try:
        while True:
            # Get current job status
            job_details = get_job_details(job_id)
            if job_details:
                await websocket.send_text(json.dumps({
                    "type": "job_update",
                    "job_id": job_id,
                    "status": job_details.get('status'),
                    "timestamp": datetime.now().isoformat(),
                    "progress": {
                        "processed": len([f for f in job_details.get('processed_files', [])
                                          if f['status'] != 'Processing']),
                        "total": job_details.get('total_files_in_job', 0)
                    }
                }))

            # Check if job is complete
            if job_details and job_details.get('status') in ['Completed', 'Failed', 'Completed with Errors']:
                await websocket.send_text(json.dumps({
                    "type": "job_complete",
                    "job_id": job_id,
                    "final_status": job_details.get('status'),
                    "timestamp": datetime.now().isoformat()
                }))
                break

            await asyncio.sleep(2)  # Update every 2 seconds
    except Exception as e:
        logging.error(f"WebSocket error for job {job_id}: {e}")
    finally:
        await websocket.close()


# Enhanced system info endpoint
@app.get("/api/v1/system/info")
async def get_system_info(auth: dict = Depends(verify_api_key)):
    """Get comprehensive system information"""
    try:
        info = {
            "api_version": "1.0.0",
            "graphrag_version": "2.1.0",
            "supported_file_types": ["PDF", "PNG", "JPEG", "TXT"],
            "ocr_providers": [],
            "features": {
                "universal_patterns": False,
                "entity_linking": True,
                "relationship_inference": True,
                "caching": True,
                "async_processing": True
            },
            "limits": {
                "max_file_size_mb": 50,
                "max_files_per_batch": 10,
                "max_query_length": 1000
            }
        }

        # Get OCR providers if available
        if enhanced_ocr_pipeline:
            info["ocr_providers"] = enhanced_ocr_pipeline.get_available_providers()

        # Check for enhanced features
        if qa_engine and hasattr(qa_engine, 'is_enhanced') and qa_engine.is_enhanced():
            info["features"]["universal_patterns"] = True
            industry_info = qa_engine.get_industry_info()
            info["detected_industry"] = industry_info.get('detected_industry', 'unknown')

        return info
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting system info: {str(e)}")


# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=datetime.now().isoformat()
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    logging.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            details=str(exc),
            timestamp=datetime.now().isoformat()
        ).dict()
    )


# Graceful shutdown
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup resources on shutdown"""
    try:
        if neo4j_exporter:
            neo4j_exporter.close()
        logging.info("GraphRAG API shutdown complete")
    except Exception as e:
        logging.error(f"Error during shutdown: {e}")


if __name__ == "__main__":
    import uvicorn

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(name)s:%(lineno)d | %(message)s'
    )

    # Run the API server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        reload=False,  # Set to True for development
        workers=1  # Single worker for shared resources
    )