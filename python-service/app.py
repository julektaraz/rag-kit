"""
FastAPI Server for RAG Pipeline

Provides HTTP API endpoints for PDF ingestion, querying, and model management.
"""

import sys
import os
import uuid
import logging
import tempfile
from pathlib import Path
from typing import Optional

import torch
import uvicorn
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

# Add the current directory to Python path to allow imports
sys.path.insert(0, str(Path(__file__).parent))

# Configure logging early
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# MIME type detection: python-magic is problematic on Windows, so we skip it
# and use header-based validation instead
HAS_MAGIC = False

from pipeline.rag_pipeline import RAGPipeline
from models.schemas import (
    ChatRequest,
    ChatResponse,
    TextIngestRequest,
    IngestResponse,
    StatsResponse,
    HealthResponse,
)
from config import MAX_FILE_SIZE

# Initialize FastAPI app
app = FastAPI(
    title="Local RAG Pipeline API",
    description="Local RAG system with PDF ingestion and querying",
    version="1.0.0",
)

# Configure CORS for Next.js frontend
# Get allowed origins from environment or use defaults
allowed_origins = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    expose_headers=["Content-Type"],
)

# Global RAG pipeline instance
rag_pipeline: Optional[RAGPipeline] = None


def validate_environment() -> None:
    """Validate required environment variables and configuration."""
    # Check Python version
    if sys.version_info < (3, 10):
        raise RuntimeError("Python 3.10 or higher is required")
    
    # Validate CUDA availability if needed
    if os.getenv("REQUIRE_GPU", "false").lower() == "true":
        if not torch.cuda.is_available():
            raise RuntimeError("GPU is required but not available")
    
    logger.info("Environment validation passed")


def get_pipeline() -> RAGPipeline:
    """Get or initialize the RAG pipeline."""
    global rag_pipeline
    
    if rag_pipeline is None:
        logger.info("Initializing RAG pipeline...")
        
        # Validate environment first
        validate_environment()
        
        # Get persistence configuration
        persistence_path = os.getenv("PERSISTENCE_PATH", "data/embeddings")
        auto_save = os.getenv("AUTO_SAVE", "true").lower() == "true"
        
        rag_pipeline = RAGPipeline(
            embedding_model_name=os.getenv(
                "EMBEDDING_MODEL",
                "sentence-transformers/all-mpnet-base-v2",
            ),
            llm_model_name=os.getenv(
                "LLM_MODEL",
                "mistralai/Mistral-7B-Instruct-v0.2",
            ),
            use_quantization=os.getenv("USE_QUANTIZATION", "true").lower() == "true",
            persistence_path=persistence_path if persistence_path else None,
            auto_save=auto_save,
        )
        logger.info("RAG pipeline initialized")
    
    return rag_pipeline


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "status": "ok",
        "message": "Local RAG Pipeline API",
        "version": "1.0.0",
    }


@app.get("/api/health", response_model=HealthResponse)
async def health():
    """Health check endpoint with GPU status."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        
        # Check GPU availability
        gpu_available = torch.cuda.is_available()
        gpu_name = None
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
        
        return HealthResponse(
            status="healthy",
            stats=StatsResponse(**stats),
            gpu_available=gpu_available,
            gpu_name=gpu_name,
        )
    except Exception as e:
        logger.error(f"Health check failed: {e}", exc_info=True)
        return JSONResponse(
            status_code=500,
            content=HealthResponse(
                status="unhealthy",
                error=str(e),
                gpu_available=torch.cuda.is_available(),
            ).model_dump(),
        )


def validate_pdf_file(file: UploadFile, content: bytes) -> None:
    """Validate PDF file by checking MIME type and size."""
    # Check file size
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )
    
    if len(content) == 0:
        raise HTTPException(
            status_code=400,
            detail="File is empty",
        )
    
    # Check filename extension
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported",
        )
    
    # Check MIME type using python-magic if available
    if HAS_MAGIC:
        try:
            mime_type = magic.from_buffer(content, mime=True)
            if mime_type != "application/pdf":
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid file type. Expected PDF, got {mime_type}",
                )
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"MIME type check failed, using header validation: {e}")
            # Fall through to header check
    
    # Fallback: Check PDF header
    if not content.startswith(b"%PDF"):
        raise HTTPException(
            status_code=400,
            detail="Invalid PDF file. File does not start with PDF header.",
        )


@app.post("/api/ingest/pdf", response_model=IngestResponse)
async def ingest_pdf(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
    sentences_per_chunk: int = Form(12, ge=1, le=100),
    overlap_sentences: int = Form(2, ge=0, le=50),
    extract_images: bool = Form(False),
):
    """
    Upload and ingest a PDF file.
    
    - **file**: PDF file to upload
    - **source**: Optional source identifier
    - **sentences_per_chunk**: Target sentences per chunk (1-100)
    - **overlap_sentences**: Sentence overlap between chunks (0-50)
    - **extract_images**: Whether to extract and OCR images
    """
    if overlap_sentences >= sentences_per_chunk:
        raise HTTPException(
            status_code=400,
            detail="overlap_sentences must be less than sentences_per_chunk",
        )
    
    tmp_path = None
    try:
        content = await file.read()
        validate_pdf_file(file, content)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        pipeline = get_pipeline()
        result = pipeline.ingest_pdf(
            pdf_path=tmp_path,
            source=source or file.filename or "uploaded_file",
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences,
            extract_images=extract_images,
        )
        
        return IngestResponse(
            success=True,
            doc_id=result["doc_id"],
            chunks=result["chunks"],
            pages=result["pages"],
            metadata=result["metadata"],
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"PDF ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to ingest PDF. Please check the file and try again.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")


@app.post("/api/ingest/image", response_model=IngestResponse)
async def ingest_image(
    file: UploadFile = File(...),
    source: Optional[str] = Form(None),
):
    """
    Upload and ingest a standalone image file (JPG, PNG, etc.).
    
    - **file**: Image file to upload (JPG, PNG, GIF, etc.)
    - **source**: Optional source identifier
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")
    
    allowed_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}
    file_ext = Path(file.filename).suffix.lower()
    
    if file_ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image format. Allowed: {', '.join(allowed_extensions)}",
        )
    
    # Check file size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024*1024):.0f}MB",
        )
    
    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    
    tmp_path = None
    try:
        # Determine file extension for temp file
        ext = file_ext or ".jpg"
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name
        
        pipeline = get_pipeline()
        result = pipeline.ingest_image(
            image_path=tmp_path,
            source=source or file.filename or "uploaded_image",
        )
        
        return IngestResponse(
            success=True,
            doc_id=result["doc_id"],
            chunks=result["chunks"],
            pages=None,
            metadata=result.get("image_metadata"),
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to ingest image. Please check the file and try again.",
        )
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Failed to delete temporary file {tmp_path}: {e}")


@app.post("/api/ingest/text", response_model=IngestResponse)
async def ingest_text(request: TextIngestRequest):
    """
    Ingest plain text content.
    
    Text will be chunked and embedded for retrieval.
    """
    try:
        pipeline = get_pipeline()
        
        # Create temporary file for text ingestion
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=".txt", encoding='utf-8') as tmp_file:
            tmp_file.write(request.text)
            tmp_path = tmp_file.name
        
        try:
            # Use PDF ingestion path (it will extract text)
            # For now, we'll chunk directly
            from ingestion import clean_text, chunk_by_sentences
            
            cleaned = clean_text(request.text)
            chunks = chunk_by_sentences(
                cleaned,
                sentences_per_chunk=request.sentences_per_chunk,
                overlap_sentences=request.overlap_sentences,
            )
            
            if not chunks:
                raise HTTPException(status_code=400, detail="No chunks created from text")
            
            # Generate embeddings
            embeddings = pipeline.embedding_model.embed_batch(chunks)
            
            # Prepare metadata
            doc_id = str(uuid.uuid4())
            chunk_metadata = []
            for idx, chunk in enumerate(chunks):
                chunk_metadata.append({
                    "text": chunk,
                    "source": request.source or "text_input",
                    "doc_id": doc_id,
                    "chunk_index": idx,
                    "total_chunks": len(chunks),
                })
            
            # Store embeddings
            pipeline.tensor_store.add_embeddings(embeddings, chunk_metadata)
            
            return IngestResponse(
                success=True,
                doc_id=doc_id,
                chunks=len(chunks),
                pages=None,
                metadata=None,
            )
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Text ingestion failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to ingest text. Please check the input and try again.",
        )


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Query the RAG system.
    
    Ask a question and get an answer based on retrieved context.
    
    - **message**: User's question (1-10000 characters)
    - **top_k**: Number of context chunks to retrieve (1-20)
    - **min_similarity**: Minimum similarity score threshold (0.0-1.0)
    - **use_hybrid_search**: Use hybrid (keyword + semantic) search
    - **use_reranking**: Re-rank results with cross-encoder
    """
    try:
        pipeline = get_pipeline()
        
        # Import defaults for hybrid search and re-ranking
        try:
            from config import DEFAULT_HYBRID_SEARCH, DEFAULT_USE_RERANKING
        except ImportError:
            DEFAULT_HYBRID_SEARCH = True
            DEFAULT_USE_RERANKING = True
        
        result = pipeline.query(
            question=request.message,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            use_hybrid_search=request.use_hybrid_search if request.use_hybrid_search is not None else DEFAULT_HYBRID_SEARCH,
            use_reranking=request.use_reranking if request.use_reranking is not None else DEFAULT_USE_RERANKING,
        )
        
        return ChatResponse(
            answer=result["answer"],
            contexts=result["contexts"],
            sources=result["sources"],
        )
        
    except Exception as e:
        logger.error(f"Chat query failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Please try again.",
        )


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """
    Stream LLM response tokens as they're generated.
    
    Returns Server-Sent Events (SSE) stream.
    """
    try:
        pipeline = get_pipeline()
        
        # Import defaults
        try:
            from config import DEFAULT_HYBRID_SEARCH, DEFAULT_USE_RERANKING, DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE
        except ImportError:
            DEFAULT_HYBRID_SEARCH = True
            DEFAULT_USE_RERANKING = True
            DEFAULT_MAX_NEW_TOKENS = 512
            DEFAULT_TEMPERATURE = 0.7
        
        # Get contexts (same as regular chat)
        result = pipeline.query(
            question=request.message,
            top_k=request.top_k,
            min_similarity=request.min_similarity,
            use_hybrid_search=request.use_hybrid_search if request.use_hybrid_search is not None else DEFAULT_HYBRID_SEARCH,
            use_reranking=request.use_reranking if request.use_reranking is not None else DEFAULT_USE_RERANKING,
        )
        
        # Format prompt
        context_texts = [ctx["text"] for ctx in result["contexts"]]
        prompt = pipeline.llm_model.format_prompt_with_context(
            request.message,
            context_texts,
        )
        
        # Stream generation
        async def generate_stream():
            try:
                # Get the pipeline from the model
                llm_pipeline = pipeline.llm_model.pipeline
                
                # Generate with streaming
                for output in llm_pipeline(
                    prompt,
                    max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    return_full_text=False,
                    pad_token_id=pipeline.llm_model.tokenizer.eos_token_id,
                    do_sample=True,
                    streamer=None,  # Would need to implement custom streamer
                ):
                    # Extract token
                    generated_text = output[0]["generated_text"]
                    # For now, send full text (proper streaming would need tokenizer streamer)
                    yield f"data: {generated_text}\n\n"
                
                # Send contexts at the end
                yield f"data: [CONTEXTS]{result['contexts']}\n\n"
                yield f"data: [SOURCES]{result['sources']}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.error(f"Streaming failed: {e}", exc_info=True)
                yield f"data: [ERROR]{str(e)}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            }
        )
        
    except Exception as e:
        logger.error(f"Chat stream failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to process query. Please try again.",
        )


@app.get("/api/stats/gpu")
async def get_gpu_stats():
    """
    Get GPU memory and performance statistics.
    
    Returns real-time GPU utilization, memory usage, and performance metrics.
    """
    try:
        from utils.gpu_monitor import get_gpu_monitor
        from utils.profiler import get_profiler
        from utils.embedding_cache import get_embedding_cache
        
        gpu_monitor = get_gpu_monitor()
        profiler = get_profiler()
        
        # Get GPU memory info
        memory_info = gpu_monitor.get_memory_info()
        
        # Get performance stats
        perf_stats = profiler.get_stats()
        
        # Get cache stats if available
        cache_stats = None
        try:
            cache = get_embedding_cache()
            cache_stats = cache.get_stats()
        except Exception:
            pass
        
        return {
            "memory": memory_info,
            "performance": perf_stats,
            "cache": cache_stats,
        }
    except ImportError:
        return {
            "error": "Performance monitoring not available",
            "memory": None,
            "performance": None,
            "cache": None,
        }
    except Exception as e:
        logger.error(f"Failed to get GPU stats: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to retrieve GPU statistics")


@app.get("/api/stats", response_model=StatsResponse)
async def get_stats():
    """Get statistics about the stored data."""
    try:
        pipeline = get_pipeline()
        stats = pipeline.get_stats()
        return StatsResponse(**stats)
    except Exception as e:
        logger.error(f"Stats retrieval failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="Failed to get stats",
        )


@app.post("/api/persistence/save")
async def save_persistence():
    """Manually save embeddings to disk."""
    try:
        pipeline = get_pipeline()
        pipeline.tensor_store.save()
        return {"success": True, "message": "Embeddings saved successfully"}
    except Exception as e:
        logger.error(f"Save failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save embeddings: {str(e)}",
        )


@app.post("/api/persistence/load")
async def load_persistence():
    """Manually load embeddings from disk."""
    try:
        pipeline = get_pipeline()
        pipeline.tensor_store.load()
        return {"success": True, "message": "Embeddings loaded successfully"}
    except Exception as e:
        logger.error(f"Load failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to load embeddings: {str(e)}",
        )


@app.post("/api/persistence/clear")
async def clear_persistence():
    """Clear all stored embeddings and metadata."""
    try:
        pipeline = get_pipeline()
        pipeline.tensor_store.clear()
        return {"success": True, "message": "Embeddings cleared successfully"}
    except Exception as e:
        logger.error(f"Clear failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to clear embeddings: {str(e)}",
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
