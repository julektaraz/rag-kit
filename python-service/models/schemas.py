"""
Pydantic models for API request/response validation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional, List, Dict, Any

# Import constants from config
try:
    from config import MAX_FILE_SIZE, MAX_TEXT_LENGTH, DEFAULT_MIN_SIMILARITY, DEFAULT_TOP_K
except ImportError:
    # Fallback if config not available
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_TEXT_LENGTH = 10 * 1024 * 1024  # 10MB
    DEFAULT_MIN_SIMILARITY = 0.3
    DEFAULT_TOP_K = 5


class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str = Field(..., min_length=1, max_length=10000, description="User's question")
    top_k: int = Field(default=DEFAULT_TOP_K, ge=1, le=20, description="Number of context chunks to retrieve")
    min_similarity: float = Field(default=DEFAULT_MIN_SIMILARITY, ge=0.0, le=1.0, description="Minimum similarity score threshold")
    use_hybrid_search: Optional[bool] = Field(default=None, description="Use hybrid (keyword + semantic) search")
    use_reranking: Optional[bool] = Field(default=None, description="Re-rank results with cross-encoder")

    @field_validator("message")
    @classmethod
    def validate_message(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Message cannot be empty")
        return v.strip()


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    answer: str
    contexts: List[Dict[str, Any]]
    sources: List[str]


class TextIngestRequest(BaseModel):
    """Request model for text ingestion."""
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="Text content to ingest")
    source: Optional[str] = Field(default=None, max_length=255, description="Optional source identifier")
    sentences_per_chunk: int = Field(default=12, ge=1, le=100, description="Target sentences per chunk")
    overlap_sentences: int = Field(default=2, ge=0, le=50, description="Sentence overlap between chunks")

    @field_validator("text")
    @classmethod
    def validate_text(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Text cannot be empty")
        return v.strip()

    @field_validator("overlap_sentences")
    @classmethod
    def validate_overlap(cls, v: int, info) -> int:
        sentences_per_chunk = info.data.get("sentences_per_chunk", 12)
        if v >= sentences_per_chunk:
            raise ValueError("overlap_sentences must be less than sentences_per_chunk")
        return v


class IngestResponse(BaseModel):
    """Response model for ingestion endpoints."""
    success: bool
    doc_id: str
    chunks: int
    pages: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None


class StatsResponse(BaseModel):
    """Response model for stats endpoint."""
    total_chunks: int
    embedding_dim: int
    device: str


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    stats: Optional[StatsResponse] = None
    gpu_available: Optional[bool] = None
    gpu_name: Optional[str] = None
    error: Optional[str] = None

