"""
Embedding Generation Module

Uses sentence-transformers to generate embeddings and stores them
as PyTorch tensors on GPU for efficient similarity search.
"""

import logging
from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

# Import default constants
try:
    from config import (
        DEFAULT_BATCH_SIZE,
        SHOW_PROGRESS_BAR,
        OPTIMIZE_BATCH_SIZE,
        MAX_BATCH_SIZE,
        MIN_BATCH_SIZE,
        ENABLE_EMBEDDING_CACHE,
        EMBEDDING_CACHE_SIZE,
        USE_MIXED_PRECISION,
        ENABLE_PROFILING,
    )
except ImportError:
    DEFAULT_BATCH_SIZE = 64
    SHOW_PROGRESS_BAR = False
    OPTIMIZE_BATCH_SIZE = True
    MAX_BATCH_SIZE = 128
    MIN_BATCH_SIZE = 16
    ENABLE_EMBEDDING_CACHE = True
    EMBEDDING_CACHE_SIZE = 1000
    USE_MIXED_PRECISION = True
    ENABLE_PROFILING = False

# Import utility modules
try:
    from utils.gpu_monitor import get_gpu_monitor
    from utils.embedding_cache import get_embedding_cache
    from utils.profiler import get_profiler
except ImportError:
    # Fallback if utils not available
    get_gpu_monitor = None
    get_embedding_cache = None
    get_profiler = None

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Manages embedding model and GPU tensor storage.
    
    This class:
    - Loads a sentence-transformers model
    - Generates embeddings for text chunks
    - Stores embeddings as PyTorch tensors on GPU
    - Provides efficient batch processing
    """
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-mpnet-base-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the embedding model.
        
        Args:
            model_name: Hugging Face model identifier
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        # Auto-detect device if not specified
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading embedding model: {model_name} on {device}")
        
        # Load the sentence-transformers model
        self.model = SentenceTransformer(model_name, device=device)
        
        # Enable mixed precision if available and configured
        if USE_MIXED_PRECISION and device == "cuda":
            try:
                # Convert model to half precision (FP16)
                self.model = self.model.half()
                logger.info("Using FP16 mixed precision for embeddings")
            except Exception as e:
                logger.warning(f"Failed to enable mixed precision: {e}")
        
        # Get embedding dimension
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        # Initialize GPU monitor and cache
        if get_gpu_monitor:
            self.gpu_monitor = get_gpu_monitor(device)
        else:
            self.gpu_monitor = None
        
        if ENABLE_EMBEDDING_CACHE and get_embedding_cache:
            self.cache = get_embedding_cache(EMBEDDING_CACHE_SIZE, device)
        else:
            self.cache = None
        
        if get_profiler:
            self.profiler = get_profiler(ENABLE_PROFILING)
        else:
            self.profiler = None
        
        logger.info(f"Embedding model loaded. Dimension: {self.embedding_dim}")
    
    def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate embedding for a single text string with caching.
        
        Args:
            text: Input text to embed
            
        Returns:
            Embedding tensor on the specified device
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(text)
            if cached is not None:
                logger.debug("Cache hit for embedding")
                return cached
        
        # Profile if enabled
        profiler_context = self.profiler.profile("embed_text") if self.profiler else None
        if profiler_context:
            profiler_context.__enter__()
        
        try:
            # Generate embedding
            embedding = self.model.encode(
                text,
                convert_to_tensor=True,
                device=self.device,
            )
            
            # Cache result
            if self.cache:
                self.cache.put(text, embedding)
        finally:
            if profiler_context:
                profiler_context.__exit__(None, None, None)
        
        return embedding
    
    def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings for a batch of texts with dynamic batch sizing.
        
        This is more efficient than embedding one at a time.
        
        Args:
            texts: List of texts to embed
            batch_size: Number of texts to process in each batch (None = auto-tune)
            
        Returns:
            Tensor of shape (num_texts, embedding_dim) on GPU
        """
        if not texts:
            return torch.empty((0, self.embedding_dim), device=self.device)
        
        # Determine optimal batch size
        if batch_size is None:
            if OPTIMIZE_BATCH_SIZE and self.gpu_monitor:
                batch_size = self.gpu_monitor.get_optimal_batch_size(
                    base_batch_size=DEFAULT_BATCH_SIZE,
                    min_batch_size=MIN_BATCH_SIZE,
                    max_batch_size=MAX_BATCH_SIZE,
                )
            else:
                batch_size = DEFAULT_BATCH_SIZE
        
        logger.info(f"Embedding {len(texts)} texts in batches of {batch_size}")
        
        # Profile if enabled
        profiler_context = self.profiler.profile("embed_batch") if self.profiler else None
        if profiler_context:
            profiler_context.__enter__()
        
        try:
            # Use memory context for better memory management
            memory_context = self.gpu_monitor.memory_context() if self.gpu_monitor else None
            if memory_context:
                memory_context.__enter__()
            
            try:
                # Generate embeddings in batches
                embeddings = self.model.encode(
                    texts,
                    convert_to_tensor=True,
                    device=self.device,
                    batch_size=batch_size,
                    show_progress_bar=SHOW_PROGRESS_BAR,
                )
            finally:
                if memory_context:
                    memory_context.__exit__(None, None, None)
        finally:
            if profiler_context:
                profiler_context.__exit__(None, None, None)
        
        # Log memory stats
        if self.gpu_monitor:
            self.gpu_monitor.log_memory_stats("embed_batch")
        
        return embeddings
    
    def get_embedding_dim(self) -> int:
        """Get the dimension of embeddings produced by this model."""
        return self.embedding_dim


__all__ = ["EmbeddingModel"]

