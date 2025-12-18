"""
Configuration constants for the RAG pipeline.

Centralizes default values and configuration parameters.
"""

# LLM Generation Parameters
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.9
DEFAULT_DO_SAMPLE = True

# Embedding Parameters
DEFAULT_BATCH_SIZE = 64  # Optimized for RTX 4080 (was 32)
OPTIMIZE_BATCH_SIZE = True  # Auto-tune batch size based on GPU memory
MAX_BATCH_SIZE = 128  # Maximum batch size for RTX 4080
MIN_BATCH_SIZE = 16  # Minimum batch size

# Chunking Parameters
DEFAULT_SENTENCES_PER_CHUNK = 12
DEFAULT_OVERLAP_SENTENCES = 2
DEFAULT_MIN_CHUNK_SIZE = 50

# Query Parameters
DEFAULT_TOP_K = 5
DEFAULT_MIN_SIMILARITY = 0.3
DEFAULT_HYBRID_SEARCH = True  # Enable hybrid (keyword + semantic) search
DEFAULT_KEYWORD_WEIGHT = 0.3  # Weight for BM25 keyword search (0-1)
DEFAULT_SEMANTIC_WEIGHT = 0.7  # Weight for semantic search (0-1)
DEFAULT_USE_RERANKING = True  # Enable re-ranking with cross-encoder
DEFAULT_RERANK_TOP_K = 20  # Number of candidates to re-rank

# Embedding Display
SHOW_PROGRESS_BAR = False  # Set to True for verbose embedding progress

# Performance Optimization Parameters
ENABLE_MEMORY_MONITORING = True  # Track GPU memory usage
CLEAR_CUDA_CACHE_INTERVAL = 10  # Clear cache every N operations (0 = disabled)
MEMORY_WARNING_THRESHOLD = 85  # Warn if GPU memory > 85%

# Embedding Cache
ENABLE_EMBEDDING_CACHE = True  # Cache query embeddings
EMBEDDING_CACHE_SIZE = 1000  # Max cached embeddings

# Mixed Precision
USE_MIXED_PRECISION = True  # Use FP16 for embeddings (faster, less memory)

# Performance Profiling
ENABLE_PROFILING = False  # Set to True for debugging/optimization 

# File Limits
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
MAX_TEXT_LENGTH = 10 * 1024 * 1024  # 10MB
MAX_MESSAGE_LENGTH = 10000  # 10k characters

# API Limits
MAX_TOP_K = 20
MIN_TOP_K = 1
MAX_SENTENCES_PER_CHUNK = 100
MIN_SENTENCES_PER_CHUNK = 1
MAX_OVERLAP_SENTENCES = 50

