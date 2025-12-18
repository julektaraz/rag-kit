"""
Retrieval Module

Handles embedding storage and similarity search using PyTorch tensors.
"""

try:
    from .tensor_store import TensorStore
    from .similarity_search import cosine_similarity_search
    from .keyword_search import BM25Search
    from .hybrid_search import hybrid_search
    from .reranking import Reranker
    from .citation_utils import enhance_citation
except ImportError:
    # Handle absolute imports when run as script
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from retrieval.tensor_store import TensorStore
    from retrieval.similarity_search import cosine_similarity_search
    from retrieval.keyword_search import BM25Search
    from retrieval.hybrid_search import hybrid_search
    from retrieval.reranking import Reranker
    from retrieval.citation_utils import enhance_citation

__all__ = [
    "TensorStore",
    "cosine_similarity_search",
    "BM25Search",
    "hybrid_search",
    "Reranker",
    "enhance_citation",
]

