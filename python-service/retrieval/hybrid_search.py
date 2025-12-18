"""
Hybrid Search Module

Combines keyword (BM25) and semantic (cosine similarity) search for better retrieval.
Hybrid search is particularly effective for technical content and code.
"""

import logging
from typing import List, Tuple, Optional
import torch
import numpy as np

from .keyword_search import BM25Search
from .similarity_search import cosine_similarity_search

# Import default constants
try:
    from config import DEFAULT_KEYWORD_WEIGHT, DEFAULT_SEMANTIC_WEIGHT
except ImportError:
    DEFAULT_KEYWORD_WEIGHT = 0.3
    DEFAULT_SEMANTIC_WEIGHT = 0.7

logger = logging.getLogger(__name__)

# Try to import FAISS for type hints
try:
    import faiss
    FAISS_INDEX_TYPE = faiss.Index
except ImportError:
    FAISS_INDEX_TYPE = None


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to 0-1 range using min-max normalization.
    
    Optimized to use vectorized operations when possible.
    
    Args:
        scores: List of scores to normalize
        
    Returns:
        Normalized scores
    """
    if not scores:
        return []
    
    # Use numpy for vectorized normalization (much faster for large lists)
    scores_array = np.array(scores, dtype=np.float32)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    if max_score == min_score:
        return [1.0] * len(scores)
    
    # Vectorized normalization
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


def hybrid_search(
    query: str,
    query_embedding: torch.Tensor,
    texts: List[str],
    chunk_embeddings: Optional[torch.Tensor] = None,
    bm25_index: Optional[BM25Search] = None,
    faiss_index: Optional[FAISS_INDEX_TYPE] = None,
    top_k: int = 10,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    min_score: float = 0.0,
) -> List[Tuple[int, float]]:
    """
    Perform hybrid search combining BM25 keyword search and semantic search.
    
    Optimized to use FAISS when available for ultra-fast semantic search.
    Uses vectorized operations for score normalization.
    
    Args:
        query: Search query string
        query_embedding: Query embedding tensor for semantic search
        texts: List of text chunks
        chunk_embeddings: Embedding tensor for all chunks (required if faiss_index is None)
        bm25_index: Optional pre-built BM25 index (will create if None)
        faiss_index: Optional FAISS index for ultra-fast semantic search
        top_k: Number of top results to return
        keyword_weight: Weight for BM25 keyword scores (0-1)
        semantic_weight: Weight for semantic similarity scores (0-1)
        min_score: Minimum combined score threshold
        
    Returns:
        List of tuples (chunk_index, combined_score) sorted by score (descending)
    """
    if not texts or (chunk_embeddings is None and faiss_index is None):
        return []
    
    # Ensure weights sum to 1.0
    total_weight = keyword_weight + semantic_weight
    if total_weight > 0:
        keyword_weight = keyword_weight / total_weight
        semantic_weight = semantic_weight / total_weight
    else:
        keyword_weight = 0.5
        semantic_weight = 0.5
    
    # 1. Keyword search (BM25)
    if bm25_index is None:
        bm25_index = BM25Search(texts)
    
    # Get more candidates than top_k for better fusion
    bm25_candidates = bm25_index.search(query, top_k=top_k * 3)
    bm25_scores_dict = {idx: score for idx, score in bm25_candidates}
    
    # 2. Semantic search (use FAISS if available for ultra-fast search)
    semantic_results = cosine_similarity_search(
        query_embedding,
        chunk_embeddings,
        top_k=top_k * 3,
        min_score=0.0,  # Don't filter here, we'll filter after fusion
        faiss_index=faiss_index,  # Pass FAISS index for faster search
    )
    semantic_scores_dict = {idx: score for idx, score in semantic_results}
    
    # 3. Combine results
    # Get all unique indices from both searches
    all_indices = list(set(bm25_scores_dict.keys()) | set(semantic_scores_dict.keys()))
    
    if not all_indices:
        return []
    
    # Vectorized normalization and combination (much faster than loops)
    bm25_scores_array = np.array([bm25_scores_dict.get(idx, 0.0) for idx in all_indices], dtype=np.float32)
    semantic_scores_array = np.array([semantic_scores_dict.get(idx, 0.0) for idx in all_indices], dtype=np.float32)
    
    # Normalize each score array to 0-1 range (vectorized)
    if len(bm25_scores_array) > 0:
        bm25_min, bm25_max = bm25_scores_array.min(), bm25_scores_array.max()
        if bm25_max > bm25_min:
            bm25_normalized = (bm25_scores_array - bm25_min) / (bm25_max - bm25_min)
        else:
            bm25_normalized = np.ones_like(bm25_scores_array)
    else:
        bm25_normalized = np.array([], dtype=np.float32)
    
    if len(semantic_scores_array) > 0:
        semantic_min, semantic_max = semantic_scores_array.min(), semantic_scores_array.max()
        if semantic_max > semantic_min:
            semantic_normalized = (semantic_scores_array - semantic_min) / (semantic_max - semantic_min)
        else:
            semantic_normalized = np.ones_like(semantic_scores_array)
    else:
        semantic_normalized = np.array([], dtype=np.float32)
    
    # Vectorized combination (much faster than Python loops)
    combined_scores_array = (
        keyword_weight * bm25_normalized +
        semantic_weight * semantic_normalized
    )
    
    # Filter by min_score and create results (vectorized)
    mask = combined_scores_array >= min_score
    filtered_indices = np.array(all_indices)[mask]
    filtered_scores = combined_scores_array[mask]
    
    # Sort by score (descending) and get top-k
    if len(filtered_scores) > 0:
        sorted_indices = np.argsort(filtered_scores)[::-1][:top_k]
        results = [
            (int(filtered_indices[i]), float(filtered_scores[i]))
            for i in sorted_indices
        ]
    else:
        results = []
    
    logger.info(
        f"Hybrid search: {len(bm25_candidates)} BM25 results, "
        f"{len(semantic_results)} semantic results, "
        f"{len(results)} combined results"
    )
    
    return results


__all__ = ["hybrid_search", "normalize_scores"]

