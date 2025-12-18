"""
Similarity Search Module

Implements cosine similarity search for retrieval.
Uses FAISS CPU when available for ultra-fast search (10-100x faster).
Falls back to PyTorch (GPU-accelerated if CUDA available) if FAISS is not available.
"""

import torch
from typing import List, Tuple, Optional
import logging

# Import default constants
try:
    from config import DEFAULT_MIN_SIMILARITY
except ImportError:
    DEFAULT_MIN_SIMILARITY = 0.3

logger = logging.getLogger(__name__)

# Try to import FAISS
try:
    import faiss
    HAS_FAISS = True
    FAISS_INDEX_TYPE = faiss.Index
except ImportError:
    HAS_FAISS = False
    FAISS_INDEX_TYPE = None


def cosine_similarity_search(
    query_embedding: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SIMILARITY,
    faiss_index: Optional[FAISS_INDEX_TYPE] = None,
) -> List[Tuple[int, float]]:
    """
    Find top-k most similar chunks using cosine similarity.
    
    Uses FAISS when available for ultra-fast search (10-100x faster than PyTorch).
    Falls back to GPU-accelerated matrix operations if FAISS is not available.
    
    Args:
        query_embedding: Query embedding tensor of shape (embedding_dim,)
        chunk_embeddings: Chunk embeddings tensor of shape (num_chunks, embedding_dim)
        top_k: Number of top results to return
        min_score: Minimum similarity score threshold (0-1)
        faiss_index: Optional FAISS index for ultra-fast search
        
    Returns:
        List of tuples (chunk_index, similarity_score) sorted by score (descending)
    """
    # Use FAISS if available and index is provided
    if faiss_index is not None and HAS_FAISS:
        return _faiss_search(query_embedding, faiss_index, top_k, min_score)
    
    # Fall back to PyTorch implementation
    return _pytorch_search(query_embedding, chunk_embeddings, top_k, min_score)


def _faiss_search(
    query_embedding: torch.Tensor,
    faiss_index: FAISS_INDEX_TYPE,
    top_k: int,
    min_score: float,
) -> List[Tuple[int, float]]:
    """FAISS-based search (ultra-fast)."""
    # Convert query to numpy
    query_np = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)
    
    # Normalize for cosine similarity (FAISS requires normalized vectors)
    faiss.normalize_L2(query_np)
    
    # Search
    scores, indices = faiss_index.search(query_np, top_k)
    
    # Filter by min_score and convert to list
    results = []
    for i in range(len(indices[0])):
        idx = int(indices[0][i])
        score = float(scores[0][i])
        if score >= min_score:
            results.append((idx, score))
    
    return results


def _pytorch_search(
    query_embedding: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    top_k: int,
    min_score: float,
) -> List[Tuple[int, float]]:
    """PyTorch-based search (fallback)."""
    # Ensure tensors are on the same device and dtype
    device = chunk_embeddings.device
    dtype = chunk_embeddings.dtype
    
    # Convert query embedding to match chunk embeddings dtype and device
    query_embedding = query_embedding.to(device=device, dtype=dtype)
    
    # Normalize query embedding for cosine similarity
    # Note: chunk_embeddings should already be normalized (from TensorStore)
    # but we normalize here as well for safety and to handle non-normalized inputs
    query_norm = torch.nn.functional.normalize(query_embedding.unsqueeze(0), dim=1)
    
    # Use pre-normalized chunk embeddings if available, otherwise normalize
    # This optimization assumes chunk_embeddings are already normalized from TensorStore
    chunks_norm = chunk_embeddings
    if chunk_embeddings.dim() == 2:
        # Check if embeddings are already normalized (L2 norm should be ~1.0)
        # If not normalized, normalize them
        # Use the same dtype for the comparison tensor
        norms = torch.norm(chunks_norm, dim=1, keepdim=True)
        ones = torch.ones_like(norms, dtype=dtype)
        if not torch.allclose(norms, ones, atol=0.01):
            chunks_norm = torch.nn.functional.normalize(chunks_norm, dim=1)
    
    # Ensure both tensors have exactly the same dtype before matrix multiplication
    # This is critical to avoid dtype mismatch errors
    query_norm = query_norm.to(dtype=dtype)
    chunks_norm = chunks_norm.to(dtype=dtype)
    
    # Compute cosine similarity (matrix multiplication)
    # Shape: (1, num_chunks)
    similarities = torch.mm(query_norm, chunks_norm.t()).squeeze(0)
    
    # Filter by minimum score
    if min_score > 0:
        mask = similarities >= min_score
        if not mask.any():
            logger.warning(f"No chunks above similarity threshold {min_score}")
            return []
        similarities = similarities[mask]
        indices = torch.arange(len(chunk_embeddings), device=device)[mask]
    else:
        indices = torch.arange(len(chunk_embeddings), device=device)
    
    # Get top-k
    if len(similarities) == 0:
        return []
    
    top_k = min(top_k, len(similarities))
    top_scores, top_indices = torch.topk(similarities, k=top_k)
    
    # Get actual indices (if filtered)
    if min_score > 0:
        actual_indices = indices[top_indices].cpu().tolist()
    else:
        actual_indices = top_indices.cpu().tolist()
    
    scores = top_scores.cpu().tolist()
    
    # Return as list of (index, score) tuples
    results = list(zip(actual_indices, scores))
    
    return results


def batch_cosine_similarity_search(
    query_embeddings: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    top_k: int = 5,
    min_score: float = DEFAULT_MIN_SIMILARITY,
    faiss_index: Optional[FAISS_INDEX_TYPE] = None,
) -> List[List[Tuple[int, float]]]:
    """
    Batch version of cosine similarity search for multiple queries.
    
    Processes multiple queries in a single pass for better efficiency.
    
    Args:
        query_embeddings: Query embedding tensor of shape (num_queries, embedding_dim)
        chunk_embeddings: Chunk embeddings tensor of shape (num_chunks, embedding_dim)
        top_k: Number of top results to return per query
        min_score: Minimum similarity score threshold (0-1)
        faiss_index: Optional FAISS index for ultra-fast search
        
    Returns:
        List of result lists, one per query. Each result list contains
        (chunk_index, similarity_score) tuples sorted by score (descending).
    """
    if faiss_index is not None and HAS_FAISS:
        return _batch_faiss_search(query_embeddings, faiss_index, top_k, min_score)
    
    return _batch_pytorch_search(query_embeddings, chunk_embeddings, top_k, min_score)


def _batch_faiss_search(
    query_embeddings: torch.Tensor,
    faiss_index: FAISS_INDEX_TYPE,
    top_k: int,
    min_score: float,
) -> List[List[Tuple[int, float]]]:
    """Batch FAISS search."""
    # Convert queries to numpy
    queries_np = query_embeddings.cpu().numpy().astype('float32')
    
    # Normalize for cosine similarity
    faiss.normalize_L2(queries_np)
    
    # Batch search
    scores, indices = faiss_index.search(queries_np, top_k)
    
    # Process results for each query
    all_results = []
    for query_idx in range(len(queries_np)):
        query_results = []
        for i in range(len(indices[query_idx])):
            idx = int(indices[query_idx][i])
            score = float(scores[query_idx][i])
            if score >= min_score:
                query_results.append((idx, score))
        all_results.append(query_results)
    
    return all_results


def _batch_pytorch_search(
    query_embeddings: torch.Tensor,
    chunk_embeddings: torch.Tensor,
    top_k: int,
    min_score: float,
) -> List[List[Tuple[int, float]]]:
    """Batch PyTorch search."""
    device = chunk_embeddings.device
    dtype = chunk_embeddings.dtype
    
    # Normalize queries
    queries_norm = torch.nn.functional.normalize(query_embeddings, dim=1)
    
    # Normalize chunks
    chunks_norm = chunk_embeddings
    if chunk_embeddings.dim() == 2:
        norms = torch.norm(chunks_norm, dim=1, keepdim=True)
        ones = torch.ones_like(norms, dtype=dtype)
        if not torch.allclose(norms, ones, atol=0.01):
            chunks_norm = torch.nn.functional.normalize(chunks_norm, dim=1)
    
    # Batch matrix multiplication: (num_queries, embedding_dim) @ (embedding_dim, num_chunks)
    # Result: (num_queries, num_chunks)
    similarities = torch.mm(queries_norm, chunks_norm.t())
    
    # Process each query
    all_results = []
    for query_idx in range(len(queries_norm)):
        query_similarities = similarities[query_idx]
        
        # Filter by min_score
        if min_score > 0:
            mask = query_similarities >= min_score
            if not mask.any():
                all_results.append([])
                continue
            filtered_similarities = query_similarities[mask]
            filtered_indices = torch.arange(len(chunk_embeddings), device=device)[mask]
        else:
            filtered_similarities = query_similarities
            filtered_indices = torch.arange(len(chunk_embeddings), device=device)
        
        # Get top-k
        if len(filtered_similarities) == 0:
            all_results.append([])
            continue
        
        k = min(top_k, len(filtered_similarities))
        top_scores, top_positions = torch.topk(filtered_similarities, k=k)
        
        if min_score > 0:
            actual_indices = filtered_indices[top_positions].cpu().tolist()
        else:
            actual_indices = top_positions.cpu().tolist()
        
        scores = top_scores.cpu().tolist()
        all_results.append(list(zip(actual_indices, scores)))
    
    return all_results


__all__ = ["cosine_similarity_search", "batch_cosine_similarity_search"]


