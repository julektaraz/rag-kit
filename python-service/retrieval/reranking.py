"""
Re-ranking Module

Uses cross-encoder models to re-rank initial retrieval results.
Cross-encoders provide better relevance by considering query-document interaction.
"""

import logging
from typing import List, Tuple, Optional
import torch

try:
    from sentence_transformers import CrossEncoder
    HAS_CROSS_ENCODER = True
except ImportError:
    HAS_CROSS_ENCODER = False
    logger = logging.getLogger(__name__)
    logger.warning("sentence-transformers not available, re-ranking disabled")

logger = logging.getLogger(__name__)


class Reranker:
    """
    Cross-encoder re-ranker for improving retrieval quality.
    
    Cross-encoders are slower than bi-encoders but provide better
    relevance by jointly encoding query and document.
    """
    
    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize the re-ranker.
        
        Args:
            model_name: Hugging Face model identifier for cross-encoder
            device: Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        if not HAS_CROSS_ENCODER:
            raise ImportError(
                "sentence-transformers is required for re-ranking. "
                "Install with: pip install sentence-transformers"
            )
        
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.model_name = model_name
        
        logger.info(f"Loading re-ranker: {model_name} on {device}")
        
        try:
            self.model = CrossEncoder(model_name, device=device)
            logger.info("Re-ranker loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load re-ranker: {e}")
            raise
    
    def rerank(
        self,
        query: str,
        texts: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank texts based on query relevance.
        
        Args:
            query: Search query
            texts: List of texts to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of tuples (text_index, relevance_score) sorted by score (descending)
        """
        if not texts:
            return []
        
        # Create query-document pairs
        pairs = [[query, text] for text in texts]
        
        # Get relevance scores from cross-encoder
        # This is slower but more accurate than bi-encoder
        scores = self.model.predict(pairs)
        
        # Convert to list of (index, score) tuples
        results = [(i, float(score)) for i, score in enumerate(scores)]
        
        # Sort by score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        
        # Return top-k
        top_k = min(top_k, len(results))
        return results[:top_k]
    
    def rerank_candidates(
        self,
        query: str,
        candidate_indices: List[int],
        candidate_texts: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """
        Re-rank pre-selected candidates.
        
        Args:
            query: Search query
            candidate_indices: Original indices of candidate texts
            candidate_texts: Texts to re-rank
            top_k: Number of top results to return
            
        Returns:
            List of tuples (original_index, relevance_score) sorted by score (descending)
        """
        if not candidate_texts or not candidate_indices:
            return []
        
        if len(candidate_indices) != len(candidate_texts):
            raise ValueError("candidate_indices and candidate_texts must have same length")
        
        # Re-rank candidates
        reranked = self.rerank(query, candidate_texts, top_k=top_k)
        
        # Map back to original indices
        results = [
            (candidate_indices[text_idx], score)
            for text_idx, score in reranked
        ]
        
        return results


__all__ = ["Reranker"]

