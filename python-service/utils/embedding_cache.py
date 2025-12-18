"""
Embedding Cache Module

Caches query embeddings to avoid re-computation for identical queries.
"""

import hashlib
import logging
from typing import Dict, Optional
import torch

logger = logging.getLogger(__name__)


class EmbeddingCache:
    """LRU cache for embeddings."""
    
    def __init__(self, max_size: int = 1000, device: Optional[str] = None):
        self.max_size = max_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.cache: Dict[str, torch.Tensor] = {}
        self.access_order: list = []
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text."""
        return hashlib.sha256(text.encode()).hexdigest()
    
    def get(self, text: str) -> Optional[torch.Tensor]:
        """Get cached embedding if exists."""
        key = self._hash_text(text)
        if key in self.cache:
            # Move to end (most recently used)
            if key in self.access_order:
                self.access_order.remove(key)
            self.access_order.append(key)
            # Return reference instead of clone (caller can clone if needed)
            # This avoids unnecessary memory copy for cache hits
            return self.cache[key]
        return None
    
    def put(self, text: str, embedding: torch.Tensor) -> None:
        """Cache an embedding."""
        key = self._hash_text(text)
        
        # Remove oldest if cache full
        if len(self.cache) >= self.max_size and key not in self.cache:
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]
        
        # Store embedding (clone only when storing to avoid modifying original)
        # Use detach() to break gradient computation graph
        self.cache[key] = embedding.detach().clone()
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)
    
    def clear(self) -> None:
        """Clear the cache."""
        self.cache.clear()
        self.access_order.clear()
        logger.info("Embedding cache cleared")
    
    def size(self) -> int:
        """Get cache size."""
        return len(self.cache)
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "hit_rate": 0.0,  # Would need to track hits/misses
        }


# Global cache instance
_embedding_cache: Optional[EmbeddingCache] = None


def get_embedding_cache(max_size: int = 1000, device: Optional[str] = None) -> EmbeddingCache:
    """Get or create global embedding cache."""
    global _embedding_cache
    if _embedding_cache is None:
        _embedding_cache = EmbeddingCache(max_size=max_size, device=device)
    return _embedding_cache


__all__ = ["EmbeddingCache", "get_embedding_cache"]

