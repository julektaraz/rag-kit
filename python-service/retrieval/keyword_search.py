"""
Keyword Search Module

Implements BM25 keyword-based search for hybrid retrieval.
BM25 is excellent for exact matches, technical terms, and code.
"""

import logging
from typing import List, Tuple, Dict
import re

# Try to import BM25, handle gracefully if not available
try:
    from rank_bm25 import BM25Okapi
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Okapi = None

logger = logging.getLogger(__name__)


def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    
    Args:
        text: Input text to tokenize
        
    Returns:
        List of lowercase tokens
    """
    # Convert to lowercase and split on non-word characters
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


class BM25Search:
    """
    BM25 keyword search index.
    
    BM25 is a ranking function used to score documents based on query terms.
    It's particularly good for exact matches and technical terminology.
    """
    
    def __init__(self, texts: List[str]):
        """
        Initialize BM25 index with texts.
        
        Args:
            texts: List of text chunks to index
        """
        if not HAS_BM25:
            raise ImportError(
                "rank-bm25 is required for keyword search. "
                "Install with: pip install rank-bm25"
            )
        
        if not texts:
            logger.warning("No texts provided for BM25 index")
            self.bm25 = None
            self.texts = []
            return
        
        # Tokenize all texts
        tokenized_texts = [tokenize(text) for text in texts]
        
        # Create BM25 index
        self.bm25 = BM25Okapi(tokenized_texts)
        self.texts = texts
        
        logger.info(f"BM25 index created with {len(texts)} documents")
    
    def search(
        self,
        query: str,
        top_k: int = 10,
    ) -> List[Tuple[int, float]]:
        """
        Search for top-k most relevant documents using BM25.
        
        Args:
            query: Search query string
            top_k: Number of top results to return
            
        Returns:
            List of tuples (document_index, bm25_score) sorted by score (descending)
        """
        if self.bm25 is None or len(self.texts) == 0:
            return []
        
        # Tokenize query
        query_tokens = tokenize(query)
        
        if not query_tokens:
            return []
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(query_tokens)
        
        # Get top-k indices
        top_k = min(top_k, len(scores))
        top_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]
        
        # Filter out zero scores
        results = [
            (idx, float(scores[idx]))
            for idx in top_indices
            if scores[idx] > 0
        ]
        
        return results
    
    def add_documents(self, texts: List[str]) -> None:
        """
        Add new documents to the BM25 index.
        
        Args:
            texts: List of new text chunks to add
        """
        if not texts:
            return
        
        # Tokenize new texts
        tokenized_texts = [tokenize(text) for text in texts]
        
        # Rebuild index with all texts
        self.texts.extend(texts)
        all_tokenized = [tokenize(text) for text in self.texts]
        self.bm25 = BM25Okapi(all_tokenized)
        
        logger.info(f"Added {len(texts)} documents to BM25 index. Total: {len(self.texts)}")


__all__ = ["BM25Search", "tokenize"]

