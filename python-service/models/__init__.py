"""
Models Module

Handles embedding generation and LLM inference.
"""

try:
    from .embeddings import EmbeddingModel
    from .llm_inference import LLMModel
except ImportError:
    # Handle absolute imports when run as script
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.embeddings import EmbeddingModel
    from models.llm_inference import LLMModel

__all__ = [
    "EmbeddingModel",
    "LLMModel",
]

