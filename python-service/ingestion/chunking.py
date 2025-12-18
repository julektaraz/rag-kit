"""
Text Chunking Module

Implements sentence-based chunking to maintain semantic context.
Chunks are created with 10-15 sentences per chunk with overlap.
"""

from typing import List

import nltk


def _ensure_nltk_data():
    """
    Ensure required NLTK data is available.

    We deliberately do not download at import/runtime here. Instead, we fail
    with a clear error message so users can install the data explicitly as
    part of their environment setup (see README).
    """
    try:
        # Newer NLTK versions use punkt_tab, older ones use punkt.
        try:
            nltk.data.find("tokenizers/punkt_tab")
        except LookupError:
            nltk.data.find("tokenizers/punkt")
    except LookupError as exc:
        raise RuntimeError(
            "NLTK sentence tokenizer data not found. Install it once with:\n"
            "  python -c \"import nltk; nltk.download('punkt')\"\n"
            "or, for newer NLTK versions:\n"
            "  python -c \"import nltk; nltk.download('punkt_tab')\""
        ) from exc


def chunk_by_sentences(
    text: str,
    sentences_per_chunk: int = 12,
    overlap_sentences: int = 2,
    min_chunk_size: int = 50,
) -> List[str]:
    """
    Split text into chunks based on sentences.
    
    This maintains semantic coherence better than word-based chunking.
    Each chunk contains approximately sentences_per_chunk sentences,
    with overlap_sentences sentences overlapping between adjacent chunks.
    
    Args:
        text: Input text to chunk
        sentences_per_chunk: Target number of sentences per chunk (default: 12)
        overlap_sentences: Number of sentences to overlap between chunks (default: 2)
        min_chunk_size: Minimum characters per chunk (default: 50)
        
    Returns:
        List of text chunks
    """
    if not text or len(text.strip()) < min_chunk_size:
        return [text] if text.strip() else []
    
    # Ensure NLTK data is available before tokenizing
    _ensure_nltk_data()
    
    # Use NLTK sentence tokenizer
    sentences = nltk.sent_tokenize(text)
    
    if not sentences:
        return []
    
    # Filter out very short sentences (likely artifacts)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    
    if not sentences:
        return []
    
    chunks = []
    start_idx = 0
    
    while start_idx < len(sentences):
        # Calculate end index for this chunk
        end_idx = min(start_idx + sentences_per_chunk, len(sentences))
        
        # Extract chunk
        chunk_sentences = sentences[start_idx:end_idx]
        chunk_text = " ".join(chunk_sentences)
        
        # Only add chunk if it meets minimum size
        if len(chunk_text.strip()) >= min_chunk_size:
            chunks.append(chunk_text.strip())
        
        # Move start index forward, accounting for overlap
        # If we're at the end, break to avoid infinite loop
        if end_idx >= len(sentences):
            break
        
        # Calculate next start index with overlap
        next_start_idx = end_idx - overlap_sentences
        
        # Safety check: ensure we always make progress
        # If overlap is too large or we're not advancing, move forward by at least 1
        if next_start_idx <= start_idx:
            next_start_idx = end_idx
        
        start_idx = next_start_idx
    
    # If no chunks were created (all too small), return the original text
    if not chunks:
        return [text.strip()] if text.strip() else []
    
    return chunks


__all__ = ["chunk_by_sentences"]

