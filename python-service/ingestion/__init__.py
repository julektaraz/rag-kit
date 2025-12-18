"""
PDF Ingestion Module

This module handles PDF text extraction, cleaning, and chunking.
"""

try:
    from .pdf_extractor import extract_text_from_pdf
    from .text_cleaner import clean_text
    from .chunking import chunk_by_sentences
    from .code_extractor import chunk_code_by_structure, detect_language
    from .image_extractor import process_pdf_images, extract_images_from_pdf
except ImportError:
    # Handle absolute imports when run as script
    import sys
    import os
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion.pdf_extractor import extract_text_from_pdf
    from ingestion.text_cleaner import clean_text
    from ingestion.chunking import chunk_by_sentences
    
    # Optional imports
    try:
        from ingestion.code_extractor import chunk_code_by_structure, detect_language
    except ImportError:
        chunk_code_by_structure = None
        detect_language = None
    
    try:
        from ingestion.image_extractor import process_pdf_images, extract_images_from_pdf
    except ImportError:
        process_pdf_images = None
        extract_images_from_pdf = None

__all__ = [
    "extract_text_from_pdf",
    "clean_text",
    "chunk_by_sentences",
    "chunk_code_by_structure",
    "detect_language",
    "process_pdf_images",
    "extract_images_from_pdf",
]

