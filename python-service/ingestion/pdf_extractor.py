"""
PDF Text Extraction Module

Uses PyMuPDF (fitz) to extract text from PDF files with metadata.
"""

import fitz  # PyMuPDF
from typing import Dict, List, Optional
from pathlib import Path


def extract_text_from_pdf(pdf_path: str) -> Dict:
    """
    Extract text and metadata from a PDF file.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Dictionary containing:
            - text: Full extracted text
            - pages: Number of pages
            - metadata: PDF metadata (title, author, etc.)
            - page_texts: List of text per page (for page number tracking)
            
    Raises:
        FileNotFoundError: If PDF file doesn't exist
        Exception: If PDF extraction fails
    """
    pdf_path = Path(pdf_path)
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    
    try:
        # Open the PDF document
        doc = fitz.open(pdf_path)
        
        # Extract metadata
        metadata = {
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "creator": doc.metadata.get("creator", ""),
            "producer": doc.metadata.get("producer", ""),
            "creation_date": doc.metadata.get("creationDate", ""),
            "modification_date": doc.metadata.get("modDate", ""),
        }
        
        # Get page count before processing
        page_count = len(doc)
        
        # Extract text from each page
        full_text = []
        page_texts = []
        
        for page_num in range(page_count):
            page = doc[page_num]
            page_text = page.get_text()
            
            # Add page marker for reference
            page_text_with_marker = f"[Page {page_num + 1}]\n{page_text}"
            page_texts.append(page_text)
            full_text.append(page_text_with_marker)
        
        # Combine all pages
        combined_text = "\n\n".join(full_text)
        
        # Store metadata and page count before closing
        result = {
            "text": combined_text,
            "raw_text": "\n\n".join(page_texts),  # Text without page markers
            "pages": page_count,
            "metadata": metadata,
            "page_texts": page_texts,  # For tracking which page chunks come from
        }
        
        # Close document after extracting all data
        doc.close()
        
        return result
        
    except fitz.FileDataError as e:
        raise ValueError(f"PDF file is corrupted or invalid: {str(e)}") from e
    except Exception as e:
        raise RuntimeError(f"Failed to extract text from PDF: {str(e)}") from e


__all__ = ["extract_text_from_pdf"]

