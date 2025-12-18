"""
Text Cleaning and Normalization Module

Handles whitespace normalization, unicode handling, and text formatting.
"""

import re


def clean_text(text: str) -> str:
    """
    Clean and normalize text extracted from PDF.
    
    This function:
    - Removes excessive whitespace
    - Normalizes unicode characters
    - Removes special control characters
    - Preserves important formatting markers
    
    Args:
        text: Raw text from PDF extraction
        
    Returns:
        Cleaned and normalized text
    """
    if not text:
        return ""
    
    # Normalize unicode (handle special characters, quotes, etc.)
    # Convert various unicode spaces to regular space
    text = text.replace("\u00A0", " ")  # Non-breaking space
    text = text.replace("\u2009", " ")  # Thin space
    text = text.replace("\u2006", " ")  # Six-per-em space
    text = text.replace("\u2002", " ")  # En space
    text = text.replace("\u2003", " ")  # Em space
    
    # Normalize line breaks - convert multiple newlines to double newline
    # This preserves paragraph breaks while removing excessive spacing
    text = re.sub(r"\n{3,}", "\n\n", text)
    
    # Remove excessive spaces (more than 2 consecutive spaces)
    text = re.sub(r" {3,}", "  ", text)
    
    # Remove control characters except newlines and tabs
    text = re.sub(r"[\x00-\x08\x0B-\x0C\x0E-\x1F\x7F]", "", text)
    
    # Normalize tabs to spaces
    text = text.replace("\t", " ")
    
    # Remove trailing whitespace from each line
    lines = [line.rstrip() for line in text.split("\n")]
    text = "\n".join(lines)
    
    # Remove leading/trailing whitespace from entire text
    text = text.strip()
    
    return text


__all__ = ["clean_text"]
