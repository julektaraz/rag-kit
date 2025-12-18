"""
Citation Utilities

Provides utilities for enhancing citations with line numbers, highlights, and context.
"""

import re
from typing import List, Dict, Optional, Tuple


def find_query_highlights(text: str, query: str, context_chars: int = 50) -> List[Dict[str, int]]:
    """
    Find positions in text where query terms appear.
    
    Args:
        text: Text to search in
        query: Search query
        context_chars: Number of characters of context around matches
        
    Returns:
        List of dicts with 'start', 'end', 'context_start', 'context_end'
    """
    if not text or not query:
        return []
    
    # Extract keywords from query (simple tokenization)
    keywords = re.findall(r'\b\w+\b', query.lower())
    
    if not keywords:
        return []
    
    highlights = []
    text_lower = text.lower()
    
    for keyword in keywords:
        # Find all occurrences of keyword
        for match in re.finditer(re.escape(keyword), text_lower):
            start = match.start()
            end = match.end()
            
            # Add context
            context_start = max(0, start - context_chars)
            context_end = min(len(text), end + context_chars)
            
            highlights.append({
                'start': start,
                'end': end,
                'context_start': context_start,
                'context_end': context_end,
            })
    
    # Merge overlapping highlights
    if highlights:
        highlights.sort(key=lambda x: x['start'])
        merged = [highlights[0]]
        
        for h in highlights[1:]:
            last = merged[-1]
            if h['start'] <= last['end']:
                # Merge overlapping highlights
                last['end'] = max(last['end'], h['end'])
                last['context_end'] = max(last['context_end'], h['context_end'])
            else:
                merged.append(h)
        
        highlights = merged
    
    return highlights


def get_line_number(text: str, char_position: int) -> int:
    """
    Get line number for a character position in text.
    
    Args:
        text: Full text
        char_position: Character position (0-indexed)
        
    Returns:
        Line number (1-indexed)
    """
    if char_position < 0 or char_position >= len(text):
        return 1
    
    # Count newlines before position
    line_num = text[:char_position].count('\n') + 1
    return line_num


def enhance_citation(
    text: str,
    query: str,
    source: Optional[str] = None,
    chunk_index: Optional[int] = None,
    score: Optional[float] = None,
) -> Dict:
    """
    Enhance a citation with highlights, line numbers, and metadata.
    
    Args:
        text: Citation text
        query: Search query (for highlighting)
        source: Source identifier
        chunk_index: Chunk index
        score: Relevance score
        
    Returns:
        Enhanced citation dict with highlights and metadata
    """
    highlights = find_query_highlights(text, query)
    
    # Get line numbers for highlights
    highlight_lines = []
    for h in highlights:
        line_start = get_line_number(text, h['start'])
        line_end = get_line_number(text, h['end'])
        highlight_lines.append({
            'line_start': line_start,
            'line_end': line_end,
            'char_start': h['start'],
            'char_end': h['end'],
        })
    
    # Get first and last line numbers of the text
    first_line = get_line_number(text, 0)
    last_line = get_line_number(text, len(text) - 1)
    
    enhanced = {
        'text': text,
        'source': source or 'unknown',
        'chunk_index': chunk_index,
        'score': score,
        'line_start': first_line,
        'line_end': last_line,
        'highlights': highlight_lines,
        'highlight_count': len(highlight_lines),
    }
    
    return enhanced


def format_citation_text(text: str, highlights: List[Dict], max_length: int = 500) -> str:
    """
    Format citation text with highlighted sections marked.
    
    Args:
        text: Original text
        highlights: List of highlight dicts
        max_length: Maximum length of formatted text
        
    Returns:
        Formatted text with highlights marked (e.g., **highlighted**)
    """
    if not highlights:
        # Just truncate if too long
        if len(text) > max_length:
            return text[:max_length] + "..."
        return text
    
    # Sort highlights by start position
    sorted_highlights = sorted(highlights, key=lambda x: x['char_start'])
    
    # Build formatted text
    result = []
    last_end = 0
    
    for h in sorted_highlights:
        # Add text before highlight
        if h['char_start'] > last_end:
            result.append(text[last_end:h['char_start']])
        
        # Add highlighted text
        highlighted = text[h['char_start']:h['char_end']]
        result.append(f"**{highlighted}**")
        
        last_end = h['char_end']
    
    # Add remaining text
    if last_end < len(text):
        result.append(text[last_end:])
    
    formatted = ''.join(result)
    
    # Truncate if too long
    if len(formatted) > max_length:
        formatted = formatted[:max_length] + "..."
    
    return formatted


__all__ = [
    "find_query_highlights",
    "get_line_number",
    "enhance_citation",
    "format_citation_text",
]

