"""
Code Extraction Module

Extracts code from files with syntax-aware chunking using tree-sitter.
Provides better code understanding than sentence-based chunking.
"""

import logging
from typing import List, Dict, Optional
from pathlib import Path
import re

logger = logging.getLogger(__name__)

# Try to import tree-sitter
try:
    from tree_sitter import Language, Parser
    HAS_TREE_SITTER = True
except ImportError:
    HAS_TREE_SITTER = False
    logger.warning("tree-sitter not available, code extraction will use simple parsing")


def detect_language(file_path: str) -> Optional[str]:
    """
    Detect programming language from file extension.
    
    Args:
        file_path: Path to code file
        
    Returns:
        Language identifier or None
    """
    extension_map = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.jsx': 'javascript',
        '.tsx': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.cs': 'csharp',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
    }
    
    ext = Path(file_path).suffix.lower()
    return extension_map.get(ext)


def extract_functions_simple(code: str, language: str = 'python') -> List[Dict]:
    """
    Extract functions/classes using simple regex (fallback when tree-sitter unavailable).
    
    Args:
        code: Source code
        language: Programming language
        
    Returns:
        List of dicts with function/class info
    """
    functions = []
    
    if language == 'python':
        # Python function/class patterns
        func_pattern = r'^(?:async\s+)?def\s+(\w+)\s*\([^)]*\)\s*:'
        class_pattern = r'^class\s+(\w+)'
        
        lines = code.split('\n')
        current_function = None
        current_class = None
        start_line = 0
        indent_level = 0
        
        for i, line in enumerate(lines, 1):
            stripped = line.lstrip()
            current_indent = len(line) - len(stripped)
            
            # Check for class definition
            class_match = re.match(class_pattern, stripped)
            if class_match:
                if current_function:
                    functions.append({
                        'name': current_function,
                        'type': 'function',
                        'class': current_class,
                        'code': '\n'.join(lines[start_line-1:i-1]),
                        'line_start': start_line,
                        'line_end': i - 1,
                    })
                current_class = class_match.group(1)
                current_function = None
                start_line = i
                indent_level = current_indent
            
            # Check for function definition
            func_match = re.match(func_pattern, stripped)
            if func_match:
                if current_function:
                    functions.append({
                        'name': current_function,
                        'type': 'function',
                        'class': current_class,
                        'code': '\n'.join(lines[start_line-1:i-1]),
                        'line_start': start_line,
                        'line_end': i - 1,
                    })
                current_function = func_match.group(1)
                start_line = i
                indent_level = current_indent
            
            # End of function/class (indent returns to previous level)
            if current_function and stripped and current_indent <= indent_level and not stripped.startswith('@'):
                if i > start_line:
                    functions.append({
                        'name': current_function,
                        'type': 'function',
                        'class': current_class,
                        'code': '\n'.join(lines[start_line-1:i]),
                        'line_start': start_line,
                        'line_end': i,
                    })
                current_function = None
        
        # Add last function if exists
        if current_function:
            functions.append({
                'name': current_function,
                'type': 'function',
                'class': current_class,
                'code': '\n'.join(lines[start_line-1:]),
                'line_start': start_line,
                'line_end': len(lines),
            })
    
    return functions


def extract_code_structures(
    code: str,
    file_path: str,
    language: Optional[str] = None,
) -> List[Dict]:
    """
    Extract code structures (functions, classes, etc.) from source code.
    
    Args:
        code: Source code content
        file_path: Path to the code file
        language: Programming language (auto-detected if None)
        
    Returns:
        List of code structure dicts with:
            - name: Function/class name
            - type: 'function', 'class', etc.
            - code: Code content
            - line_start: Starting line number
            - line_end: Ending line number
            - file_path: Source file path
    """
    if language is None:
        language = detect_language(file_path)
    
    if language is None:
        logger.warning(f"Could not detect language for {file_path}")
        return []
    
    # Use tree-sitter if available, otherwise fallback to regex
    if HAS_TREE_SITTER:
        # TODO: Implement tree-sitter parsing when language bindings are available
        # For now, use simple extraction
        logger.info(f"Using simple extraction for {language} (tree-sitter bindings not configured)")
        structures = extract_functions_simple(code, language)
    else:
        structures = extract_functions_simple(code, language)
    
    # Add file path to each structure
    for struct in structures:
        struct['file_path'] = file_path
        struct['language'] = language
    
    logger.info(f"Extracted {len(structures)} code structures from {file_path}")
    
    return structures


def chunk_code_by_structure(
    code: str,
    file_path: str,
    language: Optional[str] = None,
    min_chunk_size: int = 50,
) -> List[Dict]:
    """
    Chunk code by structure (functions, classes) rather than sentences.
    
    This maintains semantic coherence better for code.
    
    Args:
        code: Source code content
        file_path: Path to the code file
        language: Programming language (auto-detected if None)
        min_chunk_size: Minimum characters per chunk
        
    Returns:
        List of chunk dicts with metadata
    """
    structures = extract_code_structures(code, file_path, language)
    
    if not structures:
        # Fallback: chunk by lines if no structures found
        lines = code.split('\n')
        chunks = []
        chunk_size = 50  # lines per chunk
        
        for i in range(0, len(lines), chunk_size):
            chunk_lines = lines[i:i+chunk_size]
            chunk_text = '\n'.join(chunk_lines)
            
            if len(chunk_text.strip()) >= min_chunk_size:
                chunks.append({
                    'text': chunk_text,
                    'type': 'code_block',
                    'line_start': i + 1,
                    'line_end': min(i + chunk_size, len(lines)),
                    'file_path': file_path,
                    'language': language or 'unknown',
                })
        
        return chunks
    
    # Convert structures to chunks
    chunks = []
    for struct in structures:
        if len(struct['code'].strip()) >= min_chunk_size:
            chunks.append({
                'text': struct['code'],
                'type': struct['type'],
                'name': struct.get('name'),
                'class': struct.get('class'),
                'line_start': struct['line_start'],
                'line_end': struct['line_end'],
                'file_path': struct['file_path'],
                'language': struct['language'],
            })
    
    return chunks


__all__ = [
    "extract_code_structures",
    "chunk_code_by_structure",
    "detect_language",
]


