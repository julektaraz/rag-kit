"""
Multi-File Context Retrieval Module

Groups related code chunks across files for better context understanding.
"""

import logging
from typing import List, Dict, Set, Optional
from collections import defaultdict

logger = logging.getLogger(__name__)


def build_file_graph(metadata: List[Dict]) -> Dict[str, Set[str]]:
    """
    Build a graph of file relationships based on imports and references.
    
    Args:
        metadata: List of chunk metadata dicts
        
    Returns:
        Dict mapping file_path -> set of related file_paths
    """
    file_graph = defaultdict(set)
    
    # Extract imports and references from code chunks
    for chunk_meta in metadata:
        file_path = chunk_meta.get("file_path")
        if not file_path:
            continue
        
        text = chunk_meta.get("text", "")
        
        # Find imports (simple pattern matching)
        import_patterns = [
            r'from\s+([\w.]+)\s+import',
            r'import\s+([\w.]+)',
            r'require\([\'"]([^\'"]+)[\'"]\)',  # JavaScript
            r'import\s+.*from\s+[\'"]([^\'"]+)[\'"]',  # ES6 imports
        ]
        
        for pattern in import_patterns:
            import re
            matches = re.findall(pattern, text)
            for match in matches:
                # Try to resolve to file path (simplified)
                # In real implementation, would use proper module resolution
                related_file = match.replace('.', '/')
                if related_file != file_path:
                    file_graph[file_path].add(related_file)
    
    return dict(file_graph)


def get_related_files(
    file_path: str,
    file_graph: Dict[str, Set[str]],
    max_depth: int = 2,
) -> Set[str]:
    """
    Get files related to a given file through imports/references.
    
    Args:
        file_path: Starting file path
        file_graph: File relationship graph
        max_depth: Maximum depth to traverse
        
    Returns:
        Set of related file paths
    """
    related = set()
    visited = set()
    
    def traverse(current_file: str, depth: int):
        if depth > max_depth or current_file in visited:
            return
        
        visited.add(current_file)
        related.add(current_file)
        
        # Get directly related files
        for related_file in file_graph.get(current_file, set()):
            traverse(related_file, depth + 1)
    
    traverse(file_path, 0)
    return related


def group_chunks_by_file(metadata: List[Dict]) -> Dict[str, List[int]]:
    """
    Group chunk indices by file path.
    
    Args:
        metadata: List of chunk metadata dicts
        
    Returns:
        Dict mapping file_path -> list of chunk indices
    """
    file_chunks = defaultdict(list)
    
    for idx, chunk_meta in enumerate(metadata):
        file_path = chunk_meta.get("file_path")
        if file_path:
            file_chunks[file_path].append(idx)
        else:
            # Handle chunks without file_path (e.g., PDF chunks)
            source = chunk_meta.get("source", "unknown")
            file_chunks[source].append(idx)
    
    return dict(file_chunks)


def retrieve_multi_file_context(
    chunk_indices: List[int],
    metadata: List[Dict],
    file_graph: Optional[Dict[str, Set[str]]] = None,
    max_related_files: int = 3,
) -> List[Dict]:
    """
    Retrieve context chunks from related files.
    
    Args:
        chunk_indices: Indices of initially retrieved chunks
        metadata: All chunk metadata
        file_graph: Optional file relationship graph
        max_related_files: Maximum number of related files to include
        
    Returns:
        List of context dicts with file grouping information
    """
    if not chunk_indices or not metadata:
        return []
    
    # Get file paths of retrieved chunks
    retrieved_files = set()
    for idx in chunk_indices:
        file_path = metadata[idx].get("file_path") or metadata[idx].get("source")
        if file_path:
            retrieved_files.add(file_path)
    
    # Build file graph if not provided
    if file_graph is None:
        file_graph = build_file_graph(metadata)
    
    # Group chunks by file
    file_chunks_map = group_chunks_by_file(metadata)
    
    # Find related files
    all_related_files = set()
    for file_path in retrieved_files:
        related = get_related_files(file_path, file_graph, max_depth=1)
        all_related_files.update(related)
    
    # Limit to max_related_files
    related_files = list(all_related_files)[:max_related_files]
    
    # Collect chunks from related files
    related_chunks = []
    for file_path in related_files:
        chunk_indices_for_file = file_chunks_map.get(file_path, [])
        related_chunks.extend(chunk_indices_for_file[:5])  # Limit per file
    
    # Combine original and related chunks
    all_context_indices = list(set(chunk_indices + related_chunks))
    
    # Build context dicts
    contexts = []
    for idx in all_context_indices:
        chunk_meta = metadata[idx]
        contexts.append({
            **chunk_meta,
            'is_related': idx not in chunk_indices,
            'file_group': chunk_meta.get("file_path") or chunk_meta.get("source"),
        })
    
    logger.info(
        f"Multi-file context: {len(chunk_indices)} original chunks, "
        f"{len(related_chunks)} related chunks from {len(related_files)} files"
    )
    
    return contexts


__all__ = [
    "build_file_graph",
    "get_related_files",
    "group_chunks_by_file",
    "retrieve_multi_file_context",
]


