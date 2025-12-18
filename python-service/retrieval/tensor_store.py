"""
PyTorch Tensor Storage Module

Stores embeddings as PyTorch tensors on GPU for efficient similarity search.
Maintains metadata (text, source, page numbers) alongside embeddings.

Supports FAISS for ultra-fast similarity search (10-100x faster than PyTorch).
Falls back to PyTorch if FAISS is not available.
"""

import torch
from typing import List, Dict, Optional, Any
import logging
from pathlib import Path
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np

logger = logging.getLogger(__name__)

# Try to import FAISS for faster similarity search
# Note: We use FAISS CPU only. GPU acceleration is handled via PyTorch tensors.
try:
    import faiss
    HAS_FAISS = True
    FAISS_INDEX_TYPE = faiss.Index
except ImportError:
    HAS_FAISS = False
    FAISS_INDEX_TYPE = None
    logger.info("FAISS not available, using PyTorch for similarity search")

try:
    from .keyword_search import BM25Search
    HAS_BM25 = True
except ImportError:
    HAS_BM25 = False
    BM25Search = None
    logger.warning("rank-bm25 not available, keyword search disabled")


class TensorStore:
    """
    Manages storage of embeddings as PyTorch tensors on GPU.
    
    This provides fast similarity search without needing a vector database
    for datasets up to ~100k embeddings. Uses FAISS for ultra-fast search when available.
    
    Supports automatic persistence: saves embeddings to disk on updates
    and loads them on initialization if a saved file exists.
    """
    
    def __init__(
        self,
        device: Optional[str] = None,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
        use_faiss: Optional[bool] = None,
    ):
        """
        Initialize the tensor store.
        
        Args:
            device: Device to store tensors on ('cuda', 'cpu', or None for auto)
            persistence_path: Path to save/load embeddings (None = no persistence)
            auto_save: Whether to automatically save on each update
            use_faiss: Whether to use FAISS for search (None = auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.persistence_path = Path(persistence_path) if persistence_path else None
        self.auto_save = auto_save
        
        # Determine FAISS usage
        if use_faiss is None:
            self.use_faiss = HAS_FAISS
        else:
            self.use_faiss = use_faiss and HAS_FAISS
        
        # Storage: embeddings tensor and metadata
        self.embeddings: Optional[torch.Tensor] = None
        self.normalized_embeddings: Optional[torch.Tensor] = None  # Pre-normalized for faster search
        self.metadata: List[Dict] = []  # List of dicts with text, source, doc_id, etc.
        
        # FAISS index for ultra-fast similarity search (CPU only)
        self.faiss_index: Optional[FAISS_INDEX_TYPE] = None
        self._embedding_dim: Optional[int] = None
        
        # BM25 keyword search index (for hybrid search)
        self.bm25_index: Optional[BM25Search] = None if not HAS_BM25 else None
        
        # Optimized storage: use list for incremental updates, convert to tensor when needed
        self._embedding_list: List[torch.Tensor] = []  # For incremental updates
        self._needs_rebuild: bool = False  # Flag to rebuild tensor from list
        
        # Async save executor for non-blocking persistence
        self._save_executor: Optional[ThreadPoolExecutor] = None
        
        # Auto-load if persistence path exists and file is found
        if self.persistence_path and self.persistence_path.with_suffix(".pt").exists():
            try:
                self.load(str(self.persistence_path))
                logger.info(f"Auto-loaded {len(self.metadata)} embeddings from {self.persistence_path}")
            except Exception as e:
                logger.warning(f"Failed to auto-load from {self.persistence_path}: {e}")
        else:
            logger.info(f"TensorStore initialized on {device} (FAISS: {self.use_faiss})")
    
    def add_embeddings(
        self,
        embeddings: torch.Tensor,
        metadata: List[Dict],
    ) -> None:
        """
        Add new embeddings and metadata to the store.
        
        Optimized to use incremental updates and FAISS when available.
        
        Args:
            embeddings: Tensor of shape (num_chunks, embedding_dim) on GPU
            metadata: List of metadata dicts, one per embedding
                Each dict should contain: text, source, doc_id, chunk_index, etc.
        """
        if embeddings.device != self.device:
            embeddings = embeddings.to(self.device)
        
        # Store embedding dimension
        if self._embedding_dim is None:
            self._embedding_dim = embeddings.shape[1]
        
        # Ensure dtype consistency
        if self.embeddings is not None and embeddings.dtype != self.embeddings.dtype:
            embeddings = embeddings.to(dtype=self.embeddings.dtype)
        
        # Add to metadata
        self.metadata.extend(metadata)
        
        # Optimized: Add to FAISS index directly (much faster than tensor concatenation)
        if self.use_faiss:
            # Convert to numpy for FAISS
            embeddings_np = embeddings.cpu().numpy().astype('float32')
            
            # Initialize FAISS index if needed (CPU only)
            if self.faiss_index is None:
                dim = embeddings_np.shape[1]
                # Use Inner Product for cosine similarity (after normalization)
                self.faiss_index = faiss.IndexFlatIP(dim)
                logger.debug(f"FAISS index initialized (CPU) with dimension {dim}")
            
            # Normalize embeddings for cosine similarity (FAISS requires normalized vectors)
            faiss.normalize_L2(embeddings_np)
            
            # Add to FAISS index (very fast, no memory copy of existing data)
            self.faiss_index.add(embeddings_np)
            
            # Mark that we need to rebuild PyTorch tensor (lazy)
            self._needs_rebuild = True
        else:
            # PyTorch path: use optimized incremental update
            if self.embeddings is None:
                # First addition - just store
                self.embeddings = embeddings
            else:
                # Optimized concatenation: use list to avoid repeated full copies
                # Only rebuild tensor when needed (lazy evaluation)
                self._embedding_list.append(embeddings)
                self._needs_rebuild = True
        
        # Update BM25 index for keyword search
        if HAS_BM25:
            texts = [meta.get("text", "") for meta in metadata]
            if self.bm25_index is None:
                # Create new index with all texts
                all_texts = [meta.get("text", "") for meta in self.metadata]
                self.bm25_index = BM25Search(all_texts)
            else:
                # Add new texts to existing index
                self.bm25_index.add_documents(texts)
        
        logger.info(
            f"Added {len(metadata)} embeddings. Total: {len(self.metadata)}"
        )
        
        # Async auto-save if enabled (non-blocking)
        if self.auto_save and self.persistence_path:
            self._save_async()
    
    def _rebuild_tensors(self) -> None:
        """Rebuild PyTorch tensors from incremental list (lazy evaluation)."""
        if not self._needs_rebuild or not self._embedding_list:
            return
        
        # Concatenate all new embeddings at once
        new_embeddings = torch.cat(self._embedding_list, dim=0)
        
        # Now concatenate with existing
        if self.embeddings is not None:
            self.embeddings = torch.cat([self.embeddings, new_embeddings], dim=0)
        else:
            self.embeddings = new_embeddings
        
        # Clear the list
        self._embedding_list.clear()
        self._needs_rebuild = False
        
        # Normalize embeddings for faster cosine similarity search
        self.normalized_embeddings = torch.nn.functional.normalize(
            self.embeddings, dim=1
        )
    
    def get_embeddings(self) -> Optional[torch.Tensor]:
        """Get the stored embeddings tensor."""
        # Rebuild if needed
        if self._needs_rebuild and not self.use_faiss:
            self._rebuild_tensors()
        return self.embeddings
    
    def get_normalized_embeddings(self) -> Optional[torch.Tensor]:
        """Get the normalized embeddings tensor for faster cosine similarity."""
        # Rebuild if needed
        if self._needs_rebuild and not self.use_faiss:
            self._rebuild_tensors()
        return self.normalized_embeddings
    
    def get_faiss_index(self) -> Optional[FAISS_INDEX_TYPE]:
        """Get the FAISS index for ultra-fast similarity search."""
        return self.faiss_index if self.use_faiss else None
    
    def search_faiss(
        self,
        query_embedding: torch.Tensor,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> List[tuple]:
        """
        Search using FAISS (ultra-fast, 10-100x faster than PyTorch).
        
        Args:
            query_embedding: Query embedding tensor
            top_k: Number of results to return
            min_score: Minimum similarity score threshold
            
        Returns:
            List of (index, score) tuples
        """
        if not self.use_faiss or self.faiss_index is None:
            return []
        
        # Convert query to numpy
        query_np = query_embedding.cpu().numpy().astype('float32').reshape(1, -1)
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_np)
        
        # Search
        scores, indices = self.faiss_index.search(query_np, top_k)
        
        # Filter by min_score and convert to list
        results = []
        for i in range(len(indices[0])):
            idx = int(indices[0][i])
            score = float(scores[0][i])
            if score >= min_score:
                results.append((idx, score))
        
        return results
    
    def get_bm25_index(self) -> Optional[BM25Search]:
        """Get the BM25 keyword search index."""
        if HAS_BM25 and self.bm25_index is None and self.metadata:
            # Lazy initialization
            texts = [meta.get("text", "") for meta in self.metadata]
            self.bm25_index = BM25Search(texts)
        return self.bm25_index if HAS_BM25 else None
    
    def get_texts(self) -> List[str]:
        """Get all text chunks for re-ranking."""
        return [meta.get("text", "") for meta in self.metadata]
    
    def get_metadata(self) -> List[Dict]:
        """Get the stored metadata."""
        return self.metadata
    
    def _save_async(self) -> None:
        """Async save in background thread (non-blocking)."""
        if self._save_executor is None:
            self._save_executor = ThreadPoolExecutor(max_workers=1)
        
        self._save_executor.submit(self._save_sync)
    
    def _save_sync(self) -> None:
        """Synchronous save operation (called from async executor)."""
        try:
            self.save(str(self.persistence_path))
        except Exception as e:
            logger.warning(f"Failed to auto-save embeddings: {e}")
    
    def clear(self) -> None:
        """Clear all stored embeddings and metadata."""
        self.embeddings = None
        self.normalized_embeddings = None
        self.metadata = []
        self.bm25_index = None
        self.faiss_index = None
        self._embedding_list.clear()
        self._needs_rebuild = False
        self._embedding_dim = None
        logger.info("TensorStore cleared")
        
        # Auto-save if enabled (saves empty state)
        if self.auto_save and self.persistence_path:
            try:
                # Remove persistence files if they exist
                pt_file = self.persistence_path.with_suffix(".pt")
                json_file = self.persistence_path.with_suffix(".json")
                if pt_file.exists():
                    pt_file.unlink()
                if json_file.exists():
                    json_file.unlink()
                logger.info("Cleared persistence files")
            except Exception as e:
                logger.warning(f"Failed to clear persistence files: {e}")
    
    def size(self) -> int:
        """Get the number of stored embeddings."""
        return len(self.metadata) if self.metadata else 0
    
    def save(self, filepath: Optional[str] = None) -> None:
        """
        Save embeddings and metadata to disk.
        
        Args:
            filepath: Path to save the store (will create .pt and .json files).
                     If None, uses the persistence_path from initialization.
        """
        if filepath is None:
            if self.persistence_path is None:
                raise ValueError("No filepath provided and no persistence_path configured")
            filepath = self.persistence_path
        else:
            filepath = Path(filepath)
        
        filepath = Path(filepath)
        
        # Ensure parent directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        if self.embeddings is None:
            logger.warning("No embeddings to save")
            return
        
        # Save tensor
        torch.save(self.embeddings.cpu(), filepath.with_suffix(".pt"))
        
        # Save metadata
        with open(filepath.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved TensorStore to {filepath} ({len(self.metadata)} embeddings)")
    
    def load(self, filepath: Optional[str] = None) -> None:
        """
        Load embeddings and metadata from disk.
        
        Args:
            filepath: Path to load from (expects .pt and .json files).
                     If None, uses the persistence_path from initialization.
        """
        if filepath is None:
            if self.persistence_path is None:
                raise ValueError("No filepath provided and no persistence_path configured")
            filepath = self.persistence_path
        else:
            filepath = Path(filepath)
        
        filepath = Path(filepath)
        pt_file = filepath.with_suffix(".pt")
        json_file = filepath.with_suffix(".json")
        
        if not pt_file.exists() or not json_file.exists():
            raise FileNotFoundError(
                f"Persistence files not found: {pt_file} or {json_file}"
            )
        
        # Load tensor
        loaded_embeddings = torch.load(
            pt_file,
            map_location=self.device,
        )
        
        # Ensure embeddings are on the correct device
        if loaded_embeddings.device != self.device:
            loaded_embeddings = loaded_embeddings.to(self.device)
        
        self.embeddings = loaded_embeddings
        self._embedding_dim = loaded_embeddings.shape[1] if loaded_embeddings is not None else None
        
        # Normalize embeddings for faster cosine similarity search
        if self.embeddings is not None:
            self.normalized_embeddings = torch.nn.functional.normalize(
                self.embeddings, dim=1
            )
            
            # Rebuild FAISS index if using FAISS
            if self.use_faiss:
                embeddings_np = self.embeddings.cpu().numpy().astype('float32')
                dim = embeddings_np.shape[1]
                self.faiss_index = faiss.IndexFlatIP(dim)
                
                # Normalize for FAISS
                faiss.normalize_L2(embeddings_np)
                
                # Add to FAISS (CPU only)
                self.faiss_index.add(embeddings_np)
        
        # Load metadata
        with open(json_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)
        
        self._needs_rebuild = False
        self._embedding_list.clear()
        
        logger.info(
            f"Loaded TensorStore from {filepath}. "
            f"Loaded {len(self.metadata)} embeddings"
        )


__all__ = ["TensorStore"]

