"""
RAG Pipeline Orchestrator

Main class that coordinates PDF ingestion, embedding, retrieval, and generation.
"""

import uuid
from pathlib import Path
from typing import List, Dict, Optional
import logging
import sys
import os

# Import default constants
try:
    from config import (
        DEFAULT_TOP_K,
        DEFAULT_MIN_SIMILARITY,
        DEFAULT_SENTENCES_PER_CHUNK,
        DEFAULT_OVERLAP_SENTENCES,
        DEFAULT_MAX_NEW_TOKENS,
        DEFAULT_TEMPERATURE,
        DEFAULT_HYBRID_SEARCH,
        DEFAULT_KEYWORD_WEIGHT,
        DEFAULT_SEMANTIC_WEIGHT,
        DEFAULT_USE_RERANKING,
        DEFAULT_RERANK_TOP_K,
    )
except ImportError:
    DEFAULT_TOP_K = 5
    DEFAULT_MIN_SIMILARITY = 0.3
    DEFAULT_SENTENCES_PER_CHUNK = 12
    DEFAULT_OVERLAP_SENTENCES = 2
    DEFAULT_MAX_NEW_TOKENS = 512
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_HYBRID_SEARCH = True
    DEFAULT_KEYWORD_WEIGHT = 0.3
    DEFAULT_SEMANTIC_WEIGHT = 0.7
    DEFAULT_USE_RERANKING = True
    DEFAULT_RERANK_TOP_K = 20

# Handle both relative and absolute imports
try:
    from ..ingestion import extract_text_from_pdf, clean_text, chunk_by_sentences
    from ..ingestion.code_extractor import chunk_code_by_structure
    from ..ingestion.image_extractor import process_pdf_images, process_standalone_image
    from ..models.embeddings import EmbeddingModel
    from ..models.llm_inference import LLMModel
    from ..retrieval.tensor_store import TensorStore
    from ..retrieval.similarity_search import cosine_similarity_search
    from ..retrieval.hybrid_search import hybrid_search
    from ..retrieval.reranking import Reranker
    from ..retrieval.citation_utils import enhance_citation
    from .multi_file_context import retrieve_multi_file_context
except ImportError:
    # If relative imports fail, use absolute imports
    # Add parent directory to path
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ingestion import extract_text_from_pdf, clean_text, chunk_by_sentences
    from models.embeddings import EmbeddingModel
    from models.llm_inference import LLMModel
    from retrieval.tensor_store import TensorStore
    from retrieval.similarity_search import cosine_similarity_search
    
    # Optional imports for advanced features
    try:
        from ingestion.code_extractor import chunk_code_by_structure
    except ImportError:
        chunk_code_by_structure = None
    
    try:
        from ingestion.image_extractor import process_pdf_images, process_standalone_image
    except ImportError:
        process_pdf_images = None
        process_standalone_image = None
    
    try:
        from retrieval.hybrid_search import hybrid_search
    except ImportError:
        hybrid_search = None
    
    try:
        from retrieval.reranking import Reranker
    except ImportError:
        Reranker = None
    
    try:
        from retrieval.citation_utils import enhance_citation
    except ImportError:
        def enhance_citation(text, query, **kwargs):
            return {
                "text": text,
                "source": kwargs.get("source", "unknown"),
                "chunk_index": kwargs.get("chunk_index", 0),
                "score": kwargs.get("score", 0.0),
            }
    
    try:
        from pipeline.multi_file_context import retrieve_multi_file_context
    except ImportError:
        retrieve_multi_file_context = None

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    Complete RAG pipeline orchestrator.
    
    Handles:
    - PDF ingestion and chunking
    - Code file ingestion with syntax-aware chunking
    - Image extraction and OCR
    - Embedding generation and storage
    - Query retrieval with hybrid search and re-ranking
    - Answer generation
    """
    
    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-mpnet-base-v2",
        llm_model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
        use_quantization: bool = True,
        device: Optional[str] = None,
        persistence_path: Optional[str] = None,
        auto_save: bool = True,
    ):
        """
        Initialize the RAG pipeline.
        
        Args:
            embedding_model_name: Hugging Face model for embeddings
            llm_model_name: Hugging Face model for generation
            use_quantization: Whether to use 4-bit quantization for LLM
            device: Device to use ('cuda', 'cpu', or None for auto)
            persistence_path: Path to save/load embeddings (None = no persistence)
            auto_save: Whether to automatically save embeddings on updates
        """
        logger.info("Initializing RAG Pipeline...")
        
        # Initialize models
        self.embedding_model = EmbeddingModel(
            model_name=embedding_model_name,
            device=device,
        )
        
        self.llm_model = LLMModel(
            model_name=llm_model_name,
            use_quantization=use_quantization,
            device=device,
        )
        
        # Initialize storage with persistence
        self.tensor_store = TensorStore(
            device=device,
            persistence_path=persistence_path,
            auto_save=auto_save,
        )
        
        # Initialize re-ranker (optional, lazy-loaded)
        self.reranker: Optional[Reranker] = None
        
        logger.info("RAG Pipeline initialized successfully")
    
    def ingest_pdf(
        self,
        pdf_path: str,
        source: Optional[str] = None,
        sentences_per_chunk: int = DEFAULT_SENTENCES_PER_CHUNK,
        overlap_sentences: int = DEFAULT_OVERLAP_SENTENCES,
        extract_images: bool = False,
    ) -> Dict:
        """
        Ingest a PDF file: extract, chunk, embed, and store.
        
        Args:
            pdf_path: Path to PDF file
            source: Optional source identifier
            sentences_per_chunk: Target sentences per chunk
            overlap_sentences: Sentence overlap between chunks
            extract_images: Whether to extract and OCR images
            
        Returns:
            Dictionary with:
                - doc_id: Unique document identifier
                - chunks: Number of chunks created
                - pages: Number of pages in PDF
                - metadata: PDF metadata
                - images: List of processed images (if extract_images=True)
        """
        logger.info(f"Ingesting PDF: {pdf_path}")
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Extract text from PDF
        pdf_data = extract_text_from_pdf(pdf_path)
        raw_text = pdf_data["raw_text"]
        pages = pdf_data["pages"]
        metadata = pdf_data["metadata"]
        
        logger.info(f"Extracted text from {pages} pages")
        
        # Extract images if requested
        images = []
        if extract_images and process_pdf_images:
            try:
                images = process_pdf_images(pdf_path, extract_ocr=True)
                logger.info(f"Extracted {len(images)} images from PDF")
            except Exception as e:
                logger.warning(f"Image extraction failed: {e}")
        
        # Clean text
        cleaned_text = clean_text(raw_text)
        logger.info(f"Cleaned text: {len(cleaned_text)} characters")
        
        # Chunk by sentences
        chunks = chunk_by_sentences(
            cleaned_text,
            sentences_per_chunk=sentences_per_chunk,
            overlap_sentences=overlap_sentences,
        )
        
        logger.info(f"Created {len(chunks)} chunks")
        
        if not chunks:
            raise ValueError("No chunks created from PDF")
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Prepare metadata for each chunk
        chunk_metadata = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata.append({
                "text": chunk,
                "source": source or Path(pdf_path).name,
                "doc_id": doc_id,
                "chunk_index": idx,
                "total_chunks": len(chunks),
            })
        
        # Store embeddings and metadata
        self.tensor_store.add_embeddings(embeddings, chunk_metadata)
        
        logger.info(f"Successfully ingested PDF. Doc ID: {doc_id}")
        
        result = {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "pages": pages,
            "metadata": metadata,
        }
        
        if images:
            result["images"] = images
        
        return result
    
    def ingest_code(
        self,
        code_path: str,
        source: Optional[str] = None,
    ) -> Dict:
        """
        Ingest a code file with syntax-aware chunking.
        
        Args:
            code_path: Path to code file
            source: Optional source identifier
            
        Returns:
            Dictionary with:
                - doc_id: Unique document identifier
                - chunks: Number of chunks created
                - structures: Number of code structures extracted
        """
        if chunk_code_by_structure is None:
            raise ImportError(
                "Code extraction not available. Install tree-sitter: pip install tree-sitter"
            )
        
        logger.info(f"Ingesting code file: {code_path}")
        
        # Read code file
        code_path_obj = Path(code_path)
        if not code_path_obj.exists():
            raise FileNotFoundError(f"Code file not found: {code_path}")
        
        with open(code_path, 'r', encoding='utf-8') as f:
            code = f.read()
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Chunk by code structure
        chunks_data = chunk_code_by_structure(code, code_path)
        
        if not chunks_data:
            raise ValueError("No code structures extracted")
        
        # Extract texts and metadata
        chunks = [chunk["text"] for chunk in chunks_data]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Prepare metadata
        chunk_metadata = []
        for idx, chunk_data in enumerate(chunks_data):
            chunk_metadata.append({
                "text": chunk_data["text"],
                "source": source or code_path_obj.name,
                "doc_id": doc_id,
                "chunk_index": idx,
                "file_path": chunk_data.get("file_path", code_path),
                "language": chunk_data.get("language", "unknown"),
                "type": chunk_data.get("type", "code_block"),
                "name": chunk_data.get("name"),
                "class": chunk_data.get("class"),
                "line_start": chunk_data.get("line_start"),
                "line_end": chunk_data.get("line_end"),
            })
        
        # Store embeddings and metadata
        self.tensor_store.add_embeddings(embeddings, chunk_metadata)
        
        logger.info(f"Successfully ingested code file. Doc ID: {doc_id}")
        
        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "structures": len(chunks_data),
            "language": chunks_data[0].get("language", "unknown") if chunks_data else "unknown",
        }
    
    def ingest_image(
        self,
        image_path: str,
        source: Optional[str] = None,
    ) -> Dict:
        """
        Ingest a standalone image file: OCR, embed, and store.
        
        Args:
            image_path: Path to image file
            source: Optional source identifier
            
        Returns:
            Dictionary with:
                - doc_id: Unique document identifier
                - chunks: Number of chunks created (from OCR text)
                - ocr_text: Extracted OCR text
        """
        if process_standalone_image is None:
            raise ImportError(
                "Image processing not available. Install Pillow and pytesseract: "
                "pip install Pillow pytesseract"
            )
        
        logger.info(f"Ingesting image file: {image_path}")
        
        # Process image
        image_data = process_standalone_image(image_path, extract_ocr=True)
        ocr_text = image_data.get("ocr_text", "")
        
        if not ocr_text:
            logger.warning("No text extracted from image via OCR")
            # Still create a chunk with metadata
            ocr_text = f"[Image: {Path(image_path).name}]"
        
        # Generate document ID
        doc_id = str(uuid.uuid4())
        
        # Chunk OCR text (if substantial)
        if len(ocr_text) > 100:
            chunks = chunk_by_sentences(ocr_text)
        else:
            chunks = [ocr_text] if ocr_text else []
        
        if not chunks:
            chunks = [ocr_text] if ocr_text else ["[Image with no extractable text]"]
        
        # Generate embeddings
        embeddings = self.embedding_model.embed_batch(chunks)
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Prepare metadata
        chunk_metadata = []
        for idx, chunk in enumerate(chunks):
            chunk_metadata.append({
                "text": chunk,
                "source": source or Path(image_path).name,
                "doc_id": doc_id,
                "chunk_index": idx,
                "file_path": image_path,
                "type": "image",
                "image_width": image_data.get("width"),
                "image_height": image_data.get("height"),
                "image_format": image_data.get("format"),
                "has_ocr_text": image_data.get("has_text", False),
            })
        
        # Store embeddings and metadata
        self.tensor_store.add_embeddings(embeddings, chunk_metadata)
        
        logger.info(f"Successfully ingested image file. Doc ID: {doc_id}")
        
        return {
            "doc_id": doc_id,
            "chunks": len(chunks),
            "ocr_text": ocr_text,
            "image_metadata": {
                "width": image_data.get("width"),
                "height": image_data.get("height"),
                "format": image_data.get("format"),
                "has_text": image_data.get("has_text", False),
            },
        }
    
    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        min_similarity: float = DEFAULT_MIN_SIMILARITY,
        use_hybrid_search: bool = DEFAULT_HYBRID_SEARCH,
        use_reranking: bool = DEFAULT_USE_RERANKING,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
        semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    ) -> Dict:
        """
        Query the RAG system: retrieve context and generate answer.
        
        Args:
            question: User's question
            top_k: Number of context chunks to retrieve
            min_similarity: Minimum similarity score threshold
            use_hybrid_search: Whether to use hybrid (keyword + semantic) search
            use_reranking: Whether to re-rank results with cross-encoder
            keyword_weight: Weight for BM25 keyword search (0-1)
            semantic_weight: Weight for semantic search (0-1)
            
        Returns:
            Dictionary with:
                - answer: Generated answer
                - contexts: List of retrieved context chunks with scores and highlights
                - sources: List of unique sources
        """
        logger.info(f"Processing query: {question[:50]}...")
        
        # Check if we have any embeddings
        if self.tensor_store.size() == 0:
            return {
                "answer": "No documents have been ingested yet. Please upload a PDF first.",
                "contexts": [],
                "sources": [],
            }
        
        # Embed the query
        query_embedding = self.embedding_model.embed_text(question)
        logger.info("Query embedded")
        
        # Get embeddings and texts
        # Try to get FAISS index first (ultra-fast)
        faiss_index = self.tensor_store.get_faiss_index()
        chunk_embeddings = None if faiss_index is not None else self.tensor_store.get_normalized_embeddings()
        if chunk_embeddings is None and faiss_index is None:
            chunk_embeddings = self.tensor_store.get_embeddings()
        
        if chunk_embeddings is None and faiss_index is None:
            logger.warning("No embeddings available for search")
            return {
                "answer": "No documents have been ingested yet. Please upload a PDF first.",
                "contexts": [],
                "sources": [],
            }
        
        texts = self.tensor_store.get_texts()
        bm25_index = self.tensor_store.get_bm25_index()
        
        # Retrieve chunks using hybrid search or semantic search
        if use_hybrid_search and bm25_index is not None and hybrid_search is not None:
            logger.info("Using hybrid search (BM25 + semantic)")
            results = hybrid_search(
                query=question,
                query_embedding=query_embedding,
                texts=texts,
                chunk_embeddings=chunk_embeddings,
                bm25_index=bm25_index,
                faiss_index=faiss_index,  # Pass FAISS index for faster search
                top_k=DEFAULT_RERANK_TOP_K if use_reranking else top_k,
                keyword_weight=keyword_weight,
                semantic_weight=semantic_weight,
                min_score=min_similarity,
            )
        else:
            logger.info("Using semantic search only")
            results = cosine_similarity_search(
                query_embedding,
                chunk_embeddings,
                top_k=DEFAULT_RERANK_TOP_K if use_reranking else top_k,
                min_score=min_similarity,
                faiss_index=faiss_index,  # Pass FAISS index for ultra-fast search
            )
        
        logger.info(f"Retrieved {len(results)} candidate chunks")
        
        # Re-rank results if enabled
        if use_reranking and results and Reranker is not None:
            try:
                if self.reranker is None:
                    self.reranker = Reranker(device=self.embedding_model.device)
                
                # Get texts for candidates
                candidate_indices = [idx for idx, _ in results]
                candidate_texts = [texts[idx] for idx in candidate_indices]
                
                # Re-rank
                reranked_results = self.reranker.rerank_candidates(
                    query=question,
                    candidate_indices=candidate_indices,
                    candidate_texts=candidate_texts,
                    top_k=top_k,
                )
                
                results = reranked_results
                logger.info(f"Re-ranked to {len(results)} top chunks")
            except Exception as e:
                logger.warning(f"Re-ranking failed, using original results: {e}")
        
        logger.info(f"Final {len(results)} relevant chunks")
        
        # Early return if no relevant contexts found
        if not results:
            logger.warning(f"No chunks found above similarity threshold {min_similarity}")
            return {
                "answer": "I couldn't find any relevant information in the documents to answer your question. Please try rephrasing your query or lowering the similarity threshold.",
                "contexts": [],
                "sources": [],
            }
        
        # Get context texts and metadata with enhanced citations
        all_metadata = self.tensor_store.get_metadata()
        
        # Optionally retrieve multi-file context for code
        chunk_indices = [idx for idx, _ in results]
        if retrieve_multi_file_context:
            try:
                # Check if we have code chunks (have file_path)
                has_code_chunks = any(
                    all_metadata[idx].get("file_path") for idx in chunk_indices
                )
                
                if has_code_chunks:
                    # Retrieve related file contexts
                    multi_file_contexts = retrieve_multi_file_context(
                        chunk_indices=chunk_indices,
                        metadata=all_metadata,
                        max_related_files=3,
                    )
                    # Use multi-file contexts if available
                    if multi_file_contexts:
                        # Map back to scores
                        score_map = {idx: score for idx, score in results}
                        contexts = []
                        sources = set()
                        
                        for ctx in multi_file_contexts:
                            idx = ctx.get('chunk_index', 0)
                            score = score_map.get(idx, 0.0)
                            
                            enhanced = enhance_citation(
                                text=ctx.get("text", ""),
                                query=question,
                                source=ctx.get("source", "unknown"),
                                chunk_index=idx,
                                score=score,
                            )
                            enhanced['is_related'] = ctx.get('is_related', False)
                            enhanced['file_group'] = ctx.get('file_group')
                            
                            contexts.append(enhanced)
                            sources.add(ctx.get("source", "unknown"))
                    else:
                        # Fallback to regular processing
                        contexts = []
                        sources = set()
                        for chunk_idx, score in results:
                            chunk_meta = all_metadata[chunk_idx]
                            enhanced = enhance_citation(
                                text=chunk_meta["text"],
                                query=question,
                                source=chunk_meta.get("source", "unknown"),
                                chunk_index=chunk_meta.get("chunk_index", 0),
                                score=score,
                            )
                            contexts.append(enhanced)
                            sources.add(chunk_meta.get("source", "unknown"))
                else:
                    # Regular processing for non-code chunks
                    contexts = []
                    sources = set()
                    for chunk_idx, score in results:
                        chunk_meta = all_metadata[chunk_idx]
                        enhanced = enhance_citation(
                            text=chunk_meta["text"],
                            query=question,
                            source=chunk_meta.get("source", "unknown"),
                            chunk_index=chunk_meta.get("chunk_index", 0),
                            score=score,
                        )
                        contexts.append(enhanced)
                        sources.add(chunk_meta.get("source", "unknown"))
            except Exception as e:
                logger.warning(f"Multi-file context retrieval failed: {e}")
                # Fallback to regular processing
                contexts = []
                sources = set()
                for chunk_idx, score in results:
                    chunk_meta = all_metadata[chunk_idx]
                    enhanced = enhance_citation(
                        text=chunk_meta["text"],
                        query=question,
                        source=chunk_meta.get("source", "unknown"),
                        chunk_index=chunk_meta.get("chunk_index", 0),
                        score=score,
                    )
                    contexts.append(enhanced)
                    sources.add(chunk_meta.get("source", "unknown"))
        else:
            # Regular processing
            contexts = []
            sources = set()
            for chunk_idx, score in results:
                chunk_meta = all_metadata[chunk_idx]
                enhanced = enhance_citation(
                    text=chunk_meta["text"],
                    query=question,
                    source=chunk_meta.get("source", "unknown"),
                    chunk_index=chunk_meta.get("chunk_index", 0),
                    score=score,
                )
                contexts.append(enhanced)
                sources.add(chunk_meta.get("source", "unknown"))
        
        # Format prompt with context
        context_texts = [ctx["text"] for ctx in contexts]
        prompt = self.llm_model.format_prompt_with_context(
            question,
            context_texts,
        )
        
        # Generate answer
        logger.info("Generating answer with LLM...")
        answer = self.llm_model.generate(
            prompt,
            max_new_tokens=DEFAULT_MAX_NEW_TOKENS,
            temperature=DEFAULT_TEMPERATURE,
        )
        
        logger.info("Query processed successfully")
        
        return {
            "answer": answer,
            "contexts": contexts,
            "sources": list(sources),
        }
    
    def get_stats(self) -> Dict:
        """Get statistics about the stored data."""
        return {
            "total_chunks": self.tensor_store.size(),
            "embedding_dim": self.embedding_model.get_embedding_dim(),
            "device": self.tensor_store.device,
        }


__all__ = ["RAGPipeline"]
