# Python RAG Service

Local RAG pipeline built from scratch with PyTorch, sentence-transformers, and Hugging Face models.

## Features

- **PDF Ingestion**: Extract text from PDFs with metadata
- **Sentence-based Chunking**: 10-15 sentences per chunk for better context
- **GPU-accelerated Embeddings**: sentence-transformers with PyTorch tensors
- **Fast Similarity Search**: Cosine similarity on GPU
- **Quantized LLM**: 4-bit quantization for efficient GPU usage
- **Complete RAG Pipeline**: End-to-end retrieval and generation

## Setup

### Prerequisites

- Python 3.10+
- NVIDIA GPU with CUDA (recommended)
- CUDA toolkit installed

### Installation

1. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download NLTK data (first run will auto-download):
```python
python -c "import nltk; nltk.download('punkt')"
```

### Environment Variables

Create `.env` file (optional):
```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
USE_QUANTIZATION=true
PORT=8000
PERSISTENCE_PATH=data/embeddings
AUTO_SAVE=true
```

**Persistence Configuration:**
- `PERSISTENCE_PATH`: Path to save/load embeddings (default: `data/embeddings`)
  - Set to empty string to disable persistence
  - Embeddings are saved as `.pt` (PyTorch tensor) and `.json` (metadata) files
- `AUTO_SAVE`: Automatically save embeddings after each ingestion (default: `true`)
  - When enabled, embeddings are saved immediately after adding new documents
  - On startup, embeddings are automatically loaded if persistence files exist

## Running the Service

```bash
python app.py
```

Or with uvicorn directly:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

## API Endpoints

### Health Check
```
GET /api/health
```

### Ingest PDF
```
POST /api/ingest/pdf
Content-Type: multipart/form-data

file: PDF file
source: (optional) Source identifier
sentences_per_chunk: (optional) Default: 12
overlap_sentences: (optional) Default: 2
```

### Ingest Text
```
POST /api/ingest/text
Content-Type: application/json

{
  "text": "Your text here...",
  "source": "optional_source",
  "sentences_per_chunk": 12,
  "overlap_sentences": 2
}
```

### Query
```
POST /api/chat
Content-Type: application/json

{
  "message": "Your question here",
  "top_k": 5,
  "min_similarity": 0.0
}
```

### Get Stats
```
GET /api/stats
```

### Persistence Management

#### Save Embeddings
```
POST /api/persistence/save
```
Manually save embeddings to disk (useful if auto-save is disabled).

#### Load Embeddings
```
POST /api/persistence/load
```
Manually load embeddings from disk.

#### Clear Embeddings
```
POST /api/persistence/clear
```
Clear all stored embeddings and metadata (also removes persistence files if auto-save is enabled).

## Model Recommendations

### Embedding Models
- `sentence-transformers/all-mpnet-base-v2` (default) - Good balance
- `sentence-transformers/all-MiniLM-L6-v2` - Faster, smaller
- `BAAI/bge-small-en-v1.5` - High quality

### LLM Models
- `mistralai/Mistral-7B-Instruct-v0.2` (default) - Good quality
- `google/gemma-2b-it` - Smaller, faster
- `google/gemma-7b-it` - Larger, better quality

Note: Some models require Hugging Face account and access token.

## Architecture

```
PDF → Extract → Clean → Chunk → Embed → Store (GPU Tensor)
Query → Embed → Search → Retrieve → Generate Answer
```

All embeddings stored as PyTorch tensors on GPU for fast similarity search.

## Persistence

The system supports automatic persistence of embeddings:

- **Auto-save**: Embeddings are automatically saved to disk after each document ingestion
- **Auto-load**: On startup, embeddings are automatically loaded if persistence files exist
- **Manual control**: Use API endpoints to manually save/load/clear embeddings

Persistence files are stored as:
- `.pt` file: PyTorch tensor with embeddings
- `.json` file: Metadata (text, source, doc_id, etc.)

By default, embeddings are saved to `data/embeddings.pt` and `data/embeddings.json`. Configure via `PERSISTENCE_PATH` environment variable.

