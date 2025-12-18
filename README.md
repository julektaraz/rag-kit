# Local RAG Pipeline

A complete local RAG (Retrieval-Augmented Generation) pipeline built from scratch with Python and Next.js. Everything runs on your local machine with GPU acceleration — no cloud APIs, no data leaves your computer.

## Features

- **Local-first**: All processing happens on your machine. Fast, private, and fully under your control.
- **Complete RAG Pipeline**: PDF ingestion → sentence-based chunking → GPU embeddings → PyTorch tensor storage → retrieval → LLM generation
- **PDF Support**: Upload PDFs directly - automatic text extraction and processing
- **GPU-Accelerated**: PyTorch tensors on GPU for fast similarity search
- **Sentence-based Chunking**: 10-15 sentences per chunk for better semantic context

## Prerequisites

- **Python 3.10+** with pip
- **NVIDIA GPU** with CUDA support (recommended, but CPU works too)
- **Node.js 18+** and npm
- **CUDA Toolkit** (if using GPU)
- **Hugging Face Account** (optional, for gated models)

## Architecture

```
PDF → PyMuPDF Extract → Clean → Sentence Chunking (10-15 sentences)
  → Sentence-Transformers Embed → PyTorch Tensor (GPU)
  
Query → Embed → Cosine Similarity Search (GPU) → Top-K Retrieval
  → Format Prompt → Quantized LLM (4-bit) → Answer
```

## Quick Start

### 1. Setup Python Service

```bash
cd python-service

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download NLTK data (first time only)
python -c "import nltk; nltk.download('punkt')"
```

### 2. Configure Environment (Optional)

Create `python-service/.env`:

```env
EMBEDDING_MODEL=sentence-transformers/all-mpnet-base-v2
LLM_MODEL=mistralai/Mistral-7B-Instruct-v0.2
USE_QUANTIZATION=true
PORT=8000
```

### 3. Start Python Service

```bash
cd python-service
python app.py
```

The service will run on `http://localhost:8000`

### 4. Setup Next.js Frontend

In the root directory:

```bash
# Install dependencies
npm install

# Configure Python service URL (optional)
# Create .env.local:
# NEXT_PUBLIC_PYTHON_SERVICE_URL=http://localhost:8000
```

### 5. Start Next.js Dev Server

```bash
npm run dev
```

Visit `http://localhost:3000` and start uploading PDFs!

## Project Structure

```
ragkit/
├── python-service/          # Python RAG backend
│   ├── ingestion/           # PDF extraction, cleaning, chunking
│   ├── models/              # Embeddings and LLM inference
│   ├── retrieval/           # Tensor storage and similarity search
│   ├── pipeline/            # Main RAG orchestrator
│   ├── app.py               # FastAPI server
│   └── requirements.txt
├── lib/
│   ├── hf_client.ts         # Python service HTTP client
│   └── ...                  # (old files can be removed)
├── app/
│   └── api/
│       ├── ingest/          # Document ingestion endpoint
│       ├── chat/           # Query endpoint
│       └── health/          # Health check
├── components/
│   └── Chat.tsx            # Main UI with PDF upload
└── package.json
```

## Usage

### Upload PDFs

1. Click or drag & drop a PDF file in the upload area
2. The system will:
   - Extract text from all pages
   - Clean and normalize the text
   - Chunk into 10-15 sentence groups
   - Generate embeddings
   - Store in PyTorch tensors on GPU

### Upload Text

1. Paste text in the text area
2. Click "Index Text"
3. Same processing pipeline as PDFs

### Query Documents

1. Type your question in the chat
2. The system will:
   - Embed your query
   - Search for similar chunks (cosine similarity on GPU)
   - Retrieve top 5 most relevant contexts
   - Generate answer using LLM with retrieved context

## Configuration

### Embedding Models

- `sentence-transformers/all-mpnet-base-v2` (default) - Good balance
- `sentence-transformers/all-MiniLM-L6-v2` - Faster, smaller
- `BAAI/bge-small-en-v1.5` - High quality

### LLM Models

- `mistralai/Mistral-7B-Instruct-v0.2` (default) - Good quality
- `google/gemma-2b-it` - Smaller, faster
- `google/gemma-7b-it` - Larger, better quality

**Note**: Some models require Hugging Face account and access token.

### Quantization

4-bit quantization (NF4) is enabled by default to fit models in GPU memory. Disable in `.env` if you have enough VRAM:

```env
USE_QUANTIZATION=false
```

## API Endpoints

### Python Service (`http://localhost:8000`)

- `POST /api/ingest/pdf` - Upload PDF file
- `POST /api/ingest/text` - Upload plain text
- `POST /api/chat` - Query the RAG system
- `GET /api/stats` - Get storage statistics
- `GET /api/health` - Health check

### Next.js API Routes

- `POST /api/ingest` - Document ingestion (forwards to Python)
- `POST /api/chat` - Query endpoint (forwards to Python)
- `GET /api/health` - Health check

## Development

### Python Service

```bash
cd python-service
python app.py
# Or with uvicorn:
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Next.js

```bash
npm run dev
```

## Notes

- **First Run**: Models will be downloaded from Hugging Face on first use (can be several GB)
- **GPU Memory**: 4-bit quantization helps fit 7B models in ~8GB VRAM
- **Storage**: Embeddings stored in GPU memory as PyTorch tensors (fast but volatile)
- **Persistence**: Embeddings can be saved to disk and auto-loaded on startup

## Troubleshooting

### Python service won't start
- Check Python version: `python --version` (need 3.10+)
- Install CUDA toolkit if using GPU
- Check GPU availability: `python -c "import torch; print(torch.cuda.is_available())"`

### Models won't download
- Check internet connection
- For gated models, set `HUGGING_FACE_TOKEN` in `.env`
- Some models require Hugging Face account

### Out of memory errors
- Enable quantization: `USE_QUANTIZATION=true`
- Use smaller models
- Reduce batch sizes in code
