/**
 * Hugging Face / Python Service Client
 * 
 * HTTP client for communicating with the Python RAG service.
 * Handles PDF ingestion, text ingestion, and querying.
 */

const PYTHON_SERVICE_URL =
  process.env.NEXT_PUBLIC_PYTHON_SERVICE_URL ?? "http://localhost:8000";

/**
 * Ingest a PDF file into the RAG system.
 * 
 * @param file - PDF file to upload
 * @param source - Optional source identifier
 * @returns Result with doc_id, chunks, and pages
 */
export async function ingestPDF(
  file: File,
  source?: string,
): Promise<{
  success: boolean;
  doc_id: string;
  chunks: number;
  pages: number;
  metadata?: Record<string, any>;
}> {
  const formData = new FormData();
  formData.append("file", file);
  if (source) {
    formData.append("source", source);
  }

  const response = await fetch(`${PYTHON_SERVICE_URL}/api/ingest/pdf`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Ingest plain text into the RAG system.
 * 
 * @param text - Text content to ingest
 * @param source - Optional source identifier
 * @returns Result with doc_id and chunks
 */
export async function ingestText(
  text: string,
  source?: string,
): Promise<{
  success: boolean;
  doc_id: string;
  chunks: number;
}> {
  const body: Record<string, string> = { text };
  if (source) {
    body.source = source;
  }

  const response = await fetch(`${PYTHON_SERVICE_URL}/api/ingest/text`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify(body),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Ingest a standalone image file into the RAG system.
 * 
 * @param file - Image file to upload
 * @param source - Optional source identifier
 * @returns Result with doc_id and chunks
 */
export async function ingestImage(
  file: File,
  source?: string,
): Promise<{
  success: boolean;
  doc_id: string;
  chunks: number;
  metadata?: Record<string, any>;
}> {
  const formData = new FormData();
  formData.append("file", file);
  if (source) {
    formData.append("source", source);
  }

  const response = await fetch(`${PYTHON_SERVICE_URL}/api/ingest/image`, {
    method: "POST",
    body: formData,
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Query the RAG system with a question.
 * 
 * @param message - User's question
 * @param topK - Number of context chunks to retrieve (default: 5)
 * @param minSimilarity - Minimum similarity score threshold (default: 0.0)
 * @returns Answer with contexts and sources
 */
export async function queryRAG(
  message: string,
  topK: number = 5,
  minSimilarity: number = 0.0,
): Promise<{
  answer: string;
  contexts: Array<{
    text: string;
    score: number;
    source: string;
    chunk_index: number;
    line_start?: number;
    line_end?: number;
    highlights?: Array<{
      line_start: number;
      line_end: number;
      char_start: number;
      char_end: number;
    }>;
    highlight_count?: number;
  }>;
  sources: string[];
}> {
  const response = await fetch(`${PYTHON_SERVICE_URL}/api/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      message,
      top_k: topK,
      min_similarity: minSimilarity,
    }),
  });

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Get statistics about the stored data.
 * 
 * @returns Statistics including total chunks and embedding dimension
 */
export async function getStats(): Promise<{
  total_chunks: number;
  embedding_dim: number;
  device: string;
}> {
  const response = await fetch(`${PYTHON_SERVICE_URL}/api/stats`);

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: "Unknown error" }));
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return response.json();
}

/**
 * Health check for the Python service.
 * 
 * @returns Health status and stats
 */
export async function healthCheck(): Promise<{
  status: string;
  stats?: {
    total_chunks: number;
    embedding_dim: number;
    device: string;
  };
}> {
  const response = await fetch(`${PYTHON_SERVICE_URL}/api/health`);

  if (!response.ok) {
    throw new Error(`Health check failed: HTTP ${response.status}`);
  }

  return response.json();
}

