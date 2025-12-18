"use client";

import { useState, useRef } from "react";
import { ingestPDF, ingestText, ingestImage, queryRAG } from "@/lib/hf_client";

type Message = {
  role: "user" | "assistant";
  content: string;
  contexts?: Array<{
    text: string;
    score: number;
    source: string;
    chunk_index: number;
    highlights?: Array<{ char_start: number; char_end: number }>;
  }>;
  sources?: string[];
};

export default function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [docText, setDocText] = useState("");
  const [loading, setLoading] = useState(false);
  const [embedding, setEmbedding] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadingImage, setUploadingImage] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [success, setSuccess] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageInputRef = useRef<HTMLInputElement>(null);

  const hasMessages = messages.length > 0;

  /**
   * Handle sending a chat message to the RAG system.
   */
  const handleSend = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setError(null);
    setSuccess(null);

    const userMessage = input.trim();
    const nextMessages: Message[] = [
      ...messages,
      { role: "user", content: userMessage },
    ];
    setMessages(nextMessages);
    setInput("");

    try {
      const data = await queryRAG(userMessage);

      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer || "No response generated",
          contexts: data.contexts,
          sources: data.sources,
        },
      ]);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(errorMessage);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Handle embedding plain text.
   */
  const handleEmbed = async () => {
    if (!docText.trim()) return;
    setEmbedding(true);
    setError(null);
    setSuccess(null);

    try {
      const data = await ingestText(docText);

      setSuccess(
        `Document embedded successfully! ${data.chunks || 0} chunks indexed.`,
      );
      setDocText("");
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(errorMessage);
    } finally {
      setEmbedding(false);
    }
  };

  /**
   * Handle PDF file upload.
   */
  const handleFileUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Please upload a PDF file");
      return;
    }

    // Validate file size (50MB max)
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
    if (file.size > MAX_FILE_SIZE) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      return;
    }

    if (file.size === 0) {
      setError("File is empty");
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      const data = await ingestPDF(file, file.name);

      setSuccess(
        `PDF uploaded successfully! ${data.chunks || 0} chunks from ${data.pages || 0} pages indexed.`,
      );
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(errorMessage);
    } finally {
      setUploading(false);
      // Reset file input
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  /**
   * Handle image file upload.
   */
  const handleImageUpload = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    // Validate file type
    const allowedTypes = ["image/jpeg", "image/jpg", "image/png", "image/gif", "image/bmp", "image/webp"];
    if (!allowedTypes.includes(file.type)) {
      setError("Please upload a valid image file (JPG, PNG, GIF, BMP, or WEBP)");
      return;
    }

    // Validate file size (50MB max)
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
    if (file.size > MAX_FILE_SIZE) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      return;
    }

    if (file.size === 0) {
      setError("File is empty");
      return;
    }

    setUploadingImage(true);
    setError(null);
    setSuccess(null);

    try {
      const data = await ingestImage(file, file.name);

      setSuccess(
        `Image uploaded successfully! ${data.chunks || 0} chunks indexed from OCR text.`,
      );
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(errorMessage);
    } finally {
      setUploadingImage(false);
      // Reset file input
      if (imageInputRef.current) {
        imageInputRef.current.value = "";
      }
    }
  };

  /**
   * Handle drag and drop for files.
   */
  const handleDrop = async (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) {
      setError("Please drop a file");
      return;
    }

    // Validate file size (50MB max)
    const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB
    if (file.size > MAX_FILE_SIZE) {
      setError(`File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB`);
      return;
    }

    if (file.size === 0) {
      setError("File is empty");
      return;
    }

    setUploading(true);
    setError(null);
    setSuccess(null);

    try {
      if (file.name.toLowerCase().endsWith(".pdf")) {
        const data = await ingestPDF(file, file.name);
        setSuccess(
          `PDF uploaded successfully! ${data.chunks || 0} chunks from ${data.pages || 0} pages indexed.`,
        );
      } else if (file.type.startsWith("image/")) {
        setUploading(false);
        setUploadingImage(true);
        const data = await ingestImage(file, file.name);
        setSuccess(
          `Image uploaded successfully! ${data.chunks || 0} chunks indexed from OCR text.`,
        );
        setUploadingImage(false);
      } else {
        setError("Unsupported file type. Please upload a PDF or image file.");
        setUploading(false);
        return;
      }
      setTimeout(() => setSuccess(null), 5000);
    } catch (err) {
      const errorMessage =
        err instanceof Error
          ? err.message
          : "An unexpected error occurred. Please try again.";
      setError(errorMessage);
    } finally {
      setUploading(false);
      setUploadingImage(false);
    }
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
  };

  /**
   * Highlight query terms in text.
   */
  const highlightText = (text: string, highlights?: Array<{ char_start: number; char_end: number }>) => {
    if (!highlights || highlights.length === 0) {
      return text;
    }

    // Sort highlights by start position
    const sortedHighlights = [...highlights].sort((a, b) => a.char_start - b.char_start);
    
    // Build highlighted text
    let result = [];
    let lastIndex = 0;
    
    for (const highlight of sortedHighlights) {
      // Add text before highlight
      if (highlight.char_start > lastIndex) {
        result.push(text.slice(lastIndex, highlight.char_start));
      }
      // Add highlighted text
      result.push(
        <mark key={highlight.char_start} className="bg-yellow-200 text-yellow-900 px-0.5 rounded">
          {text.slice(highlight.char_start, highlight.char_end)}
        </mark>
      );
      lastIndex = highlight.char_end;
    }
    
    // Add remaining text
    if (lastIndex < text.length) {
      result.push(text.slice(lastIndex));
    }
    
    return result;
  };

  return (
    <div className="mx-auto flex min-h-screen w-full max-w-6xl flex-col gap-8 bg-gradient-to-br from-slate-50 via-white to-slate-100 px-6 py-12 text-slate-900">
      <header className="space-y-3">
        <p className="text-sm font-semibold uppercase tracking-[0.2em] text-indigo-600">
          Local RAG
        </p>
        <h1 className="text-4xl font-bold bg-gradient-to-r from-indigo-600 to-purple-600 bg-clip-text text-transparent sm:text-5xl">
          Ask your documents
        </h1>
        <p className="max-w-2xl text-slate-600 text-base">
          Upload PDFs, images, or paste text. Documents are chunked, embedded, and stored
          in PyTorch tensors on GPU. Query using retrieval-augmented generation.
        </p>
      </header>

      <section className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="col-span-1 space-y-4">
          {/* PDF Upload Section */}
          <div className="rounded-xl border border-slate-200 bg-white/80 p-5 shadow-md backdrop-blur-sm transition-all hover:shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-semibold text-slate-900">Upload PDF</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  Drag & drop or click to upload
                </p>
              </div>
              <span className="rounded-full bg-indigo-100 px-3 py-1 text-xs font-medium text-indigo-700">
                PDF
              </span>
            </div>

            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="rounded-lg border-2 border-dashed border-slate-300 bg-slate-50/50 p-6 text-center transition-all hover:border-indigo-400 hover:bg-indigo-50/30 cursor-pointer"
            >
              <input
                ref={fileInputRef}
                type="file"
                accept=".pdf"
                onChange={handleFileUpload}
                className="hidden"
                id="pdf-upload"
              />
              <label
                htmlFor="pdf-upload"
                className="cursor-pointer"
              >
                <svg
                  className="mx-auto h-10 w-10 text-slate-400 mb-2"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m7-4h-4m-4-4h4m-4 4v4m4-4v-4"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="text-sm text-slate-600 font-medium">
                  {uploading ? "Uploading..." : "Click or drag PDF here"}
                </p>
              </label>
            </div>
          </div>

          {/* Image Upload Section */}
          <div className="rounded-xl border border-slate-200 bg-white/80 p-5 shadow-md backdrop-blur-sm transition-all hover:shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-semibold text-slate-900">Upload Image</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  JPG, PNG, GIF, BMP, WEBP
                </p>
              </div>
              <span className="rounded-full bg-purple-100 px-3 py-1 text-xs font-medium text-purple-700">
                Image
              </span>
            </div>

            <div
              onDrop={handleDrop}
              onDragOver={handleDragOver}
              className="rounded-lg border-2 border-dashed border-slate-300 bg-slate-50/50 p-6 text-center transition-all hover:border-purple-400 hover:bg-purple-50/30 cursor-pointer"
            >
              <input
                ref={imageInputRef}
                type="file"
                accept="image/*"
                onChange={handleImageUpload}
                className="hidden"
                id="image-upload"
              />
              <label
                htmlFor="image-upload"
                className="cursor-pointer"
              >
                <svg
                  className="mx-auto h-10 w-10 text-slate-400 mb-2"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M28 8H12a4 4 0 00-4 4v20m32-12v8m0 0v8a4 4 0 01-4 4H12a4 4 0 01-4-4v-4m32-4l-3.172-3.172a4 4 0 00-5.656 0L28 28M8 32l9.172-9.172a4 4 0 015.656 0L28 28m0 0l4 4m7-4h-4m-4-4h4m-4 4v4m4-4v-4"
                    strokeWidth={2}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  />
                </svg>
                <p className="text-sm text-slate-600 font-medium">
                  {uploadingImage ? "Processing..." : "Click or drag image here"}
                </p>
              </label>
            </div>
          </div>

          {/* Text Upload Section */}
          <div className="rounded-xl border border-slate-200 bg-white/80 p-5 shadow-md backdrop-blur-sm transition-all hover:shadow-lg">
            <div className="flex items-center justify-between mb-4">
              <div>
                <p className="text-sm font-semibold text-slate-900">Add text</p>
                <p className="text-xs text-slate-500 mt-0.5">
                  Paste any text to index
                </p>
              </div>
              <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700">
                Text
              </span>
            </div>

            <textarea
              className="mt-2 h-32 w-full rounded-lg border border-slate-200 bg-white px-3 py-2 text-sm shadow-inner outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all"
              placeholder="Paste text to index..."
              value={docText}
              onChange={(e) => setDocText(e.target.value)}
            />
            <button
              onClick={handleEmbed}
              disabled={embedding}
              className="mt-3 inline-flex w-full items-center justify-center rounded-lg bg-indigo-600 px-4 py-2.5 text-sm font-semibold text-white shadow-sm transition-all hover:bg-indigo-500 hover:shadow-md disabled:cursor-not-allowed disabled:opacity-60"
            >
              {embedding ? "Embedding..." : "Index Text"}
            </button>
          </div>

          {/* Status Messages */}
          {success && (
            <div className="rounded-lg bg-emerald-50 border border-emerald-200 px-4 py-3 animate-in fade-in slide-in-from-top-2">
              <p className="text-xs text-emerald-700 font-medium">✓ {success}</p>
            </div>
          )}
          {error && (
            <div className="rounded-lg bg-red-50 border border-red-200 px-4 py-3 animate-in fade-in slide-in-from-top-2">
              <p className="text-xs text-red-600 font-medium">Error: {error}</p>
            </div>
          )}
        </div>

        {/* Chat Section */}
        <div className="col-span-1 lg:col-span-2 rounded-xl border border-slate-200 bg-white/80 p-6 shadow-md backdrop-blur-sm">
          <div className="flex items-center justify-between pb-4 border-b border-slate-200">
            <div>
              <p className="text-sm font-semibold text-slate-900">Chat</p>
              <p className="text-xs text-slate-500 mt-0.5">
                Ask questions about your documents using RAG
              </p>
            </div>
            <span className="rounded-full bg-emerald-100 px-3 py-1 text-xs font-medium text-emerald-700">
              RAG
            </span>
          </div>

          <div className="flex flex-col gap-4 rounded-lg mt-4">
            {hasMessages ? (
              <div className="flex max-h-[500px] flex-col gap-4 overflow-y-auto pr-2">
                {messages.map((message, idx) => (
                  <div
                    key={`${message.role}-${idx}`}
                    className={`flex ${
                      message.role === "user"
                        ? "justify-end"
                        : "justify-start"
                    } animate-in fade-in slide-in-from-bottom-2`}
                  >
                    <div
                      className={`max-w-[85%] rounded-xl px-4 py-3 shadow-sm ${
                        message.role === "user"
                          ? "bg-gradient-to-br from-indigo-600 to-indigo-700 text-white"
                          : "bg-slate-50 text-slate-900 border border-slate-200"
                      }`}
                    >
                      <p className="text-xs font-semibold uppercase tracking-wide opacity-70 mb-2">
                        {message.role}
                      </p>
                      <p className="whitespace-pre-wrap text-sm leading-relaxed">{message.content}</p>
                      
                      {/* Show contexts and sources for assistant messages */}
                      {message.role === "assistant" && message.contexts && message.contexts.length > 0 && (
                        <div className="mt-4 pt-4 border-t border-slate-200/50">
                          <p className="text-xs font-semibold text-slate-600 mb-2">Sources ({message.contexts.length}):</p>
                          <div className="space-y-2">
                            {message.contexts.slice(0, 3).map((context, ctxIdx) => (
                              <div
                                key={ctxIdx}
                                className="text-xs bg-white/60 rounded-lg p-2 border border-slate-200/50"
                              >
                                <div className="flex items-center justify-between mb-1">
                                  <span className="font-medium text-slate-700">{context.source}</span>
                                  <span className="text-slate-500">{(context.score * 100).toFixed(1)}%</span>
                                </div>
                                <p className="text-slate-600 line-clamp-2">
                                  {highlightText(context.text, context.highlights)}
                                </p>
                              </div>
                            ))}
                          </div>
                          {message.sources && message.sources.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {message.sources.map((source, srcIdx) => (
                                <span
                                  key={srcIdx}
                                  className="text-xs px-2 py-0.5 bg-indigo-100 text-indigo-700 rounded-full"
                                >
                                  {source}
                                </span>
                              ))}
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="rounded-lg border border-dashed border-slate-300 bg-slate-50/50 p-8 text-center">
                <svg
                  className="mx-auto h-12 w-12 text-slate-400 mb-3"
                  stroke="currentColor"
                  fill="none"
                  viewBox="0 0 48 48"
                >
                  <path
                    d="M8 10h32M8 18h32M8 26h16M8 34h24"
                    strokeWidth={2}
                    strokeLinecap="round"
                  />
                </svg>
                <p className="text-sm text-slate-500 font-medium">
                  No messages yet. Upload a document, then ask a question.
                </p>
              </div>
            )}

            <div className="flex items-start gap-3 pt-2">
              <textarea
                className="min-h-[70px] flex-1 rounded-lg border border-slate-200 bg-white px-4 py-3 text-sm shadow-inner outline-none focus:border-indigo-400 focus:ring-2 focus:ring-indigo-100 transition-all resize-none"
                placeholder="Ask anything about your documents..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="inline-flex h-[70px] min-w-[100px] items-center justify-center rounded-lg bg-gradient-to-r from-indigo-600 to-purple-600 px-5 text-sm font-semibold text-white shadow-md transition-all hover:from-indigo-700 hover:to-purple-700 hover:shadow-lg disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:shadow-md"
              >
                {loading ? (
                  <span className="flex items-center gap-2">
                    <svg className="animate-spin h-4 w-4" viewBox="0 0 24 24">
                      <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" fill="none" />
                      <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z" />
                    </svg>
                    Thinking...
                  </span>
                ) : (
                  "Send"
                )}
              </button>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
