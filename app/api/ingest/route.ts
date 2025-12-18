import { NextResponse } from "next/server";
import { ingestPDF, ingestText } from "@/lib/hf_client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Ingest endpoint - forwards to Python service.
 * 
 * Supports both PDF uploads and text ingestion.
 */
const MAX_FILE_SIZE = 50 * 1024 * 1024; // 50MB

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get("file") as File | null;
    const text = formData.get("text") as string | null;
    const source = formData.get("source") as string | null;

    if (file) {
      // Validate file size
      if (file.size > MAX_FILE_SIZE) {
        return NextResponse.json(
          { error: `File too large. Maximum size is ${MAX_FILE_SIZE / (1024 * 1024)}MB` },
          { status: 413 },
        );
      }

      // Handle PDF upload
      const result = await ingestPDF(file, source || undefined);
      return NextResponse.json(result);
    } else if (text) {
      // Validate text length
      if (text.length > 10 * 1024 * 1024) {
        return NextResponse.json(
          { error: "Text too long. Maximum length is 10MB" },
          { status: 413 },
        );
      }

      // Handle text ingestion
      const result = await ingestText(text, source || undefined);
      return NextResponse.json(result);
    } else {
      return NextResponse.json(
        { error: "Either 'file' or 'text' must be provided" },
        { status: 400 },
      );
    }
  } catch (error) {
    // In Next.js API routes, console.error is appropriate for server-side logging
    console.error("Ingest route error", error);
    const errorMessage =
      error instanceof Error ? error.message : "Failed to ingest document";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 },
    );
  }
}

