import { NextResponse } from "next/server";
import { z } from "zod";
import { queryRAG } from "@/lib/hf_client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

const MAX_MESSAGE_LENGTH = 10000; // 10k characters max

const ChatSchema = z.object({
  message: z
    .string()
    .min(1, "Message is required")
    .max(MAX_MESSAGE_LENGTH, `Message must be less than ${MAX_MESSAGE_LENGTH} characters`),
  top_k: z.number().int().min(1).max(20).optional().default(5),
  min_similarity: z.number().min(0).max(1).optional().default(0.0),
});

/**
 * Chat endpoint - forwards queries to Python RAG service.
 */
export async function POST(request: Request) {
  try {
    const json = await request.json();
    const parsed = ChatSchema.safeParse(json);

    if (!parsed.success) {
      return NextResponse.json(
        { error: "Validation failed", details: parsed.error.format() },
        { status: 400 },
      );
    }

    const { answer, contexts, sources } = await queryRAG(
      parsed.data.message,
      parsed.data.top_k,
      parsed.data.min_similarity,
    );

    return NextResponse.json({ answer, contexts, sources });
  } catch (error) {
    console.error("Chat route error", error);
    const errorMessage =
      error instanceof Error ? error.message : "Failed to generate answer";
    return NextResponse.json(
      { error: errorMessage },
      { status: 500 },
    );
  }
}
