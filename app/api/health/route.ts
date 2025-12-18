import { NextResponse } from "next/server";
import { healthCheck } from "@/lib/hf_client";

export const runtime = "nodejs";
export const dynamic = "force-dynamic";

/**
 * Health check endpoint - checks Python RAG service status.
 */
export async function GET() {
  try {
    const health = await healthCheck();
    const statusCode = health.status === "healthy" ? 200 : 503;
    return NextResponse.json(health, { status: statusCode });
  } catch (error) {
    return NextResponse.json(
      {
        status: "unhealthy",
        error: error instanceof Error ? error.message : "Unknown error",
        timestamp: new Date().toISOString(),
      },
      { status: 503 },
    );
  }
}
