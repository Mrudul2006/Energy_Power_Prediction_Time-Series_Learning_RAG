import { NextRequest, NextResponse } from "next/server";

const BASE_URL = process.env.MODEL_API_BASE_URL ?? "http://localhost:8000";
const TIMEOUT_MS = Number(process.env.MODEL_API_TIMEOUT_MS ?? "15000");

export async function POST(request: NextRequest): Promise<NextResponse> {
  let body: unknown;
  try {
    body = await request.json();
  } catch {
    return NextResponse.json({ error: "Invalid request body" }, { status: 400 });
  }

  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), TIMEOUT_MS);

  try {
    const upstream = await fetch(`${BASE_URL}/explain`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(body),
      signal: controller.signal,
    });

    clearTimeout(timer);

    if (!upstream.ok) {
      const text = await upstream.text().catch(() => "");
      return NextResponse.json(
        { error: `Backend returned ${upstream.status}`, detail: text },
        { status: upstream.status }
      );
    }

    const data = await upstream.json();
    return NextResponse.json(data);
  } catch (error) {
    clearTimeout(timer);
    const message = error instanceof Error ? error.message : "Explain request failed";
    return NextResponse.json({ error: message }, { status: 503 });
  }
}
