import { NextResponse } from "next/server";

const DEFAULT_API_BASE_URL = "http://localhost:8000";

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const start = searchParams.get("start");
  const end = searchParams.get("end");
  
  if (!start || !end) return NextResponse.json({ history: [] });

  try {
    const upstream = `${DEFAULT_API_BASE_URL}/api/history?start=${encodeURIComponent(start)}&end=${encodeURIComponent(end)}`;
    const response = await fetch(upstream, { cache: "no-store" });
    if (!response.ok) throw new Error("bad response from backend");
    const data = await response.json();
    return NextResponse.json(data);
  } catch (err) {
    return NextResponse.json({ history: [] });
  }
}
