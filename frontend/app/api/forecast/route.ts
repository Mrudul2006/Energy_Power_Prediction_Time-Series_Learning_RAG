import { NextResponse } from "next/server";
import {
  createFallbackForecast,
  normalizeForecastResponse,
  sanitizeForecastRequest,
  type ForecastRequest
} from "@/lib/forecast";

export const dynamic = "force-dynamic";

const DEFAULT_API_BASE_URL = "http://localhost:8000";
const DEFAULT_FORECAST_PATH = "/predict";
const DEFAULT_TIMEOUT_MS = 8000;

export async function GET() {
  const upstreamUrl = getUpstreamUrl();

  return NextResponse.json({
    ok: true,
    proxy: "/api/forecast",
    upstream: upstreamUrl.origin,
    path: upstreamUrl.pathname,
    mode: upstreamUrl.hostname === "localhost" || upstreamUrl.hostname === "127.0.0.1" ? "local" : "remote"
  });
}

export async function POST(request: Request) {
  const payload = sanitizeForecastRequest(await readRequestBody(request));
  const upstreamUrl = getUpstreamUrl();

  try {
    const upstreamPayload = await postToUpstream(upstreamUrl, payload);
    const normalized = normalizeForecastResponse(upstreamPayload, payload, {
      source: "api",
      endpoint: upstreamUrl.origin
    });

    return NextResponse.json(normalized, {
      headers: {
        "cache-control": "no-store"
      }
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : "Unable to reach forecast API.";
    const fallback = createFallbackForecast(payload, {
      source: "demo",
      endpoint: upstreamUrl.origin,
      warning: message
    });

    return NextResponse.json(fallback, {
      headers: {
        "cache-control": "no-store",
        "x-energycast-fallback": "true"
      }
    });
  }
}

async function readRequestBody(request: Request): Promise<Partial<ForecastRequest>> {
  try {
    const body = (await request.json()) as unknown;
    return typeof body === "object" && body !== null ? (body as Partial<ForecastRequest>) : {};
  } catch {
    return {};
  }
}

async function postToUpstream(url: URL, payload: ForecastRequest): Promise<unknown> {
  const controller = new AbortController();
  const timeout = windowlessTimeout(() => controller.abort(), getTimeoutMs());

  try {
    const response = await fetch(url, {
      method: "POST",
      headers: {
        "content-type": "application/json",
        accept: "application/json",
        "x-energycast-client": "next-proxy"
      },
      body: JSON.stringify(payload),
      signal: controller.signal,
      cache: "no-store"
    });

    if (!response.ok) {
      throw new Error(`Forecast API returned ${response.status}`);
    }

    return await response.json();
  } finally {
    clearTimeout(timeout);
  }
}

function getUpstreamUrl(): URL {
  const baseUrl = normalizeBaseUrl(process.env.MODEL_API_BASE_URL ?? process.env.NEXT_PUBLIC_MODEL_API_BASE_URL);
  const path = process.env.MODEL_API_FORECAST_PATH ?? DEFAULT_FORECAST_PATH;
  return new URL(path, baseUrl);
}

function normalizeBaseUrl(value: string | undefined): string {
  const base = value && value.trim().length > 0 ? value.trim() : DEFAULT_API_BASE_URL;
  return base.endsWith("/") ? base : `${base}/`;
}

function getTimeoutMs(): number {
  const parsed = Number(process.env.MODEL_API_TIMEOUT_MS);
  return Number.isFinite(parsed) && parsed > 0 ? parsed : DEFAULT_TIMEOUT_MS;
}

function windowlessTimeout(callback: () => void, ms: number): ReturnType<typeof setTimeout> {
  return setTimeout(callback, ms);
}
