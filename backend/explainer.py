"""
LLM-powered explanation service for the EnergyCast dashboard.

Auto-detects the best available backend in order:
  1. Anthropic Claude Haiku — fast, ~$0.0001/call, needs ANTHROPIC_API_KEY
  2. Ollama — local, free, needs Ollama running on localhost:11434 (or OLLAMA_BASE_URL)
  3. Template fallback — always available, zero-latency rule-based insights

Usage
─────
    service = await ExplainerService.create()
    result  = await service.explain("dlinear", context)

ExplainContext is assembled by the /api/explain endpoint from the pipeline's
existing outputs (metrics, forecast rows, drift_status, feature_importance).
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── LLM model constants ───────────────────────────────────────────────────────

GEMINI_MODEL = "gemini-2.0-flash-exp"
OLLAMA_PREFERRED = ["qwen2.5:0.5b", "tinyllama", "phi3:mini", "gemma2:2b", "llama3.2:1b"]
CACHE_TTL_SECONDS = 300   # 5 minutes


# ── context dataclass ─────────────────────────────────────────────────────────

@dataclass
class ExplainContext:
    """All pipeline outputs bundled for the LLM."""
    model_name:          str
    display_name:        str
    # metrics
    mae:                 float
    rmse:                float
    mape:                float
    online_updates:      int
    # forecast outlook
    forecast_peak_mw:    float
    forecast_peak_time:  str
    forecast_trough_mw:  float
    forecast_trough_time: str
    forecast_avg_mw:     float
    forecast_trend_pct:  float      # % change first → last forecast point
    current_mw:          float      # most recent actual observation
    # anomalies  [{time, deviation, actual, predicted}]
    top_anomalies:       List[Dict[str, Any]] = field(default_factory=list)
    # drift
    drift_count:         int  = 0
    ph_stat:             float = 0.0
    ph_threshold:        float = 50.0
    is_alarming:         bool  = False
    # features [{feature, score}]
    top_features:        List[Dict[str, Any]] = field(default_factory=list)


# ── prompts ───────────────────────────────────────────────────────────────────

_SYSTEM = (
    "You are an expert energy grid operations analyst. "
    "Interpret electricity consumption forecast data and provide concise, "
    "actionable insights for grid operators. "
    "Use specific numbers from the data. Be direct. "
    "2-3 sentences per section maximum."
)

_JSON_SCHEMA = (
    '{"forecast_summary":"...","anomaly_insight":"...","model_health":"...",'
    '"drift_interpretation":"...","top_recommendation":"...","feature_insight":"..."}'
)

def _build_prompt(ctx: ExplainContext) -> str:
    anomalies = "\n".join(
        f"  • {a['time']}: {a['deviation']:+.1f}% ({a['actual']:.0f} MW vs {a['predicted']:.0f} MW predicted)"
        for a in ctx.top_anomalies[:3]
    ) or "  • None significant"

    features = "\n".join(
        f"  • {f['feature']}: {f['score']:.0%} of model weight"
        for f in ctx.top_features[:4]
    ) or "  • Not available"

    drift_state = "⚠ ALARMING" if ctx.is_alarming else "stable"

    return f"""\
MODEL: {ctx.display_name} | MAE={ctx.mae:.1f} MW | RMSE={ctx.rmse:.1f} MW | MAPE={ctx.mape:.2f}%
ONLINE_UPDATES: {ctx.online_updates} adaptive fine-tunes since deployment
CURRENT_LOAD: {ctx.current_mw:.0f} MW

FORECAST (next 168 hours):
  Peak  : {ctx.forecast_peak_mw:.0f} MW at {ctx.forecast_peak_time}
  Trough: {ctx.forecast_trough_mw:.0f} MW at {ctx.forecast_trough_time}
  Average: {ctx.forecast_avg_mw:.0f} MW | Trend: {ctx.forecast_trend_pct:+.1f}% vs now

CONCEPT DRIFT: {ctx.drift_count} events | detector stat={ctx.ph_stat:.1f} / threshold={ctx.ph_threshold:.0f} | {drift_state}

RECENT ANOMALIES:
{anomalies}

TOP MODEL INPUT FEATURES:
{features}

Respond ONLY with valid JSON matching this schema — no markdown, no extra text:
{_JSON_SCHEMA}"""


# ── abstract backend ──────────────────────────────────────────────────────────

class LLMBackend(ABC):
    """Every concrete backend exposes a single async method."""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    async def complete(self, system: str, user: str, max_tokens: int = 512) -> str: ...

    @abstractmethod
    async def is_available(self) -> bool: ...


# ── Gemini backend ────────────────────────────────────────────────────────────

class GeminiBackend(LLMBackend):
    """Gemini Flash via the official REST API."""

    def __init__(self) -> None:
        self._api_key = "AIzaSyCtijiehUWhLrqpoCzIz_bNYjFQEKK1Ubk"

    @property
    def name(self) -> str:
        return f"gemini ({GEMINI_MODEL})"

    async def is_available(self) -> bool:
        if not self._api_key:
            return False
        import httpx
        return True

    async def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        import httpx

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{GEMINI_MODEL}:generateContent?key={self._api_key}"
        payload = {
            "system_instruction": {"parts": [{"text": system}]},
            "contents": [{"parts": [{"text": user}]}],
            "generationConfig": {
                "maxOutputTokens": max_tokens,
                "temperature": 0.3
            }
        }
        
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            return data["candidates"][0]["content"]["parts"][0]["text"]


# ── Ollama backend ────────────────────────────────────────────────────────────

class OllamaBackend(LLMBackend):
    """Local Ollama inference — auto-picks the lightest available model."""

    def __init__(self) -> None:
        self._base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self._model: Optional[str] = None

    @property
    def name(self) -> str:
        return f"ollama/{self._model or 'unknown'}"

    async def is_available(self) -> bool:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=3.0) as c:
                r = await c.get(f"{self._base}/api/tags")
                if r.status_code != 200:
                    return False
                tags = r.json().get("models", [])
                available = {m["name"].split(":")[0] for m in tags}
                available_full = {m["name"] for m in tags}
                for pref in OLLAMA_PREFERRED:
                    base = pref.split(":")[0]
                    if pref in available_full:
                        self._model = pref
                        return True
                    if base in available:
                        # Pick the first matching tag
                        self._model = next(m["name"] for m in tags if m["name"].startswith(base))
                        return True
                # Any model is better than none
                if tags:
                    self._model = tags[0]["name"]
                    return True
                return False
        except Exception:
            return False

    async def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        import httpx

        payload = {
            "model":  self._model,
            "prompt": f"<system>\n{system}\n</system>\n\n{user}",
            "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.3},
        }
        async with httpx.AsyncClient(timeout=60.0) as c:
            r = await c.post(f"{self._base}/api/generate", json=payload)
            r.raise_for_status()
            return r.json()["response"]


# ── template fallback ─────────────────────────────────────────────────────────

class TemplateFallback(LLMBackend):
    """
    Zero-latency rule-based insights — always available.
    Produces readable (if less nuanced) explanations from the raw numbers.
    """

    @property
    def name(self) -> str:
        return "template (no LLM)"

    async def is_available(self) -> bool:
        return True

    async def complete(self, system: str, user: str, max_tokens: int = 512) -> str:
        # Re-parse the context numbers from the prompt text we built.
        # Simpler than passing the context object through; good enough for fallback.
        import re

        def _get(pattern: str, default: str = "N/A") -> str:
            m = re.search(pattern, user)
            return m.group(1) if m else default

        mae        = _get(r"MAE=([\d.]+)")
        mape       = _get(r"MAPE=([\d.]+)")
        peak_mw    = _get(r"Peak\s*:\s*([\d.]+)")
        peak_time  = _get(r"Peak\s*:\s*[\d.]+ MW at (.+)")
        trough_mw  = _get(r"Trough\s*:\s*([\d.]+)")
        avg_mw     = _get(r"Average:\s*([\d.]+)")
        trend      = _get(r"Trend:\s*([+\-\d.]+)%")
        drift_stat = _get(r"stat=([\d.]+)")
        drift_thr  = _get(r"threshold=([\d.]+)")
        alarming   = "ALARMING" in user

        drift_msg = (
            f"The drift statistic ({drift_stat}) has exceeded the threshold ({drift_thr}), "
            "indicating the consumption pattern has shifted — the model has been fine-tuned to adapt."
            if alarming else
            f"The drift statistic ({drift_stat}) is below the threshold ({drift_thr}), "
            "so the model is tracking the current consumption regime reliably."
        )

        model_name = _get(r'MODEL: (\S+)')
        result = {
            "forecast_summary": (
                f"Grid consumption is forecast to peak at {peak_mw} MW around {peak_time}, "
                f"with a daily trough near {trough_mw} MW. "
                f"The 7-day average of {avg_mw} MW shows a {trend}% trend vs current load."
            ),
            "anomaly_insight": (
                "One or more consumption anomalies were detected in the recent window. "
                "Verify against scheduled industrial loads or weather events before acting."
                if "• " in user and "None significant" not in user else
                "No significant anomalies detected in the recent observation window. "
                "Forecast residuals are within normal operating bounds."
            ),
            "model_health": (
                f"The {model_name} model is operating with MAE={mae} MW and MAPE={mape}%. "
                f"At typical grid loads this translates to roughly {mape}% average forecast error, "
                "which is considered acceptable for day-ahead dispatch planning."
            ),
            "drift_interpretation": drift_msg,
            "top_recommendation": (
                f"Prepare additional reserve capacity for the peak period around {peak_time} "
                f"({peak_mw} MW forecast). "
                "Consider scheduling flexible loads to the trough period to flatten the daily curve."
            ),
            "feature_insight": (
                "The model relies most on recent consumption patterns from the past 1-2 weeks, "
                "indicating strong autocorrelation in grid load. "
                "Day-of-week and hour-of-day signals dominate short-range deviations."
            ),
        }
        return json.dumps(result)


# ── explainer service ─────────────────────────────────────────────────────────

class ExplainerService:
    """
    Builds prompts, calls the active LLM backend, caches results.

    Call ``ExplainerService.create()`` (async) to get an instance with the
    best available backend already detected.
    """

    def __init__(self, backend: LLMBackend) -> None:
        self.backend = backend
        self._cache: Dict[str, tuple] = {}   # model_name → (result_dict, timestamp)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    async def create(cls) -> "ExplainerService":
        """Auto-detect and initialise the best available LLM backend."""
        for candidate in [GeminiBackend(), OllamaBackend(), TemplateFallback()]:
            if await candidate.is_available():
                log.info("ExplainerService using backend: %s", candidate.name)
                return cls(candidate)
        return cls(TemplateFallback())   # always succeeds

    # ── public API ────────────────────────────────────────────────────────────

    def invalidate(self, model_name: str) -> None:
        """Drop cached explanation (call after drift / fine-tune)."""
        self._cache.pop(model_name, None)

    def get_cached(self, model_name: str) -> Optional[Dict[str, Any]]:
        entry = self._cache.get(model_name)
        if entry and (time.time() - entry[1]) < CACHE_TTL_SECONDS:
            return entry[0]
        return None

    async def explain(self, model_name: str, ctx: ExplainContext) -> Dict[str, Any]:
        """Return cached result or generate a fresh one from the LLM."""
        cached = self.get_cached(model_name)
        if cached:
            return cached

        prompt = _build_prompt(ctx)
        try:
            raw = await self.backend.complete(_SYSTEM, prompt, max_tokens=600)
            sections = _parse_json(raw)
        except Exception as exc:
            log.warning("LLM backend error (%s): %s — using template", self.backend.name, exc)
            fallback = TemplateFallback()
            raw = await fallback.complete(_SYSTEM, prompt)
            sections = _parse_json(raw)
            backend_name = "template (fallback: API Quota Exceeded/Error)"

        result = {
            "model":        model_name,
            "backend":      backend_name if 'backend_name' in locals() else self.backend.name,
            "generated_at": _now_iso(),
            "sections":     _to_sections(sections),
        }
        self._cache[model_name] = (result, time.time())
        return result


# ── helpers ───────────────────────────────────────────────────────────────────

def _parse_json(raw: str) -> Dict[str, str]:
    """Extract the JSON object from the LLM response (handles markdown fences)."""
    raw = raw.strip()
    # Strip markdown fences
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    # Find first { ... }
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError("No JSON object found in LLM response")
    return json.loads(raw[start:end])


_SECTION_META = {
    "forecast_summary":      {"title": "Forecast Summary",      "icon": "chart-line"},
    "anomaly_insight":       {"title": "Anomaly Analysis",      "icon": "alert-triangle"},
    "model_health":          {"title": "Model Health",          "icon": "activity"},
    "drift_interpretation":  {"title": "Drift Interpretation",  "icon": "refresh-cw"},
    "top_recommendation":    {"title": "Recommendation",        "icon": "zap"},
    "feature_insight":       {"title": "Pattern Insights",      "icon": "layers"},
}

def _to_sections(d: Dict[str, str]) -> List[Dict[str, Any]]:
    return [
        {
            "id":      key,
            "title":   _SECTION_META.get(key, {}).get("title", key),
            "icon":    _SECTION_META.get(key, {}).get("icon", "info"),
            "content": str(d.get(key, "")),
        }
        for key in _SECTION_META
        if key in d
    ]


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(tz=timezone.utc).isoformat()
