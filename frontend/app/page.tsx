"use client";

import { FormEvent, WheelEvent, useEffect, useMemo, useRef, useState } from "react";
import {
  createFallbackForecast,
  getDefaultForecastRequest,
  type ForecastPoint,
  type ForecastResponse,
  type HourObservation,
  type PointCondition
} from "@/lib/forecast";

export interface ExplainSection {
  id: string;
  title: string;
  icon: string;
  content: string;
}

export interface ExplainResponse {
  model: string;
  backend: string;
  generated_at: string;
  sections: ExplainSection[];
}

export interface BacktestPoint {
  timestamp: string;
  actual: number;
  predicted_base: number;
  predicted_online: number | null;
}

export interface BacktestMetrics {
  mae: number;
  rmse: number;
  mape: number;
}

export interface BacktestData {
  series: BacktestPoint[];
  metrics_base: BacktestMetrics;
  metrics_online: BacktestMetrics | null;
  resolution: string;
  total_hours: number;
}

const CHART_H = 340;
const CHART_PAD_X = 58;
const CHART_PAD_Y = 28;
const CHART_PAD_BTM = 48;
const CHART_PAD_R = 24;
const BUFFER_HOURS = 24;

export default function OnlineForecastPage() {
  const [history, setHistory] = useState<HourObservation[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [timestamp, setTimestamp] = useState(() => toHourInputValue(new Date()));
  const [datasetStartDate, setDatasetStartDate] = useState("");
  const [datasetEndDate, setDatasetEndDate] = useState("");
  const [actual, setActual] = useState("");
  const [selectedIndex, setSelectedIndex] = useState(0);
  const [isLoading, setIsLoading] = useState(false);
  const [currentTime, setCurrentTime] = useState(() => new Date());
  const [explanation, setExplanation] = useState<ExplainResponse | null>(null);
  const [isExplaining, setIsExplaining] = useState(false);
  const [explainError, setExplainError] = useState("");
  const [backtestResolution, setBacktestResolution] = useState("daily");
  const [backtestData, setBacktestData] = useState<BacktestData | null>(null);
  const [isBacktestLoading, setIsBacktestLoading] = useState(false);

  const [showActual, setShowActual] = useState(true);
  const [showBase, setShowBase] = useState(true);
  const [showOnline, setShowOnline] = useState(true);

  useEffect(() => {
    async function fetchBacktest() {
      setIsBacktestLoading(true);
      try {
        const res = await fetch(`/api/2021?mode=online_replay&resolution=${backtestResolution}`);
        if (res.ok) {
          const data = await res.json();
          setBacktestData(data);
        }
      } catch (e) {
        console.error("Failed to load backtest", e);
      } finally {
        setIsBacktestLoading(false);
      }
    }
    fetchBacktest();
  }, [backtestResolution]);

  async function generateExplanation() {
    if (!forecast) return;
    setIsExplaining(true);
    setExplainError("");
    try {
      const res = await fetch("/api/explain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(forecast)
      });
      if (!res.ok) {
        throw new Error(`Explain API failed with status ${res.status}`);
      }
      const data = await res.json();
      if (data.error) {
        throw new Error(data.error);
      }
      setExplanation(data);
    } catch (e) {
      setExplainError(e instanceof Error ? e.message : "Failed to generate AI explanation.");
    } finally {
      setIsExplaining(false);
    }
  }

  // Keep live clock ticking each minute
  useEffect(() => {
    const t = setInterval(() => setCurrentTime(new Date()), 60_000);
    return () => clearInterval(t);
  }, []);

  useEffect(() => {
    const fallback = createFallbackForecast(getDefaultForecastRequest());
    const nowIdx = findNowIndex(fallback.series);
    setForecast(fallback);
    setSelectedIndex(Math.max(0, nowIdx));
  }, []);

  async function runForecast(nextHistory: HourObservation[]) {
    setIsLoading(true);
    try {
      const response = await fetch("/api/forecast", {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ history: nextHistory, bufferHours: BUFFER_HOURS }),
        cache: "no-store"
      });
      if (!response.ok) throw new Error(`Forecast proxy returned ${response.status}`);
      const data = (await response.json()) as ForecastResponse;
      setForecast(data);
      const lastTs = nextHistory[nextHistory.length - 1]?.timestamp;
      const idx = data.series.findIndex((p) => p.timestamp === lastTs);
      setSelectedIndex(Math.max(0, idx >= 0 ? idx : data.series.length - 1));
    } catch (error) {
      const fallback = createFallbackForecast(
        { history: nextHistory, bufferHours: BUFFER_HOURS },
        {
          source: "demo",
          endpoint: "/api/forecast",
          warning: error instanceof Error ? error.message : "The forecast proxy is unavailable."
        }
      );
      setForecast(fallback);
      const lastTs = nextHistory[nextHistory.length - 1]?.timestamp;
      const idx = fallback.series.findIndex((p) => p.timestamp === lastTs);
      setSelectedIndex(Math.max(0, idx >= 0 ? idx : fallback.series.length - 1));
    } finally {
      setIsLoading(false);
    }
  }

  function submitObservation(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const actualValue = Number(actual);
    if (!Number.isFinite(actualValue) || actualValue <= 0) return;
    const normalized = toHourInputValue(new Date(timestamp));
    const nextHistory = upsertObservation(history, { timestamp: normalized, actual: actualValue });
    setHistory(nextHistory);
    setTimestamp(toHourInputValue(new Date(new Date(normalized).getTime() + 60 * 60 * 1000)));
    setActual("");
    void runForecast(nextHistory);
  }

  function resetSession() {
    const fallback = createFallbackForecast(getDefaultForecastRequest());
    const nowIdx = findNowIndex(fallback.series);
    setHistory([]);
    setForecast(fallback);
    setSelectedIndex(Math.max(0, nowIdx));
    setTimestamp(toHourInputValue(new Date()));
    setActual("");
  }

  async function loadHistoryFromDate(startDateStr: string, endDateStr: string) {
    if (!startDateStr || !endDateStr) return;
    setIsLoading(true);
    try {
      const res = await fetch(`/api/history?start=${encodeURIComponent(startDateStr)}&end=${encodeURIComponent(endDateStr)}`);
      const payload = await res.json();
      if (payload && payload.history && payload.history.length > 0) {
        setHistory(payload.history);
        setTimestamp(endDateStr);
        await runForecast(payload.history);
      } else {
        alert("No historical data found for this range in the dataset.");
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  }

  const selectedPoint = forecast?.series[selectedIndex] ?? forecast?.series[0] ?? null;
  const nowTs = toHourInputValue(currentTime);
  const nowPoint = forecast?.series.find((p) => p.timestamp === nowTs) ?? null;
  const metrics = forecast?.metrics;

  return (
    <main className="min-h-screen bg-canvas text-ink">
      <div className="mx-auto grid min-h-screen w-full max-w-[1800px] grid-cols-1 xl:grid-cols-[300px_minmax(0,1fr)]">

        {/* ── Sidebar ── */}
        <aside className="app-sidebar border-b border-line px-5 py-6 xl:min-h-screen xl:border-b-0 xl:border-r">
          <div>
            <p className="eyebrow">Online Learning</p>
            <h1 className="mt-1 text-2xl font-semibold text-white">EnergyCast</h1>
            <p className="mt-2 text-sm text-slate-400">
              Rolling {BUFFER_HOURS}h forecast with online updates on every actual observation.
            </p>
          </div>

          {/* Live NOW box */}
          <div className="mt-5 rounded-lg border border-white/10 p-3" style={{ background: "rgba(255,255,255,0.05)" }}>
            <p className="eyebrow text-slate-400">System Time</p>
            <p className="mono mt-1 text-lg font-semibold text-white">{formatDateTime(nowTs)}</p>
            {nowPoint && (
              <div className="mt-2 flex items-center justify-between text-sm">
                <span className="text-slate-400">Predicted Now</span>
                <span className="mono font-bold text-teal-300">{compactNumber(nowPoint.predicted)} MW</span>
              </div>
            )}
            {nowPoint && (
              <div className="mt-1 flex items-center justify-between text-sm">
                <span className="text-slate-400">Status</span>
                <span className={`text-xs font-bold ${conditionTextLight(nowPoint.condition)}`}>
                  {nowPoint.condition}
                </span>
              </div>
            )}
          </div>

          {/* Input form */}
          <form className="mt-5 space-y-4" onSubmit={submitObservation}>
            <div className="block">
              <span className="control-label mb-1">Jump to History Range</span>
              <div className="mt-2 flex flex-col gap-2">
                <input
                  className="control-field"
                  type="datetime-local"
                  step="3600"
                  value={datasetStartDate}
                  onChange={(e) => setDatasetStartDate(e.target.value.substring(0, 14) + "00")}
                  max={datasetEndDate || undefined}
                  title="Select the start date using the calendar or type it"
                  placeholder="Start Date"
                />
                <input
                  className="control-field"
                  type="datetime-local"
                  step="3600"
                  value={datasetEndDate}
                  onChange={(e) => setDatasetEndDate(e.target.value.substring(0, 14) + "00")}
                  min={datasetStartDate || undefined}
                  title="Select the end date using the calendar or type it"
                  placeholder="End Date"
                />
                
                {/* Validation error message */}
                {datasetStartDate && datasetEndDate && new Date(datasetStartDate) > new Date(datasetEndDate) && (
                  <p className="text-xs text-red-400 font-medium mt-1">Start date must be before or equal to End date.</p>
                )}

                <button
                  type="button"
                  className="primary-action"
                  onClick={() => loadHistoryFromDate(datasetStartDate, datasetEndDate)}
                  disabled={isLoading || !datasetStartDate || !datasetEndDate || new Date(datasetStartDate) > new Date(datasetEndDate)}
                >
                  Load Data
                </button>
              </div>
            </div>
            <div className="h-px bg-white/10 my-2" />
            <label className="block">
              <span className="control-label">This Hour</span>
              <input
                className="control-field mt-2"
                type="datetime-local"
                step="3600"
                value={timestamp}
                onChange={(e) => {
                  const hourVal = e.target.value ? e.target.value.substring(0, 14) + "00" : "";
                  setTimestamp(hourVal);
                  const idx = forecast?.series.findIndex((p) => p.timestamp === hourVal) ?? -1;
                  if (idx >= 0) setSelectedIndex(idx);
                }}
              />
            </label>
            <label className="block">
              <span className="control-label">Actual Demand (MW)</span>
              <input
                className="control-field mt-2"
                type="number"
                min="1"
                step="0.01"
                value={actual}
                onChange={(e) => setActual(e.target.value)}
                placeholder="e.g. 2450"
              />
            </label>
            <button className="primary-action" type="submit" disabled={isLoading}>
              {isLoading ? "Updating…" : "Submit Actual"}
            </button>
            <button className="ghost-action" type="button" onClick={resetSession}>
              Reset Session
            </button>
          </form>

          {/* Session stats */}
          <div className="mt-5 space-y-2 border-t border-white/10 pt-4 text-sm">
            {[
              ["Forecast Buffer", `Next ${BUFFER_HOURS}h`],
              ["Observations", String(history.length)],
              ["Runtime", forecast?.source === "api" ? "API" : "Demo"]
            ].map(([label, val]) => (
              <div key={label} className="flex items-center justify-between text-slate-400">
                <span>{label}</span>
                <span className="mono text-white">{val}</span>
              </div>
            ))}
          </div>

          {/* Submitted history */}
          <div className="mt-4 max-h-[220px] overflow-auto border-t border-white/10 pt-4">
            <h2 className="text-sm font-semibold text-white">Submitted Hours</h2>
            <div className="mt-2 space-y-1.5">
              {history.length === 0 ? (
                <p className="text-sm text-slate-500">No values submitted yet.</p>
              ) : (
                history
                  .slice()
                  .reverse()
                  .map((item) => (
                    <button
                      key={item.timestamp}
                      className="history-row"
                      type="button"
                      onClick={() => {
                        setTimestamp(item.timestamp);
                        setActual(String(item.actual));
                        const idx = forecast?.series.findIndex((p) => p.timestamp === item.timestamp) ?? -1;
                        if (idx >= 0) setSelectedIndex(idx);
                      }}
                    >
                      <span>{formatDateTime(item.timestamp)}</span>
                      <span className="mono">{Math.round(item.actual).toLocaleString()} MW</span>
                    </button>
                  ))
              )}
            </div>
          </div>
        </aside>

        {/* ── Main content ── */}
        <section className="px-4 py-6 sm:px-6 lg:px-8">

          {/* Header */}
          <header className="flex flex-col gap-4 border-b border-line pb-5 lg:flex-row lg:items-end lg:justify-between">
            <div>
              <p className="eyebrow text-muted">Rolling Forecast Dashboard</p>
              <h2 className="mt-1 text-3xl font-semibold">Actual vs Predicted Demand</h2>
              <p className="mt-1.5 text-sm text-muted mb-3">
                Scroll the timeline left/right · click any point to lock details.
              </p>
              <div className="flex gap-4 text-sm font-medium">
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={showActual} onChange={e => setShowActual(e.target.checked)} className="rounded text-indigo-600 focus:ring-indigo-600 border-line bg-slate-50/5" /> 
                  <span className="text-slate-300">Actual</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={showBase} onChange={e => setShowBase(e.target.checked)} className="rounded text-teal-600 focus:ring-teal-600 border-line bg-slate-50/5" /> 
                  <span className="text-slate-300">Base Model</span>
                </label>
                <label className="flex items-center gap-1.5 cursor-pointer">
                  <input type="checkbox" checked={showOnline} onChange={e => setShowOnline(e.target.checked)} className="rounded text-amber-500 focus:ring-amber-500 border-line bg-slate-50/5" /> 
                  <span className="text-slate-300">Online Corrected</span>
                </label>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-4 sm:min-w-[520px]">
              <MetricCard label="MAE" value={metrics && history.length > 0 ? metrics.mae.toFixed(1) : "--"} unit="MW" />
              <MetricCard label="RMSE" value={metrics && history.length > 0 ? metrics.rmse.toFixed(1) : "--"} unit="MW" />
              <MetricCard label="MSE" value={metrics && history.length > 0 ? compactNumber(metrics.mse) : "--"} unit="MW²" />
              <MetricCard label="MAPE" value={metrics && history.length > 0 ? metrics.mape.toFixed(2) : "--"} unit="%" />
            </div>
          </header>

          <div className="mt-5 space-y-5">

            {/* Graph 1 — Actual vs Predicted (time series) */}
            <section className="panel p-4 lg:p-5">
              <PanelHeader
                title="Actual vs Predicted"
                meta={history.length ? `${history.length} observed · ${BUFFER_HOURS}h forecast` : `${BUFFER_HOURS}h rolling forecast`}
              />
              <div className="mt-4 h-[420px]">
                <ActualVsPredictedChart
                  points={forecast?.series ?? []}
                  selectedIndex={selectedIndex}
                  nowTs={nowTs}
                  onSelect={setSelectedIndex}
                  showActual={showActual}
                  showBase={showBase}
                  showOnline={showOnline}
                />
              </div>
            </section>

            {/* Graph 2 — 24h Forecast only */}
            <section className="panel p-4 lg:p-5">
              <PanelHeader
                title="24-Hour Forecast Window"
                meta={`next ${forecast?.series.filter((p) => p.isForecast).length ?? 0} hours`}
              />
              <div className="mt-4 h-[360px]">
                <ForecastWindowChart
                  points={forecast?.series ?? []}
                  selectedIndex={selectedIndex}
                  onSelect={setSelectedIndex}
                />
              </div>
            </section>

            {/* Bottom row */}
            <div className="grid gap-5 xl:grid-cols-3">
              <aside className="panel p-4 lg:p-5">
                <PanelHeader title="Point Details" meta={selectedPoint?.condition ?? "—"} />
                <PointDetails point={selectedPoint} />
              </aside>

              <section className="panel p-4 lg:p-5">
                <PanelHeader title="Insights" meta="live · per point" />
                <div className="mt-4 space-y-2.5">
                  {selectedPoint && (
                    <div className="note-row" style={{ borderLeftColor: selectedPoint.isForecast ? "#0f766e" : "#4338ca" }}>
                      {selectedPoint.insight}
                    </div>
                  )}
                  {selectedPoint !== null && selectedPoint.actual !== null && selectedPoint.error !== null && selectedPoint.errorPercent !== null && (
                    <div className="note-row" style={{ borderLeftColor: Math.abs(selectedPoint.errorPercent) > 8 ? "#be123c" : "#0f766e" }}>
                      {Math.abs(selectedPoint.errorPercent) > 8
                        ? `High deviation: model was off by ${selectedPoint.error > 0 ? "+" : ""}${selectedPoint.error} MW (${selectedPoint.errorPercent.toFixed(1)}%) — consider submitting more actuals to improve accuracy.`
                        : `Model accuracy is good for this hour: ${selectedPoint.errorPercent.toFixed(1)}% error, within acceptable range.`}
                    </div>
                  )}
                  {(forecast?.recommendations ?? ["Submit an hourly actual to begin online learning."]).map((r) => (
                    <div key={r} className="note-row">{r}</div>
                  ))}
                </div>
              </section>

              <section className="panel overflow-hidden flex flex-col">
                <div className="border-b border-line p-4 lg:p-5">
                  <PanelHeader
                    title="Hourly Values"
                    meta={forecast ? `${forecast.series.length} rows` : "waiting"}
                  />
                </div>
                <div className="flex-1 overflow-hidden">
                  <ForecastTable
                    points={forecast?.series ?? []}
                    selectedIndex={selectedIndex}
                    onSelect={setSelectedIndex}
                  />
                </div>
              </section>
            </div>

            {/* Explainable AI Row */}
            <section className="panel p-4 lg:p-5">
              <div className="flex items-center justify-between border-b border-line pb-4 mb-4">
                <PanelHeader title="Explainable AI" meta="LLM Insights" />
                <button 
                  className="primary-action"
                  onClick={generateExplanation}
                  disabled={isExplaining || !forecast}
                >
                  {isExplaining ? "Generating..." : "Generate AI Insights"}
                </button>
              </div>
              <div>
                {explainError && <div className="text-red-400 text-sm mb-4">{explainError}</div>}
                {!explanation && !isExplaining && !explainError && (
                  <div className="text-sm text-slate-400">
                    Click "Generate AI Insights" to get an LLM-powered explanation of the current forecast and grid conditions.
                  </div>
                )}
                {explanation && (
                  <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
                    {explanation.sections.map((sec) => (
                      <div key={sec.id} className="rounded-lg border border-line bg-slate-50/5 p-4">
                        <div className="flex items-center gap-2 mb-2">
                          <span className="text-teal-400 font-bold text-lg">{IconMap(sec.icon)}</span>
                          <h4 className="font-semibold text-white">{sec.title}</h4>
                        </div>
                        <p className="text-sm text-slate-300 leading-relaxed">{sec.content}</p>
                      </div>
                    ))}
                  </div>
                )}
                {explanation && (
                  <div className="mt-4 text-xs text-slate-500 text-right">
                    Generated by {explanation.backend} at {new Date(explanation.generated_at).toLocaleString()}
                  </div>
                )}
              </div>
            </section>

            {/* Online Learning Backtest Row */}
            <section className="panel p-4 lg:p-5 mt-5">
              <div className="flex items-center justify-between border-b border-line pb-4 mb-4">
                <div>
                  <PanelHeader title="Online Learning Performance (2021 Backtest)" meta="Historical Analysis" />
                  <p className="mt-1 text-sm text-slate-400">
                    Comparing the base ensemble model against the online-corrected model over the full year.
                  </p>
                </div>
                <div className="flex gap-3">
                  <div className="flex items-center gap-3 mr-4 text-sm font-medium">
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input type="checkbox" checked={showActual} onChange={e => setShowActual(e.target.checked)} className="rounded text-indigo-600 focus:ring-indigo-600 border-line bg-slate-50/5" /> 
                      <span className="text-slate-300">Actual</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input type="checkbox" checked={showBase} onChange={e => setShowBase(e.target.checked)} className="rounded text-teal-600 focus:ring-teal-600 border-line bg-slate-50/5" /> 
                      <span className="text-slate-300">Base</span>
                    </label>
                    <label className="flex items-center gap-1.5 cursor-pointer">
                      <input type="checkbox" checked={showOnline} onChange={e => setShowOnline(e.target.checked)} className="rounded text-amber-500 focus:ring-amber-500 border-line bg-slate-50/5" /> 
                      <span className="text-slate-300">Online</span>
                    </label>
                  </div>
                  <select
                    className="control-field w-auto"
                    value={backtestResolution}
                    onChange={(e) => setBacktestResolution(e.target.value)}
                    disabled={isBacktestLoading}
                  >
                    <option value="hourly">Hourly (Detailed)</option>
                    <option value="daily">Daily (Smoothed)</option>
                    <option value="weekly">Weekly (Macro)</option>
                    <option value="monthly">Monthly (Trends)</option>
                  </select>
                </div>
              </div>
              
              {isBacktestLoading && !backtestData ? (
                <div className="flex h-[380px] items-center justify-center rounded-md border border-dashed border-line">
                  <div className="text-sm text-muted">Loading backtest data...</div>
                </div>
              ) : backtestData ? (
                <div>
                  <div className="grid grid-cols-2 gap-4 md:grid-cols-4 mb-5">
                    <div className="rounded-lg border border-line bg-slate-50/5 p-3">
                      <div className="text-xs text-slate-400 mb-1">Base MAE</div>
                      <div className="font-mono text-xl text-white">{backtestData.metrics_base.mae.toFixed(1)} MW</div>
                    </div>
                    <div className="rounded-lg border border-teal-900/30 bg-teal-900/10 p-3">
                      <div className="text-xs text-teal-400 mb-1">Online MAE</div>
                      <div className="font-mono text-xl text-teal-300">
                        {backtestData.metrics_online?.mae.toFixed(1) ?? "--"} MW
                      </div>
                    </div>
                    <div className="rounded-lg border border-line bg-slate-50/5 p-3">
                      <div className="text-xs text-slate-400 mb-1">Base RMSE</div>
                      <div className="font-mono text-xl text-white">{backtestData.metrics_base.rmse.toFixed(1)} MW</div>
                    </div>
                    <div className="rounded-lg border border-teal-900/30 bg-teal-900/10 p-3">
                      <div className="text-xs text-teal-400 mb-1">Online RMSE</div>
                      <div className="font-mono text-xl text-teal-300">
                        {backtestData.metrics_online?.rmse.toFixed(1) ?? "--"} MW
                      </div>
                    </div>
                  </div>
                  <div className="h-[380px]">
                    <BacktestChart data={backtestData} showActual={showActual} showBase={showBase} showOnline={showOnline} />
                  </div>
                </div>
              ) : (
                <div className="flex h-[380px] items-center justify-center rounded-md border border-dashed border-line">
                  <div className="text-sm text-muted">No backtest data available.</div>
                </div>
              )}
            </section>
          </div>
        </section>
      </div>
    </main>
  );
}

// ──────────────────────────────────────────────
// Graph 1: Rolling time-series (actual + predicted)
// ──────────────────────────────────────────────
function ActualVsPredictedChart({
  points,
  selectedIndex,
  nowTs,
  onSelect,
  showActual,
  showBase,
  showOnline
}: Readonly<{
  points: ForecastPoint[];
  selectedIndex: number;
  nowTs: string;
  onSelect: (i: number) => void;
  showActual: boolean;
  showBase: boolean;
  showOnline: boolean;
}>) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const [hoveredIndex, setHoveredIndex] = useState<number | null>(null);
  const POINT_GAP = 76;
  const width = Math.max(1100, points.length * POINT_GAP);
  const plotWidth = width - CHART_PAD_X - CHART_PAD_R;
  const plotHeight = CHART_H - CHART_PAD_Y - CHART_PAD_BTM;

  const nowIndex = useMemo(() => points.findIndex((p) => p.timestamp === nowTs), [points, nowTs]);

  // Center view on NOW when data loads
  useEffect(() => {
    const el = scrollRef.current;
    if (!el || nowIndex < 0 || points.length === 0) return;
    const nowX = CHART_PAD_X + (nowIndex / Math.max(1, points.length - 1)) * plotWidth;
    el.scrollLeft = Math.max(0, nowX - el.clientWidth / 2);
  }, [points.length]); // eslint-disable-line react-hooks/exhaustive-deps

  const geometry = useMemo(() => {
    if (points.length === 0) return null;
    const allValues = points.flatMap((p) => [p.predicted, p.actual ?? p.predicted, p.lower, p.upper]);
    const minVal = Math.min(...allValues);
    const maxVal = Math.max(...allValues);
    const padding = Math.max(80, (maxVal - minVal) * 0.15);
    const lowerBound = minVal - padding;
    const upperBound = maxVal + padding;
    const valueRange = Math.max(1, upperBound - lowerBound);

    const xForIndex = (i: number) => CHART_PAD_X + (i / Math.max(1, points.length - 1)) * plotWidth;
    const yForValue = (v: number) => CHART_PAD_Y + (1 - (v - lowerBound) / valueRange) * plotHeight;

    const gridValues = Array.from({ length: 5 }, (_, i) => lowerBound + (i / 4) * (upperBound - lowerBound));
    
    const actualPts: string[] = [];
    const basePts: string[] = [];
    const onlinePts: string[] = [];
    const bandLowerPts: string[] = [];
    const bandUpperPts: string[] = [];

    points.forEach((p, i) => {
      const cx = xForIndex(i);
      const cyBase = yForValue(p.predicted);
      const cyOnline = p.predictedOnline !== null ? yForValue(p.predictedOnline) : cyBase;
      
      basePts.push(`${cx},${cyBase}`);
      onlinePts.push(`${cx},${cyOnline}`);
      bandLowerPts.push(`${cx},${yForValue(p.lower)}`);
      bandUpperPts.unshift(`${cx},${yForValue(p.upper)}`);

      if (p.actual !== null) {
        actualPts.push(`${cx},${yForValue(p.actual)}`);
      }
    });

    return { xVal: xForIndex, yVal: yForValue, gridValues, actualPts: actualPts.join(" "), basePts: basePts.join(" "), onlinePts: onlinePts.join(" "), bandPts: [...bandLowerPts, ...bandUpperPts].join(" ") };
  }, [points, plotWidth, plotHeight]);

  function handleWheel(e: WheelEvent<HTMLDivElement>) {
    if (scrollRef.current) scrollRef.current.scrollLeft += e.deltaY || e.deltaX;
  }

  if (points.length === 0 || !geometry) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-line">
        <div className="text-sm text-muted">Initialising forecast…</div>
      </div>
    );
  }

  const { xVal, yVal, gridValues, actualPts, basePts, onlinePts, bandPts } = geometry;
  const hoveredPoint = hoveredIndex !== null ? points[hoveredIndex] : null;
  const selectedPoint = points[selectedIndex] ?? null;

  return (
    <div className="relative h-full">
      <div ref={scrollRef} className="chart-scroll h-full overflow-x-auto overflow-y-hidden" onWheel={handleWheel}>
      <div className="relative h-full" style={{ width }}>
        <svg className="h-full" viewBox={`0 0 ${width} ${CHART_H}`} style={{ width }} role="img">
          <rect width={width} height={CHART_H} fill="#ffffff" />

          {/* Forecast zone shading */}
          {nowIndex >= 0 && (
            <rect
              x={xVal(nowIndex)}
              y={CHART_PAD_Y}
              width={width - xVal(nowIndex) - CHART_PAD_R}
              height={plotHeight}
              fill="#f8faff"
            />
          )}

          {/* Y-axis grid lines only — labels rendered in sticky overlay */}
          {gridValues.map((v, i) => (
            <line key={i} x1={CHART_PAD_X} x2={width - CHART_PAD_R} y1={yVal(v)} y2={yVal(v)} stroke="#f1f5f9" />
          ))}

          {showBase && <polygon points={bandPts} fill="rgba(15, 118, 110, 0.05)" />}

          {/* Lines */}
          {showActual && <polyline points={actualPts} fill="none" stroke="#6366f1" strokeWidth="2.5" strokeLinejoin="round" />}
          {showBase && <polyline points={basePts} fill="none" stroke="#0f766e" strokeWidth="2" strokeDasharray="4 4" strokeLinejoin="round" />}
          {showOnline && <polyline points={onlinePts} fill="none" stroke="#f59e0b" strokeWidth="2.5" strokeLinejoin="round" />}

          {/* NOW vertical marker */}
          {nowIndex >= 0 && (
            <>
              <line
                x1={xVal(nowIndex)} x2={xVal(nowIndex)}
                y1={CHART_PAD_Y} y2={CHART_H - CHART_PAD_BTM}
                stroke="#f59e0b" strokeWidth="1.5" strokeDasharray="4 3"
              />
              <text x={xVal(nowIndex)} y={CHART_PAD_Y - 7} textAnchor="middle" fill="#f59e0b" fontSize={9} fontWeight="700">NOW</text>
            </>
          )}

          {/* Data points */}
          {points.map((p, i) => {
            const cx = xVal(i);
            const isActive = hoveredIndex === i || selectedIndex === i;
            return (
              <g
                key={`${p.timestamp}-${i}`}
                className="cursor-pointer"
                onClick={() => onSelect(i)}
                onMouseEnter={() => setHoveredIndex(i)}
                onMouseLeave={() => setHoveredIndex(null)}
              >
                {p.actual !== null && showActual && <circle cx={cx} cy={yVal(p.actual)} r={isActive ? 6 : 4} fill="#6366f1" stroke="white" strokeWidth="1.5" />}
                {showBase && <circle cx={cx} cy={yVal(p.predicted)} r={isActive ? 5 : 3} fill="#0f766e" stroke="white" strokeWidth="1.5" />}
                {showOnline && p.predictedOnline !== null && <circle cx={cx} cy={yVal(p.predictedOnline)} r={isActive ? 5 : 3} fill="#f59e0b" stroke="white" strokeWidth="1.5" />}
                <rect x={cx - 14} y={CHART_PAD_Y} width={28} height={plotHeight} fill="transparent" />
              </g>
            );
          })}

          {/* Time ticks */}
          {getTimeTicks(points).map((tick) => (
            <text key={tick.index} x={xVal(tick.index)} y={CHART_H - 12} textAnchor="middle" fill="#64748b" fontSize={10}>
              {tick.label}
            </text>
          ))}
        </svg>

        {/* Legend */}
        <div className="absolute right-3 top-3 flex flex-wrap gap-2">
          <Legend color="#6366f1" label="Actual" />
          <Legend color="#0f766e" label="Base" dashed />
          <Legend color="#f59e0b" label="Online" />
        </div>

        {/* Tooltip */}
        {hoveredPoint !== null && hoveredIndex !== null && (
          <div
            className="chart-tooltip"
            style={{
              left: Math.min(width - 240, Math.max(8, xVal(hoveredIndex) - 112)),
              top: Math.max(8, yVal(hoveredPoint.predicted) - 130)
            }}
          >
            <div className="font-semibold text-ink">{formatDateTime(hoveredPoint.timestamp)}</div>
            <div className="mt-1.5 space-y-1 text-xs">
              <div className="flex justify-between gap-4"><span className="text-muted">Actual</span><span className="mono font-semibold text-indigo-700">{hoveredPoint.actual !== null ? `${compactNumber(hoveredPoint.actual)} MW` : "—"}</span></div>
              <div className="flex justify-between gap-4"><span className="text-muted">Base</span><span className="mono font-semibold text-teal-700">{compactNumber(hoveredPoint.predicted)} MW</span></div>
              {hoveredPoint.predictedOnline !== null && <div className="flex justify-between gap-4"><span className="text-muted">Online</span><span className="mono font-semibold text-amber-600">{compactNumber(hoveredPoint.predictedOnline)} MW</span></div>}
            </div>
          </div>
        )}
      </div>
      </div>
      <div className="pointer-events-none absolute inset-y-0 left-0 z-10" style={{ width: CHART_PAD_X }}>
        <div className="relative h-full bg-white" style={{ borderRadius: "8px 0 0 8px" }}>
          {gridValues.map((v, i) => (
            <div key={i} className="mono absolute" style={{ top: `${(yVal(v) / CHART_H) * 100}%`, left: 4, fontSize: 10, color: "#94a3b8", transform: "translateY(-40%)", whiteSpace: "nowrap" }}>{compactNumber(v)}</div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ──────────────────────────────────────────────
// Graph 2: 24-hour forecast window (area chart)
// ──────────────────────────────────────────────
function ForecastWindowChart({
  points,
  selectedIndex,
  onSelect
}: Readonly<{
  points: ForecastPoint[];
  selectedIndex: number;
  onSelect: (i: number) => void;
}>) {
  const [hoveredLocal, setHoveredLocal] = useState<number | null>(null);

  const forecast = useMemo(
    () => points.map((p, i) => ({ p, origIdx: i })).filter(({ p }) => p.isForecast),
    [points]
  );

  if (forecast.length === 0) {
    return (
      <div className="flex h-full items-center justify-center rounded-md border border-dashed border-line">
        <div className="text-sm text-muted">Submit the first hourly actual to unlock the forecast window.</div>
      </div>
    );
  }

  const W = 1100;
  const H = 310;
  const PX = CHART_PAD_X;
  const PY = CHART_PAD_Y;
  const PR = CHART_PAD_R;
  const PB = CHART_PAD_BTM;
  const plotW = W - PX - PR;
  const plotH = H - PY - PB;
  const bottomY = PY + plotH;

  const vals = forecast.flatMap(({ p }) => [p.predicted, p.lower, p.upper]);
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const pad = Math.max(60, (maxVal - minVal) * 0.18);
  const lb = minVal - pad;
  const ub = maxVal + pad;
  const range = Math.max(1, ub - lb);

  const xL = (li: number) => PX + (li / Math.max(1, forecast.length - 1)) * plotW;
  const yV = (v: number) => PY + (1 - (v - lb) / range) * plotH;

  const gridVals = Array.from({ length: 5 }, (_, i) => lb + (i / 4) * (ub - lb));
  const linePts = forecast.map(({ p }, li) => `${xL(li)},${yV(p.predicted)}`).join(" ");
  const upperPts = forecast.map(({ p }, li) => `${xL(li)},${yV(p.upper)}`).join(" ");
  const lowerPts = [...forecast].reverse().map(({ p }, ri) => `${xL(forecast.length - 1 - ri)},${yV(p.lower)}`).join(" ");
  const areaD =
    `M${xL(0)},${bottomY} ` +
    forecast.map(({ p }, li) => `L${xL(li)},${yV(p.predicted)}`).join(" ") +
    ` L${xL(forecast.length - 1)},${bottomY} Z`;

  const hoveredEntry = hoveredLocal !== null ? forecast[hoveredLocal] : null;

  return (
    <div className="relative h-full w-full">
      <svg className="h-full w-full" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none" role="img">
        <defs>
          <linearGradient id="fwGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#0f766e" stopOpacity="0.2" />
            <stop offset="100%" stopColor="#0f766e" stopOpacity="0.01" />
          </linearGradient>
          <linearGradient id="zoneGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#fef3c7" stopOpacity="0.5" />
            <stop offset="100%" stopColor="#fef3c7" stopOpacity="0" />
          </linearGradient>
        </defs>

        <rect width={W} height={H} fill="#ffffff" />

        {/* Grid */}
        {gridVals.map((v, i) => (
          <g key={i}>
            <line x1={PX} x2={W - PR} y1={yV(v)} y2={yV(v)} stroke="#f1f5f9" />
            <text x={4} y={yV(v) + 4} fill="#94a3b8" fontSize={10}>{compactNumber(v)}</text>
          </g>
        ))}

        {/* Confidence band */}
        <polygon points={`${upperPts} ${lowerPts}`} fill="#0f766e" opacity="0.06" />

        {/* Area fill */}
        <path d={areaD} fill="url(#fwGrad)" />

        {/* Trend line */}
        <polyline points={linePts} fill="none" stroke="#0f766e" strokeWidth="2.5" strokeLinejoin="round" />

        {/* Condition-coloured dots */}
        {forecast.map(({ p, origIdx }, li) => {
          const isSelected = selectedIndex === origIdx;
          const isHovered = hoveredLocal === li;
          const r = isSelected || isHovered ? 7 : 4.5;
          return (
            <g
              key={`${p.timestamp}-${li}`}
              className="cursor-pointer"
              onClick={() => onSelect(origIdx)}
              onMouseEnter={() => setHoveredLocal(li)}
              onMouseLeave={() => setHoveredLocal(null)}
            >
              <circle cx={xL(li)} cy={yV(p.predicted)} r={r} fill={conditionColor(p.condition)} stroke="white" strokeWidth="2" />
              {isSelected && (
                <circle cx={xL(li)} cy={yV(p.predicted)} r={r + 5} fill="none" stroke="#be123c" strokeWidth="2" />
              )}
              <rect x={xL(li) - 14} y={PY} width={28} height={plotH} fill="transparent" />
            </g>
          );
        })}

        {/* Hour labels — every 3rd */}
        {forecast.map(({ p }, li) => {
          if (li % 3 !== 0 && li !== forecast.length - 1) return null;
          return (
            <text key={li} x={xL(li)} y={H - 12} textAnchor="middle" fill="#64748b" fontSize={10}>
              {formatShortTime(p.timestamp)}
            </text>
          );
        })}
      </svg>

      {/* Legend */}
      <div className="absolute right-3 top-3 flex flex-wrap gap-2">
        <Legend color="#2dd4bf" label="Low" />
        <Legend color="#0f766e" label="Normal" />
        <Legend color="#f59e0b" label="High" />
        <Legend color="#be123c" label="Peak" />
      </div>

      {/* Tooltip */}
      {hoveredEntry !== null && hoveredLocal !== null && (
        <div
          className="chart-tooltip"
          style={{
            left: `${Math.min(85, Math.max(1, (xL(hoveredLocal) / W) * 100 - 10))}%`,
            top: Math.max(8, yV(hoveredEntry.p.predicted) - 100)
          }}
        >
          <div className="font-semibold text-ink">{formatDateTime(hoveredEntry.p.timestamp)}</div>
          <div className="mt-1.5 space-y-1 text-xs">
            <div className="flex justify-between gap-4">
              <span className="text-muted">Predicted</span>
              <span className="mono font-semibold">{compactNumber(hoveredEntry.p.predicted)} MW</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted">Range</span>
              <span className="mono text-slate-500">{compactNumber(hoveredEntry.p.lower)}–{compactNumber(hoveredEntry.p.upper)}</span>
            </div>
            <div className="flex justify-between gap-4">
              <span className="text-muted">Status</span>
              <span className={`font-bold ${conditionTextDark(hoveredEntry.p.condition)}`}>{hoveredEntry.p.condition}</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

// ──────────────────────────────────────────────
// Supporting components
// ──────────────────────────────────────────────
function PointDetails({ point }: Readonly<{ point: ForecastPoint | null }>) {
  if (!point) {
    return <div className="mt-4 text-sm text-muted">Click a graph point to show hourly details.</div>;
  }

  return (
    <div className="mt-4 space-y-2.5">
      <DetailRow label="Date Time" value={formatDateTime(point.timestamp)} />
      <DetailRow label="Type" value={point.isForecast ? "Future forecast" : "Observed hour"} />
      {point.actual !== null && <DetailRow label="Actual" value={`${point.actual.toLocaleString()} MW`} />}
      <DetailRow label="Base Forecast" value={`${point.predicted.toLocaleString()} MW`} />
      {point.predictedOnline !== null && (
        <DetailRow label="Online Forecast" value={`${point.predictedOnline.toLocaleString()} MW`} />
      )}
      <DetailRow label="Error" value={point.error === null ? "—" : `${point.error > 0 ? "+" : ""}${point.error} MW`} />
      <DetailRow label="Error %" value={point.errorPercent === null ? "—" : `${point.errorPercent.toFixed(2)}%`} />
      <DetailRow label="Confidence" value={`${compactNumber(point.lower)} – ${compactNumber(point.upper)} MW`} />
      <div className="flex items-center justify-between py-0.5">
        <span className="text-sm text-muted">Status</span>
        <ConditionBadge condition={point.condition} />
      </div>
      <div className="rounded-lg border border-line bg-slate-50 p-3 text-sm text-muted leading-snug">{point.insight}</div>
    </div>
  );
}

function ForecastTable({
  points,
  selectedIndex,
  onSelect
}: Readonly<{
  points: ForecastPoint[];
  selectedIndex: number;
  onSelect: (i: number) => void;
}>) {
  if (points.length === 0) {
    return <div className="p-5 text-sm text-muted">Waiting for forecast data.</div>;
  }

  return (
    <div className="max-h-[390px] overflow-auto">
      <table className="w-full min-w-[700px] border-collapse text-left text-sm">
        <thead className="sticky top-0 bg-white">
          <tr className="border-b border-line text-xs text-muted">
            <th className="px-4 py-3 font-medium">Date Time</th>
            <th className="w-[110px] text-right font-medium">Actual</th>
            <th className="w-[110px] text-right font-medium">Base</th>
            <th className="w-[110px] text-right font-medium">Online</th>
            <th className="w-[110px] text-right font-medium">Error (Online)</th>
            <th className="px-4 py-3 font-medium">Status</th>
          </tr>
        </thead>
        <tbody>
          {points.map((p, i) => (
            <tr
              key={`${p.timestamp}-${i}`}
              className={selectedIndex === i ? "table-row-active" : "border-b border-line/70 hover:bg-slate-50"}
            >
              <td className="px-4 py-2.5">
                <button className="mono text-left text-ink text-xs" type="button" onClick={() => onSelect(i)}>
                  {formatDateTime(p.timestamp)}
                </button>
              </td>
              <td className="mono px-4 py-2.5 text-xs text-right text-indigo-600">{p.actual === null ? "—" : `${compactNumber(p.actual)} MW`}</td>
              <td className="mono px-4 py-2.5 text-xs text-right text-teal-600">{compactNumber(p.predicted)} MW</td>
              <td className="mono px-4 py-2.5 text-xs text-right text-amber-600">{p.predictedOnline !== null ? `${compactNumber(p.predictedOnline)} MW` : "—"}</td>
              <td className="mono px-4 py-2.5 text-xs text-right">
                {(() => {
                  if (p.actual === null) return "—";
                  const pred = p.predictedOnline !== null ? p.predictedOnline : p.predicted;
                  const err = p.actual - pred;
                  const errPct = (Math.abs(err) / Math.max(1, p.actual)) * 100;
                  return (
                    <span className={Math.abs(errPct) > 8 ? "text-rose-500" : "text-emerald-500"}>
                      {err > 0 ? "+" : ""}{compactNumber(err)} ({errPct.toFixed(1)}%)
                    </span>
                  );
                })()}
              </td>
              <td className="px-4 py-2.5">
                <ConditionBadge condition={p.condition} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function MetricCard({ label, value, unit }: Readonly<{ label: string; value: string; unit: string }>) {
  return (
    <div className="topline">
      <div className="text-xs text-muted">{label}</div>
      <div className="mt-1 flex items-end gap-1.5">
        <span className="mono text-xl font-semibold text-ink">{value}</span>
        <span className="pb-0.5 text-xs text-muted">{unit}</span>
      </div>
    </div>
  );
}

function PanelHeader({ title, meta }: Readonly<{ title: string; meta: string }>) {
  return (
    <div className="flex flex-wrap items-center justify-between gap-3">
      <h3 className="text-lg font-semibold">{title}</h3>
      <span className="pill">{meta}</span>
    </div>
  );
}

function ConditionBadge({ condition }: Readonly<{ condition: PointCondition }>) {
  const cls =
    condition === "Low" ? "risk-low"
    : condition === "High" ? "risk-watch"
    : condition === "Peak" ? "risk-high"
    : "risk-normal";
  return <span className={cls}>{condition}</span>;
}

function Legend({
  color,
  label,
  dashed
}: Readonly<{ color: string; label: string; dashed?: boolean }>) {
  return (
    <span className="legend">
      {dashed ? (
        <svg width="18" height="8" className="flex-shrink-0">
          <line x1="1" y1="4" x2="17" y2="4" stroke={color} strokeWidth="2" strokeDasharray="5 2.5" />
        </svg>
      ) : (
        <span className="h-2 w-2 flex-shrink-0 rounded-full" style={{ backgroundColor: color }} />
      )}
      {label}
    </span>
  );
}

function DetailRow({ label, value }: Readonly<{ label: string; value: string }>) {
  return (
    <div className="detail-row">
      <span className="text-muted">{label}</span>
      <span className="mono text-right font-semibold text-sm">{value}</span>
    </div>
  );
}

// ──────────────────────────────────────────────
// Helpers
// ──────────────────────────────────────────────
function conditionColor(c: PointCondition): string {
  if (c === "Low") return "#2dd4bf";
  if (c === "High") return "#f59e0b";
  if (c === "Peak") return "#be123c";
  return "#0f766e";
}

function conditionTextDark(c: PointCondition): string {
  if (c === "Low") return "text-teal-500";
  if (c === "High") return "text-amber-600";
  if (c === "Peak") return "text-rose-600";
  return "text-teal-700";
}

function conditionTextLight(c: PointCondition): string {
  if (c === "Low") return "text-teal-300";
  if (c === "High") return "text-amber-300";
  if (c === "Peak") return "text-rose-400";
  return "text-emerald-300";
}

function findNowIndex(series: ForecastPoint[]): number {
  const nowTs = toHourInputValue(new Date());
  return series.findIndex((p) => p.timestamp === nowTs);
}

function getTimeTicks(points: ForecastPoint[]): Array<{ index: number; label: string }> {
  if (points.length === 0) return [];
  const idxs = Array.from(
    new Set([0, Math.floor(points.length * 0.25), Math.floor(points.length * 0.5), Math.floor(points.length * 0.75), points.length - 1])
  );
  return idxs.map((i) => ({ index: i, label: formatShortTime(points[i].timestamp) }));
}

function upsertObservation(history: HourObservation[], obs: HourObservation): HourObservation[] {
  const map = new Map(history.map((h) => [h.timestamp, h]));
  map.set(obs.timestamp, obs);
  return Array.from(map.values()).sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime());
}

function compactNumber(value: number): string {
  return new Intl.NumberFormat("en-US", {
    notation: Math.abs(value) >= 10_000 ? "compact" : "standard",
    maximumFractionDigits: 1
  }).format(value);
}

function formatDateTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", {
    year: "numeric",
    month: "short",
    day: "2-digit",
    hour: "2-digit",
    minute: "2-digit",
    hour12: false
  }).format(new Date(value));
}

function formatShortTime(value: string): string {
  return new Intl.DateTimeFormat("en-US", { day: "2-digit", hour: "2-digit", hour12: false }).format(new Date(value));
}

function toHourInputValue(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  return `${y}-${m}-${d}T${h}:00`;
}

function BacktestChart({ data, showActual, showBase, showOnline }: { data: BacktestData, showActual: boolean, showBase: boolean, showOnline: boolean }) {
  const W = 1100;
  const H = 380;
  const PX = 60;
  const PY = 20;
  const PR = 20;
  const PB = 40;
  const plotW = W - PX - PR;
  const plotH = H - PY - PB;

  if (!data.series || data.series.length === 0) return null;

  const vals = data.series.flatMap(p => [p.actual, p.predicted_base, p.predicted_online ?? p.predicted_base]);
  const minVal = Math.min(...vals);
  const maxVal = Math.max(...vals);
  const pad = (maxVal - minVal) * 0.1;
  const lb = Math.max(0, minVal - pad);
  const ub = maxVal + pad;
  const range = Math.max(1, ub - lb);

  const xL = (i: number) => PX + (i / Math.max(1, data.series.length - 1)) * plotW;
  const yV = (v: number) => PY + (1 - (v - lb) / range) * plotH;

  const gridVals = Array.from({ length: 5 }, (_, i) => lb + (i / 4) * (ub - lb));

  const actualPts = data.series.map((p, i) => `${xL(i)},${yV(p.actual)}`).join(" ");
  const basePts = data.series.map((p, i) => `${xL(i)},${yV(p.predicted_base)}`).join(" ");
  const onlinePts = data.series.map((p, i) => `${xL(i)},${yV(p.predicted_online ?? p.predicted_base)}`).join(" ");

  return (
    <div className="relative h-full w-full overflow-x-auto chart-scroll">
      <div style={{ width: Math.max(W, data.series.length * 2), height: "100%", minWidth: "100%" }}>
        <svg className="h-full w-full" preserveAspectRatio="none" viewBox={`0 0 ${Math.max(W, data.series.length * 2)} ${H}`}>
          <rect width="100%" height={H} fill="#ffffff" />
          {gridVals.map((v, i) => (
            <g key={i}>
              <line x1={PX} x2="100%" y1={yV(v)} y2={yV(v)} stroke="#f1f5f9" />
              <text x={4} y={yV(v) + 4} fill="#94a3b8" fontSize={10}>{compactNumber(v)}</text>
            </g>
          ))}

          {/* Lines */}
          {showActual && <polyline points={actualPts} fill="none" stroke="#e2e8f0" strokeWidth="1.5" />}
          {showBase && <polyline points={basePts} fill="none" stroke="#0f766e" strokeWidth="1.5" strokeDasharray="4 4" />}
          {showOnline && <polyline points={onlinePts} fill="none" stroke="#f59e0b" strokeWidth="2" />}

          {/* Time ticks */}
          {data.series.map((p, i) => {
            const step = Math.ceil(data.series.length / 10);
            if (i % step !== 0 && i !== data.series.length - 1) return null;
            return (
              <text key={i} x={xL(i)} y={H - 12} textAnchor="middle" fill="#64748b" fontSize={10}>
                {formatShortTime(p.timestamp)}
              </text>
            );
          })}
        </svg>
      </div>
      <div className="absolute right-3 top-3 flex flex-wrap gap-2">
        <Legend color="#e2e8f0" label="Actual" />
        <Legend color="#94a3b8" label="Base Model" dashed />
        <Legend color="#f59e0b" label="Online Corrected" />
      </div>
    </div>
  );
}

function IconMap(icon: string) {
  switch (icon) {
    case "chart-line": return "📈";
    case "alert-triangle": return "⚠️";
    case "activity": return "💓";
    case "refresh-cw": return "🔄";
    case "zap": return "⚡";
    case "layers": return "📚";
    default: return "💡";
  }
}
