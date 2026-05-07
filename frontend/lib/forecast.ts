export type RuntimeSource = "api" | "demo";
export type PointCondition = "Low" | "Normal" | "High" | "Peak";

export type HourObservation = {
  timestamp: string;
  actual: number;
};

export type ForecastRequest = {
  history: HourObservation[];
  bufferHours: number;
};

export type ForecastPoint = {
  timestamp: string;
  actual: number | null;
  predicted: number;
  predictedOnline: number | null;
  onlineCorrection: number | null;
  lower: number;
  upper: number;
  error: number | null;
  errorPercent: number | null;
  condition: PointCondition;
  insight: string;
  isForecast: boolean;
};

export type ForecastMetric = {
  mae: number;
  rmse: number;
  mse: number;
  mape: number;
};

export type ForecastResponse = {
  generatedAt: string;
  source: RuntimeSource;
  endpoint: string;
  metrics: ForecastMetric;
  series: ForecastPoint[];
  recommendations: string[];
  warning?: string;
};

type RuntimeMeta = {
  source?: RuntimeSource;
  endpoint?: string;
  warning?: string;
};

const HOUR_MS = 60 * 60 * 1000;
const DEFAULT_BUFFER_HOURS = 24;

export function getDefaultForecastRequest(): ForecastRequest {
  return { history: [], bufferHours: DEFAULT_BUFFER_HOURS };
}

export function sanitizeForecastRequest(input: Partial<ForecastRequest> = {}): ForecastRequest {
  const normalizedHistory = Array.isArray(input.history)
    ? input.history
        .map(normalizeObservation)
        .filter((item): item is HourObservation => item !== null)
        .sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    : [];
  const deduped = new Map<string, HourObservation>();
  for (const item of normalizedHistory) deduped.set(item.timestamp, item);

  return {
    history: Array.from(deduped.values()),
    bufferHours: clampNumber(input.bufferHours, DEFAULT_BUFFER_HOURS, DEFAULT_BUFFER_HOURS, DEFAULT_BUFFER_HOURS)
  };
}

export function createFallbackForecast(input: Partial<ForecastRequest> = {}, meta: RuntimeMeta = {}): ForecastResponse {
  const request = sanitizeForecastRequest(input);
  const history = request.history;

  let series: ForecastPoint[];
  let metrics: ForecastMetric;

  if (history.length === 0) {
    series = buildInitialSeries(request.bufferHours);
    metrics = { mae: 0, rmse: 0, mse: 0, mape: 0 };
  } else {
    const observedPoints = history.map((_, index) => buildObservedPoint(history, index));
    const forecastPoints = buildFuturePoints(history, request.bufferHours);
    series = [...observedPoints.slice(-request.bufferHours), ...forecastPoints];
    metrics = calculateMetrics(observedPoints);
  }

  const futurePoints = series.filter((p) => p.isForecast);

  return {
    generatedAt: new Date().toISOString(),
    source: meta.source ?? "demo",
    endpoint: meta.endpoint ?? "offline-demo",
    metrics,
    series,
    recommendations: buildRecommendations(history, futurePoints, metrics),
    warning: meta.warning
  };
}

export function normalizeForecastResponse(
  payload: unknown,
  requestInput: Partial<ForecastRequest>,
  meta: RuntimeMeta = {}
): ForecastResponse {
  const fallback = createFallbackForecast(requestInput, meta);
  const record = asRecord(payload);

  if (!record) return fallback;

  const rawSeries = firstArray(record, ["series", "points", "forecast", "predictions", "data"]);
  const series = rawSeries
    .map((point, index) => normalizePoint(point, fallback.series[index] ?? fallback.series[fallback.series.length - 1]))
    .filter((point): point is ForecastPoint => point !== null);

  if (series.length === 0) {
    return { ...fallback, warning: stringFrom(record.warning) ?? stringFrom(record.message) ?? fallback.warning };
  }

  const metricsRecord = asRecord(record.metrics);

  return {
    generatedAt: stringFrom(record.generatedAt) ?? stringFrom(record.generated_at) ?? new Date().toISOString(),
    source: meta.source ?? normalizeSource(stringFrom(record.source)),
    endpoint: meta.endpoint ?? stringFrom(record.endpoint) ?? fallback.endpoint,
    metrics: {
      mae: numberFrom(metricsRecord?.mae) ?? fallback.metrics.mae,
      rmse: numberFrom(metricsRecord?.rmse) ?? fallback.metrics.rmse,
      mse: numberFrom(metricsRecord?.mse) ?? fallback.metrics.mse,
      mape: numberFrom(metricsRecord?.mape) ?? fallback.metrics.mape
    },
    series,
    recommendations: normalizeRecommendations(record.recommendations, fallback.recommendations),
    warning: meta.warning ?? stringFrom(record.warning)
  };
}

// Builds a full 48-point series (24h past + current + 24h future) anchored to system time.
// Used as the initial demo state before the user submits any actuals.
function buildInitialSeries(bufferHours: number): ForecastPoint[] {
  const now = new Date();
  const currentHour = new Date(now);
  currentHour.setMinutes(0, 0, 0);

  const startHour = new Date(currentHour.getTime() - bufferHours * HOUR_MS);
  const workingHistory: HourObservation[] = [];
  const allPoints: ForecastPoint[] = [];

  // Past context window (24h)
  for (let i = 0; i < bufferHours; i++) {
    const ts = toHourInputValue(new Date(startHour.getTime() + i * HOUR_MS));
    const predicted = Math.round(predictNextValue(workingHistory, ts));
    const band = confidenceBand(workingHistory, predicted);
    const condition = getDemandCondition(predicted, workingHistory);

    allPoints.push({
      timestamp: ts,
      actual: null,
      predicted,
      predictedOnline: null,
      onlineCorrection: null,
      lower: Math.round(predicted - band),
      upper: Math.round(predicted + band),
      error: null,
      errorPercent: null,
      condition,
      insight: buildInsight(ts, null, predicted, condition, false),
      isForecast: false
    });

    workingHistory.push({ timestamp: ts, actual: predicted });
  }

  // Current hour + future forecast window (24h)
  for (let step = 0; step <= bufferHours; step++) {
    const ts = toHourInputValue(new Date(currentHour.getTime() + step * HOUR_MS));
    const predicted = Math.round(predictNextValue(workingHistory, ts));
    const band = confidenceBand(workingHistory, predicted);
    const isForecast = step > 0;
    const condition = getDemandCondition(predicted, workingHistory);

    allPoints.push({
      timestamp: ts,
      actual: null,
      predicted,
      predictedOnline: null,
      onlineCorrection: null,
      lower: Math.round(predicted - band),
      upper: Math.round(predicted + band),
      error: null,
      errorPercent: null,
      condition,
      insight: buildInsight(ts, null, predicted, condition, isForecast),
      isForecast
    });

    workingHistory.push({ timestamp: ts, actual: predicted });
  }

  return allPoints;
}

function buildObservedPoint(history: HourObservation[], index: number): ForecastPoint {
  const item = history[index];
  const previous = history.slice(0, index);
  const predicted = Math.round(predictNextValue(previous, item.timestamp, item.actual));
  const error = Math.round(item.actual - predicted);
  const errorPercent = round((Math.abs(error) / Math.max(1, item.actual)) * 100, 2);
  const condition = getDemandCondition(item.actual, previous);

  return {
    timestamp: item.timestamp,
    actual: Math.round(item.actual),
    predicted,
    predictedOnline: null,
    onlineCorrection: null,
    lower: Math.round(predicted - confidenceBand(previous, predicted)),
    upper: Math.round(predicted + confidenceBand(previous, predicted)),
    error,
    errorPercent,
    condition,
    insight: buildInsight(item.timestamp, Math.round(item.actual), predicted, condition, false),
    isForecast: false
  };
}

function buildFuturePoints(history: HourObservation[], bufferHours: number): ForecastPoint[] {
  if (history.length === 0) return [];

  const workingHistory = [...history];
  const latest = workingHistory[workingHistory.length - 1];
  const future: ForecastPoint[] = [];

  for (let step = 1; step <= bufferHours; step++) {
    const ts = toHourInputValue(new Date(new Date(latest.timestamp).getTime() + step * HOUR_MS));
    const predicted = Math.round(predictNextValue(workingHistory, ts));
    const band = confidenceBand(workingHistory, predicted);
    const condition = getDemandCondition(predicted, workingHistory);

    future.push({
      timestamp: ts,
      actual: null,
      predicted,
      predictedOnline: null,
      onlineCorrection: null,
      lower: Math.round(predicted - band),
      upper: Math.round(predicted + band),
      error: null,
      errorPercent: null,
      condition,
      insight: buildInsight(ts, null, predicted, condition, true),
      isForecast: true
    });

    workingHistory.push({ timestamp: ts, actual: predicted });
  }

  return future;
}

function predictNextValue(history: HourObservation[], timestamp: string, fallback?: number): number {
  if (history.length === 0) return fallback ?? 2200;

  const latest = history[history.length - 1].actual;
  const lag24 = findLagValue(history, timestamp, 24);
  const rolling = mean(history.slice(-6).map((h) => h.actual));
  const hour = new Date(timestamp).getHours();
  const day = new Date(timestamp).getDay();
  const dailyShape = Math.sin(((hour - 5) / 24) * Math.PI * 2) * 70;
  const eveningLift = Math.exp(-Math.pow(hour - 19, 2) / 18) * 120;
  const weekdayLift = day > 0 && day < 6 ? 35 : -45;
  const lagComponent = lag24 ?? latest;

  return latest * 0.48 + rolling * 0.24 + lagComponent * 0.18 + dailyShape + eveningLift + weekdayLift;
}

function findLagValue(history: HourObservation[], timestamp: string, lagHours: number): number | null {
  const target = new Date(timestamp).getTime() - lagHours * HOUR_MS;
  return history.find((h) => new Date(h.timestamp).getTime() === target)?.actual ?? null;
}

function confidenceBand(history: HourObservation[], predicted: number): number {
  const values = history.slice(-12).map((h) => h.actual);
  if (values.length < 2) return Math.max(70, predicted * 0.04);
  const avg = mean(values);
  const variance = mean(values.map((v) => Math.pow(v - avg, 2)));
  return Math.max(70, Math.sqrt(variance) * 0.8);
}

// Classifies demand relative to recent rolling average.
// Low = well below average, Peak = significantly above average.
function getDemandCondition(value: number, history: HourObservation[]): PointCondition {
  const recent = history.slice(-24).map((h) => h.actual);
  if (recent.length < 3) return "Normal";
  const avg = mean(recent);
  const deviation = ((value - avg) / Math.max(1, avg)) * 100;
  if (deviation < -10) return "Low";
  if (deviation < 8) return "Normal";
  if (deviation < 22) return "High";
  return "Peak";
}

function calculateMetrics(points: ForecastPoint[]): ForecastMetric {
  const scored = points.filter((p) => p.error !== null && p.errorPercent !== null);
  if (scored.length === 0) return { mae: 0, rmse: 0, mse: 0, mape: 0 };

  const absErrors = scored.map((p) => Math.abs(p.error ?? 0));
  const squaredErrors = scored.map((p) => Math.pow(p.error ?? 0, 2));
  const pctErrors = scored.map((p) => p.errorPercent ?? 0);
  const mse = round(mean(squaredErrors), 1);

  return {
    mae: round(mean(absErrors), 1),
    rmse: round(Math.sqrt(mse), 1),
    mse,
    mape: round(mean(pctErrors), 2)
  };
}

function buildRecommendations(history: HourObservation[], future: ForecastPoint[], metrics: ForecastMetric): string[] {
  if (history.length === 0) {
    return [
      "Submit the first hourly actual value to start online learning and error tracking.",
      "The dashed curve shows the model's 24-hour rolling forecast anchored to system time.",
      "Accuracy metrics (MAE, RMSE, MSE, MAPE) appear after the first actual is recorded."
    ];
  }

  const latest = history[history.length - 1];
  const peak = future.reduce<ForecastPoint | null>((best, p) => (!best || p.predicted > best.predicted ? p : best), null);
  const trough = future.reduce<ForecastPoint | null>((best, p) => (!best || p.predicted < best.predicted ? p : best), null);
  const trend = future.length > 1 && future[future.length - 1].predicted > future[0].predicted ? "rising" : "easing";
  const alertCount = future.filter((p) => p.condition === "High" || p.condition === "Peak").length;
  const avgForecast = future.length > 0 ? Math.round(future.reduce((s, p) => s + p.predicted, 0) / future.length) : 0;

  const tips = [
    `Latest reading ${Math.round(latest.actual).toLocaleString()} MW at ${latest.timestamp.replace("T", " ")}.`,
    `Next 24h is ${trend} — peak ${peak?.predicted?.toLocaleString() ?? "--"} MW at ${peak?.timestamp.replace("T", " ") ?? "--"}, avg ${avgForecast.toLocaleString()} MW.`,
    `Accuracy: MAE ${metrics.mae.toFixed(1)} · RMSE ${metrics.rmse.toFixed(1)} · MAPE ${metrics.mape.toFixed(2)}%.`
  ];

  if (alertCount > 0) {
    tips.push(`${alertCount} forecast hour${alertCount > 1 ? "s" : ""} flagged High or Peak — review grid capacity headroom.`);
  }

  if (trough && peak && (peak.predicted - trough.predicted) > 300) {
    tips.push(`Expected demand swing of ${(peak.predicted - trough.predicted).toLocaleString()} MW — prepare ramp resources.`);
  }

  if (metrics.mape > 5) {
    tips.push(`MAPE ${metrics.mape.toFixed(2)}% is above 5% — model may need more observations to improve accuracy.`);
  }

  return tips;
}

function buildInsight(
  timestamp: string,
  actual: number | null,
  predicted: number,
  condition: PointCondition,
  isForecast: boolean
): string {
  const d = new Date(timestamp);
  const hour = d.getHours();
  const block =
    hour >= 17 && hour <= 22 ? "evening peak window"
    : hour >= 6 && hour <= 11 ? "morning ramp"
    : hour >= 0 && hour <= 5 ? "overnight off-peak"
    : "midday load period";

  if (actual !== null) {
    const errMw = Math.round(actual - predicted);
    const sign = errMw >= 0 ? "+" : "";
    const pct = ((Math.abs(errMw) / Math.max(1, actual)) * 100).toFixed(1);
    const direction = errMw >= 0 ? "under-predicted" : "over-predicted";
    return `Actual ${actual.toLocaleString()} MW vs ${predicted.toLocaleString()} MW forecast (${sign}${errMw} MW, ${pct}% ${direction}) during ${block}. Condition: ${condition}.`;
  }

  const conditionNote =
    condition === "Peak" ? " Capacity alert — demand significantly above baseline."
    : condition === "High" ? " Elevated demand — monitor grid headroom."
    : condition === "Low" ? " Demand below average — efficient operating window."
    : "";

  const verb = isForecast ? "Forecast" : "Projected";
  return `${verb} ${predicted.toLocaleString()} MW during ${block}.${conditionNote} Condition: ${condition}.`;
}

function normalizeObservation(value: unknown): HourObservation | null {
  const record = asRecord(value);
  if (!record) return null;
  const timestamp = normalizeTimestamp(stringFrom(record.timestamp) ?? stringFrom(record.time));
  const actual = numberFrom(record.actual) ?? numberFrom(record.value) ?? numberFrom(record.demand);
  if (!timestamp || actual === undefined) return null;
  return { timestamp, actual };
}

function normalizePoint(payload: unknown, fallback: ForecastPoint): ForecastPoint | null {
  const record = asRecord(payload);
  if (!record) return fallback ?? null;

  const timestamp = normalizeTimestamp(
    stringFrom(record.timestamp) ?? stringFrom(record.time) ?? stringFrom(record.ds) ?? fallback.timestamp
  );
  const predicted =
    numberFrom(record.predicted) ?? numberFrom(record.prediction) ?? numberFrom(record.yhat) ?? fallback.predicted;
  const actual = numberFrom(record.actual) ?? numberFrom(record.observed) ?? numberFrom(record.y);
  const error = actual === undefined ? null : Math.round(actual - predicted);
  const errorPercent = actual === undefined ? null : round((Math.abs(error ?? 0) / Math.max(1, actual)) * 100, 2);
  const condition = normalizeCondition(stringFrom(record.condition), actual ?? null, fallback.condition);

  if (!timestamp) return fallback;

  return {
    timestamp,
    actual: actual === undefined ? null : Math.round(actual),
    predicted: Math.round(predicted),
    predictedOnline: numberFrom(record.predicted_online) !== undefined ? Math.round(numberFrom(record.predicted_online)!) : null,
    onlineCorrection: numberFrom(record.online_correction) ?? null,
    lower: Math.round(
      numberFrom(record.lower) ??
        numberFrom(record.lowerBound) ??
        numberFrom(record.lower_bound) ??
        numberFrom(record.yhat_lower) ??
        fallback.lower
    ),
    upper: Math.round(
      numberFrom(record.upper) ??
        numberFrom(record.upperBound) ??
        numberFrom(record.upper_bound) ??
        numberFrom(record.yhat_upper) ??
        fallback.upper
    ),
    error,
    errorPercent,
    condition,
    insight: stringFrom(record.insight) ?? buildInsight(timestamp, actual ?? null, predicted, condition, actual === undefined),
    isForecast: Boolean(record.isForecast ?? record.is_forecast ?? actual === undefined)
  };
}

function normalizeTimestamp(value: string | undefined): string | null {
  if (!value) return null;
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return null;
  return toHourInputValue(parsed);
}

function normalizeRecommendations(payload: unknown, fallback: string[]): string[] {
  if (!Array.isArray(payload)) return fallback;
  const items = payload.filter((item): item is string => typeof item === "string" && item.length > 0);
  return items.length > 0 ? items : fallback;
}

function firstArray(record: Record<string, unknown>, keys: string[]): unknown[] {
  for (const key of keys) {
    if (Array.isArray(record[key])) return record[key] as unknown[];
  }
  return [];
}

function normalizeSource(value: string | undefined): RuntimeSource {
  return value === "demo" ? "demo" : "api";
}

function normalizeCondition(value: string | undefined, _actual: number | null, fallback: PointCondition): PointCondition {
  if (value === "Low" || value === "Normal" || value === "High" || value === "Peak") return value;
  // Legacy API may return "Watch" — map to High
  if (value === "Watch") return "High";
  return fallback;
}

function clampNumber(value: unknown, min: number, max: number, fallback: number): number {
  const n = typeof value === "number" && Number.isFinite(value) ? value : fallback;
  return Math.min(max, Math.max(min, Math.round(n * 100) / 100));
}

function mean(values: number[]): number {
  if (values.length === 0) return 0;
  return values.reduce((s, v) => s + v, 0) / values.length;
}

function round(value: number, decimals: number): number {
  const f = Math.pow(10, decimals);
  return Math.round(value * f) / f;
}

function toHourInputValue(date: Date): string {
  const y = date.getFullYear();
  const m = String(date.getMonth() + 1).padStart(2, "0");
  const d = String(date.getDate()).padStart(2, "0");
  const h = String(date.getHours()).padStart(2, "0");
  return `${y}-${m}-${d}T${h}:00`;
}

function asRecord(value: unknown): Record<string, unknown> | null {
  return typeof value === "object" && value !== null && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function numberFrom(value: unknown): number | undefined {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string" && value.trim().length > 0) {
    const n = Number(value);
    return Number.isFinite(n) ? n : undefined;
  }
  return undefined;
}

function stringFrom(value: unknown): string | undefined {
  return typeof value === "string" && value.length > 0 ? value : undefined;
}
