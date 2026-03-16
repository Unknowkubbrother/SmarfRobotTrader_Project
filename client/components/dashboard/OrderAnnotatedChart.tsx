"use client";

import { type ReactNode, useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CandlestickSeries,
  ColorType,
  CrosshairMode,
  createChart,
  createSeriesMarkers,
  LineStyle,
  type CandlestickData,
  type IChartApi,
  type ISeriesApi,
  type ISeriesMarkersPluginApi,
  type MouseEventParams,
  type SeriesMarker,
  type Time,
  type UTCTimestamp,
} from "lightweight-charts";
import { AlertTriangle, Loader2, RefreshCcw } from "lucide-react";

import { api } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";

interface OrderAnnotatedChartProps {
  accountId?: string | null;
  botConfigId?: string | null;
  symbol: string;
  timeframe: string;
  runtimeCandles?: ChartCandle[] | null;
  runtimeQuote?: RuntimeQuote | null;
  runtimeSourceLabel?: string | null;
  live?: boolean;
  includeArchived?: boolean;
  className?: string;
  title?: string;
  subtitle?: string;
  showMarkers?: boolean;
  pollMs?: number;
  bars?: number;
  headerContent?: ReactNode;
}

interface ChartCandle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface RuntimeQuote {
  bid?: number;
  ask?: number;
  time?: number;
}

interface ChartMarkerPayload {
  id: string;
  ticketId: number;
  symbol: string;
  time: number;
  actualTime: string;
  price: number;
  side: "BUY" | "SELL";
  action: "OPEN" | "CLOSE";
  shape: "circle" | "square" | "arrowUp" | "arrowDown";
  position: "aboveBar" | "belowBar" | "inBar" | "atPriceTop" | "atPriceBottom" | "atPriceMiddle";
  color: string;
  text: string;
  volume: number;
  profit: number;
  netProfit: number;
}

interface OrderAnnotatedChartData {
  accountId?: string | null;
  botConfigId?: string | null;
  symbol: string;
  timeframe: string;
  bars: number;
  candles: ChartCandle[];
  markers: ChartMarkerPayload[];
  sourceMode: string;
  sourceLabel: string;
  visibleFrom: string | null;
  visibleTo: string | null;
  latestOrderTime?: string | null;
  beforeTime?: string | null;
}

interface HoverTooltipState {
  x: number;
  y: number;
  markers: ChartMarkerPayload[];
  activeTicketId: number | null;
}

interface HoverPathOverlayState {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  color: string;
}

interface TradePathPayload {
  ticketId: number;
  side: "BUY" | "SELL";
  volume: number;
  openMarker: ChartMarkerPayload | null;
  closeMarker: ChartMarkerPayload | null;
  lineColor: string;
}

function formatDateTime(value?: string | null): string {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function toChartTime(value: number): UTCTimestamp {
  return Number(value || 0) as UTCTimestamp;
}

function formatMarkerValue(value: number): string {
  return Number(value || 0).toFixed(2);
}

function countFractionDigits(value: number): number {
  if (!Number.isFinite(value)) return 0;
  const text = value.toFixed(10).replace(/0+$/, "");
  const dotIndex = text.indexOf(".");
  return dotIndex >= 0 ? Math.max(0, text.length - dotIndex - 1) : 0;
}

function fallbackPricePrecision(symbol: string): number {
  const symbolText = String(symbol || "").trim().toUpperCase();
  if (/^[A-Z]{6}$/.test(symbolText)) {
    return symbolText.endsWith("JPY") ? 3 : 5;
  }
  if (symbolText.startsWith("XAU") || symbolText.startsWith("XAG")) {
    return 2;
  }
  return 2;
}

function inferPricePrecision(
  symbol: string,
  candles: ChartCandle[] | null | undefined,
  markers: ChartMarkerPayload[] | null | undefined,
  quote?: RuntimeQuote | null,
): number {
  let precision = 0;
  const candleRows = Array.isArray(candles) ? candles.slice(-240) : [];
  for (const candle of candleRows) {
    precision = Math.max(
      precision,
      countFractionDigits(Number(candle.open || 0)),
      countFractionDigits(Number(candle.high || 0)),
      countFractionDigits(Number(candle.low || 0)),
      countFractionDigits(Number(candle.close || 0)),
    );
  }
  for (const marker of markers || []) {
    precision = Math.max(precision, countFractionDigits(Number(marker.price || 0)));
  }
  precision = Math.max(
    precision,
    countFractionDigits(Number(quote?.bid || 0)),
    countFractionDigits(Number(quote?.ask || 0)),
  );
  return Math.max(0, Math.min(8, precision || fallbackPricePrecision(symbol)));
}

function minMoveForPrecision(precision: number): number {
  if (precision <= 0) return 1;
  return Number(`0.${"0".repeat(Math.max(0, precision - 1))}1`);
}

const TIMEFRAME_SECONDS: Record<string, number> = {
  M1: 60,
  M5: 300,
  M15: 900,
  M30: 1800,
  H1: 3600,
  H4: 14400,
  D1: 86400,
};

const MAX_CHART_HISTORY_PAGE_BARS = 1200;
const MIN_CHART_HISTORY_PAGE_BARS = 300;
const MAX_LIVE_REFRESH_BARS = 480;
const MAX_SYNTHETIC_GAP_BARS = 12;

function timeframeSeconds(timeframe: string): number {
  return TIMEFRAME_SECONDS[String(timeframe || "H1").trim().toUpperCase()] || 3600;
}

function normalizeQuoteEpoch(value?: number | null): number | null {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return null;
  }
  if (numeric > 1_000_000_000_000) {
    return Math.floor(numeric / 1000);
  }
  if (numeric > 10_000_000_000) {
    return Math.floor(numeric / 1000);
  }
  return Math.floor(numeric);
}

function resolveQuotePrice(quote?: RuntimeQuote | null): number | null {
  const bid = Number(quote?.bid || 0);
  const ask = Number(quote?.ask || 0);
  if (bid > 0) return bid;
  if (ask > 0) return ask;
  return null;
}

function mergeCandles(...groups: Array<ChartCandle[] | null | undefined>): ChartCandle[] {
  const byTime = new Map<number, ChartCandle>();
  for (const group of groups) {
    for (const candle of group || []) {
      const time = Number(candle?.time || 0);
      if (time <= 0) continue;
      byTime.set(time, {
        time,
        open: Number(candle.open || 0),
        high: Number(candle.high || 0),
        low: Number(candle.low || 0),
        close: Number(candle.close || 0),
        volume: Number(candle.volume || 0),
      });
    }
  }
  return Array.from(byTime.values()).sort((left, right) => left.time - right.time);
}

function mergeMarkers(...groups: Array<ChartMarkerPayload[] | null | undefined>): ChartMarkerPayload[] {
  const byId = new Map<string, ChartMarkerPayload>();
  for (const group of groups) {
    for (const marker of group || []) {
      const id = String(marker?.id || "").trim();
      if (!id) continue;
      byId.set(id, marker);
    }
  }
  return Array.from(byId.values()).sort((left, right) => {
    const timeDiff = Number(left.time || 0) - Number(right.time || 0);
    if (timeDiff !== 0) return timeDiff;

    if (left.action !== right.action) {
      return left.action === "OPEN" ? 1 : -1;
    }

    const leftActual = left.actualTime ? new Date(left.actualTime).getTime() : 0;
    const rightActual = right.actualTime ? new Date(right.actualTime).getTime() : 0;
    if (leftActual && rightActual && leftActual !== rightActual) {
      return leftActual - rightActual;
    }

    return String(left.id).localeCompare(String(right.id));
  });
}

function liveRefreshBarsForTimeframe(timeframe: string, bars: number): number {
  const tfSecs = Math.max(60, timeframeSeconds(timeframe));
  const lookbackBars = Math.ceil((6 * 3600) / tfSecs) + 6;
  return Math.max(60, Math.min(Math.max(60, bars), Math.max(lookbackBars, Math.min(Math.max(60, bars), MAX_LIVE_REFRESH_BARS))));
}

function mergeRemoteChartData(
  previous: OrderAnnotatedChartData | null,
  incoming: OrderAnnotatedChartData | null,
  targetBars: number,
): OrderAnnotatedChartData | null {
  if (!incoming) {
    return previous;
  }
  if (!previous) {
    return incoming;
  }

  const timeframe = incoming.timeframe || previous.timeframe || "H1";
  const tfSecs = Math.max(60, timeframeSeconds(timeframe));
  const mergedCandles = mergeCandles(previous.candles, incoming.candles);
  const maxBars = Math.max(
    60,
    int_or_zero(previous.bars),
    int_or_zero(incoming.bars),
    int_or_zero(targetBars),
  );
  const nextCandles = mergedCandles.length > maxBars
    ? mergedCandles.slice(-maxBars)
    : mergedCandles;
  const firstCandle = nextCandles[0] || null;
  const lastCandle = nextCandles[nextCandles.length - 1] || null;

  return {
    ...previous,
    ...incoming,
    bars: nextCandles.length,
    candles: nextCandles,
    markers: mergeMarkers(previous.markers, incoming.markers),
    visibleFrom: firstCandle ? new Date(firstCandle.time * 1000).toISOString() : null,
    visibleTo: lastCandle ? new Date((lastCandle.time + tfSecs) * 1000).toISOString() : null,
  };
}

function int_or_zero(value: unknown): number {
  const numeric = Number(value || 0);
  if (!Number.isFinite(numeric) || numeric <= 0) {
    return 0;
  }
  return Math.floor(numeric);
}

function buildDisplayCandles(
  candles: ChartCandle[] | null | undefined,
  timeframe: string,
  liveQuotePrice: number | null,
  liveQuoteEpoch: number | null,
): ChartCandle[] {
  const rows = Array.isArray(candles) ? candles : [];
  if (rows.length === 0 || liveQuotePrice == null || liveQuoteEpoch == null) {
    return rows;
  }

  const tfSecs = Math.max(60, timeframeSeconds(timeframe));
  const alignedQuoteTime = liveQuoteEpoch - (liveQuoteEpoch % tfSecs);
  const latest = rows[rows.length - 1];
  if (!latest || alignedQuoteTime < Number(latest.time || 0)) {
    return rows;
  }

  if (alignedQuoteTime === Number(latest.time || 0)) {
    const nextHigh = Math.max(Number(latest.high || 0), liveQuotePrice);
    const nextLow = Math.min(Number(latest.low || 0) || liveQuotePrice, liveQuotePrice);
    const nextClose = liveQuotePrice;
    if (nextHigh === latest.high && nextLow === latest.low && nextClose === latest.close) {
      return rows;
    }
    return [
      ...rows.slice(0, -1),
      {
        ...latest,
        high: nextHigh,
        low: nextLow,
        close: nextClose,
      },
    ];
  }

  const gapBars = Math.max(0, Math.round((alignedQuoteTime - Number(latest.time || 0)) / tfSecs));
  const carryClose = Number(latest.close || 0) > 0 ? Number(latest.close) : liveQuotePrice;
  if (gapBars > 1) {
    const syntheticGapBars = Math.max(0, gapBars - 1);
    const syntheticBarsToAdd = Math.min(syntheticGapBars, MAX_SYNTHETIC_GAP_BARS);
    const syntheticBars: ChartCandle[] = [];
    const firstSyntheticTime = alignedQuoteTime - (syntheticBarsToAdd * tfSecs);
    const startSyntheticTime = Math.max(Number(latest.time || 0) + tfSecs, firstSyntheticTime);
    for (let ts = startSyntheticTime; ts < alignedQuoteTime; ts += tfSecs) {
      syntheticBars.push({
        time: ts,
        open: carryClose,
        high: carryClose,
        low: carryClose,
        close: carryClose,
        volume: 0,
      });
    }
    const nextOpen = syntheticBars.length > 0
      ? Number(syntheticBars[syntheticBars.length - 1]?.close || carryClose)
      : carryClose;
    return [
      ...rows,
      ...syntheticBars,
      {
        time: alignedQuoteTime,
        open: nextOpen,
        high: Math.max(nextOpen, liveQuotePrice),
        low: Math.min(nextOpen, liveQuotePrice),
        close: liveQuotePrice,
        volume: 0,
      },
    ];
  }

  return [
    ...rows,
    {
      time: alignedQuoteTime,
      open: carryClose,
      high: Math.max(carryClose, liveQuotePrice),
      low: Math.min(carryClose, liveQuotePrice),
      close: liveQuotePrice,
      volume: 0,
    },
  ];
}

export default function OrderAnnotatedChart({
  accountId,
  botConfigId,
  symbol,
  timeframe,
  runtimeCandles,
  runtimeQuote,
  runtimeSourceLabel,
  live = false,
  includeArchived = true,
  className,
  title = "Order Flow Chart",
  subtitle,
  showMarkers = true,
  pollMs,
  bars = 220,
  headerContent,
}: OrderAnnotatedChartProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candleSeriesRef = useRef<ISeriesApi<"Candlestick"> | null>(null);
  const markerPluginRef = useRef<ISeriesMarkersPluginApi<Time> | null>(null);
  const livePriceLineRef = useRef<any>(null);
  const remoteDataRef = useRef<OrderAnnotatedChartData | null>(null);
  const markersByIdRef = useRef<Map<string, ChartMarkerPayload>>(new Map());
  const markersByTimeRef = useRef<Map<number, ChartMarkerPayload[]>>(new Map());
  const showMarkersRef = useRef(showMarkers);
  const hasInitializedViewportRef = useRef(false);
  const baseCandlesRef = useRef<ChartCandle[]>([]);
  const liveQuoteCandleRef = useRef<ChartCandle | null>(null);
  const historyLoadInFlightRef = useRef(false);
  const pendingHistoryViewportShiftRef = useRef<{ addedBars: number; from: number; to: number } | null>(null);
  const loadMoreHistoryRef = useRef<(() => Promise<void>) | null>(null);
  const [remoteData, setRemoteData] = useState<OrderAnnotatedChartData | null>(null);
  const [historicalData, setHistoricalData] = useState<Pick<OrderAnnotatedChartData, "candles" | "markers">>({
    candles: [],
    markers: [],
  });
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  const [loadingHistory, setLoadingHistory] = useState(false);
  const [hasMoreHistory, setHasMoreHistory] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [reloadToken, setReloadToken] = useState(0);
  const [hoverTooltip, setHoverTooltip] = useState<HoverTooltipState | null>(null);
  const [hoverPathOverlay, setHoverPathOverlay] = useState<HoverPathOverlayState | null>(null);

  const normalizedRuntimeCandles = useMemo(() => {
    const rows = Array.isArray(runtimeCandles) ? runtimeCandles : [];
    const normalized = rows
      .map((candle) => ({
        time: Number(candle?.time || 0),
        open: Number(candle?.open || 0),
        high: Number(candle?.high || 0),
        low: Number(candle?.low || 0),
        close: Number(candle?.close || 0),
        volume: Number(candle?.volume || 0),
      }))
      .filter((candle) =>
        candle.time > 0 &&
        candle.open > 0 &&
        candle.high > 0 &&
        candle.low > 0 &&
        candle.close > 0,
      )
      .sort((left, right) => left.time - right.time);
    if (normalized.length <= bars) {
      return normalized;
    }
    return normalized.slice(-bars);
  }, [bars, runtimeCandles]);

  const usesRuntimeCandles = normalizedRuntimeCandles.length > 0;
  const mergedHistoricalCandles = useMemo(() => historicalData.candles || [], [historicalData.candles]);
  const mergedHistoricalMarkers = useMemo(() => historicalData.markers || [], [historicalData.markers]);

  const data = useMemo<OrderAnnotatedChartData | null>(() => {
    const latestCandles = usesRuntimeCandles ? normalizedRuntimeCandles : (remoteData?.candles || []);
    const candles = mergeCandles(mergedHistoricalCandles, latestCandles);
    const markers = mergeMarkers(mergedHistoricalMarkers, remoteData?.markers || []);
    if (!usesRuntimeCandles && !remoteData) {
      return null;
    }
    const firstCandle = candles[0] || null;
    const lastCandle = candles[candles.length - 1] || null;
    return {
      accountId: remoteData?.accountId ?? accountId,
      botConfigId: remoteData?.botConfigId ?? botConfigId,
      symbol: remoteData?.symbol || symbol,
      timeframe: remoteData?.timeframe || timeframe,
      bars: candles.length,
      candles,
      markers,
      sourceMode: usesRuntimeCandles ? "bot_runtime_ws" : (remoteData?.sourceMode || "bot_runtime"),
      sourceLabel: usesRuntimeCandles
        ? (String(runtimeSourceLabel || "").trim() || "bot runtime websocket")
        : (remoteData?.sourceLabel || "bot runtime"),
      visibleFrom: firstCandle ? new Date(firstCandle.time * 1000).toISOString() : null,
      visibleTo: lastCandle ? new Date(lastCandle.time * 1000).toISOString() : null,
      latestOrderTime: remoteData?.latestOrderTime ?? null,
      beforeTime: remoteData?.beforeTime ?? null,
    };
  }, [
    accountId,
    botConfigId,
    mergedHistoricalCandles,
    mergedHistoricalMarkers,
    normalizedRuntimeCandles,
    remoteData,
    runtimeSourceLabel,
    symbol,
    timeframe,
    usesRuntimeCandles,
  ]);

  const markersByTime = useMemo(() => {
    const map = new Map<number, ChartMarkerPayload[]>();
    for (const marker of data?.markers || []) {
      const key = Number(marker.time || 0);
      const bucket = map.get(key) || [];
      bucket.push(marker);
      map.set(key, bucket);
    }
    return map;
  }, [data?.markers]);

  const markersById = useMemo(() => {
    const map = new Map<string, ChartMarkerPayload>();
    for (const marker of data?.markers || []) {
      map.set(String(marker.id), marker);
    }
    return map;
  }, [data?.markers]);

  const tradePathsByTicket = useMemo(() => {
    const map = new Map<number, TradePathPayload>();
    for (const marker of data?.markers || []) {
      const ticketId = Number(marker.ticketId || 0);
      if (ticketId <= 0) continue;

      const existing = map.get(ticketId) || {
        ticketId,
        side: marker.side,
        volume: marker.volume,
        openMarker: null,
        closeMarker: null,
        lineColor: marker.side === "BUY" ? "#0f766e" : "#b45309",
      };

      if (marker.action === "OPEN") {
        existing.openMarker = marker;
      } else if (marker.action === "CLOSE") {
        existing.closeMarker = marker;
        existing.lineColor = marker.netProfit >= 0 ? "#0f766e" : "#dc2626";
      }

      map.set(ticketId, existing);
    }
    return map;
  }, [data?.markers]);

  const liveQuotePrice = useMemo(() => {
    const bid = Number(runtimeQuote?.bid || 0);
    const ask = Number(runtimeQuote?.ask || 0);
    if (bid > 0) return bid;
    if (ask > 0) return ask;
    return null;
  }, [runtimeQuote?.ask, runtimeQuote?.bid]);
  const liveQuoteEpoch = useMemo(() => normalizeQuoteEpoch(runtimeQuote?.time), [runtimeQuote?.time]);
  const displayCandles = useMemo(
    () => buildDisplayCandles(data?.candles, data?.timeframe || timeframe, liveQuotePrice, liveQuoteEpoch),
    [data?.candles, data?.timeframe, liveQuoteEpoch, liveQuotePrice, timeframe],
  );
  const pricePrecision = useMemo(
    () => inferPricePrecision(data?.symbol || symbol, displayCandles, data?.markers, runtimeQuote),
    [data?.markers, data?.symbol, displayCandles, runtimeQuote, symbol],
  );
  const priceMinMove = useMemo(() => minMoveForPrecision(pricePrecision), [pricePrecision]);

  useEffect(() => {
    markersByIdRef.current = markersById;
    markersByTimeRef.current = markersByTime;
  }, [markersById, markersByTime]);

  useEffect(() => {
    remoteDataRef.current = remoteData;
  }, [remoteData]);

  useEffect(() => {
    showMarkersRef.current = showMarkers;
    if (!showMarkers) {
      setHoverTooltip(null);
      setHoverPathOverlay(null);
    }
  }, [showMarkers]);

  useEffect(() => {
    if (!symbol) {
      setRemoteData(null);
      setLoading(false);
      return;
    }

    let isActive = true;
    let controller: AbortController | null = null;
    let pollTimer: number | null = null;
    const refreshMs = Math.max(0, Number(pollMs ?? (live ? 30000 : 0)) || 0);

    const load = async (silent: boolean) => {
      controller?.abort();
      controller = new AbortController();
      if (!silent) {
        setLoading(true);
      } else {
        setRefreshing(true);
      }

      try {
        const requestBars = usesRuntimeCandles
          ? Math.min(2000, silent ? Math.min(400, Math.max(120, Math.floor(bars / 2))) : Math.max(300, bars))
          : (silent && live ? liveRefreshBarsForTimeframe(timeframe, bars) : bars);
        const response = await api.get(usesRuntimeCandles ? "/trading/chart_markers" : "/trading/chart_orders", {
          params: {
            accountId: accountId || undefined,
            botConfigId: botConfigId || undefined,
            symbol,
            timeframe,
            ...(usesRuntimeCandles
              ? { limit: requestBars }
              : { bars: requestBars, anchorMode: live ? "market" : undefined }),
            includeArchived: includeArchived ? 1 : 0,
          },
          signal: controller.signal,
        });
        if (!isActive) return;
        const incoming = (response.data?.data || null) as OrderAnnotatedChartData | null;
        setRemoteData((prev) => (
          silent && live && !usesRuntimeCandles
            ? mergeRemoteChartData(prev, incoming, Math.max(60, bars))
            : incoming
        ));
        setError(null);
      } catch (fetchError: any) {
        if (!isActive || fetchError?.isCanceled) return;
        const hasExistingData = Boolean(
          usesRuntimeCandles
            ? normalizedRuntimeCandles.length
            : ((remoteDataRef.current?.candles?.length || 0) > 0 || (remoteDataRef.current?.markers?.length || 0) > 0),
        );
        if (!silent || !hasExistingData) {
          setError(fetchError?.message || "Failed to load chart data");
        }
      } finally {
        if (!isActive) return;
        setLoading(false);
        setRefreshing(false);
      }
    };

    void load(false);
    if (refreshMs > 0) {
      pollTimer = window.setInterval(() => {
        void load(true);
      }, refreshMs);
    }

    return () => {
      isActive = false;
      controller?.abort();
      if (pollTimer) {
        window.clearInterval(pollTimer);
      }
    };
  }, [accountId, bars, botConfigId, includeArchived, live, normalizedRuntimeCandles.length, pollMs, reloadToken, symbol, timeframe, usesRuntimeCandles]);

  useEffect(() => {
    hasInitializedViewportRef.current = false;
    baseCandlesRef.current = [];
    liveQuoteCandleRef.current = null;
    historyLoadInFlightRef.current = false;
    pendingHistoryViewportShiftRef.current = null;
    setHistoricalData({ candles: [], markers: [] });
    setHasMoreHistory(true);
    setLoadingHistory(false);
    setHoverTooltip(null);
    setHoverPathOverlay(null);
  }, [symbol, timeframe]);

  const loadMoreHistory = useCallback(async () => {
    if (historyLoadInFlightRef.current || loadingHistory || !hasMoreHistory) {
      return;
    }
    if (!symbol || !botConfigId) {
      setHasMoreHistory(false);
      return;
    }

    const currentCandles = data?.candles || [];
    const oldestCandle = currentCandles[0] || null;
    if (!oldestCandle) {
      return;
    }

    const pageBars = Math.min(MAX_CHART_HISTORY_PAGE_BARS, Math.max(MIN_CHART_HISTORY_PAGE_BARS, bars));
    const beforeEpoch = Math.max(1, Number(oldestCandle.time || 0) - timeframeSeconds(timeframe));
    const previousRange = chartRef.current?.timeScale().getVisibleLogicalRange?.() || null;

    historyLoadInFlightRef.current = true;
    setLoadingHistory(true);

    try {
      const response = await api.get("/trading/chart_orders", {
        params: {
          accountId: accountId || undefined,
          botConfigId: botConfigId || undefined,
          symbol,
          timeframe,
          bars: pageBars,
          beforeTime: new Date(beforeEpoch * 1000).toISOString(),
          anchorMode: live ? "market" : undefined,
          includeArchived: includeArchived ? 1 : 0,
        },
      });
      const payload = (response.data?.data || null) as OrderAnnotatedChartData | null;
      const pageCandles = (payload?.candles || [])
        .map((candle) => ({
          time: Number(candle?.time || 0),
          open: Number(candle?.open || 0),
          high: Number(candle?.high || 0),
          low: Number(candle?.low || 0),
          close: Number(candle?.close || 0),
          volume: Number(candle?.volume || 0),
        }))
        .filter((candle) => candle.time > 0 && candle.open > 0 && candle.high > 0 && candle.low > 0 && candle.close > 0)
        .sort((left, right) => left.time - right.time);
      const uniqueOlderCandles = pageCandles.filter((candle) => candle.time < oldestCandle.time);

      if (uniqueOlderCandles.length > 0 && previousRange) {
        pendingHistoryViewportShiftRef.current = {
          addedBars: uniqueOlderCandles.length,
          from: Number(previousRange.from),
          to: Number(previousRange.to),
        };
      }

      if (uniqueOlderCandles.length > 0 || (payload?.markers || []).length > 0) {
        setHistoricalData((prev) => ({
          candles: mergeCandles(uniqueOlderCandles, prev.candles),
          markers: mergeMarkers(payload?.markers || [], prev.markers),
        }));
      }

      setHasMoreHistory(uniqueOlderCandles.length > 0);
    } catch (fetchError: any) {
      setError(fetchError?.message || "Failed to load older chart history");
    } finally {
      historyLoadInFlightRef.current = false;
      setLoadingHistory(false);
    }
  }, [accountId, bars, botConfigId, data?.candles, hasMoreHistory, includeArchived, live, loadingHistory, symbol, timeframe]);

  useEffect(() => {
    loadMoreHistoryRef.current = loadMoreHistory;
  }, [loadMoreHistory]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) {
      return;
    }

    const chart = createChart(container, {
      width: container.clientWidth,
      height: container.clientHeight,
      layout: {
        background: { type: ColorType.Solid, color: "#ffffff" },
        textColor: "#475569",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
      },
      grid: {
        vertLines: { color: "#eef2f7" },
        horzLines: { color: "#eef2f7" },
      },
      rightPriceScale: {
        borderColor: "#e2e8f0",
        scaleMargins: { top: 0.12, bottom: 0.12 },
      },
      timeScale: {
        borderColor: "#e2e8f0",
        timeVisible: true,
        secondsVisible: false,
        rightOffset: 6,
        barSpacing: 9,
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: {
          color: "#cbd5e1",
          labelBackgroundColor: "#0f172a",
        },
        horzLine: {
          visible: false,
          labelVisible: false,
          color: "#cbd5e1",
          labelBackgroundColor: "#0f172a",
        },
      },
      localization: {
        locale: typeof navigator !== "undefined" ? navigator.language : "en-US",
      },
    });

    const candleSeries = chart.addSeries(CandlestickSeries, {
      upColor: "#14b8a6",
      downColor: "#ef4444",
      borderVisible: false,
      wickUpColor: "#14b8a6",
      wickDownColor: "#ef4444",
      priceLineVisible: false,
      lastValueVisible: false,
      priceFormat: {
        type: "price",
        precision: pricePrecision,
        minMove: priceMinMove,
      },
    });
    candleSeriesRef.current = candleSeries;

    const handleCrosshairMove = (param: MouseEventParams<Time>) => {
      if (!showMarkersRef.current || !container) {
        setHoverTooltip(null);
        return;
      }

      const point = param.point;
      if (!point || point.x < 0 || point.y < 0 || point.x > container.clientWidth || point.y > container.clientHeight) {
        setHoverTooltip(null);
        return;
      }

      let hoveredMarkers: ChartMarkerPayload[] = [];
      const hoveredId = String(param.hoveredObjectId ?? "").trim();
      const markerMapById = markersByIdRef.current;
      const markerMapByTime = markersByTimeRef.current;
      
      let targetTime: number | null = null;
      if (hoveredId && markerMapById.has(hoveredId)) {
        targetTime = Number(markerMapById.get(hoveredId)!.time);
      } else if (typeof param.time === "number") {
        targetTime = Number(param.time);
      }

      if (targetTime !== null) {
        hoveredMarkers = markerMapByTime.get(targetTime) || [];
      }

      if (hoveredMarkers.length === 0) {
        setHoverTooltip(null);
        return;
      }

      const activeTicketId = Number(hoveredMarkers[0]?.ticketId || 0) || null;

      const nextTooltip: HoverTooltipState = {
        x: Math.min(point.x + 14, Math.max(12, container.clientWidth - 260)),
        y: Math.min(point.y + 14, Math.max(12, container.clientHeight - 180)),
        markers: hoveredMarkers,
        activeTicketId,
      };

      setHoverTooltip((prev) => {
        if (
          prev &&
          prev.x === nextTooltip.x &&
          prev.y === nextTooltip.y &&
          prev.activeTicketId === nextTooltip.activeTicketId &&
          prev.markers.length === nextTooltip.markers.length &&
          prev.markers.every((marker, index) => marker.id === nextTooltip.markers[index]?.id)
        ) {
          return prev;
        }
        return nextTooltip;
      });
    };
    chart.subscribeCrosshairMove(handleCrosshairMove);

    const handleVisibleLogicalRangeChange = (range: { from: number; to: number } | null) => {
      if (!range || historyLoadInFlightRef.current) {
        return;
      }
      if (Number(range.from) > 25) {
        return;
      }
      void loadMoreHistoryRef.current?.();
    };
    chart.timeScale().subscribeVisibleLogicalRangeChange(handleVisibleLogicalRangeChange);

    const resizeObserver = new ResizeObserver((entries) => {
      const next = entries[0];
      if (!next) return;
      const { width, height } = next.contentRect;
      chart.applyOptions({
        width: Math.max(320, Math.floor(width)),
        height: Math.max(260, Math.floor(height)),
      });
    });
    resizeObserver.observe(container);

    chartRef.current = chart;

    return () => {
      chart.unsubscribeCrosshairMove(handleCrosshairMove);
      chart.timeScale().unsubscribeVisibleLogicalRangeChange(handleVisibleLogicalRangeChange);
      setHoverTooltip(null);
      setHoverPathOverlay(null);
      resizeObserver.disconnect();
      chart.remove();
      chartRef.current = null;
      candleSeriesRef.current = null;
      markerPluginRef.current = null;
      livePriceLineRef.current = null;
    };
  }, [priceMinMove, pricePrecision, symbol, timeframe]);

  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) {
      return;
    }

    chart.applyOptions({
      rightPriceScale: {
        borderColor: "#e2e8f0",
        scaleMargins: { top: 0.12, bottom: 0.12 },
        minimumWidth: pricePrecision >= 5 ? 92 : pricePrecision >= 3 ? 82 : 72,
      },
    });
    candleSeries.applyOptions({
      priceFormat: {
        type: "price",
        precision: pricePrecision,
        minMove: priceMinMove,
      },
    });
  }, [priceMinMove, pricePrecision]);

  useEffect(() => {
    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    if (!chart || !candleSeries) {
      return;
    }

    const candleData: CandlestickData<UTCTimestamp>[] = displayCandles.map((candle) => ({
      time: toChartTime(candle.time),
      open: candle.open,
      high: candle.high,
      low: candle.low,
      close: candle.close,
    }));
    baseCandlesRef.current = displayCandles;
    liveQuoteCandleRef.current = displayCandles.length > 0
      ? displayCandles[Math.max(0, displayCandles.length - 1)]
      : null;
    candleSeries.setData(candleData);

    const pendingViewportShift = pendingHistoryViewportShiftRef.current;
    if (pendingViewportShift && pendingViewportShift.addedBars > 0) {
      chart.timeScale().setVisibleLogicalRange({
        from: pendingViewportShift.from + pendingViewportShift.addedBars,
        to: pendingViewportShift.to + pendingViewportShift.addedBars,
      });
      pendingHistoryViewportShiftRef.current = null;
      return;
    }

    if (!hasInitializedViewportRef.current && candleData.length > 0) {
      const totalBars = candleData.length;
      const defaultVisibleBars = showMarkers ? 90 : 110;
      const from = Math.max(0, totalBars - defaultVisibleBars);
      const to = Math.max(defaultVisibleBars, totalBars + 6);
      chart.timeScale().setVisibleLogicalRange({ from, to });
      chart.timeScale().scrollToRealTime();
      hasInitializedViewportRef.current = true;
    }
  }, [displayCandles, showMarkers]);

  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) {
      return;
    }

    if (showMarkers) {
      const markerData: SeriesMarker<Time>[] = (data?.markers || []).map((marker) => ({
        id: marker.id,
        time: toChartTime(marker.time),
        position: marker.position,
        shape: marker.shape,
        color: marker.color,
        price: marker.price,
        size: marker.action === "OPEN" ? 1.0 : 1.3,
      }));

      if (!markerPluginRef.current) {
        markerPluginRef.current = createSeriesMarkers(candleSeries, markerData, {
          zOrder: "top",
        });
      } else {
        markerPluginRef.current.setMarkers(markerData);
      }
    } else if (markerPluginRef.current) {
      markerPluginRef.current.setMarkers([]);
    }
  }, [data?.markers, showMarkers]);

  useEffect(() => {
    const candleSeries = candleSeriesRef.current;
    if (!candleSeries) {
      return;
    }

    const livePrice = liveQuotePrice
      ?? (displayCandles.length ? displayCandles[displayCandles.length - 1]?.close ?? null : null);

    if (livePrice == null) {
      if (livePriceLineRef.current) {
        candleSeries.removePriceLine(livePriceLineRef.current);
      }
      livePriceLineRef.current = null;
      return;
    }

    const lineOptions = {
      price: livePrice,
      color: "#14b8a6",
      lineWidth: 1 as const,
      lineStyle: LineStyle.Dashed,
      axisLabelVisible: true,
      title: "",
    };

    if (livePriceLineRef.current) {
      candleSeries.removePriceLine(livePriceLineRef.current);
    }

    livePriceLineRef.current = candleSeries.createPriceLine(lineOptions);
  }, [displayCandles, liveQuotePrice, liveQuoteEpoch]);

  useEffect(() => {
    if (!showMarkers) {
      setHoverPathOverlay(null);
      return;
    }

    const chart = chartRef.current;
    const candleSeries = candleSeriesRef.current;
    const activeTicketId = hoverTooltip?.activeTicketId || null;
    const activeTrade = activeTicketId ? tradePathsByTicket.get(activeTicketId) || null : null;
    const latestVisibleCandle = liveQuoteCandleRef.current || displayCandles[displayCandles.length - 1] || null;

    if (!chart || !candleSeries || !activeTrade?.openMarker) {
      setHoverPathOverlay(null);
      return;
    }

    const endTime = activeTrade.closeMarker?.time ?? latestVisibleCandle?.time ?? null;
    const endPrice = activeTrade.closeMarker?.price ?? latestVisibleCandle?.close ?? null;
    if (endTime == null || endPrice == null) {
      setHoverPathOverlay(null);
      return;
    }

    const x1 = chart.timeScale().timeToCoordinate(toChartTime(activeTrade.openMarker.time));
    const x2 = chart.timeScale().timeToCoordinate(toChartTime(Number(endTime)));
    const y1 = candleSeries.priceToCoordinate(activeTrade.openMarker.price);
    const y2 = candleSeries.priceToCoordinate(Number(endPrice));

    if (x1 == null || x2 == null || y1 == null || y2 == null) {
      setHoverPathOverlay(null);
      return;
    }

    setHoverPathOverlay({
      x1,
      y1,
      x2,
      y2,
      color: activeTrade.lineColor,
    });
  }, [displayCandles, hoverTooltip?.activeTicketId, runtimeQuote, showMarkers, tradePathsByTicket]);

  const visibleMarkers = data?.markers || [];
  const latestCandle = liveQuoteCandleRef.current || displayCandles[displayCandles.length - 1] || null;
  const latestPrice = liveQuotePrice ?? latestCandle?.close ?? null;
  const latestBarEpoch = liveQuoteEpoch ?? latestCandle?.time ?? null;
  const latestBarTimeIso = latestBarEpoch ? new Date(latestBarEpoch * 1000).toISOString() : null;
  const latestBarTime = latestBarTimeIso ? formatDateTime(latestBarTimeIso) : "-";
  const activeTradePath = hoverTooltip?.activeTicketId
    ? tradePathsByTicket.get(hoverTooltip.activeTicketId) || null
    : null;
  const liveBadgeText = pollMs && pollMs > 0 ? `${Math.max(1, Math.round(pollMs / 1000))}s refresh` : "Live refresh";
  const resolvedSourceLabel = data?.sourceLabel || "Loading chart source";
  const resolvedSubtitle = subtitle
    ? `${subtitle} • ${resolvedSourceLabel}`
    : `${data?.symbol || symbol} • ${data?.timeframe || timeframe} • ${resolvedSourceLabel}`;

  return (
    <div className={cn("bg-white border rounded-xl shadow-sm overflow-hidden", className)}>
      <div className="border-b border-border px-4 py-3">
        <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
          <div className="space-y-1">
            <div className="flex items-center gap-2">
              <h3 className="text-sm font-semibold text-foreground">{title}</h3>
              {(live || (pollMs ?? 0) > 0) && (
                <span className="inline-flex items-center gap-1 rounded-full bg-success/10 px-2 py-0.5 text-[11px] font-medium text-success">
                  <span className="h-1.5 w-1.5 rounded-full bg-success animate-pulse" />
                  {liveBadgeText}
                </span>
              )}
            </div>
            <p className="text-xs text-muted-foreground">{resolvedSubtitle}</p>
          </div>

          <div className="flex flex-wrap items-center justify-end gap-2 text-xs text-muted-foreground">
            {headerContent}
            {refreshing && <Loader2 className="h-3.5 w-3.5 animate-spin" />}
            {loadingHistory && <span>Loading history...</span>}
            <span>{showMarkers ? `${visibleMarkers.length} order events` : `${data?.candles.length || 0} candles`}</span>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="h-8 gap-1.5"
              onClick={() => setReloadToken((prev) => prev + 1)}
            >
              <RefreshCcw className="h-3.5 w-3.5" />
              Refresh
            </Button>
          </div>
        </div>

        {showMarkers ? (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            <span className="inline-flex items-center gap-1 rounded-full bg-emerald-50 px-2 py-1 text-emerald-700">
              <span className="h-2 w-2 rounded-full bg-emerald-600" />
              Open buy
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-teal-50 px-2 py-1 text-teal-700">
              <span className="h-2 w-2 rounded-full bg-teal-700" />
              Close buy
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-rose-50 px-2 py-1 text-rose-700">
              <span className="h-2 w-2 rounded-full bg-rose-600" />
              Open sell
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-amber-50 px-2 py-1 text-amber-700">
              <span className="h-2 w-2 rounded-full bg-amber-600" />
              Close sell
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1 text-slate-600">
              Latest order {formatDateTime(data?.latestOrderTime)}
            </span>
          </div>
        ) : (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            <span className="inline-flex items-center gap-1 rounded-full bg-sky-50 px-2 py-1 text-sky-700">
              <span className="h-2 w-2 rounded-full bg-sky-600" />
              Last price {latestPrice != null ? latestPrice.toFixed(pricePrecision) : "-"}
            </span>
            <span className="inline-flex items-center gap-1 rounded-full bg-slate-100 px-2 py-1 text-slate-600">
              Latest bar {latestBarTime}
            </span>
          </div>
        )}
      </div>

      <div className="relative h-[440px]">
        {loading && (
          <div className="absolute inset-0 z-10 flex items-center justify-center bg-white/90">
            <div className="flex items-center gap-2 text-sm text-muted-foreground">
              <Loader2 className="h-4 w-4 animate-spin" />
              Loading market data...
            </div>
          </div>
        )}

        {!loading && error && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-6 text-center">
            <AlertTriangle className="h-6 w-6 text-amber-500" />
            <div className="space-y-1">
              <p className="text-sm font-medium text-foreground">Chart unavailable</p>
              <p className="text-sm text-muted-foreground">{error}</p>
            </div>
          </div>
        )}

        {!loading && !error && (!data || data.candles.length === 0) && (
          <div className="absolute inset-0 flex items-center justify-center px-6 text-center text-sm text-muted-foreground">
            No candle data returned for this symbol yet.
          </div>
        )}

        {!loading && !error && hoverTooltip && showMarkers && (
          <div
            className="pointer-events-none absolute z-20 w-[240px] rounded-xl border border-slate-200 bg-white/95 p-3 shadow-xl backdrop-blur"
            style={{ left: hoverTooltip.x, top: hoverTooltip.y }}
          >
            <div className="mb-2 text-[11px] font-semibold uppercase tracking-[0.16em] text-slate-500">
              Order details
            </div>
            <div className="space-y-2">
              {hoverTooltip.markers.map((marker) => (
                <div key={marker.id} className="rounded-lg bg-slate-50 px-2 py-2 text-xs text-slate-700">
                  <div className="flex items-center justify-between gap-2">
                    <span className="font-semibold" style={{ color: marker.color }}>
                      {marker.action} {marker.side}
                    </span>
                    <span>#{marker.ticketId}</span>
                  </div>
                  <div className="mt-1">Volume {marker.volume.toFixed(2)} at {marker.price.toFixed(pricePrecision)}</div>
                  <div>Time {formatDateTime(marker.actualTime)}</div>
                  {marker.action === "CLOSE" && (
                    <div>Net {marker.netProfit >= 0 ? "+" : ""}{formatMarkerValue(marker.netProfit)}</div>
                  )}
                </div>
              ))}
              {activeTradePath?.openMarker && (
                <div className="rounded-lg border border-slate-200 bg-white px-2 py-2 text-xs text-slate-700">
                  <div className="mb-1 font-semibold text-slate-900">Trade path</div>
                  <div>
                    From {formatDateTime(activeTradePath.openMarker.actualTime)} @ {activeTradePath.openMarker.price.toFixed(pricePrecision)}
                  </div>
                  <div>
                    To {activeTradePath.closeMarker?.actualTime ? formatDateTime(activeTradePath.closeMarker.actualTime) : latestBarTime} @ {(
                      activeTradePath.closeMarker?.price ?? latestPrice ?? activeTradePath.openMarker.price
                    ).toFixed(pricePrecision)}
                  </div>
                </div>
              )}
            </div>
          </div>
        )}

        <div ref={containerRef} className="h-full w-full" />
        {!loading && !error && hoverPathOverlay && showMarkers && (
          <svg className="pointer-events-none absolute inset-0 z-10 h-full w-full overflow-visible">
            <line
              x1={hoverPathOverlay.x1}
              y1={hoverPathOverlay.y1}
              x2={hoverPathOverlay.x2}
              y2={hoverPathOverlay.y2}
              stroke={hoverPathOverlay.color}
              strokeWidth="2.5"
              strokeDasharray="6 6"
              strokeLinecap="round"
              opacity="0.95"
            />
            <circle cx={hoverPathOverlay.x1} cy={hoverPathOverlay.y1} r="4" fill={hoverPathOverlay.color} />
            <circle cx={hoverPathOverlay.x2} cy={hoverPathOverlay.y2} r="4" fill={hoverPathOverlay.color} />
          </svg>
        )}
      </div>
    </div>
  );
}
