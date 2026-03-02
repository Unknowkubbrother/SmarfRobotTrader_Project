import Link from "next/link";
import { useState, useEffect, useMemo, useRef } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { ArrowUpRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BotLiveState } from "@/hooks/useBotLiveState";
import { api } from "@/lib/api";

interface ChartDataPoint {
  balance: number;
  equity: number;
  timestamp: number;
}

interface ChartRenderPoint extends ChartDataPoint {
  label: string;
}

interface CalendarDayPoint {
  date: number;
  profit: number;
  trades: number;
}

const timeframes = ["1D", "1W", "1M", "3M", "1Y", "ALL"] as const;
type ChartTimeframe = (typeof timeframes)[number];

const timeframeMs: Record<Exclude<ChartTimeframe, "ALL">, number> = {
  "1D": 24 * 60 * 60 * 1000,
  "1W": 7 * 24 * 60 * 60 * 1000,
  "1M": 30 * 24 * 60 * 60 * 1000,
  "3M": 90 * 24 * 60 * 60 * 1000,
  "1Y": 365 * 24 * 60 * 60 * 1000,
};

const formatLabel = (ts: number, timeframe: ChartTimeframe): string => {
  const date = new Date(ts);
  if (timeframe === "1D") {
    return date.toLocaleTimeString("en-US", { hour12: false, hour: "2-digit", minute: "2-digit" });
  }
  if (timeframe === "1W") {
    return date.toLocaleString("en-US", { weekday: "short", hour: "2-digit" });
  }
  return date.toLocaleDateString("en-US", { month: "short", day: "numeric" });
};

const filterByTimeframe = (rows: ChartDataPoint[], timeframe: ChartTimeframe): ChartDataPoint[] => {
  if (timeframe === "ALL" || rows.length === 0) return rows;
  const cutoff = Date.now() - timeframeMs[timeframe];
  const filtered = rows.filter((row) => row.timestamp >= cutoff);
  return filtered.length > 0 ? filtered : rows.slice(-1);
};

interface PerformanceChartProps {
  liveState?: BotLiveState;
  botId?: string;
  className?: string;
}

export function PerformanceChart({ liveState, botId, className }: PerformanceChartProps) {
  const [selectedTimeframe, setSelectedTimeframe] = useState<ChartTimeframe>("1M");
  const [historyData, setHistoryData] = useState<ChartDataPoint[]>([]);
  const lastBotIdRef = useRef<string>("");
  const [historyLoaded, setHistoryLoaded] = useState(false);

  const isLive = !!liveState?.connected;
  const activeBotId = String(botId || liveState?.bot_config_id || "").trim();

  useEffect(() => {
    if (!activeBotId) return;
    if (lastBotIdRef.current && lastBotIdRef.current !== activeBotId) {
      // Avoid mixing equity histories when user switches selected bot.
      setHistoryData([]);
      setHistoryLoaded(false);
    }
    lastBotIdRef.current = activeBotId;
  }, [activeBotId]);

  useEffect(() => {
    if (!activeBotId || historyLoaded) return;
    let cancelled = false;

    const loadCalendarHistory = async () => {
      try {
        const now = new Date();
        const monthRefs = Array.from({ length: 12 }, (_, idx) => {
          const dt = new Date(now.getFullYear(), now.getMonth() - (11 - idx), 1);
          return { year: dt.getFullYear(), month: dt.getMonth() + 1 };
        });

        const responses = await Promise.all(
          monthRefs.map(({ year, month }) =>
            api
              .get("/trading/calendar", { params: { year, month } })
              .then((res) => ({ ok: true, data: res.data, year, month }))
              .catch(() => ({ ok: false, data: null, year, month }))
          )
        );

        const seeded: ChartDataPoint[] = [];
        let cumulative = 0.0;
        for (const item of responses) {
          if (!item.ok) continue;
          const rows: CalendarDayPoint[] = Array.isArray(item.data?.data) ? item.data.data : [];
          const monthRows = rows
            .map((r) => ({
              date: Number(r?.date || 0),
              profit: Number(r?.profit || 0),
              trades: Number(r?.trades || 0),
            }))
            .filter((r) => Number.isFinite(r.date) && r.date > 0 && r.trades > 0)
            .sort((a, b) => a.date - b.date);

          for (const row of monthRows) {
            cumulative += row.profit;
            const ts = Date.UTC(item.year, item.month - 1, row.date, 12, 0, 0);
            seeded.push({
              balance: cumulative,
              equity: cumulative,
              timestamp: ts,
            });
          }
        }

        if (cancelled || seeded.length === 0) return;

        const liveBalance = Number(liveState?.balance);
        if (Number.isFinite(liveBalance) && liveBalance > 0) {
          const offset = liveBalance - seeded[seeded.length - 1].balance;
          for (const point of seeded) {
            point.balance += offset;
            point.equity += offset;
          }
        }

        setHistoryData((prev) => {
          if (prev.length === 0) return seeded;
          const oldestLiveTs = prev[0].timestamp;
          const base = seeded.filter((p) => p.timestamp < oldestLiveTs);
          return [...base, ...prev];
        });
      } catch {
        // Fallback silently: chart will continue to work from live stream.
      } finally {
        if (!cancelled) setHistoryLoaded(true);
      }
    };

    loadCalendarHistory();
    return () => {
      cancelled = true;
    };
  }, [activeBotId, historyLoaded, liveState?.balance]);

  useEffect(() => {
    if (!isLive || !liveState) return;
    const balance = Number(liveState.balance);
    const equity = Number(liveState.equity);
    if (!Number.isFinite(balance) || !Number.isFinite(equity) || balance <= 0 || equity <= 0) {
      return;
    }

    setHistoryData((prev) => {
      const now = Date.now();
      const newPoint: ChartDataPoint = {
        balance,
        equity,
        timestamp: now,
      };

      const last = prev[prev.length - 1];
      if (!last) return [newPoint];
      const sameValue = last.balance === newPoint.balance && last.equity === newPoint.equity;
      if (sameValue && now - last.timestamp < 5000) {
        return prev;
      }
      if (now - last.timestamp < 2000) {
        // Replace latest point when updates are too frequent to keep chart smooth.
        return [...prev.slice(0, -1), newPoint];
      }

      const next = [...prev, newPoint];
      const maxKeep = 3000;
      return next.length > maxKeep ? next.slice(next.length - maxKeep) : next;
    });
  }, [isLive, liveState?.balance, liveState?.equity]);

  const chartData = useMemo<ChartRenderPoint[]>(() => {
    const scoped = filterByTimeframe(historyData, selectedTimeframe);
    return scoped.map((row) => ({
      ...row,
      label: formatLabel(row.timestamp, selectedTimeframe),
    }));
  }, [historyData, selectedTimeframe]);

  return (
    <div className={cn("bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full min-h-0 flex flex-col", className)}>
      <div className="flex items-center justify-between mb-6">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="font-semibold text-foreground">Portfolio Performance</h3>
            {isLive && (
              <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
                <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                Live
              </span>
            )}
          </div>
          <p className="text-sm text-muted-foreground">Balance & Equity over time</p>
        </div>
        <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary">
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition-all",
                selectedTimeframe === tf
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 min-h-[280px]">
        {chartData.length === 0 ? (
          <div className="h-full flex items-center justify-center text-sm text-muted-foreground">
            <div className="text-center space-y-2">
              <div>{isLive ? "Waiting for live balance/equity data..." : "Connect bot to show performance chart"}</div>
              <Link href="/calendar" className="inline-flex items-center gap-1 text-primary hover:underline">
                View history in Profit Calendar
                <ArrowUpRight className="w-3.5 h-3.5" />
              </Link>
            </div>
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(220, 90%, 56%)" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="hsl(220, 90%, 56%)" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="hsl(142, 72%, 42%)" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="hsl(142, 72%, 42%)" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" vertical={false} />
              <XAxis
                dataKey="label"
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 9%, 46%)", fontSize: 11 }}
                minTickGap={18}
              />
              <YAxis
                axisLine={false}
                tickLine={false}
                tick={{ fill: "hsl(220, 9%, 46%)", fontSize: 11 }}
                tickFormatter={(value) => `$${Number(value).toLocaleString()}`}
                domain={["auto", "auto"]}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: "hsl(0, 0%, 100%)",
                  border: "1px solid hsl(220, 13%, 91%)",
                  borderRadius: "8px",
                  boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
                }}
                labelFormatter={(_, payload) => {
                  const ts = payload?.[0]?.payload?.timestamp;
                  if (!ts) return "";
                  return new Date(ts).toLocaleString("en-US", { hour12: false });
                }}
                formatter={(value: number, name: string) => [`$${value.toLocaleString()}`, name]}
              />
              <Legend
                verticalAlign="top"
                align="right"
                height={36}
                iconType="circle"
              />
              <Area
                type="monotone"
                dataKey="balance"
                name="Balance"
                stroke="hsl(220, 90%, 56%)"
                strokeWidth={2}
                fill="url(#balanceGradient)"
                isAnimationActive={false}
              />
              <Area
                type="monotone"
                dataKey="equity"
                name="Equity"
                stroke="hsl(142, 72%, 42%)"
                strokeWidth={2}
                fill="url(#equityGradient)"
                isAnimationActive={false}
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  );
}
