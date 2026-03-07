import { useState, useEffect, useMemo } from "react";
import { ChevronLeft, ChevronRight, TrendingUp, TrendingDown, BarChart3, Loader2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { api } from "@/lib/api";
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from "@/components/ui/dialog";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Switch } from "@/components/ui/switch";
import { Label } from "@/components/ui/label";

interface DayData {
  date: number;
  profit: number | null;
  trades: number;
  winRate: number;
}

interface CalendarSummary {
  totalProfit: number;
  totalTrades: number;
  tradingDays: number;
  profitableDays: number;
  averageWinRate: number;
}

interface TradeHistoryRow {
  ticketId: number;
  accountId: string;
  symbol: string;
  type: string;
  status: string;
  volume: number;
  openPrice: number | null;
  closePrice: number | null;
  commission: number;
  swap: number;
  profit: number;
  netProfit?: number;
  openTime: string | null;
  closeTime: string | null;
}

interface TradeHistorySummary {
  date: string;
  totalTrades: number;
  netProfit: number;
  wins: number;
  losses: number;
}

interface TradingAccountSource {
  id: string;
  broker_name: string | null;
  server_name: string | null;
  mt5_login_id: string | null;
}

const weekDays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
const ALL_ACCOUNTS_FILTER = "__all_accounts__";
const INCLUDE_ARCHIVED_STORAGE_KEY = "profit-calendar-include-archived";
const toDateKey = (year: number, monthZeroBased: number, day: number): string => {
  const mm = String(monthZeroBased + 1).padStart(2, "0");
  const dd = String(day).padStart(2, "0");
  return `${year}-${mm}-${dd}`;
};

const formatPriceOrDash = (value: number | null | undefined): string => {
  const parsed = Number(value ?? 0);
  if (!Number.isFinite(parsed) || parsed <= 0) return "-";
  return parsed.toFixed(5);
};

export default function ProfitCalendar() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDay, setSelectedDay] = useState<DayData | null>(null);
  const [monthData, setMonthData] = useState<DayData[]>([]);
  const [summary, setSummary] = useState<CalendarSummary>({
    totalProfit: 0,
    totalTrades: 0,
    tradingDays: 0,
    profitableDays: 0,
    averageWinRate: 0,
  });
  const [loading, setLoading] = useState(true);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [tradeHistory, setTradeHistory] = useState<TradeHistoryRow[]>([]);
  const [historyLoadedForDate, setHistoryLoadedForDate] = useState<string>("");
  const [historyOpen, setHistoryOpen] = useState(false);
  const [historySummary, setHistorySummary] = useState<TradeHistorySummary | null>(null);
  const [tradingAccounts, setTradingAccounts] = useState<TradingAccountSource[]>([]);
  const [calendarAccountFilter, setCalendarAccountFilter] = useState<string>(ALL_ACCOUNTS_FILTER);
  const [includeArchived, setIncludeArchived] = useState<boolean>(() => {
    if (typeof window === "undefined") return true;
    const raw = window.localStorage.getItem(INCLUDE_ARCHIVED_STORAGE_KEY);
    if (raw === "0") return false;
    if (raw === "1") return true;
    return true;
  });

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const selectedAccountId = calendarAccountFilter === ALL_ACCOUNTS_FILTER ? null : calendarAccountFilter;

  const firstDayOfMonth = new Date(year, month, 1).getDay();
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const monthName = currentDate.toLocaleString("default", { month: "long", year: "numeric" });

  useEffect(() => {
    if (typeof window === "undefined") return;
    window.localStorage.setItem(INCLUDE_ARCHIVED_STORAGE_KEY, includeArchived ? "1" : "0");
  }, [includeArchived]);

  useEffect(() => {
    let cancelled = false;
    const fetchTradingAccounts = async () => {
      try {
        const res = await api.get("/trading/accounts_with_bots", {
          params: { includeArchived: includeArchived ? 1 : 0 },
        });
        const rows = Array.isArray(res.data?.data) ? res.data.data : [];
        if (cancelled) return;
        const mapped = rows.map((row: any) => ({
          id: String(row?.id || ""),
          broker_name: row?.broker_name ? String(row.broker_name) : null,
          server_name: row?.server_name ? String(row.server_name) : null,
          mt5_login_id: row?.mt5_login_id ? String(row.mt5_login_id) : null,
        })).filter((row: TradingAccountSource) => row.id);
        setTradingAccounts(mapped);
      } catch (error) {
        console.error("Failed to fetch trading accounts for calendar source", error);
      }
    };

    fetchTradingAccounts();
    return () => {
      cancelled = true;
    };
  }, [includeArchived]);

  useEffect(() => {
    const fetchCalendarData = async () => {
      setLoading(true);
      try {
        const params: Record<string, number | string> = {
          year,
          month: month + 1,
          includeArchived: includeArchived ? 1 : 0,
        };
        if (selectedAccountId) {
          params.accountId = selectedAccountId;
        }
        const res = await api.get("/trading/calendar", {
          params,
        });
        const result = res.data || {};
        const apiData: { date: number; profit: number; trades: number; winRate: number }[] = Array.isArray(result.data)
          ? result.data
          : [];
        const apiSummary: CalendarSummary | null = result.summary && typeof result.summary === "object"
          ? {
            totalProfit: Number(result.summary.totalProfit || 0),
            totalTrades: Number(result.summary.totalTrades || 0),
            tradingDays: Number(result.summary.tradingDays || 0),
            profitableDays: Number(result.summary.profitableDays || 0),
            averageWinRate: Number(result.summary.averageWinRate || 0),
          }
          : null;

        const byDay = new Map(apiData.map((d) => [Number(d.date), d]));
        const data: DayData[] = [];
        for (let day = 1; day <= daysInMonth; day++) {
          const apiDay = byDay.get(day);
          if (apiDay) {
            data.push({
              date: day,
              profit: Number(apiDay.profit || 0),
              trades: Number(apiDay.trades || 0),
              winRate: Number(apiDay.winRate || 0),
            });
          } else {
            data.push({ date: day, profit: null, trades: 0, winRate: 0 });
          }
        }
        setMonthData(data);

        if (apiSummary) {
          setSummary(apiSummary);
        } else {
          const fallbackProfit = data.reduce((sum, d) => sum + (d.profit || 0), 0);
          const fallbackTrades = data.reduce((sum, d) => sum + d.trades, 0);
          const fallbackTradingDays = data.filter((d) => d.profit !== null && d.trades > 0).length;
          const fallbackProfitDays = data.filter((d) => d.profit !== null && d.trades > 0 && (d.profit || 0) > 0).length;
          const fallbackAvgWin = fallbackTradingDays > 0
            ? data.reduce((sum, d) => sum + (d.trades > 0 ? d.winRate : 0), 0) / fallbackTradingDays
            : 0;
          setSummary({
            totalProfit: fallbackProfit,
            totalTrades: fallbackTrades,
            tradingDays: fallbackTradingDays,
            profitableDays: fallbackProfitDays,
            averageWinRate: fallbackAvgWin,
          });
        }
      } catch (error) {
        console.error("Failed to fetch calendar data", error);
      } finally {
        setLoading(false);
      }
    };

    fetchCalendarData();
  }, [year, month, daysInMonth, includeArchived, selectedAccountId]);

  const navigateMonth = (direction: number) => {
    setCurrentDate(new Date(year, month + direction, 1));
    setSelectedDay(null);
    setTradeHistory([]);
    setHistoryLoadedForDate("");
  };

  const selectedDateKey = selectedDay
    ? toDateKey(year, month, selectedDay.date)
    : "";

  const accountById = useMemo(() => {
    const map = new Map<string, TradingAccountSource>();
    for (const account of tradingAccounts) {
      if (!account.id) continue;
      map.set(account.id, account);
    }
    return map;
  }, [tradingAccounts]);

  const formatAccountLabel = (accountId: string): string => {
    const safeId = String(accountId || "").trim();
    if (!safeId) return "Unknown account";
    const account = accountById.get(safeId);
    if (!account) return `Account ${safeId.slice(0, 8)}`;

    const broker = String(account.broker_name || "").trim();
    const server = String(account.server_name || "").trim();
    const login = String(account.mt5_login_id || "").trim();

    const left = [broker, server].filter(Boolean).join(" / ");
    if (left && login) return `${left} (${login})`;
    if (left) return left;
    if (login) return `MT5 ${login}`;
    return `Account ${safeId.slice(0, 8)}`;
  };

  const historyAccountLabels = useMemo(() => {
    const ids = Array.from(
      new Set(
        tradeHistory
          .map((row) => String(row.accountId || "").trim())
          .filter((id) => id.length > 0)
      )
    );
    return ids.map((id) => formatAccountLabel(id));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [tradeHistory, accountById]);

  const calendarAccountOptions = useMemo(
    () =>
      tradingAccounts
        .filter((account) => Boolean(account.id))
        .map((account) => ({ id: account.id, label: formatAccountLabel(account.id) }))
        .sort((a, b) => a.label.localeCompare(b.label)),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [tradingAccounts, accountById]
  );

  const effectiveHistorySummary = useMemo<TradeHistorySummary>(() => {
    if (historySummary) {
      return {
        date: String(historySummary.date || selectedDateKey),
        totalTrades: Number(historySummary.totalTrades || tradeHistory.length || 0),
        netProfit: Number(historySummary.netProfit || 0),
        wins: Number(historySummary.wins || 0),
        losses: Number(historySummary.losses || 0),
      };
    }
    const wins = tradeHistory.filter((r) => Number(r.profit || 0) > 0).length;
    const losses = tradeHistory.filter((r) => Number(r.profit || 0) < 0).length;
    const netProfit = tradeHistory.reduce((sum, r) => sum + Number(r.profit || 0), 0);
    return {
      date: selectedDateKey,
      totalTrades: tradeHistory.length,
      netProfit,
      wins,
      losses,
    };
  }, [historySummary, tradeHistory, selectedDateKey]);

  useEffect(() => {
    if (calendarAccountFilter === ALL_ACCOUNTS_FILTER) return;
    if (!accountById.has(calendarAccountFilter)) {
      setCalendarAccountFilter(ALL_ACCOUNTS_FILTER);
    }
  }, [calendarAccountFilter, accountById]);

  useEffect(() => {
    setTradeHistory([]);
    setHistoryLoadedForDate("");
    setHistorySummary(null);
    setHistoryOpen(false);
  }, [selectedDateKey]);

  useEffect(() => {
    setSelectedDay(null);
    setTradeHistory([]);
    setHistoryLoadedForDate("");
    setHistorySummary(null);
    setHistoryOpen(false);
  }, [includeArchived, selectedAccountId]);

  const loadTradeHistory = async () => {
    if (!selectedDay) return;
    setHistoryLoading(true);
    try {
      const params: Record<string, number | string> = {
        year,
        month: month + 1,
        day: selectedDay.date,
        includeArchived: includeArchived ? 1 : 0,
      };
      if (selectedAccountId) {
        params.accountId = selectedAccountId;
      }
      const res = await api.get("/trading/history_by_day", {
        params,
      });
      const rows = Array.isArray(res.data?.data) ? res.data.data : [];
      const summary = res.data?.summary && typeof res.data.summary === "object" ? res.data.summary : null;
      setTradeHistory(rows as TradeHistoryRow[]);
      if (summary) {
        setHistorySummary({
          date: String(summary.date || selectedDateKey),
          totalTrades: Number(summary.totalTrades || rows.length || 0),
          netProfit: Number(summary.netProfit || 0),
          wins: Number(summary.wins || 0),
          losses: Number(summary.losses || 0),
        });
      } else {
        const normalizeRowNetProfit = (row: TradeHistoryRow) =>
          Number(row.netProfit ?? Number(row.profit || 0) + Number(row.commission || 0) + Number(row.swap || 0));
        const wins = (rows as TradeHistoryRow[]).filter((r) => normalizeRowNetProfit(r) > 0).length;
        const losses = (rows as TradeHistoryRow[]).filter((r) => normalizeRowNetProfit(r) < 0).length;
        const netProfit = (rows as TradeHistoryRow[]).reduce((sum, r) => sum + normalizeRowNetProfit(r), 0);
        setHistorySummary({
          date: selectedDateKey,
          totalTrades: rows.length,
          netProfit,
          wins,
          losses,
        });
      }
      setHistoryLoadedForDate(selectedDateKey);
      setHistoryOpen(true);
    } catch (error) {
      console.error("Failed to fetch trade history", error);
      setTradeHistory([]);
      setHistorySummary({
        date: selectedDateKey,
        totalTrades: 0,
        netProfit: 0,
        wins: 0,
        losses: 0,
      });
      setHistoryLoadedForDate(selectedDateKey);
      setHistoryOpen(true);
    } finally {
      setHistoryLoading(false);
    }
  };

  const getIntensity = (profit: number | null): string => {
    if (profit === null || profit === 0) return "bg-secondary hover:bg-secondary/80"; // Gray for no trades/data
    if (profit > 0) {
      if (profit > 300) return "bg-success";
      if (profit > 100) return "bg-success/70";
      return "bg-success/40";
    } else {
      if (profit < -300) return "bg-destructive";
      if (profit < -100) return "bg-destructive/70";
      return "bg-destructive/40";
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-foreground">Profit Calendar</h1>
          <p className="text-sm text-muted-foreground">Visualize your daily trading performance</p>
        </div>
        <div className="w-full max-w-[420px] rounded-lg border border-border bg-card px-3 py-3 space-y-3">
          <div className="space-y-1">
            <p className="text-xs text-muted-foreground">Filter by Account</p>
            <Select value={calendarAccountFilter} onValueChange={setCalendarAccountFilter}>
              <SelectTrigger className="h-9">
                <SelectValue placeholder="Select account" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value={ALL_ACCOUNTS_FILTER}>All accounts</SelectItem>
                {calendarAccountOptions.map((option) => (
                  <SelectItem key={option.id} value={option.id}>
                    {option.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <div className="flex items-center gap-2">
            <Switch
              id="calendar-include-archived"
              checked={includeArchived}
              onCheckedChange={setIncludeArchived}
            />
            <Label htmlFor="calendar-include-archived" className="cursor-pointer text-sm font-medium">
              Include archived bots/accounts
            </Label>
          </div>
        </div>
      </div>

      {loading ? (
        <div className="flex h-96 items-center justify-center">
          <Loader2 className="w-8 h-8 animate-spin text-primary" />
        </div>
      ) : (
        <>
          {/* Monthly Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="bg-white border text-card-foreground shadow-sm rounded-xl p-4 animate-slide-up">
              <p className="text-sm text-muted-foreground mb-1">Monthly P&L</p>
              <p className={cn("text-2xl font-bold font-mono", summary.totalProfit >= 0 ? "text-success" : "text-destructive")}>
                {summary.totalProfit >= 0 ? "+" : ""}${summary.totalProfit.toFixed(2)}
              </p>
            </div>
            <div className="bg-white border text-card-foreground shadow-sm rounded-xl p-4 animate-slide-up" style={{ animationDelay: "50ms" }}>
              <p className="text-sm text-muted-foreground mb-1">Total Trades</p>
              <p className="text-2xl font-bold font-mono text-foreground">{summary.totalTrades}</p>
            </div>
            <div className="bg-white border text-card-foreground shadow-sm rounded-xl p-4 animate-slide-up" style={{ animationDelay: "100ms" }}>
              <p className="text-sm text-muted-foreground mb-1">Avg Win Rate</p>
              <p className="text-2xl font-bold font-mono text-success">
                {summary.averageWinRate.toFixed(1)}%
              </p>
            </div>
            <div className="bg-white border text-card-foreground shadow-sm rounded-xl p-4 animate-slide-up" style={{ animationDelay: "150ms" }}>
              <p className="text-sm text-muted-foreground mb-1">Trading / Profit Days</p>
              <p className="text-2xl font-bold font-mono text-foreground">
                {summary.tradingDays} / {summary.profitableDays}
              </p>
            </div>
          </div>

          <div className="grid lg:grid-cols-3 gap-6">
            {/* Calendar */}
            <div className="lg:col-span-2 bg-white border text-card-foreground shadow-sm rounded-xl p-6 animate-slide-up" style={{ animationDelay: "200ms" }}>
              {/* Navigation */}
              <div className="flex items-center justify-between mb-6">
                <Button variant="ghost" size="icon" onClick={() => navigateMonth(-1)}>
                  <ChevronLeft className="w-5 h-5" />
                </Button>
                <h3 className="text-xl font-semibold">{monthName}</h3>
                <Button variant="ghost" size="icon" onClick={() => navigateMonth(1)}>
                  <ChevronRight className="w-5 h-5" />
                </Button>
              </div>

              {/* Week Days Header */}
              <div className="grid grid-cols-7 gap-2 mb-2">
                {weekDays.map((day) => (
                  <div key={day} className="text-center text-xs font-medium text-muted-foreground py-2">
                    {day}
                  </div>
                ))}
              </div>

              {/* Calendar Grid */}
              <div className="grid grid-cols-7 gap-2">
                {/* Empty cells for days before the first of the month */}
                {Array.from({ length: firstDayOfMonth }).map((_, i) => (
                  <div key={`empty-${i}`} className="aspect-square" />
                ))}

                {/* Day cells */}
                {monthData.map((day) => (
                  <button
                    key={day.date}
                    onClick={() => day.profit !== null && setSelectedDay(day)}
                    disabled={day.profit === null}
                    className={cn(
                      "aspect-square rounded-lg flex flex-col items-center justify-center transition-all p-1",
                      getIntensity(day.profit),
                      day.profit !== null && "hover:ring-2 hover:ring-primary cursor-pointer",
                      selectedDay?.date === day.date && "ring-2 ring-primary"
                    )}
                  >
                    <span className="text-xs text-white/90 font-medium">{day.date}</span>
                    {day.profit !== null && (
                      <span className="text-[10px] font-mono font-semibold mt-0.5 text-white">
                        {day.profit >= 0 ? "+" : ""}${Math.abs(day.profit).toFixed(2)}
                      </span>
                    )}
                  </button>
                ))}
              </div>

              {/* Legend */}
              <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-border">
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-success" />
                  <span className="text-xs text-muted-foreground">Profit</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-destructive" />
                  <span className="text-xs text-muted-foreground">Loss</span>
                </div>
                <div className="flex items-center gap-2">
                  <div className="w-3 h-3 rounded bg-muted/30" />
                  <span className="text-xs text-muted-foreground">No Trading</span>
                </div>
              </div>
            </div>

            {/* Day Details */}
            <div className="bg-white border text-card-foreground shadow-sm rounded-xl p-6 animate-slide-up" style={{ animationDelay: "250ms" }}>
              <h3 className="text-lg font-semibold mb-4">Day Details</h3>

              {selectedDay ? (
                <div className="space-y-6">
                  <div className="text-center p-6 rounded-lg bg-secondary/50">
                    <p className="text-sm text-muted-foreground mb-2">
                      {new Date(year, month, selectedDay.date).toLocaleDateString("en-US", {
                        weekday: "long",
                        month: "long",
                        day: "numeric",
                      })}
                    </p>
                    <p
                      className={cn(
                        "text-4xl font-bold font-mono",
                        selectedDay.profit! >= 0 ? "text-success" : "text-destructive"
                      )}
                    >
                      {selectedDay.profit! >= 0 ? "+" : ""}${Math.abs(selectedDay.profit!).toFixed(2)}
                    </p>
                  </div>

                  <div className="space-y-4">
                    <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                      <div className="flex items-center gap-2">
                        <BarChart3 className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Trades</span>
                      </div>
                      <span className="font-mono font-medium">{selectedDay.trades}</span>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                      <div className="flex items-center gap-2">
                        <TrendingUp className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Win Rate</span>
                      </div>
                      <span className="font-mono font-medium text-success">{selectedDay.winRate}%</span>
                    </div>

                    <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                      <div className="flex items-center gap-2">
                        <TrendingDown className="w-4 h-4 text-muted-foreground" />
                        <span className="text-sm text-muted-foreground">Growth</span>
                      </div>
                      <span
                        className={cn(
                          "font-mono font-medium",
                          selectedDay.profit! >= 0 ? "text-success" : "text-destructive"
                        )}
                      >
                        {/* Placeholder growth calc */}
                        {selectedDay.profit! >= 0 ? "+" : ""}
                        {((selectedDay.profit! / 10000) * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={loadTradeHistory}
                    disabled={historyLoading}
                  >
                    {historyLoading ? (
                      <span className="inline-flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin" />
                        Loading...
                      </span>
                    ) : (
                      "View Trade History"
                    )}
                  </Button>

                </div>
              ) : (
                <div className="text-center py-12 text-muted-foreground">
                  <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <p>Select a day to view details</p>
                </div>
              )}
            </div>
          </div>

          <Dialog open={historyOpen} onOpenChange={setHistoryOpen}>
            <DialogContent className="max-w-5xl max-h-[85vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>Trade History - {selectedDateKey || "-"}</DialogTitle>
                <DialogDescription className="space-y-1">
                  <span className="block">Detailed closed orders for the selected day from your MT5-synced history.</span>
                  <span className="block text-xs">
                    Source trading account{historyAccountLabels.length > 1 ? "s" : ""}:{" "}
                    {historyAccountLabels.length > 0 ? historyAccountLabels.join(", ") : "No trade rows on this day"}
                  </span>
                </DialogDescription>
              </DialogHeader>

              <div className="text-xs text-muted-foreground">
                {selectedAccountId
                  ? `Filtered account: ${formatAccountLabel(selectedAccountId)}`
                  : "Filtered account: All accounts"}
              </div>

              <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
                <div className="rounded-lg border border-border p-3">
                  <p className="text-xs text-muted-foreground">Net PnL</p>
                  <p className={cn("text-lg font-bold font-mono", (effectiveHistorySummary.netProfit || 0) >= 0 ? "text-success" : "text-destructive")}>
                    {(effectiveHistorySummary.netProfit || 0) >= 0 ? "+" : ""}${Math.abs(effectiveHistorySummary.netProfit || 0).toFixed(2)}
                  </p>
                </div>
                <div className="rounded-lg border border-border p-3">
                  <p className="text-xs text-muted-foreground">Total Trades</p>
                  <p className="text-lg font-bold font-mono">{effectiveHistorySummary.totalTrades || 0}</p>
                </div>
                <div className="rounded-lg border border-border p-3">
                  <p className="text-xs text-muted-foreground">Wins</p>
                  <p className="text-lg font-bold font-mono text-success">{effectiveHistorySummary.wins || 0}</p>
                </div>
                <div className="rounded-lg border border-border p-3">
                  <p className="text-xs text-muted-foreground">Losses</p>
                  <p className="text-lg font-bold font-mono text-destructive">{effectiveHistorySummary.losses || 0}</p>
                </div>
              </div>

              <div className="rounded-lg border border-border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Ticket</TableHead>
                      <TableHead>Account</TableHead>
                      <TableHead>Symbol</TableHead>
                      <TableHead>Type</TableHead>
                      <TableHead>Lot</TableHead>
                      <TableHead>Open</TableHead>
                      <TableHead>Close</TableHead>
                      <TableHead>Fees</TableHead>
                      <TableHead className="text-right">Profit</TableHead>
                      <TableHead>Closed Time</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {historyLoadedForDate === selectedDateKey && tradeHistory.length === 0 && (
                      <TableRow>
                        <TableCell colSpan={10} className="text-center text-muted-foreground py-8">
                          No trade history found for this day
                        </TableCell>
                      </TableRow>
                    )}
                    {tradeHistory.map((row) => {
                      const fee = Number(row.commission || 0) + Number(row.swap || 0);
                      const rowType = String(row.type || "").toUpperCase();
                      return (
                        <TableRow key={`${row.ticketId}-${row.closeTime || ""}`}>
                          <TableCell className="font-mono text-xs">#{row.ticketId}</TableCell>
                          <TableCell className="text-xs text-muted-foreground max-w-[180px] break-words">
                            {formatAccountLabel(row.accountId)}
                          </TableCell>
                          <TableCell>{row.symbol || "-"}</TableCell>
                          <TableCell>
                            <span className={cn("text-xs font-semibold", rowType === "BUY" ? "text-success" : rowType === "SELL" ? "text-destructive" : "text-foreground")}>
                              {rowType || "-"}
                            </span>
                          </TableCell>
                          <TableCell className="font-mono">{Number(row.volume || 0).toFixed(2)}</TableCell>
                          <TableCell className="font-mono">{formatPriceOrDash(row.openPrice)}</TableCell>
                          <TableCell className="font-mono">{formatPriceOrDash(row.closePrice)}</TableCell>
                          <TableCell className="font-mono">{fee >= 0 ? "+" : ""}${Math.abs(fee).toFixed(2)}</TableCell>
                          <TableCell className={cn("font-mono text-right", Number(row.profit || 0) >= 0 ? "text-success" : "text-destructive")}>
                            {Number(row.profit || 0) >= 0 ? "+" : ""}${Math.abs(Number(row.profit || 0)).toFixed(2)}
                          </TableCell>
                          <TableCell className="font-mono text-xs">
                            {row.closeTime
                              ? new Date(row.closeTime).toLocaleString("en-US", { hour12: false })
                              : "-"}
                          </TableCell>
                        </TableRow>
                      );
                    })}
                  </TableBody>
                </Table>
              </div>
            </DialogContent>
          </Dialog>
        </>
      )}
    </div>
  );
}
