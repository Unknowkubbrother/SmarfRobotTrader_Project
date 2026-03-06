import { Terminal, TrendingUp, AlertTriangle, CheckCircle, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import { Badge } from "@/components/ui/badge";
import type { BotLiveIntrabarReview, BotLiveLogEntry, BotLiveState } from "@/hooks/useBotLiveState";

interface LogEntry {
  id: string;
  timestamp: string;
  type: "info" | "analysis" | "action" | "warning" | "success";
  message: string;
  phase?: string;
  event?: string;
  isInsufficientFunds?: boolean;
}

function parseUtcLike(raw: string): Date | null {
  const text = String(raw || "").trim();
  if (!text) return null;

  const direct = new Date(text);
  if (!Number.isNaN(direct.getTime())) return direct;

  // Handle "YYYY-MM-DD HH:mm:ssZ" (space before time + trailing Z)
  if (/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}Z$/i.test(text)) {
    const patched = text.replace(" ", "T");
    const parsed = new Date(patched);
    if (!Number.isNaN(parsed.getTime())) return parsed;
  }
  return null;
}

function toLocalClockText(rawUtc?: string, fallback?: string): string {
  const parsed = parseUtcLike(String(rawUtc || ""));
  if (!parsed) {
    return String(fallback || "--:--:--");
  }
  return parsed.toLocaleTimeString([], { hour12: false });
}

function toLocalDateTimeText(rawUtc?: string): string {
  const parsed = parseUtcLike(String(rawUtc || ""));
  if (!parsed) {
    return String(rawUtc || "").trim();
  }
  const dateText = parsed.toLocaleDateString();
  const timeText = parsed.toLocaleTimeString([], { hour12: false });
  return `${dateText} ${timeText}`;
}

const typeConfig = {
  info: { icon: Bot, color: "text-primary" },
  analysis: { icon: TrendingUp, color: "text-muted-foreground" },
  action: { icon: Terminal, color: "text-foreground" },
  warning: { icon: AlertTriangle, color: "text-warning" },
  success: { icon: CheckCircle, color: "text-success" },
};

const HIDDEN_PHASES = new Set(["SEM", "MODEL", "GATE", "DECISION"]);

function normalizeMeta(meta: unknown): Record<string, unknown> | undefined {
  if (!meta || typeof meta !== "object" || Array.isArray(meta)) return undefined;
  const entries = Object.entries(meta as Record<string, unknown>);
  if (entries.length === 0) return undefined;
  return Object.fromEntries(entries.map(([k, v]) => [String(k), v]));
}

function toNum(value: unknown): number | null {
  if (typeof value === "number" && Number.isFinite(value)) return value;
  if (typeof value === "string") {
    const parsed = Number(value);
    if (Number.isFinite(parsed)) return parsed;
  }
  return null;
}

const INSUFFICIENT_FUNDS_HINTS = [
  "no money",
  "not enough money",
  "insufficient funds",
  "insufficient margin",
  "free margin",
];

function isInsufficientFundsLog(log: Partial<BotLiveLogEntry>): boolean {
  const phase = String(log?.phase || "").trim().toUpperCase();
  const event = String(log?.event || "").trim().toLowerCase();
  const message = String(log?.message || "").trim().toLowerCase();
  const reason = String(normalizeMeta(log?.meta)?.reason || "").trim().toLowerCase();
  if (phase === "ORDER" && (event === "open_blocked_insufficient_funds" || event === "open_failed_insufficient_funds")) {
    return true;
  }
  if (phase === "ORDER" && event === "open_failed") {
    const text = `${message} ${reason}`;
    return INSUFFICIENT_FUNDS_HINTS.some((hint) => text.includes(hint));
  }
  return false;
}

function compactLlmText(raw: string, limit = 280): string {
  const cleaned = String(raw || "").replace(/\s+/g, " ").trim();
  if (!cleaned) return "";
  if (cleaned.length <= limit) return cleaned;
  return `${cleaned.slice(0, Math.max(0, limit - 1))}...`;
}

function formatSignedNumber(value: unknown, digits = 2, suffix = ""): string {
  const parsed = toNum(value);
  if (parsed === null) return "-";
  const sign = parsed > 0 ? "+" : parsed < 0 ? "-" : "";
  return `${sign}${Math.abs(parsed).toFixed(digits)}${suffix}`;
}

function formatPositiveNumber(value: unknown, digits = 2, suffix = ""): string {
  const parsed = toNum(value);
  if (parsed === null) return "-";
  return `${Math.abs(parsed).toFixed(digits)}${suffix}`;
}

function toIntrabarSourceLabel(value: unknown): string {
  const source = String(value || "").trim().toLowerCase();
  if (!source) return "reviewed at candle close";
  if (source === "tick_bid_at_or_before_close" || source === "tick_ask_at_or_before_close") {
    return "reviewed with close tick";
  }
  if (source === "tick_bid_after_close" || source === "tick_ask_after_close") {
    return "reviewed with first tick after close";
  }
  if (source === "bar_close_bid_fallback") return "reviewed with bar close fallback";
  if (source === "bar_close_plus_spread_estimate") return "reviewed with spread estimate fallback";
  return `review source: ${source}`;
}

function toReviewOutcomeLabel(value: unknown): string {
  const outcome = String(value || "").trim().toLowerCase();
  if (outcome === "intrabar_better") return "Early Exit Better";
  if (outcome === "hold_better") return "Hold Better";
  if (outcome === "flat") return "Nearly Flat";
  return "Review";
}

function toReviewOutcomeClasses(value: unknown): string {
  const outcome = String(value || "").trim().toLowerCase();
  if (outcome === "intrabar_better") {
    return "border-success/30 bg-success/10 text-success";
  }
  if (outcome === "hold_better") {
    return "border-warning/30 bg-warning/10 text-warning";
  }
  return "border-border bg-secondary text-muted-foreground";
}

function toReviewHeadline(review: Partial<BotLiveIntrabarReview>): string {
  const outcome = String(review.review_outcome || "").trim().toLowerCase();
  const delta = Math.abs(toNum(review.delta_vs_hold_money) || 0);
  if (outcome === "intrabar_better") {
    return delta > 0
      ? `Early exit protected ${formatPositiveNumber(delta, 2)} more profit than holding`
      : "Early exit finished better than holding";
  }
  if (outcome === "hold_better") {
    return delta > 0
      ? `Holding to candle close would add ${formatPositiveNumber(delta, 2)} more profit`
      : "Holding to candle close finished better";
  }
  return "Early exit and candle close finished almost the same";
}

function shouldHideLog(log: Partial<BotLiveLogEntry>): boolean {
  const phase = String(log?.phase || "").trim().toUpperCase();
  const event = String(log?.event || "").trim().toLowerCase();
  if (HIDDEN_PHASES.has(phase)) return true;
  if (phase === "WS" && event === "llm_result") return true;
  return false;
}

function toFriendlyMessage(log: Partial<BotLiveLogEntry>): string {
  const phase = String(log?.phase || "").trim().toUpperCase();
  const event = String(log?.event || "").trim().toLowerCase();
  const message = String(log?.message || "").trim();
  const meta = normalizeMeta(log?.meta);

  if (phase === "BOOT" && event === "start") return "Live trading runtime started";
  if (phase === "BOOT" && event === "mt5_connected") {
    const symbol = String(meta?.symbol || "").trim();
    const timeframe = String(meta?.timeframe || "").trim();
    if (symbol && timeframe) return `Connected to MT5 (${symbol}/${timeframe})`;
    return "Connected to MT5";
  }
  if (phase === "BOOT" && event === "ready") return "System is ready";

  if (phase === "WS" && event === "connected") return "Connected to server";
  if (phase === "WS" && event === "registered") return "Bot registered with server";
  if (phase === "WS" && event === "disconnected") return "Server disconnected, reconnecting";

  if (phase === "CLOCK" && event === "waiting_new_candle") return "Waiting for next candle";
  if (phase === "CLOCK" && event === "new_closed_candle") {
    const ts = String(meta?.closed_utc || "").trim();
    return ts ? `New candle closed (${toLocalDateTimeText(ts)})` : "New candle closed";
  }

  if (phase === "STARTUP" && event === "snapshot_no_order") {
    return "Open exposure found: startup runs without sending duplicate orders";
  }
  if (phase === "STARTUP" && event === "startup_immediate") {
    return "No open exposure: startup can execute immediately";
  }
  if (phase === "STARTUP" && event === "no_state_bar") {
    return "No previous state: processing latest closed bar";
  }

  if (phase === "FLOW" && event === "catchup_replay") {
    const bars = toNum(meta?.bars);
    return bars !== null ? `Running catch-up for ${bars} bar(s)` : "Running catch-up replay";
  }
  if (phase === "BAR" && event === "bar_start") {
    const endUtc = String(meta?.end_utc || "").trim();
    return endUtc ? `Evaluating closed bar (${toLocalDateTimeText(endUtc)})` : "Evaluating closed bar";
  }
  if (phase === "BAR" && event === "bar_complete") {
    const finalAction = String(meta?.final_action || "").trim().toUpperCase();
    const status = String(meta?.status || "").trim().toLowerCase();
    const orderOk = meta?.order_ok;
    const statusText = status ? ` | ${status}` : "";
    if (typeof orderOk === "boolean" && finalAction && finalAction !== "HOLD") {
      return `Bar evaluation completed: ${finalAction}${statusText} | order ${orderOk ? "sent" : "failed"}`;
    }
    if (finalAction) {
      return `Bar evaluation completed: ${finalAction}${statusText}`;
    }
    return "Bar evaluation completed";
  }

  if (phase === "ORDER" && event === "open_done") {
    const side = String(meta?.side || "").toUpperCase().trim();
    const price = toNum(meta?.price);
    const lot = toNum(meta?.lot);
    const sideText = side || "ORDER";
    const priceText = price !== null ? price.toFixed(5) : "-";
    const lotText = lot !== null ? lot.toFixed(2) : "-";
    return `Opened ${sideText} @ ${priceText} (lot=${lotText})`;
  }
  if (phase === "ORDER" && event === "close_done") {
    const ticket = String(meta?.ticket || "").trim();
    const pnl = toNum(meta?.pnl);
    const pnlText = pnl !== null ? pnl.toFixed(2) : "-";
    return ticket ? `Closed position #${ticket} (PnL ${pnlText})` : `Closed position (PnL ${pnlText})`;
  }
  if (phase === "ORDER" && event === "open_failed") {
    const reason = String(meta?.reason || "").trim();
    return reason ? `Open order failed: ${reason}` : "Open order failed";
  }
  if (phase === "ORDER" && event === "open_blocked_insufficient_funds") {
    const reason = String(meta?.reason || "").trim();
    const freeMargin = toNum(meta?.free_margin);
    const requiredMargin = toNum(meta?.required_margin);
    const marginText = freeMargin !== null
      ? `free margin=${freeMargin.toFixed(2)}`
      : "";
    const requiredText = requiredMargin !== null && requiredMargin > 0
      ? ` required=${requiredMargin.toFixed(2)}`
      : "";
    const detail = reason || `${marginText}${requiredText}`.trim();
    return detail
      ? `Order blocked (insufficient funds): ${detail}`
      : "Order blocked (insufficient funds)";
  }
  if (phase === "ORDER" && event === "open_failed_insufficient_funds") {
    const reason = String(meta?.reason || "").trim();
    return reason
      ? `Open order failed (insufficient funds): ${reason}`
      : "Open order failed (insufficient funds)";
  }
  if (phase === "ORDER" && event === "close_failed") {
    const reason = String(meta?.reason || "").trim();
    return reason ? `Close order failed: ${reason}` : "Close order failed";
  }
  if (phase === "ORDER" && event === "close_no_positions") return "No open positions to close";
  if (phase === "ORDER" && event === "close_skipped_no_tick") return "Close skipped: live tick is unavailable";

  if (phase === "INTRABAR" && event === "enabled") {
    const parts: string[] = [];
    const changePct = toNum(meta?.take_profit_change_pct);
    const pips = toNum(meta?.take_profit_pips);
    const money = toNum(meta?.take_profit_money);
    if (changePct !== null && changePct > 0) parts.push(`change ${changePct.toFixed(3)}%`);
    if (pips !== null && pips > 0) parts.push(`pips ${pips.toFixed(1)}`);
    if (money !== null && money > 0) parts.push(`money ${money.toFixed(2)}`);
    return parts.length > 0
      ? `Intrabar take-profit active (${parts.join(" | ")})`
      : "Intrabar take-profit active";
  }
  if (phase === "INTRABAR" && event === "take_profit_hit") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const changePct = formatSignedNumber(meta?.change_pct, 3, "%");
    const pnlMoney = formatSignedNumber(meta?.pnl_money, 2);
    const reasons = String(meta?.reasons || "").trim();
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const suffix = reasons ? ` | trigger: ${reasons}` : "";
    return `Intrabar take-profit hit: closing ${subject || "position"} | change ${changePct} | PnL ${pnlMoney}${suffix}`;
  }
  if (phase === "INTRABAR" && event === "review_pending") {
    const ticket = String(meta?.ticket || "").trim();
    const side = String(meta?.side || "").trim().toUpperCase();
    const barEndUtc = String(meta?.bar_end_utc || "").trim();
    const whenText = barEndUtc ? ` at candle close (${toLocalDateTimeText(barEndUtc)})` : " at candle close";
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    return `Intrabar exit stored for review${subject ? `: ${subject}` : ""} | compare again${whenText}`;
  }
  if (phase === "INTRABAR" && event === "review") {
    const outcome = String(meta?.outcome || "").trim().toLowerCase();
    const delta = Math.abs(toNum(meta?.delta_vs_hold_money) || 0);
    const actual = formatSignedNumber(meta?.actual_pnl_money, 2);
    const hold = formatSignedNumber(meta?.hold_to_close_pnl_money, 2);
    if (outcome === "intrabar_better") {
      return `Intrabar review: early exit was better by ${formatPositiveNumber(delta, 2)} | actual ${actual} vs candle close ${hold}`;
    }
    if (outcome === "hold_better") {
      return `Intrabar review: holding to candle close would be better by ${formatPositiveNumber(delta, 2)} | actual ${actual} vs candle close ${hold}`;
    }
    return `Intrabar review: little difference | actual ${actual} vs candle close ${hold}`;
  }
  if (phase === "INTRABAR" && event === "cooldown_reset") {
    return "Intrabar exit completed: ready for the next candle prediction";
  }

  if (phase === "SCHEDULE" && event === "blocked") return "New order blocked by trading schedule";

  if (phase === "CONFIG" && event === "runtime_updated") {
    const riskLevel = String(meta?.risk_level || "").trim();
    const riskPercent = toNum(meta?.risk_percent);
    if (riskLevel && riskPercent !== null) return `Risk updated: ${riskLevel} (${riskPercent.toFixed(2)}%)`;
    return "Runtime configuration updated";
  }

  return message;
}

interface AIConsoleProps {
  botName?: string;
  liveState?: BotLiveState;
}

export function AIConsole({ botName = "Bot", liveState }: AIConsoleProps) {
  const isLive = !!liveState?.connected;
  const llmText = liveState?.llm_text || "";
  const pendingIntrabarReviewCount = Math.max(0, Number(liveState?.pending_intrabar_review_count || 0));

  const logs: LogEntry[] = (() => {
    const recentLogs = Array.isArray(liveState?.recent_logs) ? liveState.recent_logs : [];
    const dedup = new Set<string>();
    const rows: LogEntry[] = [];

    for (let i = 0; i < recentLogs.length; i += 1) {
      const row = recentLogs[i];
      const type = row?.type && row.type in typeConfig ? row.type : "info";
      const phase = String(row?.phase || "").trim().toUpperCase();
      const event = String(row?.event || "").trim().toLowerCase();
      const insufficientFunds = isInsufficientFundsLog(row);
      const timestamp = toLocalClockText(
        typeof row?.timestamp_utc === "string" ? row.timestamp_utc : undefined,
        String(row?.timestamp || "")
      );
      if (shouldHideLog(row)) continue;
      const message = toFriendlyMessage(row);
      if (!message) continue;
      const key = `${timestamp}|${type}|${message}`;
      if (dedup.has(key)) continue;
      dedup.add(key);
      rows.push({
        id: `${key}|${i}`,
        timestamp: timestamp || "--:--:--",
        type,
        message,
        phase,
        event,
        isInsufficientFunds: insufficientFunds,
      });
    }

    const llmSummary = compactLlmText(llmText);
    if (llmSummary) {
      const ts = rows.length > 0 ? rows[rows.length - 1].timestamp : "--:--:--";
      const message = `AI Analysis: ${llmSummary}`;
      rows.push({
        id: `llm-analysis|${message}`,
        timestamp: ts,
        type: "analysis",
        message,
      });
    }

    return rows.slice(-120);
  })();

  const recentIntrabarReviews = (
    Array.isArray(liveState?.recent_intrabar_reviews) ? liveState.recent_intrabar_reviews : []
  )
    .filter((row): row is BotLiveIntrabarReview => !!row && typeof row === "object")
    .slice(-3)
    .reverse();

  return (
    <div
      className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full min-h-0 overflow-hidden flex flex-col"
      style={{ animationDelay: "150ms" }}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          <h3 className="font-semibold text-foreground">Activity Log</h3>
        </div>
        {isLive && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
            <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
            Live
          </span>
        )}
      </div>

      {(pendingIntrabarReviewCount > 0 || recentIntrabarReviews.length > 0) && (
        <div className="mb-3 space-y-2 shrink-0">
          {pendingIntrabarReviewCount > 0 && (
            <div className="rounded-lg border border-warning/25 bg-warning/10 px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-xs font-semibold text-warning">Intrabar review pending</span>
                <Badge variant="outline" className="border-warning/30 bg-white/60 text-warning">
                  {pendingIntrabarReviewCount} waiting
                </Badge>
              </div>
              <p className="mt-1 text-xs text-warning/90">
                Closed positions from the current candle will be compared against the candle-close result as soon as the bar ends.
              </p>
            </div>
          )}

          {recentIntrabarReviews.map((review, index) => {
            const ticket = String(review.ticket || "").trim();
            const side = String(review.side || "").trim().toUpperCase();
            const actualMoney = formatSignedNumber(review.actual_pnl_money, 2);
            const holdMoney = formatSignedNumber(review.hold_to_close_pnl_money, 2);
            const actualChange = formatSignedNumber(review.actual_change_pct, 3, "%");
            const holdChange = formatSignedNumber(review.hold_to_close_change_pct, 3, "%");
            const reviewedAt = toLocalDateTimeText(
              typeof review.reviewed_at_bar_end_utc === "string" ? review.reviewed_at_bar_end_utc : review.bar_end_utc
            );
            const triggerReasons = Array.isArray(review.trigger_reasons)
              ? review.trigger_reasons.filter((item) => String(item || "").trim()).join(" | ")
              : "";

            return (
              <div
                key={review.review_id || `${ticket}|${reviewedAt}|${index}`}
                className="rounded-lg border border-border bg-slate-50/80 px-3 py-2"
              >
                <div className="flex items-start justify-between gap-2">
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-foreground">{toReviewHeadline(review)}</p>
                    <p className="mt-0.5 text-[11px] text-muted-foreground">
                      {[side, ticket ? `#${ticket}` : "", reviewedAt].filter(Boolean).join(" | ")}
                    </p>
                  </div>
                  <Badge variant="outline" className={cn("shrink-0", toReviewOutcomeClasses(review.review_outcome))}>
                    {toReviewOutcomeLabel(review.review_outcome)}
                  </Badge>
                </div>

                <div className="mt-2 flex flex-wrap gap-x-3 gap-y-1 text-xs text-muted-foreground">
                  <span>Actual {actualMoney} ({actualChange})</span>
                  <span>Close {holdMoney} ({holdChange})</span>
                  <span>{toIntrabarSourceLabel(review.bar_close_exit_source)}</span>
                </div>

                {triggerReasons && (
                  <p className="mt-1 text-[11px] text-muted-foreground">Trigger: {triggerReasons}</p>
                )}
              </div>
            );
          })}
        </div>
      )}

      <div className="bg-secondary/50 rounded-lg p-3 flex-1 min-h-0 overflow-auto overscroll-contain scrollbar-thin">
        {logs.length === 0 ? (
          <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
            {isLive ? `Waiting for runtime logs from ${botName}` : `Connect ${botName} to stream logs`}
          </div>
        ) : (
          logs.map((log, index) => {
            const config = typeConfig[log.type];
            const Icon = log.message.startsWith("AI Analysis:") ? Bot : config.icon;
            const isLlmLog = log.message.startsWith("AI Analysis:");
            return (
              <div
                key={log.id}
                className={cn(
                  "flex items-start gap-2 py-1.5 text-sm transition-colors",
                  index === logs.length - 1 && "bg-primary/5 -mx-1 px-1 rounded font-medium",
                  isLlmLog && "bg-primary/5 -mx-1 px-1 rounded"
                )}
              >
                <span className="text-xs text-muted-foreground font-mono shrink-0">{log.timestamp}</span>
                <Icon className={cn("w-3.5 h-3.5 shrink-0 mt-0.5", config.color)} />
                <div className="min-w-0 flex-1">
                  {log.isInsufficientFunds && (
                    <span className="inline-flex items-center rounded px-1.5 py-0.5 mr-1.5 text-[10px] font-semibold tracking-wide bg-destructive/10 text-destructive border border-destructive/30">
                      INSUFFICIENT FUNDS
                    </span>
                  )}
                  <span className={cn("text-sm break-all", config.color)}>{log.message}</span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
