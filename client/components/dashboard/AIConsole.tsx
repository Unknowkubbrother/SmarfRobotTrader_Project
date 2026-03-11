import { Terminal, TrendingUp, AlertTriangle, CheckCircle, Bot } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BotLiveIntrabarReview, BotLiveLogEntry, BotLiveState } from "@/hooks/useBotLiveState";

interface LogEntry {
  id: string;
  timestamp: string;
  type: "info" | "analysis" | "action" | "warning" | "success";
  message: string;
  phase?: string;
  event?: string;
  isInsufficientFunds?: boolean;
  _sortKey?: Date | null;
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

  // Treat bare bot timestamps as UTC instead of browser-local time.
  if (/^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$/i.test(text)) {
    const patched = `${text.replace(" ", "T")}Z`;
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

function normalizeBrokerDateTimeText(raw?: string): string {
  const text = String(raw || "").trim();
  if (!text) return "";

  const matched = text.match(/^(\d{4})-(\d{2})-(\d{2})[ T](\d{2}:\d{2}:\d{2})(?:Z)?$/i);
  if (matched) {
    const [, year, month, day, clock] = matched;
    return `${day}/${month}/${year} ${clock}`;
  }

  return text.replace("T", " ").replace(/Z$/i, "").trim();
}

const typeConfig = {
  info: { icon: Bot, color: "text-primary" },
  analysis: { icon: TrendingUp, color: "text-muted-foreground" },
  action: { icon: Terminal, color: "text-foreground" },
  warning: { icon: AlertTriangle, color: "text-warning" },
  success: { icon: CheckCircle, color: "text-success" },
};

const HIDDEN_PHASES = new Set(["SEM", "MODEL", "GATE", "DECISION"]);
const AI_ANALYSIS_LOG_EVENTS = new Set(["embedding_ready", "cache_hit", "cache_alias_hit"]);

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

function findLatestAiAnalysisTimestamp(logs: BotLiveLogEntry[]): string {
  for (let i = logs.length - 1; i >= 0; i -= 1) {
    const row = logs[i];
    const phase = String(row?.phase || "").trim().toUpperCase();
    const event = String(row?.event || "").trim().toLowerCase();
    const timestampUtc = typeof row?.timestamp_utc === "string" ? row.timestamp_utc : "";
    if (phase === "SEM" && AI_ANALYSIS_LOG_EVENTS.has(event) && timestampUtc) {
      return timestampUtc;
    }
  }
  return "";
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

function formatRatioPercent(value: unknown): string {
  const parsed = toNum(value);
  if (parsed === null) return "-";
  return `${Math.round(parsed * 100)}%`;
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

function toReviewLogType(value: unknown): LogEntry["type"] {
  const outcome = String(value || "").trim().toLowerCase();
  if (outcome === "intrabar_better") return "success";
  if (outcome === "hold_better") return "warning";
  return "analysis";
}

function toReviewLogMessage(review: Partial<BotLiveIntrabarReview>): string {
  const ticket = String(review.ticket || "").trim();
  const side = String(review.side || "").trim().toUpperCase();
  const headline = toReviewHeadline(review);
  const actualMoney = formatSignedNumber(review.actual_pnl_money, 2);
  const holdMoney = formatSignedNumber(review.hold_to_close_pnl_money, 2);
  const actualChange = formatSignedNumber(review.actual_change_pct, 3, "%");
  const holdChange = formatSignedNumber(review.hold_to_close_change_pct, 3, "%");
  const sourceLabel = toIntrabarSourceLabel(review.bar_close_exit_source);
  const triggerReasons = Array.isArray(review.trigger_reasons)
    ? review.trigger_reasons.filter((item) => String(item || "").trim()).join(" | ")
    : "";
  const subject = [toReviewOutcomeLabel(review.review_outcome), side, ticket ? `#${ticket}` : ""]
    .filter(Boolean)
    .join(" | ");
  const triggerText = triggerReasons ? ` | trigger: ${triggerReasons}` : "";
  return `${subject} | ${headline} | actual ${actualMoney} (${actualChange}) vs close ${holdMoney} (${holdChange}) | ${sourceLabel}${triggerText}`;
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
    return ts ? `New candle close detected | broker candle time ${normalizeBrokerDateTimeText(ts)}` : "New candle close detected";
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
    return endUtc ? `Evaluating closed bar | broker candle time ${normalizeBrokerDateTimeText(endUtc)}` : "Evaluating closed bar";
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
    const trailingEnabled = Boolean(meta?.trailing_enabled);
    const trendKeep = formatRatioPercent(meta?.trail_keep_ratio_trend);
    const normalKeep = formatRatioPercent(meta?.trail_keep_ratio_normal);
    const tightKeep = formatRatioPercent(meta?.trail_keep_ratio_tight);
    const bufferRatio = formatRatioPercent(meta?.trail_arm_buffer_ratio);
    const confirmPolls = toNum(meta?.trail_confirm_polls);
    if (changePct !== null && changePct > 0) parts.push(`change ${changePct.toFixed(3)}%`);
    if (pips !== null && pips > 0) parts.push(`pips ${pips.toFixed(1)}`);
    if (money !== null && money > 0) parts.push(`money ${money.toFixed(2)}`);
    if (trailingEnabled) {
      parts.push(`trailing lock ${trendKeep}/${normalKeep}/${tightKeep}`);
      if (bufferRatio !== "-") parts.push(`buffer ${bufferRatio}`);
      if (confirmPolls !== null) parts.push(`confirm ${confirmPolls.toFixed(0)} polls`);
    }
    const label = trailingEnabled ? "Intrabar profit lock active" : "Intrabar take-profit active";
    return parts.length > 0 ? `${label} (${parts.join(" | ")})` : label;
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
  if (phase === "INTRABAR" && event === "trail_armed") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const regime = String(meta?.regime || "").trim().toLowerCase();
    const keepRatio = formatRatioPercent(meta?.keep_ratio);
    const changePct = formatSignedNumber(meta?.change_pct, 3, "%");
    const pnlMoney = formatSignedNumber(meta?.pnl_money, 2);
    const reasons = String(meta?.reasons || "").trim();
    const floorChangePct = toNum(meta?.initial_floor_change_pct);
    const activationChangePct = toNum(meta?.activation_peak_change_pct);
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const regimeText = regime ? ` | regime ${regime}` : "";
    const keepText = keepRatio !== "-" ? ` | lock ${keepRatio} of extra profit` : "";
    const floorText = floorChangePct !== null ? ` | soft floor ${floorChangePct.toFixed(3)}%` : "";
    const activationText = activationChangePct !== null ? ` | trail after ${activationChangePct.toFixed(3)}%` : "";
    const reasonText = reasons ? ` | trigger: ${reasons}` : "";
    return `Intrabar trailing armed for ${subject || "position"}${regimeText}${keepText}${floorText}${activationText} | now ${changePct} | PnL ${pnlMoney}${reasonText}`;
  }
  if (phase === "INTRABAR" && event === "trail_regime") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const previousRegime = String(meta?.previous_regime || "").trim().toLowerCase();
    const regime = String(meta?.regime || "").trim().toLowerCase();
    const keepRatio = formatRatioPercent(meta?.keep_ratio);
    const floorChangePct = toNum(meta?.floor_change_pct);
    const peakChangePct = toNum(meta?.peak_change_pct);
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const fromText = previousRegime ? `${previousRegime} -> ` : "";
    const floorText = floorChangePct !== null ? ` | floor ${floorChangePct.toFixed(3)}%` : "";
    const peakText = peakChangePct !== null ? ` | peak ${peakChangePct.toFixed(3)}%` : "";
    return `Intrabar trailing regime updated for ${subject || "position"} | ${fromText}${regime || "unknown"} | lock ${keepRatio}${peakText}${floorText}`;
  }
  if (phase === "INTRABAR" && event === "trail_floor_pending") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const confirmCount = toNum(meta?.confirm_count);
    const confirmPolls = toNum(meta?.confirm_polls);
    const changePct = formatSignedNumber(meta?.change_pct, 3, "%");
    const floorChangePct = toNum(meta?.floor_change_pct);
    const reasons = String(meta?.reasons || "").trim();
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const confirmText = confirmCount !== null && confirmPolls !== null
      ? ` | confirm ${confirmCount.toFixed(0)}/${confirmPolls.toFixed(0)}`
      : "";
    const floorText = floorChangePct !== null ? ` | floor ${floorChangePct.toFixed(3)}%` : "";
    const reasonText = reasons ? ` | trigger: ${reasons}` : "";
    return `Intrabar floor breached for ${subject || "position"}${confirmText}${floorText} | now ${changePct}${reasonText}`;
  }
  if (phase === "INTRABAR" && event === "trail_floor_recovered") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const recoveredAfter = toNum(meta?.recovered_after);
    const changePct = formatSignedNumber(meta?.change_pct, 3, "%");
    const floorChangePct = toNum(meta?.floor_change_pct);
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const afterText = recoveredAfter !== null ? ` after ${recoveredAfter.toFixed(0)} watch poll(s)` : "";
    const floorText = floorChangePct !== null ? ` | floor ${floorChangePct.toFixed(3)}%` : "";
    return `Intrabar trailing recovered for ${subject || "position"}${afterText}${floorText} | now ${changePct}`;
  }
  if (phase === "INTRABAR" && event === "trail_floor_hit") {
    const side = String(meta?.side || "").trim().toUpperCase();
    const ticket = String(meta?.ticket || "").trim();
    const regime = String(meta?.regime || "").trim().toLowerCase();
    const changePct = formatSignedNumber(meta?.change_pct, 3, "%");
    const pnlMoney = formatSignedNumber(meta?.pnl_money, 2);
    const peakChangePct = toNum(meta?.peak_change_pct);
    const floorChangePct = toNum(meta?.floor_change_pct);
    const confirmCount = toNum(meta?.confirm_count);
    const confirmPolls = toNum(meta?.confirm_polls);
    const reasons = String(meta?.reasons || "").trim();
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const regimeText = regime ? ` | regime ${regime}` : "";
    const peakText = peakChangePct !== null ? ` | peak ${formatSignedNumber(peakChangePct, 3, "%")}` : "";
    const floorText = floorChangePct !== null ? ` | floor ${formatSignedNumber(floorChangePct, 3, "%")}` : "";
    const confirmText = confirmCount !== null && confirmPolls !== null
      ? ` | confirm ${confirmCount.toFixed(0)}/${confirmPolls.toFixed(0)}`
      : "";
    const reasonText = reasons ? ` | trigger: ${reasons}` : "";
    return `Intrabar trailing floor hit: closing ${subject || "position"}${regimeText}${confirmText} | now ${changePct} | PnL ${pnlMoney}${peakText}${floorText}${reasonText}`;
  }
  if (phase === "INTRABAR" && event === "review_pending") {
    const ticket = String(meta?.ticket || "").trim();
    const side = String(meta?.side || "").trim().toUpperCase();
    const barEndUtc = String(meta?.bar_end_utc || "").trim();
    const exitMode = String(meta?.exit_mode || "").trim().toLowerCase();
    const whenText = barEndUtc ? ` at broker candle close (${normalizeBrokerDateTimeText(barEndUtc)})` : " at broker candle close";
    const subject = [side, ticket ? `#${ticket}` : ""].filter(Boolean).join(" ");
    const exitLabel = exitMode === "trailing_floor_hit" ? "Trailing exit" : "Intrabar exit";
    return `${exitLabel} stored for review${subject ? `: ${subject}` : ""} | compare again${whenText}`;
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

  const logs: LogEntry[] = (() => {
    const recentLogs = Array.isArray(liveState?.recent_logs) ? liveState.recent_logs : [];
    const recentIntrabarReviews = (
      Array.isArray(liveState?.recent_intrabar_reviews) ? liveState.recent_intrabar_reviews : []
    )
      .filter((row): row is BotLiveIntrabarReview => !!row && typeof row === "object")
      .slice(-5);
    const reviewLogTimestampByTicket = new Map<string, string>();
    for (let i = 0; i < recentLogs.length; i += 1) {
      const row = recentLogs[i];
      const phase = String(row?.phase || "").trim().toUpperCase();
      const event = String(row?.event || "").trim().toLowerCase();
      if (phase !== "INTRABAR" || event !== "review") continue;
      const ticket = String(normalizeMeta(row?.meta)?.ticket || "").trim();
      const timestampUtc = typeof row?.timestamp_utc === "string" ? row.timestamp_utc : "";
      if (ticket && timestampUtc) {
        reviewLogTimestampByTicket.set(ticket, timestampUtc);
      }
    }
    const dedup = new Set<string>();
    const rows: LogEntry[] = [];

    for (let i = 0; i < recentLogs.length; i += 1) {
      const row = recentLogs[i];
      const type = row?.type && row.type in typeConfig ? row.type : "info";
      const phase = String(row?.phase || "").trim().toUpperCase();
      const event = String(row?.event || "").trim().toLowerCase();
      if (phase === "INTRABAR" && event === "review" && recentIntrabarReviews.length > 0) continue;
      const insufficientFunds = isInsufficientFundsLog(row);
      const rawUtc = typeof row?.timestamp_utc === "string" ? row.timestamp_utc : undefined;
      const timestamp = toLocalClockText(
        rawUtc,
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
        _sortKey: parseUtcLike(rawUtc || ""),
      });
    }

    for (let i = 0; i < recentIntrabarReviews.length; i += 1) {
      const review = recentIntrabarReviews[i];
      const ticketKey = String(review.ticket || "").trim();
      const reviewLogTimestampUtc = ticketKey ? reviewLogTimestampByTicket.get(ticketKey) : "";
      const timestampUtc = reviewLogTimestampUtc
        || (typeof review.reviewed_at_bar_end_utc === "string" ? review.reviewed_at_bar_end_utc : review.bar_end_utc);
      const timestamp = toLocalClockText(timestampUtc, "--:--:--");
      const message = toReviewLogMessage(review);
      const type = toReviewLogType(review.review_outcome);
      const key = `${timestamp}|${type}|${message}`;
      if (dedup.has(key)) continue;
      dedup.add(key);
      rows.push({
        id: `review|${review.review_id || key}|${i}`,
        timestamp,
        type,
        message,
        phase: "INTRABAR",
        event: "review_summary",
        _sortKey: parseUtcLike(timestampUtc || ""),
      });
    }

    const llmSummary = compactLlmText(llmText);
    if (llmSummary) {
      const latestVisibleTimestampUtc = (() => {
        for (let i = recentLogs.length - 1; i >= 0; i -= 1) {
          const rawUtc = typeof recentLogs[i]?.timestamp_utc === "string" ? recentLogs[i].timestamp_utc : "";
          if (rawUtc) return rawUtc;
        }
        return "";
      })();
      const llmTimestampUtc =
        findLatestAiAnalysisTimestamp(recentLogs)
        || latestVisibleTimestampUtc
        || (typeof liveState?.last_bar_time === "string" ? liveState.last_bar_time : "");
      const llmSortKey = parseUtcLike(llmTimestampUtc);
      const ts = llmSortKey
        ? toLocalClockText(llmTimestampUtc)
        : (rows.length > 0 ? rows[rows.length - 1].timestamp : "--:--:--");
      const message = `AI Analysis: ${llmSummary}`;
      rows.push({
        id: `llm-analysis|${llmTimestampUtc}|${message}`,
        timestamp: ts,
        type: "analysis",
        message,
        _sortKey: llmSortKey,
      });
    }

    // Sort all entries chronologically
    rows.sort((a, b) => {
      const ta = a._sortKey?.getTime() ?? Infinity;
      const tb = b._sortKey?.getTime() ?? Infinity;
      return ta - tb;
    });

    return rows.slice(-120);
  })();

  return (
    <div
      className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full min-h-0 overflow-hidden flex flex-col"
      style={{ animationDelay: "150ms" }}
    >
      <div className="flex items-center justify-between mb-4">
        <div>
          <div className="flex items-center gap-2">
            <Terminal className="w-4 h-4 text-muted-foreground" />
            <h3 className="font-semibold text-foreground">Activity Log</h3>
          </div>
        </div>
        {isLive && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
            <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
            Live
          </span>
        )}
      </div>

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
