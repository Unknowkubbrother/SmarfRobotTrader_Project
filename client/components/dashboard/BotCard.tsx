import { useState } from "react";
import { Power, Play, Activity, TrendingUp, Trash2, AlertTriangle, DownloadCloud, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import type { BotConfigWithVersion } from "@/hooks/useTradingAccounts";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

interface BotCardProps {
  bot: BotConfigWithVersion;
  onToggleStatus: (botId: string, newStatus: "running" | "stopped") => void;
  onDelete?: (botId: string) => void;
  onSelect?: (botId: string) => void;
  isSelected?: boolean;
  showDelete?: boolean;
  isBusy?: boolean;
  startBlockedReason?: string | null;
}

export function BotCard({
  bot,
  onToggleStatus,
  onDelete,
  onSelect,
  isSelected = false,
  showDelete = true,
  isBusy = false,
  startBlockedReason = null,
}: BotCardProps) {
  const [showStopConfirm, setShowStopConfirm] = useState(false);
  const status = String(bot.status || bot.container_status || "stopped").toLowerCase();
  const isStarting = status === "starting";
  const isRunning = status === "running";
  const isVersionInactive = bot.bot_version?.is_active === false;
  const isStartLockedByVersion = isVersionInactive && !isRunning && !isStarting;
  const isStartBlockedBySubscription = Boolean(startBlockedReason) && !isRunning && !isStarting;
  const isActionLocked = isBusy || isStarting;
  const isToggleDisabled = isActionLocked || isStartLockedByVersion || isStartBlockedBySubscription;
  const symbol = bot.bot_version?.symbol || "N/A";
  const label = bot.bot_version?.label || "No Model";
  const riskLevel = bot.risk_level || "medium";

  const handleToggleClick = (e: React.MouseEvent) => {
    e.stopPropagation();
    if (isToggleDisabled) return;
    if (status === "running") {
      setShowStopConfirm(true);
    } else {
      onToggleStatus(bot.id, "running");
    }
  };

  const handleConfirmStop = () => {
    onToggleStatus(bot.id, "stopped");
    setShowStopConfirm(false);
  };

  const getStatusBadge = (status: string | null) => {
    switch (status) {
      case "running":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-success/10 text-success">
            <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
            Running
          </span>
        );
      case "starting":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">
            <RefreshCw className="w-3 h-3 animate-spin" />
            Starting
          </span>
        );
      case "error":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-destructive/10 text-destructive">
            <AlertTriangle className="w-3 h-3" />
            Failed
          </span>
        );
      case "paused":
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">
            Paused
          </span>
        );
      default:
        return (
          <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-muted text-muted-foreground">
            <span className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />
            Stopped
          </span>
        );
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case "low": return "text-success";
      case "high": return "text-destructive";
      default: return "text-warning";
    }
  };

  return (
    <>
      <div
        className={cn(
          "bg-white border rounded-xl shadow-sm p-5 transition-all cursor-pointer relative overflow-hidden group hover:shadow-md",
          isSelected ? "ring-2 ring-primary border-primary" : "hover:border-primary/50"
        )}
        onClick={() => onSelect?.(bot.id)}
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-center gap-3 flex-1 min-w-0">
            <div className={cn(
              "w-10 h-10 rounded-xl flex items-center justify-center shrink-0",
              status === "running"
                ? "bg-success/10"
                : status === "starting"
                  ? "bg-warning/10"
                  : status === "error"
                    ? "bg-destructive/10"
                    : "bg-secondary"
            )}>
              <Activity className={cn(
                "w-5 h-5",
                status === "running"
                  ? "text-success"
                  : status === "starting"
                    ? "text-warning"
                    : status === "error"
                      ? "text-destructive"
                      : "text-muted-foreground"
              )} />
            </div>
            <div className="min-w-0 flex-1">
              <div className="flex items-center gap-2 flex-wrap">
                <h4 className="font-medium text-sm truncate">{label}</h4>
                {getStatusBadge(status)}
                {isBusy && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">
                    <RefreshCw className="w-3 h-3 animate-spin" />
                    Processing
                  </span>
                )}
                {isVersionInactive && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-destructive/10 text-destructive">
                    <AlertTriangle className="w-3 h-3" />
                    Locked
                  </span>
                )}
                {isStartBlockedBySubscription && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">
                    <AlertTriangle className="w-3 h-3" />
                    Billing Hold
                  </span>
                )}
                {bot.has_pending_update && (
                  <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-primary/10 text-primary">
                    <DownloadCloud className="w-3 h-3" />
                    Update Available
                  </span>
                )}
              </div>
              <p className="text-xs text-muted-foreground mt-0.5">
                <span className="font-mono">{symbol}</span>
                <span className="mx-1.5">•</span>
                <span className={getRiskColor(riskLevel)}>{riskLevel} risk</span>
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {bot.today_pnl !== 0 && (
              <div className="flex items-center gap-1 px-2 py-1 rounded-md bg-secondary">
                <TrendingUp className={cn(
                  "w-3 h-3",
                  bot.today_pnl >= 0 ? "text-success" : "text-destructive"
                )} />
                <span className={cn(
                  "text-xs font-mono font-medium",
                  bot.today_pnl >= 0 ? "text-success" : "text-destructive"
                )}>
                  {bot.today_pnl >= 0 ? "+" : ""}${bot.today_pnl.toFixed(2)}
                </span>
              </div>
            )}

            <Button
              variant="ghost"
              size="sm"
              className="h-8 w-8 p-0"
              onClick={handleToggleClick}
              disabled={isToggleDisabled}
              title={
                isStartLockedByVersion
                  ? "This bot model is inactive. Change model to an active version first."
                  : isStartBlockedBySubscription
                  ? startBlockedReason ?? "Resolve subscription billing before starting bots."
                  : undefined
              }
            >
              {status === "running" ? (
                <Power className="w-4 h-4 text-destructive" />
              ) : isStartBlockedBySubscription ? (
                <AlertTriangle className="w-4 h-4 text-warning" />
              ) : isStartLockedByVersion ? (
                <AlertTriangle className="w-4 h-4 text-destructive" />
              ) : status === "starting" ? (
                <RefreshCw className="w-4 h-4 text-warning animate-spin" />
              ) : (
                <Play className="w-4 h-4 text-success" />
              )}
            </Button>

            {showDelete && onDelete && (
              <Button
                variant="ghost"
                size="sm"
                className="h-8 w-8 p-0 hover:bg-destructive/10"
                onClick={(e) => {
                  e.stopPropagation();
                  if (isActionLocked) return;
                  onDelete(bot.id);
                }}
                disabled={isActionLocked}
              >
                <Trash2 className="w-4 h-4 text-muted-foreground hover:text-destructive" />
              </Button>
            )}
          </div>
        </div>
      </div>

      <AlertDialog open={showStopConfirm} onOpenChange={setShowStopConfirm}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop Bot?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to stop this bot? It will stop processing new signals.
              <br />
              Open positions will need to be managed manually.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={(e) => e.stopPropagation()}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={(e) => {
                e.stopPropagation();
                handleConfirmStop();
              }}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isActionLocked}
            >
              Stop Bot
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </>
  );
}
