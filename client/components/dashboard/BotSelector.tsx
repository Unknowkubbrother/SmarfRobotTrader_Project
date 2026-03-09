import { useState } from "react";
import { ChevronDown, Bot, Plus, Check, Play, Square, AlertTriangle } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { AddBotDialog } from "@/components/dialogs/AddBotDialog";
import { BotConfigWithVersion } from "@/hooks/useTradingAccounts";
import { Button } from "@/components/ui/button";

interface BotSelectorProps {
  bots: BotConfigWithVersion[];
  selectedBotId: string | null;
  onBotSelect: (botId: string) => void;
  accountId: string;
  onBotAdded: () => void;
  getBotPnl?: (botId: string, fallbackPnl: number) => number;
  addBotDisabled?: boolean;
  addBotDisabledReason?: string | null;
}

export function BotSelector({
  bots,
  selectedBotId,
  onBotSelect,
  accountId,
  onBotAdded,
  getBotPnl,
  addBotDisabled = false,
  addBotDisabledReason = null,
}: BotSelectorProps) {
  const [dialogOpen, setDialogOpen] = useState(false);

  const selectedBot = bots.find(b => b.id === selectedBotId);
  const runningCount = bots.filter((b) => {
    const status = String(b.status || b.container_status || "").toLowerCase();
    return status === "running" || status === "starting";
  }).length;
  const selectedBotVersionInactive = selectedBot?.bot_version?.is_active === false;
  const selectedBotStatus = String(selectedBot?.status || selectedBot?.container_status || "").toLowerCase();
  const resolveBotPnl = (bot: BotConfigWithVersion): number =>
    Number(getBotPnl?.(bot.id, Number(bot.today_pnl ?? 0)) ?? bot.today_pnl ?? 0);

  return (
    <div className="flex items-center gap-3">
      <DropdownMenu>
        <DropdownMenuTrigger asChild>
          <button className="flex items-center gap-3 px-4 py-3 rounded-lg bg-card border border-border hover:border-primary/50 transition-all w-full md:w-auto min-w-[280px]">
            <div className="w-10 h-10 rounded-lg bg-secondary flex items-center justify-center">
              <Bot className="w-5 h-5 text-muted-foreground" />
            </div>
            <div className="flex-1 text-left">
              {selectedBot ? (
                <>
                  <div className="flex items-center gap-2">
                    <p className="text-sm font-medium">{selectedBot.bot_version?.label || "Unknown Bot"}</p>
                    <span className={cn(
                      "w-2 h-2 rounded-full",
                      selectedBotVersionInactive
                        ? "bg-destructive"
                        : selectedBotStatus === "running"
                        ? "bg-success"
                        : selectedBotStatus === "starting"
                          ? "bg-warning"
                          : selectedBotStatus === "error"
                            ? "bg-destructive"
                          : "bg-muted-foreground"
                    )} />
                    {selectedBotVersionInactive && (
                      <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive">
                        <AlertTriangle className="h-2.5 w-2.5" />
                        Locked
                      </span>
                    )}
                  </div>
                  <p className="text-xs text-muted-foreground">
                    {selectedBot.bot_version?.symbol || "No Pair"} • {selectedBot.bot_version?.timeframe || "TF"}
                  </p>
                </>
              ) : (
                <div className="flex flex-col">
                  <span className="text-sm font-medium">Select Bot</span>
                  <span className="text-xs text-muted-foreground">
                    {bots.length > 0 ? `${runningCount} Running` : "No bots"}
                  </span>
                </div>
              )}
            </div>
            <ChevronDown className="w-4 h-4 text-muted-foreground" />
          </button>
        </DropdownMenuTrigger>
        <DropdownMenuContent align="start" className="w-[320px]">
          {bots.length === 0 ? (
            <div className="p-4 text-center text-sm text-muted-foreground flex flex-col items-center gap-2">
              <Bot className="w-8 h-8 text-muted-foreground/50" />
              <p>No bots found in this account.</p>
              <Button
                variant="outline"
                size="sm"
                onClick={() => setDialogOpen(true)}
                className="w-full mt-2"
                disabled={addBotDisabled}
                title={addBotDisabled ? addBotDisabledReason ?? undefined : undefined}
              >
                Create Your First Bot
              </Button>
            </div>
          ) : (
            <>
              <div className="p-2 text-xs font-medium text-muted-foreground uppercase tracking-wider">
                Available Bots
              </div>
              {bots.map((bot) => {
                const status = String(bot.status || bot.container_status || "").toLowerCase();
                const botPnl = resolveBotPnl(bot);
                return (
                  <DropdownMenuItem
                    key={bot.id}
                    onClick={() => onBotSelect(bot.id)}
                    className="flex items-center gap-3 p-3 cursor-pointer"
                  >
                    <div className="w-8 h-8 rounded-lg bg-secondary flex items-center justify-center relative">
                      {status === "running" ? (
                        <Play className="w-4 h-4 text-success" />
                      ) : status === "starting" ? (
                        <Play className="w-4 h-4 text-warning" />
                      ) : status === "error" ? (
                        <AlertTriangle className="w-4 h-4 text-destructive" />
                      ) : (
                        <Square className="w-4 h-4 text-muted-foreground" />
                      )}
                    </div>
                    <div className="flex-1">
                      <div className="flex items-center justify-between">
                        <p className="text-sm font-medium">{bot.bot_version?.label || "Bot"}</p>
                        <div className="flex items-center gap-2">
                          {bot.bot_version?.is_active === false && (
                            <span className="inline-flex items-center gap-1 rounded-full bg-destructive/10 px-2 py-0.5 text-[10px] font-medium text-destructive">
                              <AlertTriangle className="h-2.5 w-2.5" />
                              Locked
                            </span>
                          )}
                          <span className={cn(
                            "text-xs font-mono",
                            botPnl >= 0 ? "text-success" : "text-destructive"
                          )}>
                            {botPnl >= 0 ? "+" : ""}${botPnl.toFixed(2)}
                          </span>
                        </div>
                      </div>
                      <div className="flex items-center gap-2 text-xs text-muted-foreground">
                        <span>{bot.bot_version?.symbol}</span>
                        <span>•</span>
                        <span>{bot.bot_version?.timeframe}</span>
                      </div>
                    </div>
                    {selectedBotId === bot.id && (
                      <Check className="w-4 h-4 text-primary" />
                    )}
                  </DropdownMenuItem>
                );
              })}
            </>
          )}
          {bots.length > 0 && (
            <>
              <DropdownMenuSeparator />
              <DropdownMenuItem
                onClick={() => {
                  if (!addBotDisabled) {
                    setDialogOpen(true);
                  }
                }}
                className="flex items-center gap-3 p-3 cursor-pointer text-primary"
                disabled={addBotDisabled}
                title={addBotDisabled ? addBotDisabledReason ?? undefined : undefined}
              >
                <Plus className="w-4 h-4" />
                <span className="text-sm font-medium">Add New Bot</span>
              </DropdownMenuItem>
            </>
          )}
        </DropdownMenuContent>
      </DropdownMenu>

      <AddBotDialog
        open={dialogOpen}
        onOpenChange={setDialogOpen}
        accountId={accountId}
        existingBots={bots}
        blockedReason={addBotDisabled ? addBotDisabledReason : null}
        onBotAdded={() => {
          onBotAdded();
          setDialogOpen(false);
        }}
      />
    </div>
  );
}
