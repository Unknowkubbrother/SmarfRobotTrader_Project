import { useState } from "react";
import { ChevronDown, Bot, Plus, Check, Activity } from "lucide-react";
import { cn } from "@/lib/utils";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import type { BotConfigWithVersion } from "@/hooks/useTradingAccounts";

interface BotSelectorProps {
  bots: BotConfigWithVersion[];
  selectedBot: BotConfigWithVersion | null;
  onBotChange: (bot: BotConfigWithVersion) => void;
  onAddBot?: () => void;
}

export function BotSelector({ bots, selectedBot, onBotChange, onAddBot }: BotSelectorProps) {
  const getStatusColor = (status: string | null) => {
    switch (status) {
      case "running": return "bg-success";
      case "paused": return "bg-warning";
      default: return "bg-muted-foreground";
    }
  };

  const getBotSymbol = (bot: BotConfigWithVersion) => {
    return bot.bot_version?.symbol || "N/A";
  };

  const getBotLabel = (bot: BotConfigWithVersion) => {
    return bot.bot_version?.label || "No Model";
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <button className="flex items-center gap-3 px-4 py-3 rounded-lg bg-card border border-border hover:border-primary/50 transition-all w-full md:w-auto min-w-[280px]">
          <div className="w-9 h-9 rounded-lg bg-primary/10 flex items-center justify-center">
            <Bot className="w-4 h-4 text-primary" />
          </div>
          <div className="flex-1 text-left">
            {selectedBot ? (
              <>
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium">{getBotLabel(selectedBot)}</p>
                  <span className={cn("w-2 h-2 rounded-full", getStatusColor(selectedBot.status))} />
                </div>
                <p className="text-xs text-muted-foreground">
                  {getBotSymbol(selectedBot)} • {selectedBot.risk_level || "medium"} risk
                </p>
              </>
            ) : (
              <p className="text-sm text-muted-foreground">
                {bots.length > 0 ? "Select Bot" : "No bots configured"}
              </p>
            )}
          </div>
          <ChevronDown className="w-4 h-4 text-muted-foreground" />
        </button>
      </DropdownMenuTrigger>
      <DropdownMenuContent align="start" className="w-[280px]">
        {bots.length === 0 ? (
          <div className="p-4 text-center text-sm text-muted-foreground">
            No bots found. Add one to get started.
          </div>
        ) : (
          bots.map((bot) => (
            <DropdownMenuItem
              key={bot.id}
              onClick={() => onBotChange(bot)}
              className="flex items-center gap-3 p-3 cursor-pointer"
            >
              <div className={cn(
                "w-8 h-8 rounded-lg flex items-center justify-center",
                bot.status === "running" ? "bg-success/10" : "bg-secondary"
              )}>
                <Activity className={cn(
                  "w-4 h-4",
                  bot.status === "running" ? "text-success" : "text-muted-foreground"
                )} />
              </div>
              <div className="flex-1">
                <div className="flex items-center gap-2">
                  <p className="text-sm font-medium">{getBotLabel(bot)}</p>
                  <span className={cn("w-1.5 h-1.5 rounded-full", getStatusColor(bot.status))} />
                </div>
                <div className="flex items-center gap-2 text-xs text-muted-foreground">
                  <span className="font-mono">{getBotSymbol(bot)}</span>
                  <span>•</span>
                  <span>{bot.risk_level || "medium"} risk</span>
                </div>
              </div>
              {selectedBot?.id === bot.id && (
                <Check className="w-4 h-4 text-primary" />
              )}
            </DropdownMenuItem>
          ))
        )}
        {onAddBot && (
          <>
            <DropdownMenuSeparator />
            <DropdownMenuItem 
              onClick={onAddBot}
              className="flex items-center gap-3 p-3 cursor-pointer text-primary"
            >
              <Plus className="w-4 h-4" />
              <span className="text-sm font-medium">Add New Bot</span>
            </DropdownMenuItem>
          </>
        )}
      </DropdownMenuContent>
    </DropdownMenu>
  );
}
