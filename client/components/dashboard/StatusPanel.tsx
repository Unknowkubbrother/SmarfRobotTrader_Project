import { Server, TrendingUp, Shield, Wallet, Activity } from "lucide-react";
import { AccountWithBots } from "@/hooks/useTradingAccounts";
import type { BotLiveState } from "@/hooks/useBotLiveState";

interface StatusPanelProps {
  account?: AccountWithBots | null;
  liveState?: BotLiveState;
}

export function StatusPanel({ account, liveState }: StatusPanelProps) {
  const isLive = !!liveState?.connected;

  const balance = liveState?.balance ?? account?.balance ?? 0;
  const equity = liveState?.equity ?? account?.equity ?? balance;
  const freeMargin = liveState?.free_margin ?? account?.margin_free ?? equity * 0.67;
  const marginLevel = liveState?.margin_level ?? account?.margin_level ?? (freeMargin > 0 ? (equity / (equity - freeMargin)) * 100 : 0);
  const leverage = liveState?.leverage ?? account?.leverage ?? 100;
  const serverName = (isLive && liveState?.server) ? liveState.server : (account?.server_name || "-");
  const loginId = (isLive && liveState?.login) ? liveState.login : account?.mt5_login_id;
  const currency = (isLive && liveState?.currency) ? liveState.currency : "USD";

  const statusItems = [
    { icon: Server, label: "Broker", value: account?.broker_name || "-", color: "text-primary" },
    {
      icon: Activity, label: "Server", value: serverName,
      color: isLive ? "text-success" : (account ? "text-success" : "text-muted-foreground"),
    },
    { icon: TrendingUp, label: "Leverage", value: `1:${leverage}`, color: "text-foreground" },
    {
      icon: Shield, label: "Margin Level",
      value: marginLevel > 0 ? `${marginLevel.toFixed(0)}%` : "-",
      color: marginLevel > 100 ? "text-success" : "text-warning",
    },
    {
      icon: Wallet, label: "Free Margin",
      value: `${currency === "USD" ? "$" : ""}${freeMargin.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`,
      color: "text-foreground",
    },
  ];

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full flex flex-col" style={{ animationDelay: "150ms" }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="text-lg font-semibold text-foreground">Account Status</h3>
          {isLive && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
              <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
              Live
            </span>
          )}
        </div>
        {loginId && (
          <span className="text-xs bg-secondary px-2 py-1 rounded font-mono">
            #{loginId}
          </span>
        )}
      </div>
      <div className="space-y-4">
        {statusItems.map((item) => (
          <div key={item.label} className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 rounded-lg bg-secondary/50 flex items-center justify-center">
                <item.icon className="w-4 h-4 text-muted-foreground" />
              </div>
              <span className="text-sm text-muted-foreground">{item.label}</span>
            </div>
            <span className={`text-sm font-medium font-mono ${item.color}`}>{item.value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
