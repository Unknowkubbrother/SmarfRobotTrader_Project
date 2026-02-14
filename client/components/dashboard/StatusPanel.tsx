import { Server, TrendingUp, Shield, Wallet, Activity } from "lucide-react";
import { AccountWithBots } from "@/hooks/useTradingAccounts";

interface StatusPanelProps {
  account?: AccountWithBots | null;
}

export function StatusPanel({ account }: StatusPanelProps) {
  const balance = account?.balance || 0;
  const equity = account?.equity || 0;
  const freeMargin = equity * 0.67;
  const marginLevel = freeMargin > 0 ? (equity / (equity - freeMargin)) * 100 : 0;

  const statusItems = [
    { icon: Server, label: "Broker", value: account?.broker_name || "-", color: "text-primary" },
    { icon: Activity, label: "Server", value: account?.server_name || "-", color: account ? "text-success" : "text-muted-foreground" },
    { icon: TrendingUp, label: "Leverage", value: "1:100", color: "text-foreground" },
    { icon: Shield, label: "Margin Level", value: marginLevel > 0 ? `${marginLevel.toFixed(0)}%` : "-", color: marginLevel > 100 ? "text-success" : "text-warning" },
    { icon: Wallet, label: "Free Margin", value: account ? `$${freeMargin.toLocaleString(undefined, { minimumFractionDigits: 2, maximumFractionDigits: 2 })}` : "-", color: "text-foreground" },
  ];

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full flex flex-col" style={{ animationDelay: "150ms" }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-foreground">Account Status</h3>
        {account && (
          <span className="text-xs bg-secondary px-2 py-1 rounded font-mono">
            #{account.mt5_login_id}
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
