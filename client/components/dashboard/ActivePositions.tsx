import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BotLiveState, MT5Position } from "@/hooks/useBotLiveState";

interface ActivePositionsProps {
  accountId?: string;
  liveState?: BotLiveState;
}

export function ActivePositions({ accountId, liveState }: ActivePositionsProps) {
  const isLive = !!liveState?.connected;
  const positions: MT5Position[] = isLive ? (liveState!.positions || []) : [];
  const totalProfit = positions.reduce((sum, p) => sum + p.profit, 0);

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up" style={{ animationDelay: "200ms" }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <h3 className="font-semibold text-foreground">Active Positions</h3>
          {isLive && (
            <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
              <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
              Live
            </span>
          )}
        </div>
        <div className="flex items-center gap-3">
          <span className={cn(
            "text-sm font-mono font-medium",
            totalProfit >= 0 ? "text-success" : "text-destructive"
          )}>
            {totalProfit >= 0 ? "+" : ""}${totalProfit.toFixed(2)}
          </span>
          <span className="text-xs text-muted-foreground bg-secondary px-2 py-1 rounded">
            {positions.length} open
          </span>
        </div>
      </div>

      {positions.length === 0 ? (
        <div className="text-center py-8 text-muted-foreground text-sm">
          {isLive ? "No open positions" : "Connect a bot to see live positions"}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="border-b border-border">
                <th className="text-left text-xs font-medium text-muted-foreground py-2 px-2">Symbol</th>
                <th className="text-left text-xs font-medium text-muted-foreground py-2 px-2">Type</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">Volume</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">Open</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">Current</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">SL</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">TP</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">Swap</th>
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={position.ticket} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
                  <td className="py-2.5 px-2">
                    <span className="font-mono text-sm">{position.symbol}</span>
                  </td>
                  <td className="py-2.5 px-2">
                    <span
                      className={cn(
                        "inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium",
                        position.type === "BUY"
                          ? "bg-success/10 text-success"
                          : "bg-destructive/10 text-destructive"
                      )}
                    >
                      {position.type === "BUY" ? (
                        <ArrowUpRight className="w-3 h-3" />
                      ) : (
                        <ArrowDownRight className="w-3 h-3" />
                      )}
                      {position.type}
                    </span>
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm">{position.volume}</td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm text-muted-foreground">
                    {position.price_open}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm">{position.price_current}</td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm text-muted-foreground">
                    {position.sl > 0 ? position.sl : "—"}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm text-muted-foreground">
                    {position.tp > 0 ? position.tp : "—"}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm text-muted-foreground">
                    {position.swap !== 0 ? position.swap.toFixed(2) : "0.00"}
                  </td>
                  <td className="py-2.5 px-2 text-right">
                    <span
                      className={cn(
                        "font-mono text-sm font-medium",
                        position.profit >= 0 ? "text-success" : "text-destructive"
                      )}
                    >
                      {position.profit >= 0 ? "+" : ""}${position.profit.toFixed(2)}
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}