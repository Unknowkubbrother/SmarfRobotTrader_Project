import { ArrowUpRight, ArrowDownRight } from "lucide-react";
import { cn } from "@/lib/utils";

interface Position {
  id: number;
  symbol: string;
  type: "BUY" | "SELL";
  volume: number;
  openPrice: number;
  currentPrice: number;
  profit: number;
  openTime: string;
  accountId: string;
}

const allPositions: Position[] = [
  {
    id: 1,
    symbol: "XAUUSD",
    type: "BUY",
    volume: 0.5,
    openPrice: 2028.45,
    currentPrice: 2031.82,
    profit: 168.5,
    openTime: "14:32:15",
    accountId: "1",
  },
  {
    id: 2,
    symbol: "XAUUSD",
    type: "BUY",
    volume: 0.3,
    openPrice: 2025.10,
    currentPrice: 2031.82,
    profit: 201.6,
    openTime: "13:15:42",
    accountId: "1",
  },
  {
    id: 3,
    symbol: "EURUSD",
    type: "SELL",
    volume: 1.0,
    openPrice: 1.0892,
    currentPrice: 1.0872,
    profit: 200.0,
    openTime: "13:45:22",
    accountId: "2",
  },
  {
    id: 4,
    symbol: "BTCUSD",
    type: "BUY",
    volume: 0.1,
    openPrice: 42150.00,
    currentPrice: 42080.00,
    profit: -70.0,
    openTime: "12:18:44",
    accountId: "3",
  },
];

interface ActivePositionsProps {
  accountId?: string;
  botId?: string;
}

export function ActivePositions({ accountId }: ActivePositionsProps) {
  const positions = accountId 
    ? allPositions.filter(p => p.accountId === accountId)
    : allPositions;

  const totalProfit = positions.reduce((sum, p) => sum + p.profit, 0);

  return (
    <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "200ms" }}>
      <div className="flex items-center justify-between mb-4">
        <h3 className="font-semibold text-foreground">Active Positions</h3>
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
          No open positions
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
                <th className="text-right text-xs font-medium text-muted-foreground py-2 px-2">P&L</th>
              </tr>
            </thead>
            <tbody>
              {positions.map((position) => (
                <tr key={position.id} className="border-b border-border/50 hover:bg-secondary/30 transition-colors">
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
                    {position.openPrice}
                  </td>
                  <td className="py-2.5 px-2 text-right font-mono text-sm">{position.currentPrice}</td>
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