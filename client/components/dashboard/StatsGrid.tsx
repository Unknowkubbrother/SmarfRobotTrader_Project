import { TrendingUp, TrendingDown, BarChart3, Target, Percent, DollarSign } from "lucide-react";
import { cn } from "@/lib/utils";
import { useDailyAggregates } from "@/hooks/useDailyAggregates";
import { Skeleton } from "@/components/ui/skeleton";

interface StatsGridProps {
  accountId?: string;
}

export function StatsGrid({ accountId }: StatsGridProps) {
  const { aggregates, loading, getStats } = useDailyAggregates(accountId);
  
  const stats = getStats();
  
  // Calculate additional stats from aggregates
  const todayProfit = aggregates[0]?.daily_net_profit || 0;
  const todayTrades = aggregates[0]?.total_trades || 0;
  
  // Calculate max drawdown from recent data
  let maxDrawdown = 0;
  let runningPeak = 0;
  aggregates.slice().reverse().forEach(agg => {
    runningPeak = Math.max(runningPeak, agg.daily_net_profit || 0);
    const drawdown = runningPeak > 0 ? ((runningPeak - (agg.daily_net_profit || 0)) / runningPeak) * 100 : 0;
    maxDrawdown = Math.max(maxDrawdown, drawdown);
  });

  // Calculate profit factor
  const totalGrossProfit = aggregates.filter(a => (a.daily_net_profit || 0) > 0).reduce((sum, a) => sum + (a.daily_net_profit || 0), 0);
  const totalGrossLoss = Math.abs(aggregates.filter(a => (a.daily_net_profit || 0) < 0).reduce((sum, a) => sum + (a.daily_net_profit || 0), 0));
  const profitFactor = totalGrossLoss > 0 ? totalGrossProfit / totalGrossLoss : totalGrossProfit > 0 ? 999 : 0;

  const statsData = [
    {
      label: "Total Balance",
      value: `$${stats.totalProfit.toFixed(2)}`,
      change: stats.totalProfit >= 0 ? `+${((stats.totalProfit / 10000) * 100).toFixed(2)}%` : `${((stats.totalProfit / 10000) * 100).toFixed(2)}%`,
      isPositive: stats.totalProfit >= 0,
      icon: DollarSign,
    },
    {
      label: "Today's P&L",
      value: `${todayProfit >= 0 ? '+' : ''}$${todayProfit.toFixed(2)}`,
      change: todayTrades > 0 ? `${todayTrades} trades` : "No trades",
      isPositive: todayProfit >= 0,
      icon: TrendingUp,
    },
    {
      label: "Win Rate",
      value: `${stats.winRate.toFixed(1)}%`,
      change: `${stats.profitableDays}/${stats.tradingDays} days`,
      isPositive: stats.winRate >= 50,
      icon: Target,
    },
    {
      label: "Max Drawdown",
      value: `-${maxDrawdown.toFixed(1)}%`,
      change: maxDrawdown < 10 ? "Low risk" : maxDrawdown < 20 ? "Moderate" : "High risk",
      isPositive: maxDrawdown < 10,
      icon: TrendingDown,
    },
    {
      label: "Total Trades",
      value: stats.totalTrades.toLocaleString(),
      change: `${stats.tradingDays} days`,
      isPositive: true,
      icon: BarChart3,
    },
    {
      label: "Profit Factor",
      value: profitFactor > 100 ? "∞" : profitFactor.toFixed(2),
      change: profitFactor >= 2 ? "Excellent" : profitFactor >= 1.5 ? "Good" : profitFactor >= 1 ? "Fair" : "Poor",
      isPositive: profitFactor >= 1.5,
      icon: Percent,
    },
  ];

  if (loading) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
        {[...Array(6)].map((_, i) => (
          <div key={i} className="glass-card p-4">
            <div className="flex items-center justify-between mb-2">
              <Skeleton className="w-4 h-4 rounded" />
              <Skeleton className="w-10 h-4" />
            </div>
            <Skeleton className="w-20 h-6 mb-1" />
            <Skeleton className="w-16 h-3" />
          </div>
        ))}
      </div>
    );
  }

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
      {statsData.map((stat, index) => (
        <div
          key={stat.label}
          className="glass-card p-4 animate-slide-up"
          style={{ animationDelay: `${index * 50}ms` }}
        >
          <div className="flex items-center justify-between mb-2">
            <stat.icon className="w-4 h-4 text-muted-foreground" />
            <span
              className={cn(
                "text-xs font-medium",
                stat.isPositive ? "text-success" : "text-destructive"
              )}
            >
              {stat.change}
            </span>
          </div>
          <p className="text-xl font-semibold font-mono">{stat.value}</p>
          <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
        </div>
      ))}
    </div>
  );
}
