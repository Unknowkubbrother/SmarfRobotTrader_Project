import { useState } from "react";
import { Play, Settings2, BarChart3, TrendingUp, Target, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";
import { cn } from "@/lib/utils";

const generateEquityCurve = () => {
  const data = [];
  let equity = 10000;
  
  for (let i = 0; i < 100; i++) {
    const change = (Math.random() - 0.4) * 150;
    equity += change;
    data.push({
      trade: i + 1,
      equity: Math.round(equity * 100) / 100,
    });
  }
  return data;
};

const equityData = generateEquityCurve();

const modelVersions = [
  { id: "v2.4.1", name: "Current (v2.4.1)", date: "2024-01-10" },
  { id: "v2.4.0", name: "v2.4.0", date: "2024-01-05" },
  { id: "v2.3.5", name: "v2.3.5", date: "2023-12-20" },
];

const backtestResults = {
  totalProfit: 2847.50,
  maxDrawdown: -8.4,
  winRate: 68.5,
  sharpeRatio: 1.87,
  totalTrades: 142,
  avgWin: 89.30,
  avgLoss: -42.15,
  profitFactor: 2.12,
};

export default function Backtesting() {
  const [isRunning, setIsRunning] = useState(false);
  const [selectedModel, setSelectedModel] = useState("v2.4.1");
  const [initialBalance, setInitialBalance] = useState("10000");
  const [riskLevel, setRiskLevel] = useState("medium");
  const [hasResults, setHasResults] = useState(true);

  const handleRunBacktest = () => {
    setIsRunning(true);
    setTimeout(() => {
      setIsRunning(false);
      setHasResults(true);
    }, 3000);
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Backtesting & Simulation</h1>
        <p className="text-sm text-muted-foreground">Test AI strategies on historical market data</p>
      </div>

      {/* Configuration */}
      <div className="glass-card p-6 animate-slide-up">
        <div className="flex items-center gap-2 mb-6">
          <Settings2 className="w-5 h-5 text-primary" />
          <h3 className="text-lg font-semibold">Test Configuration</h3>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
          <div>
            <label className="block text-sm text-muted-foreground mb-2">Model Version</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
            >
              {modelVersions.map((v) => (
                <option key={v.id} value={v.id}>
                  {v.name}
                </option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm text-muted-foreground mb-2">Initial Balance</label>
            <input
              type="text"
              value={initialBalance}
              onChange={(e) => setInitialBalance(e.target.value)}
              className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm font-mono focus:outline-none focus:border-primary/50"
            />
          </div>

          <div>
            <label className="block text-sm text-muted-foreground mb-2">Risk Level</label>
            <select
              value={riskLevel}
              onChange={(e) => setRiskLevel(e.target.value)}
              className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
            >
              <option value="low">Low</option>
              <option value="medium">Medium</option>
              <option value="high">High</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-muted-foreground mb-2">Date Range</label>
            <select className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50">
              <option>Last 3 Months</option>
              <option>Last 6 Months</option>
              <option>Last 1 Year</option>
              <option>All Time</option>
            </select>
          </div>
        </div>

        <Button onClick={handleRunBacktest} disabled={isRunning} size="lg">
          {isRunning ? (
            <>
              <div className="w-4 h-4 border-2 border-primary-foreground border-t-transparent rounded-full animate-spin" />
              Running Simulation...
            </>
          ) : (
            <>
              <Play className="w-4 h-4" />
              Run Backtest
            </>
          )}
        </Button>
      </div>

      {hasResults && (
        <>
          {/* Results Summary */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "50ms" }}>
              <div className="flex items-center gap-2 mb-2">
                <TrendingUp className="w-4 h-4 text-success" />
                <span className="text-sm text-muted-foreground">Total Profit</span>
              </div>
              <p className="text-2xl font-bold font-mono profit-text">
                +${backtestResults.totalProfit.toFixed(2)}
              </p>
            </div>

            <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "100ms" }}>
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-warning" />
                <span className="text-sm text-muted-foreground">Max Drawdown</span>
              </div>
              <p className="text-2xl font-bold font-mono text-warning">
                {backtestResults.maxDrawdown}%
              </p>
            </div>

            <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "150ms" }}>
              <div className="flex items-center gap-2 mb-2">
                <Target className="w-4 h-4 text-primary" />
                <span className="text-sm text-muted-foreground">Win Rate</span>
              </div>
              <p className="text-2xl font-bold font-mono">{backtestResults.winRate}%</p>
            </div>

            <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "200ms" }}>
              <div className="flex items-center gap-2 mb-2">
                <BarChart3 className="w-4 h-4 text-accent" />
                <span className="text-sm text-muted-foreground">Sharpe Ratio</span>
              </div>
              <p className="text-2xl font-bold font-mono">{backtestResults.sharpeRatio}</p>
            </div>
          </div>

          {/* Equity Curve */}
          <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "250ms" }}>
            <h3 className="text-lg font-semibold mb-6">Equity Curve</h3>
            <div className="h-[300px]">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={equityData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                  <defs>
                    <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="hsl(187, 100%, 50%)" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="hsl(187, 100%, 50%)" stopOpacity={0} />
                    </linearGradient>
                  </defs>
                  <CartesianGrid strokeDasharray="3 3" stroke="hsl(222, 30%, 16%)" vertical={false} />
                  <XAxis
                    dataKey="trade"
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 12 }}
                  />
                  <YAxis
                    axisLine={false}
                    tickLine={false}
                    tick={{ fill: "hsl(215, 20%, 55%)", fontSize: 12 }}
                    tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
                  />
                  <Tooltip
                    contentStyle={{
                      backgroundColor: "hsl(222, 47%, 10%)",
                      border: "1px solid hsl(222, 30%, 16%)",
                      borderRadius: "8px",
                    }}
                    formatter={(value: number) => [`$${value.toLocaleString()}`, "Equity"]}
                  />
                  <Area
                    type="monotone"
                    dataKey="equity"
                    stroke="hsl(187, 100%, 50%)"
                    strokeWidth={2}
                    fill="url(#equityGradient)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>

          {/* Detailed Stats */}
          <div className="grid md:grid-cols-2 gap-6">
            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "300ms" }}>
              <h3 className="text-lg font-semibold mb-4">Trade Statistics</h3>
              <div className="space-y-3">
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Total Trades</span>
                  <span className="font-mono font-medium">{backtestResults.totalTrades}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Average Win</span>
                  <span className="font-mono font-medium profit-text">
                    +${backtestResults.avgWin.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Average Loss</span>
                  <span className="font-mono font-medium loss-text">
                    ${backtestResults.avgLoss.toFixed(2)}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-muted-foreground">Profit Factor</span>
                  <span className="font-mono font-medium">{backtestResults.profitFactor}</span>
                </div>
              </div>
            </div>

            <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "350ms" }}>
              <h3 className="text-lg font-semibold mb-4">Model Comparison</h3>
              <div className="space-y-3">
                {modelVersions.map((v, i) => (
                  <div
                    key={v.id}
                    className={cn(
                      "flex items-center justify-between p-3 rounded-lg",
                      v.id === selectedModel ? "bg-primary/10 border border-primary/20" : "bg-secondary/30"
                    )}
                  >
                    <div>
                      <p className="font-medium">{v.name}</p>
                      <p className="text-xs text-muted-foreground">{v.date}</p>
                    </div>
                    <span className={cn("font-mono", i === 0 ? "profit-text" : "text-muted-foreground")}>
                      {i === 0 ? "+28.5%" : i === 1 ? "+24.2%" : "+19.8%"}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
