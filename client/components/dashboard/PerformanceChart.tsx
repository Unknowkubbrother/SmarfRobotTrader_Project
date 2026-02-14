import { useState } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend,
} from "recharts";
import { cn } from "@/lib/utils";

const generateData = () => {
  const data = [];
  let balance = 10000;
  let equity = 10000;

  for (let i = 0; i < 30; i++) {
    const date = new Date();
    date.setDate(date.getDate() - (29 - i));

    const balanceChange = (Math.random() - 0.4) * 200;
    const equityChange = (Math.random() - 0.4) * 250;

    balance += balanceChange;
    equity = balance + (Math.random() - 0.5) * 500;

    data.push({
      date: date.toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      balance: Math.round(balance * 100) / 100,
      equity: Math.round(equity * 100) / 100,
    });
  }
  return data;
};

const data = generateData();

const timeframes = ["1D", "1W", "1M", "3M", "1Y", "ALL"];

export function PerformanceChart() {
  const [selectedTimeframe, setSelectedTimeframe] = useState("1M");

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h3 className="font-semibold text-foreground">Portfolio Performance</h3>
          <p className="text-sm text-muted-foreground">Balance & Equity over time</p>
        </div>
        <div className="flex items-center gap-1 p-1 rounded-lg bg-secondary">
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => setSelectedTimeframe(tf)}
              className={cn(
                "px-2.5 py-1 rounded text-xs font-medium transition-all",
                selectedTimeframe === tf
                  ? "bg-primary text-primary-foreground"
                  : "text-muted-foreground hover:text-foreground"
              )}
            >
              {tf}
            </button>
          ))}
        </div>
      </div>

      <div className="h-[280px]">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
            <defs>
              <linearGradient id="balanceGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(220, 90%, 56%)" stopOpacity={0.2} />
                <stop offset="100%" stopColor="hsl(220, 90%, 56%)" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="equityGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="0%" stopColor="hsl(142, 72%, 42%)" stopOpacity={0.2} />
                <stop offset="100%" stopColor="hsl(142, 72%, 42%)" stopOpacity={0} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="hsl(220, 13%, 91%)" vertical={false} />
            <XAxis
              dataKey="date"
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(220, 9%, 46%)", fontSize: 11 }}
            />
            <YAxis
              axisLine={false}
              tickLine={false}
              tick={{ fill: "hsl(220, 9%, 46%)", fontSize: 11 }}
              tickFormatter={(value) => `$${(value / 1000).toFixed(1)}k`}
            />
            <Tooltip
              contentStyle={{
                backgroundColor: "hsl(0, 0%, 100%)",
                border: "1px solid hsl(220, 13%, 91%)",
                borderRadius: "8px",
                boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
              }}
              formatter={(value: number) => [`$${value.toLocaleString()}`, ""]}
            />
            <Legend
              verticalAlign="top"
              align="right"
              height={36}
              iconType="circle"
            />
            <Area
              type="monotone"
              dataKey="balance"
              name="Balance"
              stroke="hsl(220, 90%, 56%)"
              strokeWidth={2}
              fill="url(#balanceGradient)"
            />
            <Area
              type="monotone"
              dataKey="equity"
              name="Equity"
              stroke="hsl(142, 72%, 42%)"
              strokeWidth={2}
              fill="url(#equityGradient)"
            />
          </AreaChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}