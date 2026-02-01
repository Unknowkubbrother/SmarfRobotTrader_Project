import { useState } from "react";
import { ChevronLeft, ChevronRight, TrendingUp, TrendingDown, BarChart3 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";

interface DayData {
  date: number;
  profit: number | null;
  trades: number;
  winRate: number;
}

const generateMonthData = (year: number, month: number): DayData[] => {
  const daysInMonth = new Date(year, month + 1, 0).getDate();
  const today = new Date();
  const data: DayData[] = [];

  for (let day = 1; day <= daysInMonth; day++) {
    const date = new Date(year, month, day);
    const isWeekend = date.getDay() === 0 || date.getDay() === 6;
    const isFuture = date > today;

    if (isWeekend || isFuture) {
      data.push({ date: day, profit: null, trades: 0, winRate: 0 });
    } else {
      const profit = (Math.random() - 0.35) * 600;
      data.push({
        date: day,
        profit: Math.round(profit * 100) / 100,
        trades: Math.floor(Math.random() * 15) + 1,
        winRate: Math.floor(Math.random() * 40) + 50,
      });
    }
  }

  return data;
};

const weekDays = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];

export default function ProfitCalendar() {
  const [currentDate, setCurrentDate] = useState(new Date());
  const [selectedDay, setSelectedDay] = useState<DayData | null>(null);

  const year = currentDate.getFullYear();
  const month = currentDate.getMonth();
  const monthData = generateMonthData(year, month);

  const firstDayOfMonth = new Date(year, month, 1).getDay();
  const monthName = currentDate.toLocaleString("default", { month: "long", year: "numeric" });

  const totalProfit = monthData.reduce((sum, day) => sum + (day.profit || 0), 0);
  const tradingDays = monthData.filter((d) => d.profit !== null).length;
  const profitableDays = monthData.filter((d) => d.profit !== null && d.profit > 0).length;
  const totalTrades = monthData.reduce((sum, day) => sum + day.trades, 0);

  const navigateMonth = (direction: number) => {
    setCurrentDate(new Date(year, month + direction, 1));
    setSelectedDay(null);
  };

  const getIntensity = (profit: number | null): string => {
    if (profit === null) return "bg-muted/30";
    if (profit === 0) return "bg-neutral";
    if (profit > 0) {
      if (profit > 300) return "bg-success";
      if (profit > 100) return "bg-success/70";
      return "bg-success/40";
    } else {
      if (profit < -300) return "bg-destructive";
      if (profit < -100) return "bg-destructive/70";
      return "bg-destructive/40";
    }
  };

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div>
        <h1 className="text-2xl font-bold text-foreground">Profit Calendar</h1>
        <p className="text-sm text-muted-foreground">Visualize your daily trading performance</p>
      </div>

      {/* Monthly Stats */}
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="glass-card p-4 animate-slide-up">
          <p className="text-sm text-muted-foreground mb-1">Monthly P&L</p>
          <p className={cn("text-2xl font-bold font-mono", totalProfit >= 0 ? "profit-text" : "loss-text")}>
            {totalProfit >= 0 ? "+" : ""}${totalProfit.toFixed(2)}
          </p>
        </div>
        <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "50ms" }}>
          <p className="text-sm text-muted-foreground mb-1">Total Trades</p>
          <p className="text-2xl font-bold font-mono text-foreground">{totalTrades}</p>
        </div>
        <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "100ms" }}>
          <p className="text-sm text-muted-foreground mb-1">Win Rate</p>
          <p className="text-2xl font-bold font-mono text-success">
            {tradingDays > 0 ? ((profitableDays / tradingDays) * 100).toFixed(1) : 0}%
          </p>
        </div>
        <div className="glass-card p-4 animate-slide-up" style={{ animationDelay: "150ms" }}>
          <p className="text-sm text-muted-foreground mb-1">Trading Days</p>
          <p className="text-2xl font-bold font-mono text-foreground">{tradingDays}</p>
        </div>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        {/* Calendar */}
        <div className="lg:col-span-2 glass-card p-6 animate-slide-up" style={{ animationDelay: "200ms" }}>
          {/* Navigation */}
          <div className="flex items-center justify-between mb-6">
            <Button variant="ghost" size="icon" onClick={() => navigateMonth(-1)}>
              <ChevronLeft className="w-5 h-5" />
            </Button>
            <h3 className="text-xl font-semibold">{monthName}</h3>
            <Button variant="ghost" size="icon" onClick={() => navigateMonth(1)}>
              <ChevronRight className="w-5 h-5" />
            </Button>
          </div>

          {/* Week Days Header */}
          <div className="grid grid-cols-7 gap-2 mb-2">
            {weekDays.map((day) => (
              <div key={day} className="text-center text-xs font-medium text-muted-foreground py-2">
                {day}
              </div>
            ))}
          </div>

          {/* Calendar Grid */}
          <div className="grid grid-cols-7 gap-2">
            {/* Empty cells for days before the first of the month */}
            {Array.from({ length: firstDayOfMonth }).map((_, i) => (
              <div key={`empty-${i}`} className="aspect-square" />
            ))}

            {/* Day cells */}
            {monthData.map((day) => (
              <button
                key={day.date}
                onClick={() => day.profit !== null && setSelectedDay(day)}
                disabled={day.profit === null}
                className={cn(
                  "aspect-square rounded-lg flex flex-col items-center justify-center transition-all p-1",
                  getIntensity(day.profit),
                  day.profit !== null && "hover:ring-2 hover:ring-primary cursor-pointer",
                  selectedDay?.date === day.date && "ring-2 ring-primary"
                )}
              >
                <span className="text-xs text-white/90 font-medium">{day.date}</span>
                {day.profit !== null && (
                  <span className="text-[10px] font-mono font-semibold mt-0.5 text-white">
                    {day.profit >= 0 ? "+" : ""}${Math.abs(day.profit).toFixed(0)}
                  </span>
                )}
              </button>
            ))}
          </div>

          {/* Legend */}
          <div className="flex items-center justify-center gap-6 mt-6 pt-4 border-t border-border">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-success" />
              <span className="text-xs text-muted-foreground">Profit</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-destructive" />
              <span className="text-xs text-muted-foreground">Loss</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded bg-muted/30" />
              <span className="text-xs text-muted-foreground">No Trading</span>
            </div>
          </div>
        </div>

        {/* Day Details */}
        <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "250ms" }}>
          <h3 className="text-lg font-semibold mb-4">Day Details</h3>

          {selectedDay ? (
            <div className="space-y-6">
              <div className="text-center p-6 rounded-lg bg-secondary/50">
                <p className="text-sm text-muted-foreground mb-2">
                  {new Date(year, month, selectedDay.date).toLocaleDateString("en-US", {
                    weekday: "long",
                    month: "long",
                    day: "numeric",
                  })}
                </p>
                <p
                  className={cn(
                    "text-4xl font-bold font-mono",
                    selectedDay.profit! >= 0 ? "profit-text" : "loss-text"
                  )}
                >
                  {selectedDay.profit! >= 0 ? "+" : ""}${selectedDay.profit!.toFixed(2)}
                </p>
              </div>

              <div className="space-y-4">
                <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                  <div className="flex items-center gap-2">
                    <BarChart3 className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Trades</span>
                  </div>
                  <span className="font-mono font-medium">{selectedDay.trades}</span>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                  <div className="flex items-center gap-2">
                    <TrendingUp className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Win Rate</span>
                  </div>
                  <span className="font-mono font-medium text-success">{selectedDay.winRate}%</span>
                </div>

                <div className="flex items-center justify-between p-3 rounded-lg bg-secondary/30">
                  <div className="flex items-center gap-2">
                    <TrendingDown className="w-4 h-4 text-muted-foreground" />
                    <span className="text-sm text-muted-foreground">Growth</span>
                  </div>
                  <span
                    className={cn(
                      "font-mono font-medium",
                      selectedDay.profit! >= 0 ? "text-success" : "text-destructive"
                    )}
                  >
                    {selectedDay.profit! >= 0 ? "+" : ""}
                    {((selectedDay.profit! / 10000) * 100).toFixed(2)}%
                  </span>
                </div>
              </div>

              <Button variant="outline" className="w-full">
                View Trade History
              </Button>
            </div>
          ) : (
            <div className="text-center py-12 text-muted-foreground">
              <BarChart3 className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <p>Select a day to view details</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
