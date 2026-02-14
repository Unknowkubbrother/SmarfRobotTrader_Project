import { useState, useEffect } from "react";
import { Terminal, Bot, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface LogEntry {
  id: number;
  timestamp: string;
  type: "info" | "analysis" | "action" | "warning" | "success";
  message: string;
}

const generateInitialLogs = (botName: string): LogEntry[] => [
  { id: 1, timestamp: "14:05:23", type: "analysis", message: `Analysing XAUUSD... Found support at 2030.00` },
  { id: 2, timestamp: "14:05:24", type: "info", message: `Processing chart pattern recognition` },
  { id: 3, timestamp: "14:05:25", type: "success", message: `Bullish engulfing pattern detected on H4` },
  { id: 4, timestamp: "14:05:26", type: "action", message: `Opening BUY position: XAUUSD @ 2028.45` },
  { id: 5, timestamp: "14:05:27", type: "info", message: `Confidence: 87.3%` },
];

const generateNewLogMessages = () => [
  { type: "info" as const, message: `Market sentiment: Bullish bias on majors` },
  { type: "analysis" as const, message: `Evaluating momentum indicators...` },
  { type: "success" as const, message: `Take profit triggered +$200` },
  { type: "action" as const, message: `Adjusting stop loss to break even` },
  { type: "warning" as const, message: `Approaching daily risk limit (78%)` },
];

const typeConfig = {
  info: { icon: Bot, color: "text-primary" },
  analysis: { icon: TrendingUp, color: "text-muted-foreground" },
  action: { icon: Terminal, color: "text-foreground" },
  warning: { icon: AlertTriangle, color: "text-warning" },
  success: { icon: CheckCircle, color: "text-success" },
};

interface AIConsoleProps {
  botName?: string;
}

export function AIConsole({ botName = "Bot" }: AIConsoleProps) {
  const [logs, setLogs] = useState<LogEntry[]>(() => generateInitialLogs(botName));

  useEffect(() => {
    setLogs(generateInitialLogs(botName));
  }, [botName]);

  useEffect(() => {
    const newLogMessages = generateNewLogMessages();
    const interval = setInterval(() => {
      const randomLog = newLogMessages[Math.floor(Math.random() * newLogMessages.length)];
      const now = new Date();
      const timestamp = `${now.getHours().toString().padStart(2, "0")}:${now.getMinutes().toString().padStart(2, "0")}:${now.getSeconds().toString().padStart(2, "0")}`;

      setLogs((prev) => [
        ...prev.slice(-8),
        {
          id: Date.now(),
          timestamp,
          ...randomLog,
        },
      ]);
    }, 4000);

    return () => clearInterval(interval);
  }, [botName]);

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full flex flex-col" style={{ animationDelay: "150ms" }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          <h3 className="font-semibold text-foreground">Activity Log</h3>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-1.5 h-1.5 rounded-full bg-success" />
          <span className="text-xs text-muted-foreground">Live</span>
        </div>
      </div>

      <div className="bg-secondary/50 rounded-lg p-3 flex-1 overflow-y-auto scrollbar-thin min-h-[200px]">
        {logs.map((log, index) => {
          const config = typeConfig[log.type];
          const Icon = config.icon;
          return (
            <div
              key={log.id}
              className={cn(
                "flex items-start gap-2 py-1.5 text-sm",
                index === logs.length - 1 && "bg-primary/5 -mx-1 px-1 rounded"
              )}
            >
              <span className="text-xs text-muted-foreground font-mono shrink-0">{log.timestamp}</span>
              <Icon className={cn("w-3.5 h-3.5 shrink-0 mt-0.5", config.color)} />
              <span className={cn("text-sm", config.color)}>{log.message}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}