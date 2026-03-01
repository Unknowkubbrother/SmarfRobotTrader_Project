import { useMemo } from "react";
import { Terminal, Bot, TrendingUp, AlertTriangle, CheckCircle } from "lucide-react";
import { cn } from "@/lib/utils";
import type { BotLiveState } from "@/hooks/useBotLiveState";

interface LogEntry {
  id: string;
  timestamp: string;
  type: "info" | "analysis" | "action" | "warning" | "success";
  message: string;
}

const typeConfig = {
  info: { icon: Bot, color: "text-primary" },
  analysis: { icon: TrendingUp, color: "text-muted-foreground" },
  action: { icon: Terminal, color: "text-foreground" },
  warning: { icon: AlertTriangle, color: "text-warning" },
  success: { icon: CheckCircle, color: "text-success" },
};

interface AIConsoleProps {
  botName?: string;
  liveState?: BotLiveState;
}

export function AIConsole({ botName = "Bot", liveState }: AIConsoleProps) {
  const isLive = !!liveState?.connected;
  const llmText = liveState?.llm_text || "";

  const logs = useMemo<LogEntry[]>(() => {
    const wsLogs = Array.isArray(liveState?.recent_logs) ? liveState!.recent_logs : [];
    const dedup = new Set<string>();
    const rows: LogEntry[] = [];

    for (let i = 0; i < wsLogs.length; i += 1) {
      const row = wsLogs[i];
      const type = row?.type && row.type in typeConfig ? row.type : "info";
      const timestamp = String(row?.timestamp || "");
      const message = String(row?.message || "").trim();
      if (!message) continue;
      const key = `${timestamp}|${type}|${message}`;
      if (dedup.has(key)) continue;
      dedup.add(key);
      rows.push({
        id: `${key}|${i}`,
        timestamp: timestamp || "--:--:--",
        type,
        message,
      });
    }

    return rows.slice(-80);
  }, [liveState?.recent_logs]);

  return (
    <div className="bg-white border rounded-xl shadow-sm p-6 animate-slide-up h-full flex flex-col" style={{ animationDelay: "150ms" }}>
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-2">
          <Terminal className="w-4 h-4 text-muted-foreground" />
          <h3 className="font-semibold text-foreground">Activity Log</h3>
        </div>
        {isLive && (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
            <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
            Live
          </span>
        )}
      </div>

      <div className="flex flex-col gap-2 flex-1 min-h-[220px] h-[300px]">
        {isLive && llmText && (
          <div className="bg-primary/5 border border-primary/20 rounded-lg p-3 text-xs text-foreground/80 font-mono whitespace-pre-wrap flex-shrink-0 max-h-[110px] overflow-y-auto scrollbar-thin">
            <div className="text-primary font-semibold mb-1 flex items-center gap-1.5">
              <Bot className="w-3.5 h-3.5" /> AI Analysis
            </div>
            {llmText}
          </div>
        )}

        <div className="bg-secondary/50 rounded-lg p-3 flex-1 overflow-y-auto scrollbar-thin">
          {logs.length === 0 ? (
            <div className="h-full flex items-center justify-center text-xs text-muted-foreground">
              {isLive ? `Waiting for runtime logs from ${botName}` : `Connect ${botName} to stream logs`}
            </div>
          ) : (
            logs.map((log, index) => {
              const config = typeConfig[log.type];
              const Icon = config.icon;
              return (
                <div
                  key={log.id}
                  className={cn(
                    "flex items-start gap-2 py-1.5 text-sm transition-colors",
                    index === logs.length - 1 && "bg-primary/5 -mx-1 px-1 rounded font-medium"
                  )}
                >
                  <span className="text-xs text-muted-foreground font-mono shrink-0">{log.timestamp}</span>
                  <Icon className={cn("w-3.5 h-3.5 shrink-0 mt-0.5", config.color)} />
                  <span className={cn("text-sm", config.color)}>{log.message}</span>
                </div>
              );
            })
          )}
        </div>
      </div>
    </div>
  );
}
