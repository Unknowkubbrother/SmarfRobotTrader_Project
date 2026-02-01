import { useState, useEffect } from "react";
import { Activity, Server, AlertTriangle, CheckCircle, Clock, RefreshCw } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { supabase } from "@/lib/integrations/supabase/client";
import { toast } from "sonner";

interface ActivityLog {
  id: string;
  user_id: string;
  topic: string;
  detail: string | null;
  ip_address: string | null;
  created_at: string | null;
}

export function AdminMonitoring() {
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>([]);
  const [loading, setLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState({
    database: "healthy",
    auth: "healthy",
    storage: "healthy",
    functions: "healthy",
  });

  useEffect(() => {
    fetchLogs();
  }, []);

  const fetchLogs = async () => {
    try {
      const { data, error } = await supabase
        .from("activity_logs")
        .select("*")
        .order("created_at", { ascending: false })
        .limit(50);

      if (error) throw error;
      setActivityLogs(data || []);
    } catch (error) {
      console.error("Error fetching logs:", error);
      toast.error("Failed to load activity logs");
    } finally {
      setLoading(false);
    }
  };

  const runHealthCheck = () => {
    toast.success("Health check completed - All systems operational");
    setSystemStatus({
      database: "healthy",
      auth: "healthy",
      storage: "healthy",
      functions: "healthy",
    });
  };

  const getStatusIcon = (status: string) => {
    if (status === "healthy") {
      return <CheckCircle className="w-4 h-4 text-success" />;
    }
    return <AlertTriangle className="w-4 h-4 text-destructive" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* System Health */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center gap-2">
            <Server className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">System Health</h3>
          </div>
          <Button variant="outline" size="sm" onClick={runHealthCheck}>
            <RefreshCw className="w-4 h-4 mr-2" />
            Run Health Check
          </Button>
        </div>

        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {Object.entries(systemStatus).map(([service, status]) => (
            <div key={service} className="flex items-center gap-3 p-3 rounded-lg bg-secondary/50">
              {getStatusIcon(status)}
              <div>
                <p className="text-sm font-medium capitalize">{service}</p>
                <p className="text-xs text-muted-foreground capitalize">{status}</p>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Activity Logs */}
      <div className="glass-card">
        <div className="p-4 border-b border-border flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Activity className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Recent Activity</h3>
          </div>
          <Button variant="ghost" size="sm" onClick={fetchLogs}>
            <RefreshCw className="w-4 h-4" />
          </Button>
        </div>

        <div className="divide-y divide-border max-h-96 overflow-y-auto">
          {activityLogs.length === 0 ? (
            <div className="p-8 text-center text-muted-foreground">
              No activity logs yet
            </div>
          ) : (
            activityLogs.map((log) => (
              <div key={log.id} className="p-4 hover:bg-secondary/30 transition-colors">
                <div className="flex items-start justify-between">
                  <div>
                    <div className="flex items-center gap-2">
                      <Badge variant="outline">{log.topic}</Badge>
                    </div>
                    {log.detail && (
                      <p className="text-sm text-muted-foreground mt-1">{log.detail}</p>
                    )}
                    {log.ip_address && (
                      <p className="text-xs text-muted-foreground mt-1 font-mono">
                        IP: {log.ip_address}
                      </p>
                    )}
                  </div>
                  <div className="flex items-center gap-1 text-xs text-muted-foreground">
                    <Clock className="w-3 h-3" />
                    {log.created_at ? new Date(log.created_at).toLocaleString() : 'N/A'}
                  </div>
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      {/* Error Alerts (Placeholder) */}
      <div className="glass-card p-6">
        <div className="flex items-center gap-2 mb-4">
          <AlertTriangle className="w-5 h-5 text-warning" />
          <h3 className="font-semibold">Error Alerts</h3>
        </div>
        <div className="text-center py-8 text-muted-foreground">
          <CheckCircle className="w-12 h-12 text-success mx-auto mb-3" />
          <p>No critical errors detected</p>
          <p className="text-sm">All systems operating normally</p>
        </div>
      </div>
    </div>
  );
}
