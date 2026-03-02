import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Users, Bot, CreditCard, Activity, Ticket, AlertTriangle } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import { useAuth } from "@/contexts/AuthContext";
import { api } from "@/lib/api";
import { toast } from "sonner";
import { AdminUserManagement } from "@/components/admin/AdminUserManagement";
import { AdminBotVersions } from "@/components/admin/AdminBotVersions";
import { AdminSubscriptions } from "@/components/admin/AdminSubscriptions";
import { AdminMonitoring } from "@/components/admin/AdminMonitoring";

type AdminTab = "overview" | "users" | "bots" | "subscriptions" | "monitoring";

export default function Admin() {
  const { isAdmin, loading } = useAuth();
  const router = useRouter();
  const [activeTab, setActiveTab] = useState<AdminTab>("overview");
  const [stats, setStats] = useState({
    totalUsers: 0,
    totalMt5Accounts: 0,
    activeSubscriptions: 0,
    totalBots: 0,
    totalRevenue: 0,
    pendingTickets: 0,
    runningBots: 0,
  });

  useEffect(() => {
    if (!loading && !isAdmin) {
      toast.error("Access denied. Admin privileges required.");
      router.push("/");
    }
  }, [isAdmin, loading, router]);

  useEffect(() => {
    if (isAdmin) {
      fetchStats();
    }
  }, [isAdmin]);

  const fetchStats = async () => {
    try {
      const { data } = await api.get("/admin/stats");

      setStats({
        totalUsers: data?.total_users || 0,
        totalMt5Accounts: data?.total_mt5_accounts || 0,
        activeSubscriptions: data?.active_subscriptions || 0,
        totalBots: data?.total_bot_versions || 0,
        totalRevenue: data?.monthly_revenue || 0,
        pendingTickets: data?.pending_tickets || 0,
        runningBots: data?.running_bots || 0,
      });
    } catch (error: any) {
      console.error("Error fetching stats:", error);
      toast.error(error?.message || "Failed to fetch admin stats");
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-96">
        <div className="animate-spin w-8 h-8 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  if (!isAdmin) {
    return null;
  }

  const tabs = [
    { id: "overview" as AdminTab, label: "Overview", icon: Activity },
    { id: "users" as AdminTab, label: "Users", icon: Users },
    { id: "bots" as AdminTab, label: "Bot Versions", icon: Bot },
    { id: "subscriptions" as AdminTab, label: "Subscriptions", icon: CreditCard },
    { id: "monitoring" as AdminTab, label: "Support Tickets", icon: Ticket },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Admin Panel</h1>
          <p className="text-sm text-muted-foreground">Manage users, bots, and system settings</p>
        </div>
        <Badge variant="secondary" className="bg-primary/10 text-primary">
          Administrator
        </Badge>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-border pb-4">
        {tabs.map((tab) => (
          <Button
            key={tab.id}
            variant={activeTab === tab.id ? "default" : "ghost"}
            size="sm"
            onClick={() => setActiveTab(tab.id)}
            className="gap-2"
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </Button>
        ))}
      </div>

      {/* Content */}
      {activeTab === "overview" && (
        <div className="space-y-6">
          {/* Stats Grid */}
          <div className="grid grid-cols-2 lg:grid-cols-5 gap-4">
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <Users className="w-5 h-5 text-primary" />
                <Badge variant="outline">Total</Badge>
              </div>
              <p className="text-2xl font-bold mt-2">{stats.totalUsers}</p>
              <p className="text-sm text-muted-foreground">Users</p>
            </div>
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <Bot className="w-5 h-5 text-success" />
                <Badge variant="outline">Total</Badge>
              </div>
              <p className="text-2xl font-bold mt-2">{stats.totalMt5Accounts}</p>
              <p className="text-sm text-muted-foreground">Total MT5 accounts</p>
            </div>
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <CreditCard className="w-5 h-5 text-warning" />
                <Badge variant="outline">Revenue</Badge>
              </div>
              <p className="text-2xl font-bold mt-2">${stats.totalRevenue.toLocaleString()}</p>
              <p className="text-sm text-muted-foreground">This Month</p>
            </div>
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <AlertTriangle className="w-5 h-5 text-destructive" />
                <Badge variant="outline">Pending</Badge>
              </div>
              <p className="text-2xl font-bold mt-2">{stats.pendingTickets}</p>
              <p className="text-sm text-muted-foreground">Support Tickets</p>
            </div>
            <div className="glass-card p-4">
              <div className="flex items-center justify-between">
                <Bot className="w-5 h-5 text-warning" />
                <Badge variant="outline">Bot</Badge>
              </div>
              <p className="text-2xl font-bold mt-2">{stats.runningBots}</p>
              <p className="text-sm text-muted-foreground">Running Bot</p>
            </div>
          </div>

          {/* Quick Actions */}
          <div className="glass-card p-6">
            <h3 className="font-semibold mb-4">Quick Actions</h3>
            <div className="grid grid-cols-2 lg:grid-cols-4 gap-3">
              <Button variant="outline" onClick={() => setActiveTab("users")} className="justify-start">
                <Users className="w-4 h-4 mr-2" />
                Manage Users
              </Button>
              <Button variant="outline" onClick={() => setActiveTab("bots")} className="justify-start">
                <Bot className="w-4 h-4 mr-2" />
                Bot Versions
              </Button>
              <Button variant="outline" onClick={() => setActiveTab("subscriptions")} className="justify-start">
                <CreditCard className="w-4 h-4 mr-2" />
                Billing Settings
              </Button>
              <Button variant="outline" onClick={() => setActiveTab("monitoring")} className="justify-start">
                <Activity className="w-4 h-4 mr-2" />
                Ticket Queue
              </Button>
            </div>
          </div>
        </div>
      )}

      {activeTab === "users" && <AdminUserManagement />}
      {activeTab === "bots" && <AdminBotVersions />}
      {activeTab === "subscriptions" && <AdminSubscriptions />}
      {activeTab === "monitoring" && <AdminMonitoring />}
    </div>
  );
}
