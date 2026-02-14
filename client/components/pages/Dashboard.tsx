import { useState, useEffect } from "react";
import { Power, Play, TrendingUp, TrendingDown, Activity, Wallet, Target, BarChart3, Calculator, Plus } from "lucide-react";
import { PerformanceChart } from "@/components/dashboard/PerformanceChart";
import { StatusPanel } from "@/components/dashboard/StatusPanel";
import { ActivePositions } from "@/components/dashboard/ActivePositions";
import { AIConsole } from "@/components/dashboard/AIConsole";
import { AccountSelector, AccountWithBots } from "@/components/dashboard/AccountSelector";
import { BotCard } from "@/components/dashboard/BotCard";
import { AddBotDialog } from "@/components/dialogs/AddBotDialog";
import { AddAccountDialog } from "@/components/dialogs/AddAccountDialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useTradingAccounts, BotConfigWithVersion } from "@/hooks/useTradingAccounts";
import { BotSelector } from "@/components/dashboard/BotSelector";

export default function Dashboard() {
  const { accounts, loading, updateBotStatus, deleteBot, refetch } = useTradingAccounts();
  const [selectedAccount, setSelectedAccount] = useState<AccountWithBots | null>(null);
  const [selectedBotId, setSelectedBotId] = useState<string | null>(null);
  const [addBotOpen, setAddBotOpen] = useState(false);
  const [addAccountOpen, setAddAccountOpen] = useState(false);

  // Auto-select first account when accounts load
  useEffect(() => {
    if (accounts.length > 0 && !selectedAccount) {
      setSelectedAccount(accounts[0]);
    }
  }, [accounts, selectedAccount]);

  // Update selected account when accounts change
  useEffect(() => {
    if (selectedAccount) {
      const updated = accounts.find(a => a.id === selectedAccount.id);
      if (updated) {
        setSelectedAccount(updated);
      }
    }
  }, [accounts, selectedAccount]);

  // Reset bot selection when account changes
  useEffect(() => {
    setSelectedBotId(null);
  }, [selectedAccount?.id]);

  const handleToggleBotStatus = async (botId: string, newStatus: "running" | "stopped") => {
    const success = await updateBotStatus(botId, newStatus);
    if (success) {
      toast.success(newStatus === "running" ? "Bot started" : "Bot stopped");
    }
  };

  const handleDeleteBot = async (botId: string) => {
    const success = await deleteBot(botId);
    if (success) {
      toast.success("Bot removed");
    }
  };

  // Mock stats data matching the design
  const stats = [
    { icon: Wallet, label: "Total Balance", value: "$12,482.35", change: "+8.24%", changeType: "positive" as const },
    { icon: TrendingUp, label: "Today's P&L", value: `$${(selectedAccount?.total_today_pnl || 0).toFixed(2)}`, change: "+2.35%", changeType: "positive" as const },
    { icon: Target, label: "Win Rate", value: "72.4%", change: "+3.2%", changeType: "positive" as const },
    { icon: TrendingDown, label: "Max Drawdown", value: "-4.2%", change: "-0.8%", changeType: "negative" as const },
    { icon: BarChart3, label: "Total Trades", value: "1,247", change: "+12", changeType: "positive" as const },
    { icon: Calculator, label: "Profit Factor", value: "2.14", change: "+0.12", changeType: "positive" as const },
  ];

  const bots = selectedAccount?.bot_configurations || [];

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Dashboard</h1>
          <p className="text-sm text-muted-foreground">Monitor your trading performance</p>
        </div>

        {/* Selectors Row */}
        <div className="flex flex-col md:flex-row gap-4">
          <AccountSelector
            selectedAccount={selectedAccount}
            onAccountChange={setSelectedAccount}
            accounts={accounts}
            isLoading={loading}
            onRefresh={refetch}
          />

          {selectedAccount && (
            <BotSelector
              bots={selectedAccount.bot_configurations}
              selectedBotId={selectedBotId}
              onBotSelect={setSelectedBotId}
              accountId={selectedAccount.id}
              onBotAdded={refetch}
            />
          )}
        </div>
      </div>

      {/* Main Content Area */}
      {!selectedAccount ? (
        <div className="glass-card p-12 text-center animate-slide-up flex flex-col items-center justify-center min-h-[400px]">
          <div className="w-16 h-16 bg-secondary rounded-2xl flex items-center justify-center mb-6">
            <Wallet className="w-8 h-8 text-muted-foreground" />
          </div>
          <h2 className="text-xl font-semibold mb-2">Connect Your Trading Account</h2>
          <p className="text-muted-foreground max-w-md mx-auto mb-8">
            Link your MT5 broker account to start trading with our AI-powered bots.
          </p>
          <Button onClick={() => setAddAccountOpen(true)} className="gap-2">
            <Plus className="w-4 h-4" />
            Add Trading Account
          </Button>
        </div>
      ) : !selectedBotId ? (
        <div className="space-y-6 animate-slide-up">
          <div className="flex items-center justify-between">
            <h2 className="text-lg font-semibold text-foreground">
              Select a Bot to View Dashboard
            </h2>
            <Button onClick={() => setAddBotOpen(true)} className="gap-2">
              <Plus className="w-4 h-4" />
              Create New Bot
            </Button>
          </div>

          {bots.length === 0 ? (
            <div className="glass-card p-12 text-center flex flex-col items-center justify-center min-h-[300px]">
              <div className="w-16 h-16 bg-secondary rounded-2xl flex items-center justify-center mb-6">
                <Activity className="w-8 h-8 text-muted-foreground" />
              </div>
              <h3 className="text-xl font-medium mb-2">No bots configured</h3>
              <p className="text-muted-foreground mb-6">
                Create a bot to start automated trading and view performance analytics.
              </p>
              <Button onClick={() => setAddBotOpen(true)} className="gap-2">
                <Plus className="w-4 h-4" />
                Create Your First Bot
              </Button>
            </div>
          ) : (
            <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-4">
              {bots.map((bot, index) => (
                <div
                  key={bot.id}
                  className="animate-slide-up"
                  style={{ animationDelay: `${index * 50}ms` }}
                >
                  <BotCard
                    bot={bot}
                    onToggleStatus={handleToggleBotStatus}
                    onDelete={handleDeleteBot}
                    onSelect={setSelectedBotId}
                    showDelete={true}
                  />
                </div>
              ))}
            </div>
          )}
        </div>
      ) : (
        <div className="space-y-6 animate-fade-in">
          <div className="flex items-center gap-2 mb-2">
            <Button variant="ghost" size="sm" onClick={() => setSelectedBotId(null)} className="text-muted-foreground hover:text-foreground">
              ← Back to Bots
            </Button>
          </div>
          {/* Stats Grid */}
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {stats.map((stat, index) => (
              <div
                key={stat.label}
                className="glass-card p-4 animate-slide-up"
                style={{ animationDelay: `${index * 50}ms` }}
              >
                <div className="flex items-center justify-between mb-2">
                  <stat.icon className="w-5 h-5 text-muted-foreground" />
                  <span className={cn(
                    "text-xs font-medium",
                    stat.changeType === "positive" ? "text-success" : "text-destructive"
                  )}>
                    {stat.change}
                  </span>
                </div>
                <p className="text-xl font-bold font-mono">{stat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              </div>
            ))}
          </div>

          {/* Charts Row */}
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <PerformanceChart />
            </div>
            <div>
              <StatusPanel account={selectedAccount} />
            </div>
          </div>

          {/* Second Row - Activity Log */}
          <div className="grid lg:grid-cols-3 gap-6">
            <div className="lg:col-span-2">
              <ActivePositions accountId={selectedAccount?.id} />
            </div>
            <div>
              <AIConsole botName={bots.find(b => b.id === selectedBotId)?.bot_version?.label || "Trading Bot"} />
            </div>
          </div>
        </div>
      )}

      {/* Add Bot Dialog */}
      {selectedAccount && (
        <AddBotDialog
          open={addBotOpen}
          onOpenChange={setAddBotOpen}
          accountId={selectedAccount.id}
          onBotAdded={refetch}
        />
      )}

      <AddAccountDialog
        open={addAccountOpen}
        onOpenChange={setAddAccountOpen}
        onAccountAdded={refetch}
      />
    </div>
  );
}
