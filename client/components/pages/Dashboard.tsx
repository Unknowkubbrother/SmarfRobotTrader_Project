import { useState, useEffect } from "react";
import { TrendingUp, TrendingDown, Activity, Wallet, Target, BarChart3, Plus, BellRing, DownloadCloud } from "lucide-react";
import { PerformanceChart } from "@/components/dashboard/PerformanceChart";
import { StatusPanel } from "@/components/dashboard/StatusPanel";
import { ActivePositions } from "@/components/dashboard/ActivePositions";
import { AIConsole } from "@/components/dashboard/AIConsole";
import { AccountSelector, AccountWithBots } from "@/components/dashboard/AccountSelector";
import { BotCard } from "@/components/dashboard/BotCard";
import TradingViewWidget from "@/components/dashboard/TradingViewWidget";
import { AddBotDialog } from "@/components/dialogs/AddBotDialog";
import { AddAccountDialog } from "@/components/dialogs/AddAccountDialog";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { useTradingAccounts } from "@/hooks/useTradingAccounts";
import { useBotLiveState } from "@/hooks/useBotLiveState";
import { BotSelector } from "@/components/dashboard/BotSelector";
import Link from "next/link";

export default function Dashboard() {
  const { accounts, loading, updateBotStatus, deleteBot, updateAccount, deleteAccount, refetch, getPendingUpdatesCount } = useTradingAccounts();
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
      } else {
        setSelectedAccount(accounts[0] || null);
        setSelectedBotId(null);
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

  const { getBotState } = useBotLiveState();
  const liveState = selectedBotId ? getBotState(selectedBotId) : undefined;
  const isLive = !!liveState?.connected;

  // Safe live state properties with fallbacks
  const accountBalance = selectedAccount?.balance ?? 0;
  const accountEquity = selectedAccount?.equity ?? accountBalance;
  const accountTodayPnl = selectedAccount?.total_today_pnl ?? 0;
  const liveEquity = liveState?.equity ?? accountEquity;
  const liveTotalPnl = liveState?.total_pnl ?? accountTodayPnl;
  const liveUnrealized = liveState?.unrealized_pnl ?? accountTodayPnl;
  const liveWins = liveState?.wins ?? 0;
  const liveTrades = liveState?.trades ?? 0;
  const livePosition = liveState?.position ?? 0;
  const liveLotSize = liveState?.lot_size ?? 0;
  const liveAction = liveState?.last_action || "—";
  const liveMarginLevel = liveState?.margin_level ?? selectedAccount?.margin_level ?? 0;

  // Live stats — use WS data when available, otherwise show static/DB data
  const stats = [
    {
      icon: Wallet,
      label: "Total Balance",
      value: `$${liveEquity.toFixed(2)}`,
      change: liveTotalPnl >= 0 ? `+${liveTotalPnl.toFixed(2)}` : liveTotalPnl.toFixed(2),
      changeType: liveTotalPnl >= 0 ? "positive" as const : "negative" as const,
    },
    {
      icon: TrendingUp,
      label: "Today's P&L",
      value: `$${liveUnrealized.toFixed(2)}`,
      change: (liveUnrealized >= 0 ? "+" : "") + liveUnrealized.toFixed(2),
      changeType: liveUnrealized >= 0 ? "positive" as const : "negative" as const,
    },
    {
      icon: Target,
      label: "Win Rate",
      value: isLive && liveTrades > 0
        ? `${((liveWins / liveTrades) * 100).toFixed(1)}%`
        : "—",
      change: isLive ? `${liveWins}W` : "—",
      changeType: "positive" as const,
    },
    {
      icon: TrendingDown,
      label: "Position",
      value: isLive
        ? (livePosition === 1 ? "LONG" : livePosition === -1 ? "SHORT" : "FLAT")
        : "—",
      change: isLive ? `Lot ${liveLotSize.toFixed(2)}` : "",
      changeType: (isLive && livePosition === 1) ? "positive" as const : (isLive && livePosition === -1) ? "negative" as const : "positive" as const,
    },
    {
      icon: BarChart3,
      label: "Total Trades",
      value: isLive ? `${liveTrades}` : "0",
      change: isLive ? `${liveWins} wins` : "",
      changeType: "positive" as const,
    },
    {
      icon: Activity,
      label: "Last Action",
      value: isLive ? liveAction : "—",
      change: isLive && liveState?.last_bar_time ? liveState.last_bar_time.slice(11, 19) : "",
      changeType: (isLive && (liveAction === "BUY")) ? "positive" as const : (isLive && liveAction === "SELL") ? "negative" as const : "positive" as const,
    },
  ];

  const bots = selectedAccount?.bot_configurations || [];
  const currentBot = bots.find(b => b.id === selectedBotId);
  const currentSymbol = currentBot?.bot_version?.symbol || "XAUUSD";
  const selectedAccountPendingUpdates = selectedAccount ? getPendingUpdatesCount(selectedAccount) : 0;
  const totalPendingUpdates = accounts.reduce((sum, account) => sum + getPendingUpdatesCount(account), 0);

  return (
    <div className="space-y-6">
      {/* Page Header */}
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Dashboard</h1>
          <p className="text-sm text-muted-foreground">
            Monitor your trading performance
            {totalPendingUpdates > 0 ? ` • ${totalPendingUpdates} bot update${totalPendingUpdates > 1 ? "s" : ""} available` : ""}
          </p>
        </div>

        {/* Selectors Row */}
        <div className="flex flex-col md:flex-row gap-4">
          <AccountSelector
            selectedAccount={selectedAccount}
            onAccountChange={setSelectedAccount}
            accounts={accounts}
            isLoading={loading}
            onRefresh={refetch}
            onUpdateAccount={updateAccount}
            onDeleteAccount={deleteAccount}
            onAccountDeleted={(accountId) => {
              if (selectedAccount?.id === accountId) {
                setSelectedAccount(null);
                setSelectedBotId(null);
              }
            }}
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

      {selectedAccountPendingUpdates > 0 && (
        <div className="rounded-xl border border-primary/30 bg-primary/5 px-4 py-3">
          <div className="flex flex-col gap-2 md:flex-row md:items-center md:justify-between">
            <div className="flex items-center gap-2 text-sm">
              <BellRing className="w-4 h-4 text-primary" />
              <span className="font-medium text-primary">
                {selectedAccountPendingUpdates} bot update{selectedAccountPendingUpdates > 1 ? "s" : ""} waiting in this account
              </span>
            </div>
            <Button asChild size="sm" className="gap-2">
              <Link href="/bot-control">
                <DownloadCloud className="w-4 h-4" />
                Open Bot Control
              </Link>
            </Button>
          </div>
        </div>
      )}

      {/* Main Content Area */}
      {/* Main Content Area */}
      {!selectedAccount ? (
        <div className="bg-white border rounded-xl shadow-sm p-12 text-center animate-slide-up flex flex-col items-center justify-center min-h-[400px]">
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
            <div className="bg-white border rounded-xl shadow-sm p-12 text-center flex flex-col items-center justify-center min-h-[300px]">
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
          <div className="flex items-center gap-2 mb-1">
            {isLive && (
              <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
                <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                Live
              </span>
            )}
          </div>
          <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-6 gap-4">
            {stats.map((stat, index) => (
              <div
                key={stat.label}
                className="bg-white border rounded-xl shadow-sm p-4 animate-slide-up hover:shadow-md transition-shadow"
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
                <p className="text-xl font-bold font-mono text-foreground">{stat.value}</p>
                <p className="text-xs text-muted-foreground mt-1">{stat.label}</p>
              </div>
            ))}
          </div>

          {/* Main Grid: Chart & Status */}
          <div className="grid lg:grid-cols-3 gap-6">
            {/* TradingView Chart (2/3) */}
            <div className="lg:col-span-2 bg-white border rounded-xl shadow-sm p-1 overflow-hidden animate-slide-up h-[500px]">
              <TradingViewWidget symbol={currentSymbol} theme="light" />
            </div>

            {/* Account Status (1/3) */}
            <div className="lg:col-span-1">
              <StatusPanel account={selectedAccount} liveState={liveState} />
            </div>
          </div>

          {/* Performance & Activity Grid */}
          <div className="grid lg:grid-cols-3 gap-6 items-stretch lg:h-[430px]">
            {/* Portfolio Performance (2/3) */}
            <div className="lg:col-span-2 h-full min-h-0">
              <PerformanceChart liveState={liveState} botId={selectedBotId || undefined} />
            </div>
            {/* Activity Log (1/3) */}
            <div className="lg:col-span-1 h-full min-h-0">
              <AIConsole
                botName={bots.find(b => b.id === selectedBotId)?.bot_version?.label || "Trading Bot"}
                liveState={liveState}
              />
            </div>
          </div>

          {/* Active Positions (Full width at bottom) */}
          <div className="bg-white border rounded-xl shadow-sm overflow-hidden">
            <ActivePositions accountId={selectedAccount?.id} liveState={liveState} />
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
