import { useState, useEffect } from "react";
import { Power, AlertOctagon, Clock, Shield, Play, Activity, TrendingUp, RefreshCw, Sparkles, Plus, Trash2 } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { AccountSelector, AccountWithBots } from "@/components/dashboard/AccountSelector";
import { BotSelector } from "@/components/dashboard/BotSelector";
import { AddBotDialog } from "@/components/dialogs/AddBotDialog";
import { useTradingAccounts, BotConfigWithVersion, BotVersion } from "@/hooks/useTradingAccounts";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";

const riskLevels = [
  { id: "low", label: "Low", description: "Conservative trading with minimal risk", color: "text-success" },
  { id: "medium", label: "Medium", description: "Balanced approach with moderate risk", color: "text-warning" },
  { id: "high", label: "High", description: "Aggressive trading strategy", color: "text-destructive" },
];

const defaultSchedule = [
  { day: "Mon", enabled: true },
  { day: "Tue", enabled: true },
  { day: "Wed", enabled: true },
  { day: "Thu", enabled: true },
  { day: "Fri", enabled: true },
];

export default function BotControl() {
  const {
    accounts,
    loading,
    updateBotStatus,
    updateBotRisk,
    updateBotSchedule,
    changeModel,
    deleteBot,
    refetch,
    getBotVersions
  } = useTradingAccounts();

  const [selectedAccount, setSelectedAccount] = useState<AccountWithBots | null>(null);
  const [selectedBot, setSelectedBot] = useState<BotConfigWithVersion | null>(null);
  const [selectedRisk, setSelectedRisk] = useState("medium");
  const [schedule, setSchedule] = useState(defaultSchedule);
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [addBotOpen, setAddBotOpen] = useState(false);
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [availableModels, setAvailableModels] = useState<BotVersion[]>([]);

  // Load available bot versions on mount
  useEffect(() => {
    getBotVersions().then(setAvailableModels);
  }, []);

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
        // Update selected bot if it exists
        if (selectedBot) {
          const updatedBot = updated.bot_configurations.find(b => b.id === selectedBot.id);
          if (updatedBot) {
            setSelectedBot(updatedBot);
          } else {
            // Bot was deleted, select first available
            setSelectedBot(updated.bot_configurations[0] || null);
          }
        }
      }
    }
  }, [accounts]);

  // Auto-select first bot when account changes
  useEffect(() => {
    if (selectedAccount && selectedAccount.bot_configurations.length > 0) {
      if (!selectedBot || !selectedAccount.bot_configurations.find(b => b.id === selectedBot.id)) {
        setSelectedBot(selectedAccount.bot_configurations[0]);
      }
    } else {
      setSelectedBot(null);
    }
  }, [selectedAccount]);

  // Load bot settings when selected bot changes
  useEffect(() => {
    if (selectedBot) {
      setSelectedRisk(selectedBot.risk_level || "medium");
      const savedSchedule = selectedBot.trading_schedule as Record<string, boolean> | null;
      if (savedSchedule) {
        setSchedule(defaultSchedule.map(s => ({
          ...s,
          enabled: savedSchedule[s.day.toLowerCase()] ?? s.enabled
        })));
      }
    }
  }, [selectedBot]);

  const handleToggleBot = async () => {
    if (!selectedBot) return;
    const currentStatus = selectedBot.status || "stopped";
    const newStatus = currentStatus === "running" ? "stopped" : "running";
    const success = await updateBotStatus(selectedBot.id, newStatus);
    if (success) {
      toast.success(newStatus === "running" ? "Bot started successfully" : "Bot stopped successfully");
    }
  };

  const handlePanicButton = async () => {
    if (!selectedBot) return;
    await updateBotStatus(selectedBot.id, "stopped");
    toast.error("Emergency: Bot stopped and closing all positions...");
  };

  const toggleDay = async (day: string) => {
    if (!selectedBot) return;
    const newSchedule = schedule.map(s => s.day === day ? { ...s, enabled: !s.enabled } : s);
    setSchedule(newSchedule);
    const scheduleObj = Object.fromEntries(newSchedule.map(s => [s.day.toLowerCase(), s.enabled]));
    await updateBotSchedule(selectedBot.id, scheduleObj);
  };

  const handleRiskChange = async (riskId: string) => {
    if (!selectedBot) return;
    setSelectedRisk(riskId);
    await updateBotRisk(selectedBot.id, riskId);
  };

  const handleChangeModel = async (modelId: string) => {
    if (!selectedBot) return;
    const success = await changeModel(selectedBot.id, modelId);
    if (success) {
      setShowModelDialog(false);
    }
  };

  const handleDeleteBot = async () => {
    if (!selectedBot) return;
    const success = await deleteBot(selectedBot.id);
    if (success) {
      setDeleteConfirmOpen(false);
    }
  };

  const botStatus = selectedBot?.status || "stopped";
  const botSymbol = selectedBot?.bot_version?.symbol || "N/A";
  const botLabel = selectedBot?.bot_version?.label || "No Bot Selected";
  const bots = selectedAccount?.bot_configurations || [];

  const getStatusBadge = (status: string | null) => {
    switch (status) {
      case "running":
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-success/10 text-success"><span className="w-1.5 h-1.5 rounded-full bg-success" />Running</span>;
      case "paused":
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">Paused</span>;
      default:
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-muted text-muted-foreground"><span className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />Stopped</span>;
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4">
        <div>
          <h1 className="text-xl font-semibold text-foreground">Bot Control</h1>
          <p className="text-sm text-muted-foreground">Configure your trading bot settings</p>
        </div>

        {/* Account & Bot Selectors */}
        <div className="flex flex-wrap items-center gap-3">
          <AccountSelector
            selectedAccount={selectedAccount}
            onAccountChange={setSelectedAccount}
            accounts={accounts}
            isLoading={loading}
            onRefresh={refetch}
          />
          {selectedAccount && (
            <BotSelector
              bots={bots}
              selectedBotId={selectedBot?.id || null}
              onBotSelect={(botId) => {
                const bot = bots.find(b => b.id === botId);
                if (bot) setSelectedBot(bot);
              }}
              accountId={selectedAccount.id}
              onBotAdded={refetch}
            />
          )}
        </div>
      </div>

      {selectedAccount && (
        <>
          {/* Bot Status Card */}
          <div className="glass-card p-5 animate-slide-up">
            <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4">
              <div className="flex items-center gap-4">
                <div className="w-12 h-12 rounded-xl bg-primary/10 flex items-center justify-center">
                  <Activity className="w-6 h-6 text-primary" />
                </div>
                <div>
                  <div className="flex items-center gap-3">
                    <h3 className="font-semibold">{selectedBot ? botLabel : "No Bot Selected"}</h3>
                    {selectedBot && getStatusBadge(botStatus)}
                  </div>
                  {selectedBot && (
                    <p className="text-sm text-muted-foreground">Trading <span className="font-mono font-medium">{botSymbol}</span></p>
                  )}
                </div>
              </div>
              <div className="flex items-center gap-3 flex-wrap">
                {selectedBot && (
                  <>
                    <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary">
                      <TrendingUp className="w-4 h-4 text-success" />
                      <span className={cn("font-mono font-medium text-success")}>
                        +${selectedBot.today_pnl.toFixed(2)}
                      </span>
                    </div>
                    <Button
                      variant="outline"
                      className="gap-2 bg-warning/10 text-warning border-warning/30 hover:bg-warning/20"
                      onClick={() => setShowModelDialog(true)}
                    >
                      <RefreshCw className="w-4 h-4" />
                      Change Model
                    </Button>
                    <Button variant="outline" className="gap-2" onClick={handleToggleBot}>
                      {botStatus === "running" ? <><Power className="w-4 h-4" />Stop</> : <><Play className="w-4 h-4" />Start</>}
                    </Button>
                    <Button variant="destructive" onClick={handlePanicButton} className="gap-2">
                      <AlertOctagon className="w-4 h-4" />
                      Emergency Stop
                    </Button>
                    <Button
                      variant="ghost"
                      className="gap-2 text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={() => setDeleteConfirmOpen(true)}
                    >
                      <Trash2 className="w-4 h-4" />
                      Delete Bot
                    </Button>
                  </>
                )}
                {!selectedBot && (
                  <Button className="gap-2 bg-warning text-warning-foreground hover:bg-warning/90" onClick={() => setAddBotOpen(true)}>
                    <Sparkles className="w-4 h-4" />
                    Add First Bot
                  </Button>
                )}
              </div>
            </div>
          </div>

          {/* Risk Level & Schedule */}
          {selectedBot && (
            <div className="grid lg:grid-cols-2 gap-6">
              {/* Risk Level */}
              <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "50ms" }}>
                <div className="flex items-center gap-2 mb-6">
                  <Shield className="w-5 h-5 text-primary" />
                  <h3 className="font-semibold">Risk Level</h3>
                </div>
                <div className="space-y-3">
                  {riskLevels.map((level) => (
                    <button
                      key={level.id}
                      onClick={() => handleRiskChange(level.id)}
                      className={cn(
                        "w-full p-4 rounded-xl border text-left transition-all",
                        selectedRisk === level.id
                          ? "border-primary bg-primary/5"
                          : "border-border hover:border-primary/50 hover:bg-secondary/50"
                      )}
                    >
                      <div className="flex items-center justify-between mb-1">
                        <span className={cn("font-semibold", level.color)}>{level.label}</span>
                        {selectedRisk === level.id && (
                          <span className="text-xs bg-primary text-primary-foreground px-2 py-0.5 rounded-full">Active</span>
                        )}
                      </div>
                      <p className="text-sm text-muted-foreground">{level.description}</p>
                    </button>
                  ))}
                </div>
              </div>

              {/* Trading Schedule */}
              <div className="glass-card p-6 animate-slide-up" style={{ animationDelay: "100ms" }}>
                <div className="flex items-center gap-2 mb-6">
                  <Clock className="w-5 h-5 text-primary" />
                  <h3 className="font-semibold">Trading Schedule</h3>
                </div>
                <div className="grid grid-cols-5 gap-2 mb-6">
                  {schedule.map((day) => (
                    <button
                      key={day.day}
                      onClick={() => toggleDay(day.day)}
                      className={cn(
                        "py-3 rounded-xl text-sm font-medium transition-all",
                        day.enabled
                          ? "bg-primary text-primary-foreground"
                          : "bg-secondary text-muted-foreground hover:text-foreground"
                      )}
                    >
                      {day.day}
                    </button>
                  ))}
                </div>
                <div className="p-4 rounded-xl bg-secondary/50 space-y-3">
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Trading Hours</span>
                    <span className="text-sm font-medium font-mono">00:00 - 23:59 UTC</span>
                  </div>
                  <div className="flex items-center justify-between">
                    <span className="text-sm text-muted-foreground">Symbol</span>
                    <span className="text-sm font-medium font-mono">{botSymbol}</span>
                  </div>
                </div>
              </div>
            </div>
          )}
        </>
      )}

      {/* Model Selection Dialog */}
      <Dialog open={showModelDialog} onOpenChange={setShowModelDialog}>
        <DialogContent className="max-w-4xl">
          <DialogHeader>
            <DialogTitle className="text-2xl">MODEL Robot Trading</DialogTitle>
            <p className="text-muted-foreground">Select your automated trading companion</p>
          </DialogHeader>
          <div className="grid md:grid-cols-3 gap-4 mt-4 max-h-[60vh] overflow-y-auto">
            {availableModels.map((model) => (
              <div
                key={model.model_id}
                className={cn(
                  "rounded-2xl border border-border overflow-hidden hover:border-primary transition-colors cursor-pointer",
                  selectedBot?.model_id === model.model_id && "ring-2 ring-primary border-primary"
                )}
                onClick={() => handleChangeModel(model.model_id)}
              >
                <div className="bg-gradient-to-r from-[#0f3460] to-[#16537e] p-4 text-white">
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <Activity className="w-5 h-5" />
                      <span className="font-semibold truncate">{model.label}</span>
                    </div>
                    <div className="flex items-center gap-1">
                      <span className="text-xs">{model.timeframe}</span>
                      <span className="px-2 py-0.5 rounded-full bg-white/20 text-xs">{model.version_tag}</span>
                    </div>
                  </div>
                </div>
                <div className="p-4 bg-card space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground flex items-center gap-2">
                      <Clock className="w-4 h-4" /> Latest update
                    </span>
                    <span>30 Dec 2568</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Symbol</span>
                    <span className="font-mono">{model.symbol}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-muted-foreground">Timeframe</span>
                    <span>{model.timeframe}</span>
                  </div>
                  <div className="bg-primary/10 rounded-lg p-3 mt-3">
                    <p className="text-xs font-medium text-primary mb-2">Release Notes</p>
                    <ul className="space-y-1">
                      {model.release_notes.map((note, i) => (
                        <li key={i} className="text-xs text-muted-foreground flex items-start gap-2">
                          <span className="w-1 h-1 rounded-full bg-primary mt-1.5 shrink-0" />
                          {note}
                        </li>
                      ))}
                    </ul>
                  </div>
                  <Button
                    className="w-full gap-2"
                    onClick={(e) => {
                      e.stopPropagation();
                      handleChangeModel(model.model_id);
                    }}
                    variant={selectedBot?.model_id === model.model_id ? "secondary" : "default"}
                    disabled={selectedBot?.model_id === model.model_id}
                  >
                    <Activity className="w-4 h-4" />
                    {selectedBot?.model_id === model.model_id ? "Current Model" : "Select Model"}
                  </Button>
                </div>
              </div>
            ))}
            {availableModels.length === 0 && (
              <div className="col-span-3 text-center py-10 text-muted-foreground">
                No trading models available.
              </div>
            )}
          </div>
        </DialogContent>
      </Dialog>

      {/* Add Bot Dialog */}
      {selectedAccount && (
        <AddBotDialog
          open={addBotOpen}
          onOpenChange={setAddBotOpen}
          accountId={selectedAccount.id}
          onBotAdded={refetch}
        />
      )}

      {/* Delete Confirmation */}
      <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete Bot</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to delete "{botLabel}"? This action cannot be undone.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleDeleteBot} className="bg-destructive text-destructive-foreground hover:bg-destructive/90">
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
