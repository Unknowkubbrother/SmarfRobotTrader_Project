import { useState, useEffect, useRef } from "react";
import { Power, AlertOctagon, Clock, Shield, Play, Activity, TrendingUp, RefreshCw, Sparkles, Plus, Trash2, AlertTriangle, DownloadCloud } from "lucide-react";
import { Button } from "@/components/ui/button";
import { cn } from "@/lib/utils";
import { toast } from "sonner";
import { AccountSelector, AccountWithBots } from "@/components/dashboard/AccountSelector";
import { BotSelector } from "@/components/dashboard/BotSelector";
import { AddBotDialog } from "@/components/dialogs/AddBotDialog";
import {
  useTradingAccounts,
  BotConfigWithVersion,
  BotOperationLogEntry,
  BotRuntimeHealth,
  BotVersion,
} from "@/hooks/useTradingAccounts";
import { useBotLiveState, type BotLifecycleEvent } from "@/hooks/useBotLiveState";
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
import {
  BOT_UI_STORAGE_KEY,
  readBotUiState,
  saveActiveBotAction,
  saveBotUiState,
  type PersistedBotAction,
  type PersistedBotLog,
} from "@/lib/botOperationStore";

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
  { day: "Sat", enabled: false },
  { day: "Sun", enabled: false },
];

const DAY_ALIAS_TO_KEY: Record<string, string> = {
  mon: "mon",
  monday: "mon",
  tue: "tue",
  tues: "tue",
  tuesday: "tue",
  wed: "wed",
  weds: "wed",
  wednesday: "wed",
  thu: "thu",
  thur: "thu",
  thurs: "thu",
  thursday: "thu",
  fri: "fri",
  friday: "fri",
  sat: "sat",
  saturday: "sat",
  sun: "sun",
  sunday: "sun",
};

const normalizeTradingSchedule = (value: unknown): Record<string, boolean> => {
  const normalized: Record<string, boolean> = {
    mon: true,
    tue: true,
    wed: true,
    thu: true,
    fri: true,
    sat: false,
    sun: false,
  };

  let payload: Record<string, unknown> = {};
  if (typeof value === "string") {
    try {
      const parsed = JSON.parse(value);
      if (parsed && typeof parsed === "object") {
        payload = parsed as Record<string, unknown>;
      }
    } catch {
      payload = {};
    }
  } else if (value && typeof value === "object") {
    payload = value as Record<string, unknown>;
  }

  for (const [rawKey, rawValue] of Object.entries(payload)) {
    const dayKey = DAY_ALIAS_TO_KEY[String(rawKey).trim().toLowerCase()];
    if (!dayKey) continue;
    normalized[dayKey] = Boolean(rawValue);
  }

  return normalized;
};

const mapLifecycleActionTitle = (action: string): string => {
  switch (String(action || "").toLowerCase()) {
    case "start":
    case "restart":
      return "Starting Bot";
    case "stop":
      return "Stopping Bot";
    case "emergency_stop":
      return "Emergency Stop";
    case "change_model":
      return "Changing Model";
    case "apply_update":
      return "Applying Update";
    case "delete":
      return "Deleting Bot";
    default:
      return "Bot Operation";
  }
};

export default function BotControl() {
  const {
    accounts,
    loading,
    updateBotStatus,
    updateBotRisk,
    updateBotSchedule,
    changeModel,
    emergencyStopBot,
    applyBotUpdate,
    deleteBot,
    getBotRuntimeHealth,
    getBotOperationLogs,
    updateAccount,
    deleteAccount,
    refetch,
    getBotVersions
  } = useTradingAccounts();

  const [selectedAccount, setSelectedAccount] = useState<AccountWithBots | null>(null);
  const [selectedBot, setSelectedBot] = useState<BotConfigWithVersion | null>(null);
  const [selectedRisk, setSelectedRisk] = useState("medium");
  const [schedule, setSchedule] = useState(defaultSchedule);
  const [showModelDialog, setShowModelDialog] = useState(false);
  const [addBotOpen, setAddBotOpen] = useState(false);

  // Confirmation States
  const [deleteConfirmOpen, setDeleteConfirmOpen] = useState(false);
  const [stopConfirmOpen, setStopConfirmOpen] = useState(false);
  const [panicConfirmOpen, setPanicConfirmOpen] = useState(false);
  const [riskConfirmOpen, setRiskConfirmOpen] = useState(false);
  const [pendingRiskId, setPendingRiskId] = useState<string | null>(null);
  const [modelConfirmOpen, setModelConfirmOpen] = useState(false);
  const [pendingModelId, setPendingModelId] = useState<string | null>(null);
  const [scheduleConfirmOpen, setScheduleConfirmOpen] = useState(false);
  const [pendingDay, setPendingDay] = useState<string | null>(null);
  const [updateConfirmOpen, setUpdateConfirmOpen] = useState(false);
  const [activeAction, setActiveAction] = useState<PersistedBotAction | null>(null);
  const [actionLogs, setActionLogs] = useState<PersistedBotLog[]>([]);
  const [runtimeHealth, setRuntimeHealth] = useState<BotRuntimeHealth | null>(null);
  const [storeReady, setStoreReady] = useState(false);
  const healthLogKeysRef = useRef<Set<string>>(new Set());
  const processedLifecycleIdsRef = useRef<Set<string>>(new Set());
  const lastLifecycleRefetchAtRef = useRef<number>(0);
  const lastLogsRefreshAtRef = useRef<number>(0);
  const applyUpdateInFlightRef = useRef(false);

  const [availableModels, setAvailableModels] = useState<BotVersion[]>([]);
  const activeActionForSelectedBot =
    activeAction && selectedBot && activeAction.botId === selectedBot.id
      ? activeAction
      : null;
  const selectedBotStatusNormalized = String(
    selectedBot?.status || selectedBot?.container_status || ""
  ).toLowerCase();
  const isStatusStarting = selectedBotStatusNormalized === "starting";
  const isBusy = Boolean(activeActionForSelectedBot) || isStatusStarting;
  const isApplyingUpdateAction = activeActionForSelectedBot?.kind === "apply_update";

  const { getBotState, lifecycleEvents } = useBotLiveState();
  const liveState = selectedBot ? getBotState(selectedBot.id) : undefined;

  // Safe live state properties with fallbacks
  const botTodayPnl = selectedBot?.today_pnl ?? 0;
  const liveEquity = liveState?.equity ?? selectedAccount?.equity ?? selectedAccount?.balance ?? 0;
  const liveTotalPnl = liveState?.total_pnl ?? botTodayPnl;
  const liveUnrealized = liveState?.unrealized_pnl ?? botTodayPnl;
  const liveWins = liveState?.wins ?? 0;
  const liveTrades = liveState?.trades ?? 0;
  const livePosition = liveState?.position ?? 0;
  const liveLotSize = liveState?.lot_size ?? 0;
  const liveAction = liveState?.last_action || "—";
  const liveScheduleSummary = (() => {
    const src = normalizeTradingSchedule(liveState?.trading_schedule ?? selectedBot?.trading_schedule);
    const days = [
      ["mon", "Mon"],
      ["tue", "Tue"],
      ["wed", "Wed"],
      ["thu", "Thu"],
      ["fri", "Fri"],
      ["sat", "Sat"],
      ["sun", "Sun"],
    ] as const;
    const enabled = days.filter(([k]) => src[k]).map(([, l]) => l);
    return enabled.length > 0 ? enabled.join(" ") : "None";
  })();

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
      } else {
        setSelectedAccount(accounts[0] || null);
        setSelectedBot(null);
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
      const savedSchedule = normalizeTradingSchedule(selectedBot.trading_schedule);
      setSchedule(defaultSchedule.map((s) => ({
        ...s,
        enabled: savedSchedule[s.day.toLowerCase()] ?? s.enabled,
      })));
    }
  }, [selectedBot]);

  useEffect(() => {
    setRuntimeHealth(null);
  }, [selectedBot?.id]);

  useEffect(() => {
    const botId = selectedBot?.id;
    if (!botId) {
      setActionLogs([]);
      return;
    }
    void refreshSelectedBotLogs(botId, true);
  }, [selectedBot?.id]);

  const appendActionLog = (
    level: PersistedBotLog["level"],
    message: string,
    botId?: string
  ) => {
    const now = new Date();
    const at = now.toLocaleTimeString([], { hour12: false });
    const id = `${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`;
    setActionLogs((prev) => [{ id, level, message, at, botId }, ...prev].slice(0, 20));
  };

  const mapServerLogToUi = (entry: BotOperationLogEntry): PersistedBotLog => {
    const normalizedLevel = String(entry.level || "").toLowerCase();
    const level: PersistedBotLog["level"] =
      normalizedLevel === "success" ? "success" : normalizedLevel === "error" ? "error" : "info";
    const createdDate = entry.created_at ? new Date(entry.created_at) : new Date();
    const at = createdDate.toLocaleTimeString([], { hour12: false });
    const sourceLabel =
      entry.source === "admin" ? "Admin" : entry.source === "user" ? "User" : "System";
    const fallbackMessage = `${entry.action || "operation"} ${entry.phase || "updated"}`.trim();
    return {
      id: `srv-${entry.id}`,
      level,
      message: `${sourceLabel}: ${entry.message || fallbackMessage}`,
      at,
      botId: entry.bot_config_id,
      action: String(entry.action || "").trim().toLowerCase() || undefined,
      phase: String(entry.phase || "").trim().toLowerCase() || undefined,
      ts: createdDate.getTime(),
    };
  };

  const refreshSelectedBotLogs = async (botId: string, force = false) => {
    if (!botId) return;
    const now = Date.now();
    if (!force && now - lastLogsRefreshAtRef.current < 1000) {
      return;
    }
    lastLogsRefreshAtRef.current = now;
    const logs = await getBotOperationLogs(botId, 60);
    if (logs.length === 0) return;
    setActionLogs(logs.map(mapServerLogToUi));
  };

  const startUiAction = (
    title: string,
    detail: string,
    startLog: string,
    meta: Partial<PersistedBotAction> = {}
  ) => {
    const nextAction: PersistedBotAction = {
      title,
      detail,
      startedAt: Date.now(),
      ...meta,
    };
    setActiveAction(nextAction);
    saveActiveBotAction(nextAction);
    appendActionLog("info", startLog, meta.botId);
  };

  const finishUiAction = () => {
    setActiveAction(null);
    saveActiveBotAction(null);
  };

  useEffect(() => {
    const latest = readBotUiState();
    setActiveAction(latest.activeAction);
    setStoreReady(true);
  }, []);

  useEffect(() => {
    if (!storeReady) return;
    saveBotUiState({
      activeAction,
      logs: [],
    });
  }, [activeAction, storeReady]);

  useEffect(() => {
    const onStorage = (event: StorageEvent) => {
      if (event.key !== BOT_UI_STORAGE_KEY) return;
      const latest = readBotUiState();
      setActiveAction(latest.activeAction);
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    if (lifecycleEvents.length === 0) return;

    const ownedBotIds = new Set(
      accounts.flatMap((account) => (account.bot_configurations || []).map((bot) => bot.id))
    );
    const selectedBotId = selectedBot?.id || null;
    const pendingSelectedEvents: BotLifecycleEvent[] = [];
    let shouldRefetch = false;
    let shouldRefreshSelectedLogs = false;

    const orderedEvents = [...lifecycleEvents].reverse();
    for (const event of orderedEvents) {
      if (processedLifecycleIdsRef.current.has(event.id)) {
        continue;
      }
      processedLifecycleIdsRef.current.add(event.id);

      if (!ownedBotIds.has(event.bot_config_id)) {
        continue;
      }

      shouldRefetch = true;
      if (selectedBotId && event.bot_config_id === selectedBotId) {
        pendingSelectedEvents.push(event);
        shouldRefreshSelectedLogs = true;
      }
    }

    for (const event of pendingSelectedEvents) {
      const sourceLabel =
        event.source === "admin" ? "Admin" : event.source === "user" ? "User" : "System";
      const actionTitle = mapLifecycleActionTitle(event.action);
      const detailMessage =
        String(event.detail || "").trim() ||
        (event.phase === "requested"
          ? `${actionTitle} requested`
          : event.phase === "failed"
            ? `${actionTitle} failed`
            : `${actionTitle} completed`);
      const prefixedLog = `${sourceLabel}: ${detailMessage}`;

      if (event.phase === "requested") {
        appendActionLog("info", prefixedLog, event.bot_config_id);
      } else if (event.phase === "succeeded") {
        appendActionLog("success", prefixedLog, event.bot_config_id);
        if (activeAction?.botId === event.bot_config_id) {
          finishUiAction();
        }
      } else if (event.phase === "failed") {
        appendActionLog("error", prefixedLog, event.bot_config_id);
        if (activeAction?.botId === event.bot_config_id) {
          finishUiAction();
        }
      } else {
        appendActionLog("info", prefixedLog, event.bot_config_id);
      }
    }

    if (processedLifecycleIdsRef.current.size > 1200) {
      processedLifecycleIdsRef.current.clear();
      for (const item of lifecycleEvents.slice(0, 400)) {
        processedLifecycleIdsRef.current.add(item.id);
      }
    }

    if (shouldRefetch) {
      const now = Date.now();
      if (now - lastLifecycleRefetchAtRef.current > 1200) {
        lastLifecycleRefetchAtRef.current = now;
        void refetch();
      }
    }
    if (selectedBotId && shouldRefreshSelectedLogs) {
      void refreshSelectedBotLogs(selectedBotId);
    }
  }, [accounts, lifecycleEvents, selectedBot?.id, activeAction, refetch]);

  useEffect(() => {
    if (!activeAction?.botId || !activeAction?.expectedStatus) return;
    const shouldSyncByStatus =
      activeAction.kind === "starting" ||
      activeAction.kind === "stopping" ||
      activeAction.kind === "deleting" ||
      activeAction.kind === "emergency";
    if (!shouldSyncByStatus) return;

    const allBots = accounts.flatMap((account) => account.bot_configurations || []);
    const targetBot = allBots.find((bot) => bot.id === activeAction.botId);
    const currentStatus = String(targetBot?.status || targetBot?.container_status || "").toLowerCase();
    let isDone = false;
    if (activeAction.expectedStatus === "deleted") {
      isDone = !targetBot;
    } else if (targetBot) {
      isDone = currentStatus === activeAction.expectedStatus;
    }

    if (isDone) {
      const completedAction = activeAction;
      appendActionLog("success", `${completedAction.title} synchronized`, completedAction.botId);
      finishUiAction();

      if (completedAction.kind === "starting" && completedAction.botId) {
        const actionKey = `${completedAction.botId}:${completedAction.startedAt}:post_start_health`;
        if (!healthLogKeysRef.current.has(actionKey)) {
          healthLogKeysRef.current.add(actionKey);
          void (async () => {
            let health: BotRuntimeHealth | null = null;
            for (let attempt = 1; attempt <= 6; attempt += 1) {
              health = await getBotRuntimeHealth(completedAction.botId as string);
              if (health && health.trade_allowed !== null) {
                break;
              }
              if (attempt < 6) {
                await new Promise((resolve) => window.setTimeout(resolve, 2000));
              }
            }
            if (!health) {
              appendActionLog("info", "Bot started. Runtime health check is pending...", completedAction.botId);
              return;
            }
            if (selectedBot?.id === health.bot_config_id) {
              setRuntimeHealth(health);
            }
            if (health.trade_allowed === true) {
              appendActionLog("success", "MT5 algorithmic trading is enabled", completedAction.botId);
            } else if (health.trade_allowed === false) {
              appendActionLog("error", "MT5 algorithmic trading is disabled", completedAction.botId);
            } else {
              appendActionLog(
                "info",
                `MT5 runtime health: ${health.health_detail || "waiting for data"}`,
                completedAction.botId,
              );
            }
          })();
        }
      }
    }
  }, [accounts, activeAction, selectedBot?.id]);

  useEffect(() => {
    if (!activeAction) return;
    const timer = window.setInterval(() => {
      void refetch();
    }, 4000);
    return () => window.clearInterval(timer);
  }, [activeAction, refetch]);

  useEffect(() => {
    if (!selectedBot) return;
    const normalizedStatus = String(selectedBot.status || selectedBot.container_status || "").toLowerCase();
    if (normalizedStatus !== "starting") return;
    const timer = window.setInterval(() => {
      void refetch();
      void refreshSelectedBotLogs(selectedBot.id);
    }, 3000);
    return () => window.clearInterval(timer);
  }, [selectedBot?.id, selectedBot?.status, selectedBot?.container_status, refetch]);

  useEffect(() => {
    if (!selectedBot) return;
    const selectedBotId = selectedBot.id;
    const normalizedStatus = String(selectedBot.status || selectedBot.container_status || "").toLowerCase();
    const shouldPoll = normalizedStatus === "running" || activeAction?.botId === selectedBotId;
    if (!shouldPoll) return;

    let cancelled = false;
    const pollRuntimeHealth = async () => {
      const health = await getBotRuntimeHealth(selectedBotId);
      if (!cancelled && health) {
        setRuntimeHealth(health);
      }
    };

    void pollRuntimeHealth();
    const timer = window.setInterval(() => {
      void pollRuntimeHealth();
    }, 5000);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [selectedBot, activeAction?.botId]);

  useEffect(() => {
    if (!activeAction?.startedAt) return;
    const ttlMs = 15 * 60 * 1000;
    const ageMs = Date.now() - Number(activeAction.startedAt || 0);
    if (ageMs >= ttlMs) {
      appendActionLog("error", `${activeAction.title} timeout. Please check latest bot status.`, activeAction.botId);
      setActiveAction(null);
      return;
    }
    const timer = window.setTimeout(() => {
      appendActionLog("error", `${activeAction.title} timeout. Please check latest bot status.`, activeAction.botId);
      setActiveAction(null);
    }, ttlMs - ageMs);
    return () => window.clearTimeout(timer);
  }, [activeAction]);

  const handleToggleBotClick = () => {
    if (isBusy) return;
    if (!selectedBot) return;
    const currentStatus = selectedBot.status || "stopped";
    if (currentStatus === "running") {
      setStopConfirmOpen(true);
    } else {
      performBotToggle("running");
    }
  };

  const performBotToggle = async (newStatus: "running" | "stopped") => {
    if (isBusy) return;
    if (!selectedBot) return;
    const actionTitle = newStatus === "running" ? "Starting Bot" : "Stopping Bot";
    startUiAction(
      actionTitle,
      "Waiting for Docker and MT5 runtime to finish operation...",
      `${actionTitle} for ${selectedBot.bot_version?.label || "bot"}`,
      {
        botId: selectedBot.id,
        expectedStatus: newStatus,
        kind: newStatus === "running" ? "starting" : "stopping",
      }
    );
    const result = await updateBotStatus(selectedBot.id, newStatus);
    if (result.success) {
      appendActionLog("success", `${actionTitle} completed`, selectedBot.id);
      toast.success(newStatus === "running" ? "Bot started successfully" : "Bot stopped successfully");
      finishUiAction();
    } else if (result.pending) {
      appendActionLog("info", `${actionTitle} request accepted. Waiting for backend sync...`, selectedBot.id);
      toast.info("Operation still in progress. Tracking status from backend...");
    } else {
      appendActionLog("error", `${actionTitle} failed`, selectedBot.id);
      finishUiAction();
    }
    setStopConfirmOpen(false);
  };

  const handlePanicClick = () => {
    if (isBusy) return;
    setPanicConfirmOpen(true);
  };

  const performPanicStop = async () => {
    if (isBusy) return;
    if (!selectedBot) return;
    startUiAction(
      "Emergency Stop",
      "Sending close-all command to bot, then stopping Docker container...",
      `Emergency stop requested for ${selectedBot.bot_version?.label || "bot"}`,
      {
        botId: selectedBot.id,
        expectedStatus: "stopped",
        kind: "emergency",
      }
    );
    const result = await emergencyStopBot(selectedBot.id);
    if (result.success) {
      const warning = result.data?.warning;
      if (warning) {
        appendActionLog("error", `Emergency stop completed with warning: ${warning}`, selectedBot.id);
        toast.warning(`Emergency stop completed. ${warning}`);
      } else {
        appendActionLog("success", "Emergency stop completed successfully", selectedBot.id);
        toast.success("Emergency stop completed. Bot stopped and close-position command finished.");
      }
    } else {
      appendActionLog("error", `Emergency stop failed: ${result.error || "unknown error"}`, selectedBot.id);
      toast.error(result.error || "Emergency stop failed");
    }
    setPanicConfirmOpen(false);
    finishUiAction();
  };

  const handleDayClick = (day: string) => {
    if (isBusy) return;
    if (!selectedBot) return;
    setPendingDay(day);
    setScheduleConfirmOpen(true);
  };

  const performScheduleToggle = async () => {
    if (isBusy) return;
    if (!selectedBot || !pendingDay) return;

    startUiAction(
      "Updating Schedule",
      "Sending schedule update to live bot runtime...",
      `Updating trading schedule (${pendingDay})`,
      {
        botId: selectedBot.id,
        kind: "update_schedule",
      }
    );
    const newSchedule = schedule.map(s => s.day === pendingDay ? { ...s, enabled: !s.enabled } : s);
    setSchedule(newSchedule);

    const scheduleObj = Object.fromEntries(
      newSchedule.map((s) => [DAY_ALIAS_TO_KEY[s.day.toLowerCase()] || s.day.toLowerCase(), s.enabled]),
    );
    const success = await updateBotSchedule(selectedBot.id, scheduleObj);
    if (success) {
      appendActionLog("success", "Trading schedule updated", selectedBot.id);
    } else {
      appendActionLog("error", "Trading schedule update failed", selectedBot.id);
    }

    setScheduleConfirmOpen(false);
    setPendingDay(null);
    finishUiAction();
  };

  const handleRiskClick = (riskId: string) => {
    if (isBusy) return;
    if (riskId === selectedRisk) return;
    setPendingRiskId(riskId);
    setRiskConfirmOpen(true);
  };

  const performRiskChange = async () => {
    if (isBusy) return;
    if (!selectedBot || !pendingRiskId) return;
    startUiAction(
      "Updating Risk",
      "Applying risk profile to running bot...",
      `Updating risk to ${pendingRiskId.toUpperCase()}`,
      {
        botId: selectedBot.id,
        kind: "update_risk",
      }
    );
    setSelectedRisk(pendingRiskId);
    const success = await updateBotRisk(selectedBot.id, pendingRiskId);
    if (success) {
      appendActionLog("success", "Risk level updated", selectedBot.id);
    } else {
      appendActionLog("error", "Risk level update failed", selectedBot.id);
    }
    setRiskConfirmOpen(false);
    setPendingRiskId(null);
    finishUiAction();
  };

  const handleModelSelect = (modelId: string) => {
    if (isBusy) return;
    if (modelId === selectedBot?.model_id) return;
    setPendingModelId(modelId);
    setModelConfirmOpen(true);
  };

  const performModelChange = async () => {
    if (isBusy) return;
    if (!selectedBot || !pendingModelId) return;
    startUiAction(
      "Changing Model",
      "Updating model metadata and restarting bot if currently running...",
      `Changing model to ${availableModels.find((m) => m.model_id === pendingModelId)?.label || pendingModelId}`,
      {
        botId: selectedBot.id,
        expectedStatus: (selectedBot.status === "running" ? "running" : "stopped"),
        kind: "change_model",
      }
    );
    const success = await changeModel(selectedBot.id, pendingModelId);
    if (success) {
      appendActionLog("success", "Model changed successfully", selectedBot.id);
      setShowModelDialog(false);
    } else {
      appendActionLog("error", "Model change failed", selectedBot.id);
    }
    setModelConfirmOpen(false);
    setPendingModelId(null);
    finishUiAction();
  };

  const handleApplyUpdate = async () => {
    if (isBusy) return;
    if (isApplyUpdatePendingFromLogs) return;
    if (!selectedBot) return;
    if (applyUpdateInFlightRef.current) return;
    applyUpdateInFlightRef.current = true;
    startUiAction(
      "Applying Update",
      "Applying latest version and restarting bot if required...",
      `Applying latest version for ${selectedBot.bot_version?.label || "bot"}`,
      {
        botId: selectedBot.id,
        kind: "apply_update",
      }
    );
    setUpdateConfirmOpen(false);
    try {
      const result = await applyBotUpdate(selectedBot.id);
      if (result.success) {
        appendActionLog("success", "Bot update applied", selectedBot.id);
        finishUiAction();
        return;
      }
      if (result.pending) {
        appendActionLog("info", "Apply update request accepted. Waiting for backend sync...", selectedBot.id);
        toast.info("Update is still in progress. Tracking status from backend...");
        return;
      }
      appendActionLog("error", "Bot update failed", selectedBot.id);
      finishUiAction();
    } finally {
      applyUpdateInFlightRef.current = false;
    }
  };

  const handleDeleteBot = async () => {
    if (isBusy) return;
    if (!selectedBot) return;
    startUiAction(
      "Deleting Bot",
      "Stopping container and removing bot configuration...",
      `Deleting ${selectedBot.bot_version?.label || "bot"}`,
      {
        botId: selectedBot.id,
        expectedStatus: "deleted",
        kind: "deleting",
      }
    );
    const success = await deleteBot(selectedBot.id);
    if (success) {
      appendActionLog("success", "Bot deleted", selectedBot.id);
      setDeleteConfirmOpen(false);
    } else {
      appendActionLog("error", "Delete bot failed", selectedBot.id);
    }
    finishUiAction();
  };

  const botStatus = selectedBot?.status || "stopped";
  const normalizedBotStatus = String(botStatus || "").toLowerCase();
  const isRunningStatus = normalizedBotStatus === "running";
  const isStartingStatus = normalizedBotStatus === "starting";
  const hasLiveStream = Boolean(liveState?.connected);
  const isLiveConnecting =
    !hasLiveStream && (isRunningStatus || isStartingStatus || activeActionForSelectedBot?.kind === "starting");
  const botSymbol = selectedBot?.bot_version?.symbol || "N/A";
  const botLabel = selectedBot?.bot_version?.label || "No Bot Selected";
  const bots = selectedAccount?.bot_configurations || [];
  const visibleActionLogs = selectedBot
    ? actionLogs.filter((log) => log.botId === selectedBot.id)
    : [];
  const latestApplyUpdateLog = visibleActionLogs.find((log) => log.action === "apply_update");
  const isApplyUpdatePendingFromLogs = Boolean(
    latestApplyUpdateLog?.phase === "requested" &&
    typeof latestApplyUpdateLog?.ts === "number" &&
    Date.now() - latestApplyUpdateLog.ts <= 15 * 60 * 1000
  );
  const isApplyingUpdateVisual = Boolean(isApplyingUpdateAction || isApplyUpdatePendingFromLogs);

  const getStatusBadge = (status: string | null) => {
    switch (status) {
      case "running":
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-success/10 text-success"><span className="w-1.5 h-1.5 rounded-full bg-success" />Running</span>;
      case "starting":
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning"><RefreshCw className="w-3 h-3 animate-spin" />Starting</span>;
      case "paused":
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-warning/10 text-warning">Paused</span>;
      default:
        return <span className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium bg-muted text-muted-foreground"><span className="w-1.5 h-1.5 rounded-full bg-muted-foreground" />Stopped</span>;
    }
  };

  const pendingRiskLabel = riskLevels.find(r => r.id === pendingRiskId)?.label;
  const pendingModelLabel = availableModels.find(m => m.model_id === pendingModelId)?.label;

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
            onUpdateAccount={updateAccount}
            onDeleteAccount={deleteAccount}
            onAccountDeleted={(accountId) => {
              if (selectedAccount?.id === accountId) {
                setSelectedAccount(null);
                setSelectedBot(null);
              }
            }}
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

      {activeActionForSelectedBot && (
        <div className="rounded-xl border border-warning/30 bg-warning/5 p-4">
          <div className="flex items-start gap-3">
            <RefreshCw className="mt-0.5 h-4 w-4 animate-spin text-warning" />
            <div>
              <p className="text-sm font-semibold text-foreground">{activeActionForSelectedBot.title}</p>
              <p className="text-xs text-muted-foreground">{activeActionForSelectedBot.detail}</p>
            </div>
          </div>
        </div>
      )}

      {visibleActionLogs.length > 0 && (
        <div className="glass-card p-4">
          <div className="mb-2 flex items-center justify-between">
            <p className="text-sm font-semibold">Operation Log</p>
            <span className="text-xs text-muted-foreground">latest {visibleActionLogs.length} entries</span>
          </div>
          <div className="max-h-52 space-y-2 overflow-y-auto pr-1">
            {visibleActionLogs.map((log) => (
              <div key={log.id} className="flex items-center gap-2 text-xs">
                <span className="font-mono text-muted-foreground">{log.at}</span>
                <span
                  className={cn(
                    "font-medium",
                    log.level === "success"
                      ? "text-success"
                      : log.level === "error"
                        ? "text-destructive"
                        : "text-warning"
                  )}
                >
                  {log.level.toUpperCase()}
                </span>
                <span className="text-muted-foreground">{log.message}</span>
              </div>
            ))}
          </div>
        </div>
      )}

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
                    {selectedBot.has_pending_update && (
                      <Button
                        className="gap-2 bg-primary text-primary-foreground hover:bg-primary/90"
                        onClick={() => setUpdateConfirmOpen(true)}
                        disabled={isBusy || isApplyUpdatePendingFromLogs}
                      >
                        {isApplyingUpdateVisual ? (
                          <RefreshCw className="w-4 h-4 animate-spin" />
                        ) : (
                          <DownloadCloud className="w-4 h-4" />
                        )}
                        {isApplyingUpdateVisual ? "Updating..." : "Update Bot"}
                      </Button>
                    )}
                    <div className="flex items-center gap-2 px-4 py-2 rounded-lg bg-secondary">
                      <TrendingUp className="w-4 h-4 text-success" />
                      <span
                        className={cn(
                          "font-mono font-medium",
                          selectedBot.today_pnl >= 0 ? "text-success" : "text-destructive"
                        )}
                      >
                        {selectedBot.today_pnl >= 0 ? "+" : ""}${selectedBot.today_pnl.toFixed(2)}
                      </span>
                    </div>
                    <Button
                      variant="outline"
                      className="gap-2 bg-warning/10 text-warning border-warning/30 hover:bg-warning/20"
                      onClick={() => setShowModelDialog(true)}
                      disabled={isBusy}
                    >
                      <RefreshCw className="w-4 h-4" />
                      Change Model
                    </Button>
                    <Button variant="outline" className="gap-2" onClick={handleToggleBotClick} disabled={isBusy}>
                      {botStatus === "running" ? (
                        <>
                          <Power className="w-4 h-4" />
                          Stop
                        </>
                      ) : botStatus === "starting" ? (
                        <>
                          <RefreshCw className="w-4 h-4 animate-spin" />
                          Starting...
                        </>
                      ) : (
                        <>
                          <Play className="w-4 h-4" />
                          Start
                        </>
                      )}
                    </Button>
                    <Button variant="destructive" onClick={handlePanicClick} className="gap-2" disabled={isBusy}>
                      <AlertOctagon className="w-4 h-4" />
                      Emergency Stop
                    </Button>
                    <Button
                      variant="ghost"
                      className="gap-2 text-destructive hover:text-destructive hover:bg-destructive/10"
                      onClick={() => setDeleteConfirmOpen(true)}
                      disabled={isBusy}
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

          {/* Live Status Card */}
          {selectedBot && (
            <div className="glass-card p-5 animate-slide-up" style={{ animationDelay: "25ms" }}>
              <div className="flex items-center gap-2 mb-4">
                <Activity className="w-5 h-5 text-primary" />
                <h3 className="font-semibold">Live Status</h3>
                {liveState?.connected ? (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-success/10 text-success">
                    <span className="w-1.5 h-1.5 rounded-full bg-success animate-pulse" />
                    Live
                  </span>
                ) : isLiveConnecting ? (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-warning/10 text-warning">
                    <RefreshCw className="h-3 w-3 animate-spin" />
                    Connecting
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-muted text-muted-foreground">
                    Offline
                  </span>
                )}
              </div>
              {liveState ? (
                <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Position</p>
                    <p className={cn(
                      "text-lg font-semibold font-mono",
                      livePosition === 1 ? "text-success" : livePosition === -1 ? "text-destructive" : "text-muted-foreground"
                    )}>
                      {livePosition === 1 ? "LONG" : livePosition === -1 ? "SHORT" : "FLAT"}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Total PnL</p>
                    <p className={cn(
                      "text-lg font-semibold font-mono",
                      liveTotalPnl >= 0 ? "text-success" : "text-destructive"
                    )}>
                      {liveTotalPnl >= 0 ? "+" : ""}{liveTotalPnl.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Unrealized PnL</p>
                    <p className={cn(
                      "text-lg font-semibold font-mono",
                      liveUnrealized >= 0 ? "text-success" : "text-destructive"
                    )}>
                      {liveUnrealized >= 0 ? "+" : ""}{liveUnrealized.toFixed(2)}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Equity</p>
                    <p className="text-lg font-semibold font-mono">${liveEquity.toFixed(2)}</p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Trades / Wins</p>
                    <p className="text-lg font-semibold font-mono">
                      {liveTrades} / <span className="text-success">{liveWins}</span>
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Win Rate</p>
                    <p className="text-lg font-semibold font-mono">
                      {liveTrades > 0 ? ((liveWins / liveTrades) * 100).toFixed(1) : "0.0"}%
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Last Action</p>
                    <p className={cn(
                      "text-lg font-semibold",
                      liveAction === "BUY" ? "text-success" : liveAction === "SELL" ? "text-destructive" : "text-muted-foreground"
                    )}>
                      {liveAction}
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50">
                    <p className="text-xs text-muted-foreground mb-1">Lot / Last Bar</p>
                    <p className="text-sm font-mono">
                      {liveLotSize.toFixed(2)} · <span className="text-muted-foreground">{liveState?.last_bar_time?.slice(11, 19) || "—"}</span>
                    </p>
                  </div>
                  <div className="p-3 rounded-xl bg-secondary/50 md:col-span-2">
                    <p className="text-xs text-muted-foreground mb-1">Trading Schedule</p>
                    <p className="text-sm font-mono">{liveScheduleSummary}</p>
                  </div>
                </div>
              ) : (
                <div className="text-center py-6 text-muted-foreground">
                  {isLiveConnecting ? (
                    <>
                      <p className="text-sm">Bot is starting up</p>
                      <p className="text-xs mt-1">
                        Container is running. Waiting for live data stream...
                        {runtimeHealth?.health_detail ? ` (${runtimeHealth.health_detail})` : ""}
                      </p>
                    </>
                  ) : (
                    <>
                      <p className="text-sm">Bot is currently offline</p>
                      <p className="text-xs mt-1">Start the bot to begin live updates. This panel will connect automatically.</p>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* Risk Level & Schedule */}
          {selectedBot && (
            <div className="grid lg:grid-cols-2 gap-6">
              {selectedBot.has_pending_update && (
                <div className="lg:col-span-2 rounded-xl border border-primary/30 bg-primary/5 p-4">
                  <div className="flex flex-col gap-3 md:flex-row md:items-center md:justify-between">
                    <div>
                      <p className="text-sm font-semibold text-primary">New bot version available</p>
                      <p className="text-xs text-muted-foreground">
                        Installed: {selectedBot.installed_version_tag || "-"} | Latest: {selectedBot.latest_version_tag || "-"}
                      </p>
                      {selectedBot.latest_release_notes.length > 0 && (
                        <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
                          {selectedBot.latest_release_notes.slice(0, 3).map((note, index) => (
                            <li key={`${selectedBot.id}-update-note-${index}`} className="flex items-start gap-2">
                              <span className="mt-1 h-1.5 w-1.5 shrink-0 rounded-full bg-primary" />
                              <span>{note}</span>
                            </li>
                          ))}
                        </ul>
                      )}
                    </div>
                    <Button className="gap-2" onClick={() => setUpdateConfirmOpen(true)} disabled={isBusy || isApplyUpdatePendingFromLogs}>
                      {isApplyingUpdateVisual ? (
                        <RefreshCw className="w-4 h-4 animate-spin" />
                      ) : (
                        <DownloadCloud className="w-4 h-4" />
                      )}
                      {isApplyingUpdateVisual ? "Updating..." : "Apply Update"}
                    </Button>
                  </div>
                </div>
              )}
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
                      onClick={() => handleRiskClick(level.id)}
                      disabled={isBusy}
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
                <div className="grid grid-cols-4 sm:grid-cols-7 gap-2 mb-6">
                  {schedule.map((day) => (
                    <button
                      key={day.day}
                      onClick={() => handleDayClick(day.day)}
                      disabled={isBusy}
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
                  isBusy && "pointer-events-none opacity-60",
                  selectedBot?.model_id === model.model_id && "ring-2 ring-primary border-primary"
                )}
                onClick={() => handleModelSelect(model.model_id)}
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
                      handleModelSelect(model.model_id);
                    }}
                    variant={selectedBot?.model_id === model.model_id ? "secondary" : "default"}
                    disabled={selectedBot?.model_id === model.model_id || isBusy}
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
            <AlertDialogAction
              onClick={handleDeleteBot}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isBusy}
            >
              Delete
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Stop Confirmation */}
      <AlertDialog open={stopConfirmOpen} onOpenChange={setStopConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Stop Bot?</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to stop this bot? It will stop processing new signals immediately.
              <br />
              Existing positions will need to be managed manually.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={() => performBotToggle("stopped")}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isBusy}
            >
              Stop Bot
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Emergency Stop Confirmation */}
      <AlertDialog open={panicConfirmOpen} onOpenChange={setPanicConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle className="flex items-center gap-2 text-destructive">
              <AlertTriangle className="w-5 h-5" />
              EMERGENCY STOP
            </AlertDialogTitle>
            <AlertDialogDescription>
              This will immediately STOP the bot and attempt to CLOSE ALL OPEN POSITIONS.
              <br />
              Use this only in emergency situations. Are you sure?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction
              onClick={performPanicStop}
              className="bg-destructive text-destructive-foreground hover:bg-destructive/90"
              disabled={isBusy}
            >
              CONFIRM EMERGENCY STOP
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Risk Change Confirmation */}
      <AlertDialog open={riskConfirmOpen} onOpenChange={setRiskConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Change Risk Level</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to change the risk level to <strong>{pendingRiskLabel}</strong>?
              <br />
              This will affect position sizing and stop-loss parameters for future trades.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPendingRiskId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={performRiskChange} disabled={isBusy}>
              Confirm Change
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Model Change Confirmation */}
      <AlertDialog open={modelConfirmOpen} onOpenChange={setModelConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Change Trading Model</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to switch to <strong>{pendingModelLabel}</strong>?
              <br />
              The bot will be updated with the new model's logic and parameters.
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPendingModelId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={performModelChange} disabled={isBusy}>
              Confirm Switch
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Schedule Change Confirmation */}
      <AlertDialog open={scheduleConfirmOpen} onOpenChange={setScheduleConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Update Trading Schedule</AlertDialogTitle>
            <AlertDialogDescription>
              Are you sure you want to <strong>{schedule.find(s => s.day === pendingDay)?.enabled ? "disable" : "enable"}</strong> trading on <strong>{pendingDay}</strong>?
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPendingDay(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={performScheduleToggle} disabled={isBusy}>
              Confirm Update
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      {/* Bot Update Confirmation */}
      <AlertDialog open={updateConfirmOpen} onOpenChange={setUpdateConfirmOpen}>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Apply Bot Update</AlertDialogTitle>
            <AlertDialogDescription>
              This will apply the latest version tag to this bot and restart automatically if it is currently running.
              <br />
              Latest version: <strong>{selectedBot?.latest_version_tag || "-"}</strong>
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel>Cancel</AlertDialogCancel>
            <AlertDialogAction onClick={handleApplyUpdate} disabled={isBusy || isApplyUpdatePendingFromLogs}>
              {isApplyingUpdateVisual ? "Updating..." : "Update Bot"}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}
