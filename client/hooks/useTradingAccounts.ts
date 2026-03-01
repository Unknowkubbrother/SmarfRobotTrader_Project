import { useState, useEffect, useCallback } from "react";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import { api } from "@/lib/api";

export interface Create_Trading_Account {
  brokerName: string;
  serverName: string;
  mt5LoginId: string;
  mt5Password: string;
}

export interface BotVersion {
  model_id: string;
  label: string | null;
  docker_image_id: string | null;
  version_tag: string | null;
  symbol: string | null;
  timeframe: string | null;
  release_notes: string[];
}

export interface PendingBotUpdate {
  has_pending_update: boolean;
  installed_docker_image_id: string | null;
  latest_docker_image_id: string | null;
  latest_version_tag: string | null;
  latest_release_notes: string[];
  latest_release_date: string | null;
}

export interface BotConfigWithVersion {
  id: string;
  account_id: string;
  model_id: string;
  bot_instance_id: number;
  risk_level: string | null;
  trading_schedule: any;
  is_active: boolean;
  docker_container_id: string | null;
  container_status: string | null;
  status: string | null;
  has_pending_update: boolean;
  installed_docker_image_id: string | null;
  latest_docker_image_id: string | null;
  latest_version_tag: string | null;
  latest_release_notes: string[];
  latest_release_date: string | null;
  bot_version: BotVersion | null;
  today_pnl: number;
}

export interface AccountWithBots {
  id: string;
  user_id: string;
  broker_name: string | null;
  server_name: string | null;
  mt5_login_id: string | null;
  balance: number;
  equity: number;
  leverage: number | null;
  margin: number;
  margin_free: number;
  margin_level: number;
  created_at: string | null;
  bot_configurations: BotConfigWithVersion[];
  total_today_pnl: number;
}

// Legacy interface for backward compatibility
export interface AccountWithBot {
  id: string;
  broker_name: string | null;
  server_name: string | null;
  mt5_login_id: string | null;
  bot_configuration: (BotConfigWithVersion & { bot_version: BotVersion | null }) | null;
  today_pnl: number;
}

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
    const key = DAY_ALIAS_TO_KEY[String(rawKey).trim().toLowerCase()];
    if (!key) continue;
    normalized[key] = Boolean(rawValue);
  }

  return normalized;
};

export function useTradingAccounts() {
  const { user, loading: authLoading } = useAuth();
  const [accounts, setAccounts] = useState<AccountWithBots[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAccounts = useCallback(async () => {
    // Wait for auth to finish loading before deciding
    if (authLoading) return;

    if (!user) {
      setAccounts([]);
      setLoading(false);
      return;
    }

    try {
      setLoading(true);
      const result = await api.get("/trading/accounts_with_bots");

      const accountsData: AccountWithBots[] = (result.data?.data || []).map((account: any) => ({
        ...account,
        bot_configurations: (account.bot_configurations || []).map((config: any) => ({
          ...config,
          trading_schedule: normalizeTradingSchedule(config.trading_schedule),
          status: config.container_status || "stopped",
          has_pending_update: Boolean(config.has_pending_update),
          installed_docker_image_id: config.installed_docker_image_id || null,
          latest_docker_image_id: config.latest_docker_image_id || null,
          latest_version_tag: config.latest_version_tag || null,
          latest_release_notes: Array.isArray(config.latest_release_notes) ? config.latest_release_notes : [],
          latest_release_date: config.latest_release_date || null,
          today_pnl: 0,
        })),
      }));

      setAccounts(accountsData);
    } catch (error) {
      console.error("Error fetching accounts:", error);
      toast.error("Failed to fetch trading accounts");
    } finally {
      setLoading(false);
    }
  }, [user, authLoading]);

  useEffect(() => {
    fetchAccounts();
  }, [fetchAccounts]);

  // Update bot status (start/stop)
  const updateBotStatus = async (botConfigId: string, status: string) => {
    try {
      await api.patch("/bot/update_status", { botConfigId, status });
      await fetchAccounts();
      return true;
    } catch (error) {
      console.error("Error updating bot status:", error);
      toast.error("Failed to update bot status");
      return false;
    }
  };

  // Update bot risk level
  const updateBotRisk = async (botConfigId: string, riskLevel: string) => {
    try {
      await api.patch("/bot/update_risk", { botConfigId, riskLevel });
      await fetchAccounts();
      toast.success("Risk level updated");
      return true;
    } catch (error) {
      console.error("Error updating risk level:", error);
      toast.error("Failed to update risk level");
      return false;
    }
  };

  // Update bot trading schedule
  const updateBotSchedule = async (botConfigId: string, tradingSchedule: Record<string, boolean>) => {
    try {
      await api.patch("/bot/update_schedule", { botConfigId, tradingSchedule });
      await fetchAccounts();
      toast.success("Trading schedule updated");
      return true;
    } catch (error) {
      console.error("Error updating schedule:", error);
      toast.error("Failed to update trading schedule");
      return false;
    }
  };

  // Change bot model
  const changeModel = async (botConfigId: string, newModelId: string) => {
    try {
      await api.patch("/bot/change_model", { botConfigId, newModelId });
      await fetchAccounts();
      toast.success("Bot model changed successfully");
      return true;
    } catch (error) {
      console.error("Error changing model:", error);
      toast.error("Failed to change bot model");
      return false;
    }
  };

  // Apply pending bot image update
  const applyBotUpdate = async (botConfigId: string) => {
    try {
      await api.patch("/bot/apply_update", { botConfigId });
      await fetchAccounts();
      toast.success("Bot update applied successfully");
      return true;
    } catch (error) {
      console.error("Error applying bot update:", error);
      toast.error("Failed to apply bot update");
      return false;
    }
  };

  // Create a new bot configuration for an account
  const createBot = async (
    accountId: string,
    modelId: string,
    riskLevel: string = "medium"
  ) => {
    try {
      await api.post("/bot/create_bot_configuration", {
        accountId,
        modelId,
        riskLevel,
      });

      await fetchAccounts();
      toast.success("Bot added successfully");
      return true;
    } catch (error: any) {
      console.error("Error creating bot:", error);
      toast.error(error.message || "Failed to add bot");
      return false;
    }
  };

  // Delete a bot configuration
  const deleteBot = async (botConfigId: string) => {
    try {
      await api.delete("/bot/delete", { data: { botConfigId } });
      await fetchAccounts();
      toast.success("Bot removed successfully");
      return true;
    } catch (error) {
      console.error("Error deleting bot:", error);
      toast.error("Failed to remove bot");
      return false;
    }
  };

  // Create a new trading account
  const createAccount = async (data: Create_Trading_Account) => {
    try {
      const response = await api.post("/trading/create_account", data);
      await fetchAccounts();
      toast.success("Trading account added successfully. Add a bot to start trading.");
      return { success: true, data: response.data };
    } catch (error: any) {
      console.error("Error creating account:", error);
      toast.error(error.message || "Failed to add account");
      return { success: false, error: error.message };
    }
  };

  // Helper to get running bots count for an account
  const getRunningBotsCount = (account: AccountWithBots) => {
    return account.bot_configurations.filter((b) => b.container_status === "running").length;
  };

  const getPendingUpdatesCount = (account: AccountWithBots) => {
    return account.bot_configurations.filter((b) => b.has_pending_update).length;
  };

  // Get all available bot versions
  const getBotVersions = async () => {
    try {
      const response = await api.get("/bot/versions");
      return response.data.data;
    } catch (error) {
      console.error("Error fetching bot versions:", error);
      return [];
    }
  };

  return {
    accounts,
    loading,
    refetch: fetchAccounts,
    createAccount,
    updateBotStatus,
    updateBotRisk,
    updateBotSchedule,
    changeModel,
    applyBotUpdate,
    createBot,
    deleteBot,
    getBotVersions,
    getRunningBotsCount,
    getPendingUpdatesCount,
  };
}
