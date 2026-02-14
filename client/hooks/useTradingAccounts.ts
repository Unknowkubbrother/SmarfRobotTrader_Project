import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
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

export function useTradingAccounts() {
  const { user } = useAuth();
  const [accounts, setAccounts] = useState<AccountWithBots[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAccounts = useCallback(async () => {
    if (!user) {
      setAccounts([]);
      setLoading(false);
      return;
    }

    try {
      const result = await api.get("/trading/accounts_with_bots");

      const accountsData: AccountWithBots[] = (result.data?.data || []).map((account: any) => ({
        ...account,
        bot_configurations: (account.bot_configurations || []).map((config: any) => ({
          ...config,
          status: config.container_status || "stopped",
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
  }, [user]);

  useEffect(() => {
    fetchAccounts();
  }, [fetchAccounts]);

  // Update bot status by bot config ID (still uses Supabase — backend endpoint TBD)
  const updateBotStatus = async (
    botConfigId: string,
    status: string
  ) => {
    try {
      const { error } = await supabase
        .from("bot_configurations")
        .update({ status: status as any, is_active: status === "running" })
        .eq("id", botConfigId);

      if (error) throw error;

      await fetchAccounts();
      return true;
    } catch (error) {
      console.error("Error updating bot status:", error);
      toast.error("Failed to update bot status");
      return false;
    }
  };

  // Update bot config by bot config ID (still uses Supabase — backend endpoint TBD)
  const updateBotConfig = async (
    botConfigId: string,
    updates: Record<string, any>
  ) => {
    try {
      const { error } = await supabase
        .from("bot_configurations")
        .update(updates)
        .eq("id", botConfigId);

      if (error) throw error;

      await fetchAccounts();
      toast.success("Bot configuration updated");
      return true;
    } catch (error) {
      console.error("Error updating bot config:", error);
      toast.error("Failed to update configuration");
      return false;
    }
  };

  // Create a new bot configuration for an account via FastAPI
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

  // Delete a bot configuration (still uses Supabase — backend endpoint TBD)
  const deleteBot = async (botConfigId: string) => {
    try {
      const { error } = await supabase
        .from("bot_configurations")
        .delete()
        .eq("id", botConfigId);

      if (error) throw error;

      await fetchAccounts();
      toast.success("Bot removed successfully");
      return true;
    } catch (error) {
      console.error("Error deleting bot:", error);
      toast.error("Failed to remove bot");
      return false;
    }
  };

  // Create a new trading account via FastAPI
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

  return {
    accounts,
    loading,
    refetch: fetchAccounts,
    createAccount,
    updateBotStatus,
    updateBotConfig,
    createBot,
    deleteBot,
    getRunningBotsCount,
  };
}
