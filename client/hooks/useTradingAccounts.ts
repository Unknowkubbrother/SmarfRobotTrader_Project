import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { Database } from "@/lib/integrations/supabase/types";

type TradingAccount = Database["public"]["Tables"]["trading_accounts"]["Row"];
type BotConfiguration = Database["public"]["Tables"]["bot_configurations"]["Row"];
type BotVersion = Database["public"]["Tables"]["bot_versions"]["Row"];

export interface BotConfigWithVersion extends BotConfiguration {
  bot_version: BotVersion | null;
  today_pnl: number;
}

export interface AccountWithBots extends TradingAccount {
  bot_configurations: BotConfigWithVersion[];
  total_today_pnl: number;
}

// Legacy interface for backward compatibility
export interface AccountWithBot extends TradingAccount {
  bot_configuration: (BotConfiguration & { bot_version: BotVersion | null }) | null;
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
      // Fetch trading accounts
      const { data: accountsData, error: accountsError } = await supabase
        .from("trading_accounts")
        .select("*")
        .eq("user_id", user.id);

      if (accountsError) throw accountsError;

      if (!accountsData || accountsData.length === 0) {
        setAccounts([]);
        setLoading(false);
        return;
      }

      // Fetch all bot configurations for these accounts (multiple per account)
      const accountIds = accountsData.map((a) => a.id);
      const { data: botConfigs } = await supabase
        .from("bot_configurations")
        .select("*, bot_versions(*)")
        .in("account_id", accountIds);

      // Fetch today's daily aggregates
      const today = new Date().toISOString().split("T")[0];
      const { data: dailyAggregates } = await supabase
        .from("daily_aggregates")
        .select("*")
        .in("account_id", accountIds)
        .eq("date", today);

      // Combine data - now supporting multiple bots per account
      const accountsWithBots: AccountWithBots[] = accountsData.map((account) => {
        // Get all bot configs for this account
        const accountBotConfigs = botConfigs?.filter((c) => c.account_id === account.id) || [];
        const todayAgg = dailyAggregates?.find((d) => d.account_id === account.id);

        // Map bot configs with their versions
        const botConfigsWithVersion: BotConfigWithVersion[] = accountBotConfigs.map((config) => ({
          ...config,
          bot_version: config.bot_versions as BotVersion | null,
          today_pnl: 0, // TODO: Calculate per-bot P&L when bot_instance_id is used in daily_aggregates
        }));

        return {
          ...account,
          bot_configurations: botConfigsWithVersion,
          total_today_pnl: todayAgg?.daily_net_profit || 0,
        };
      });

      setAccounts(accountsWithBots);
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

  // Update bot status by bot config ID (not account ID)
  const updateBotStatus = async (
    botConfigId: string,
    status: Database["public"]["Enums"]["bot_status"]
  ) => {
    try {
      const { error } = await supabase
        .from("bot_configurations")
        .update({ status, is_active: status === "running" })
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

  // Update bot config by bot config ID (not account ID)
  const updateBotConfig = async (
    botConfigId: string,
    updates: Partial<BotConfiguration>
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

  // Create a new bot configuration for an account
  const createBot = async (
    accountId: string,
    modelId: string,
    riskLevel: string = "medium"
  ) => {
    try {
      // Generate unique bot instance ID
      const botInstanceId = `bot_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      const { error } = await supabase.from("bot_configurations").insert({
        account_id: accountId,
        model_id: modelId,
        risk_level: riskLevel,
        is_active: false,
        status: "stopped",
        bot_instance_id: botInstanceId,
        container_status: "stopped",
      });

      if (error) throw error;

      await fetchAccounts();
      toast.success("Bot added successfully");
      return true;
    } catch (error) {
      console.error("Error creating bot:", error);
      toast.error("Failed to add bot");
      return false;
    }
  };

  // Delete a bot configuration
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

  // Helper to get running bots count for an account
  const getRunningBotsCount = (account: AccountWithBots) => {
    return account.bot_configurations.filter((b) => b.status === "running").length;
  };

  return {
    accounts,
    loading,
    refetch: fetchAccounts,
    updateBotStatus,
    updateBotConfig,
    createBot,
    deleteBot,
    getRunningBotsCount,
  };
}
