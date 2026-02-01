import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import type { Database } from "@/lib/integrations/supabase/types";

type DailyAggregate = Database["public"]["Tables"]["daily_aggregates"]["Row"];

interface MonthlyData {
  date: number;
  profit: number | null;
  trades: number;
  winRate: number;
}

export function useDailyAggregates(accountId?: string) {
  const { user } = useAuth();
  const [aggregates, setAggregates] = useState<DailyAggregate[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchAggregates = useCallback(async () => {
    if (!user) {
      setAggregates([]);
      setLoading(false);
      return;
    }

    try {
      // First get user's trading accounts
      let accountIds: string[] = [];
      
      if (accountId) {
        accountIds = [accountId];
      } else {
        const { data: accounts } = await supabase
          .from("trading_accounts")
          .select("id")
          .eq("user_id", user.id);
        
        accountIds = accounts?.map(a => a.id) || [];
      }

      if (accountIds.length === 0) {
        setAggregates([]);
        setLoading(false);
        return;
      }

      const { data, error } = await supabase
        .from("daily_aggregates")
        .select("*")
        .in("account_id", accountIds)
        .order("date", { ascending: false });

      if (error) throw error;

      setAggregates(data || []);
    } catch (error) {
      console.error("Error fetching daily aggregates:", error);
    } finally {
      setLoading(false);
    }
  }, [user, accountId]);

  useEffect(() => {
    fetchAggregates();
  }, [fetchAggregates]);

  const getMonthData = (year: number, month: number): MonthlyData[] => {
    const daysInMonth = new Date(year, month + 1, 0).getDate();
    const today = new Date();
    const data: MonthlyData[] = [];

    for (let day = 1; day <= daysInMonth; day++) {
      const date = new Date(year, month, day);
      const dateStr = date.toISOString().split("T")[0];
      const isWeekend = date.getDay() === 0 || date.getDay() === 6;
      const isFuture = date > today;

      if (isWeekend || isFuture) {
        data.push({ date: day, profit: null, trades: 0, winRate: 0 });
      } else {
        const dayAggregate = aggregates.find(a => a.date === dateStr);
        if (dayAggregate) {
          const winRate = dayAggregate.total_trades 
            ? Math.min(100, Math.max(0, (dayAggregate.daily_net_profit || 0) > 0 ? 70 : 30))
            : 0;
          data.push({
            date: day,
            profit: dayAggregate.daily_net_profit || 0,
            trades: dayAggregate.total_trades || 0,
            winRate,
          });
        } else {
          data.push({ date: day, profit: null, trades: 0, winRate: 0 });
        }
      }
    }

    return data;
  };

  const getStats = () => {
    const totalProfit = aggregates.reduce((sum, a) => sum + (a.daily_net_profit || 0), 0);
    const totalTrades = aggregates.reduce((sum, a) => sum + (a.total_trades || 0), 0);
    const tradingDays = aggregates.length;
    const profitableDays = aggregates.filter(a => (a.daily_net_profit || 0) > 0).length;
    const winRate = tradingDays > 0 ? (profitableDays / tradingDays) * 100 : 0;

    return { totalProfit, totalTrades, tradingDays, profitableDays, winRate };
  };

  return {
    aggregates,
    loading,
    refetch: fetchAggregates,
    getMonthData,
    getStats,
  };
}
