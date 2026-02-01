import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { Database } from "@/lib/integrations/supabase/types";

type TradingJournal = Database["public"]["Tables"]["trading_journals"]["Row"];
type OrderHistory = Database["public"]["Tables"]["orders_history"]["Row"];

export interface JournalWithOrder extends TradingJournal {
  order?: OrderHistory | null;
}

export function useTradingJournal() {
  const { user } = useAuth();
  const [journals, setJournals] = useState<JournalWithOrder[]>([]);
  const [orders, setOrders] = useState<OrderHistory[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchJournals = useCallback(async () => {
    if (!user) {
      setJournals([]);
      setOrders([]);
      setLoading(false);
      return;
    }

    try {
      // Fetch journals
      const { data: journalsData, error: journalsError } = await supabase
        .from("trading_journals")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false });

      if (journalsError) throw journalsError;

      // Get user's trading accounts to fetch orders
      const { data: accounts } = await supabase
        .from("trading_accounts")
        .select("id")
        .eq("user_id", user.id);

      const accountIds = accounts?.map(a => a.id) || [];

      // Fetch orders for user's accounts
      let ordersData: OrderHistory[] = [];
      if (accountIds.length > 0) {
        const { data } = await supabase
          .from("orders_history")
          .select("*")
          .in("account_id", accountIds)
          .order("open_time", { ascending: false });
        ordersData = data || [];
      }

      setOrders(ordersData);

      // Combine journals with orders
      const journalsWithOrders: JournalWithOrder[] = (journalsData || []).map(journal => ({
        ...journal,
        order: ordersData.find(o => o.id === journal.ticket_id) || null,
      }));

      setJournals(journalsWithOrders);
    } catch (error) {
      console.error("Error fetching journals:", error);
      toast.error("Failed to fetch journal entries");
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchJournals();
  }, [fetchJournals]);

  const createJournal = async (data: {
    trade_rationale?: string;
    mistake_lesson?: string;
    tags?: string[];
    ticket_id?: string;
    attachment_urls?: string[];
  }) => {
    if (!user) return null;

    try {
      const { data: journal, error } = await supabase
        .from("trading_journals")
        .insert({
          user_id: user.id,
          ...data,
        })
        .select()
        .single();

      if (error) throw error;

      toast.success("Journal entry created");
      await fetchJournals();
      return journal;
    } catch (error) {
      console.error("Error creating journal:", error);
      toast.error("Failed to create journal entry");
      return null;
    }
  };

  const updateJournal = async (id: string, data: Partial<TradingJournal>) => {
    try {
      const { error } = await supabase
        .from("trading_journals")
        .update(data)
        .eq("id", id);

      if (error) throw error;

      toast.success("Journal entry updated");
      await fetchJournals();
      return true;
    } catch (error) {
      console.error("Error updating journal:", error);
      toast.error("Failed to update journal entry");
      return false;
    }
  };

  const deleteJournal = async (id: string) => {
    try {
      const { error } = await supabase
        .from("trading_journals")
        .delete()
        .eq("id", id);

      if (error) throw error;

      toast.success("Journal entry deleted");
      await fetchJournals();
      return true;
    } catch (error) {
      console.error("Error deleting journal:", error);
      toast.error("Failed to delete journal entry");
      return false;
    }
  };

  return {
    journals,
    orders,
    loading,
    refetch: fetchJournals,
    createJournal,
    updateJournal,
    deleteJournal,
  };
}
