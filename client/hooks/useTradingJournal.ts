import { useCallback, useEffect, useMemo, useState } from "react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export interface TradingJournalRow {
  journalId: string | null;
  ticketId: number;
  accountId: string;
  symbol: string;
  type: string;
  status: string;
  volume: number;
  openPrice: number;
  closePrice: number;
  profit: number;
  commission: number;
  swap: number;
  openTime: string | null;
  closeTime: string | null;
  tradeRationale: string | null;
  mistakeLesson: string | null;
  tags: string[];
  attachmentUrls: string[];
  journalCreatedAt: string | null;
  journalUpdatedAt: string | null;
}

export interface TradingJournalSummary {
  totalRows: number;
  withJournal: number;
  withoutJournal: number;
}

export interface UpsertTradingJournalPayload {
  ticketId: number;
  tradeRationale?: string;
  mistakeLesson?: string;
  tags?: string[];
  attachmentUrls?: string[];
}

export function useTradingJournal() {
  const { user, loading: authLoading } = useAuth();
  const [rows, setRows] = useState<TradingJournalRow[]>([]);
  const [summary, setSummary] = useState<TradingJournalSummary>({
    totalRows: 0,
    withJournal: 0,
    withoutJournal: 0,
  });
  const [loading, setLoading] = useState<boolean>(true);
  const [query, setQuery] = useState<string>("");

  const fetchJournalFeed = useCallback(
    async (nextQuery?: string) => {
      if (authLoading) return;
      if (!user) {
        setRows([]);
        setSummary({ totalRows: 0, withJournal: 0, withoutJournal: 0 });
        setLoading(false);
        return;
      }

      try {
        setLoading(true);
        const q = typeof nextQuery === "string" ? nextQuery : query;
        const res = await api.get("/trading/journal_feed", {
          params: {
            q: q || undefined,
            limit: 400,
          },
        });
        const data = Array.isArray(res.data?.data) ? res.data.data : [];
        const s = res.data?.summary || {};
        setRows(data as TradingJournalRow[]);
        setSummary({
          totalRows: Number(s.totalRows || data.length || 0),
          withJournal: Number(s.withJournal || 0),
          withoutJournal: Number(s.withoutJournal || 0),
        });
      } catch (error: any) {
        console.error("Error fetching trading journal feed:", error);
        toast.error(error?.message || "Failed to fetch trading journal");
      } finally {
        setLoading(false);
      }
    },
    [authLoading, query, user]
  );

  useEffect(() => {
    fetchJournalFeed();
  }, [fetchJournalFeed]);

  const upsertJournal = useCallback(
    async (payload: UpsertTradingJournalPayload) => {
      try {
        await api.post("/trading/journal/upsert", payload);
        toast.success("Journal saved");
        await fetchJournalFeed();
        return true;
      } catch (error: any) {
        console.error("Error saving journal:", error);
        toast.error(error?.message || "Failed to save journal");
        return false;
      }
    },
    [fetchJournalFeed]
  );

  const deleteJournal = useCallback(
    async (journalId: string) => {
      try {
        await api.delete(`/trading/journal/${journalId}`);
        toast.success("Journal deleted");
        await fetchJournalFeed();
        return true;
      } catch (error: any) {
        console.error("Error deleting journal:", error);
        toast.error(error?.message || "Failed to delete journal");
        return false;
      }
    },
    [fetchJournalFeed]
  );

  const stats = useMemo(() => {
    const totalProfit = rows.reduce((acc, row) => acc + Number(row.profit || 0), 0);
    const wins = rows.filter((row) => Number(row.profit || 0) > 0).length;
    const losses = rows.filter((row) => Number(row.profit || 0) < 0).length;
    return {
      totalProfit,
      wins,
      losses,
    };
  }, [rows]);

  return {
    rows,
    summary,
    stats,
    loading,
    query,
    setQuery,
    refetch: fetchJournalFeed,
    upsertJournal,
    deleteJournal,
  };
}
