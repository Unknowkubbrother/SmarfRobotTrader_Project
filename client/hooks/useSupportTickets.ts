import { useState, useEffect, useCallback } from "react";
import { supabase } from "@/lib/integrations/supabase/client";
import { useAuth } from "@/contexts/AuthContext";
import { toast } from "sonner";
import type { Database } from "@/lib/integrations/supabase/types";

type SupportTicket = Database["public"]["Tables"]["support_tickets"]["Row"];

export function useSupportTickets() {
  const { user } = useAuth();
  const [tickets, setTickets] = useState<SupportTicket[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchTickets = useCallback(async () => {
    if (!user) {
      setTickets([]);
      setLoading(false);
      return;
    }

    try {
      const { data, error } = await supabase
        .from("support_tickets")
        .select("*")
        .eq("user_id", user.id)
        .order("created_at", { ascending: false });

      if (error) throw error;

      setTickets(data || []);
    } catch (error) {
      console.error("Error fetching support tickets:", error);
    } finally {
      setLoading(false);
    }
  }, [user]);

  useEffect(() => {
    fetchTickets();
  }, [fetchTickets]);

  const createTicket = async (subject: string, message: string) => {
    if (!user) {
      toast.error("Please sign in to create a ticket");
      return null;
    }

    if (!subject.trim() || !message.trim()) {
      toast.error("Please fill in all fields");
      return null;
    }

    try {
      const { data, error } = await supabase
        .from("support_tickets")
        .insert({
          user_id: user.id,
          subject,
          message,
        })
        .select()
        .single();

      if (error) throw error;

      toast.success("Support ticket created successfully");
      await fetchTickets();
      return data;
    } catch (error) {
      console.error("Error creating support ticket:", error);
      toast.error("Failed to create support ticket");
      return null;
    }
  };

  return {
    tickets,
    loading,
    refetch: fetchTickets,
    createTicket,
  };
}
