import { useCallback, useEffect, useState } from "react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { useAuth } from "@/contexts/AuthContext";

export type TicketStatus = "open" | "in_progress" | "resolved" | "closed";

export interface SupportTicketMessageItem {
  role: "user" | "admin";
  text: string;
  created_at: string | null;
  sender_name: string | null;
  sender_email: string | null;
}

export interface SupportTicketItem {
  id: string;
  user_id: string;
  user_email: string | null;
  user_name: string | null;
  subject: string;
  category: string | null;
  user_message: string;
  admin_reply: string | null;
  messages: SupportTicketMessageItem[];
  status: TicketStatus;
  created_at: string | null;
  updated_at: string | null;
}

interface CreateTicketPayload {
  subject: string;
  category?: string;
  message: string;
}

interface AdminReplyPayload {
  reply: string;
  status?: TicketStatus;
}

interface UserReplyPayload {
  message: string;
}

export function useSupportTickets() {
  const { user, isAdmin, loading: authLoading } = useAuth();
  const [tickets, setTickets] = useState<SupportTicketItem[]>([]);
  const [loading, setLoading] = useState(true);

  const fetchMyTickets = useCallback(async () => {
    if (authLoading) return;
    if (!user) {
      setTickets([]);
      setLoading(false);
      return;
    }
    try {
      setLoading(true);
      const { data } = await api.get<SupportTicketItem[]>("/support/tickets");
      setTickets(Array.isArray(data) ? data : []);
    } catch (error: any) {
      console.error("Error fetching my support tickets:", error);
      toast.error(error?.message || "Failed to fetch support tickets");
    } finally {
      setLoading(false);
    }
  }, [authLoading, user]);

  const fetchAdminTickets = useCallback(
    async (status: "all" | TicketStatus = "all") => {
      if (authLoading) return;
      if (!user || !isAdmin) {
        setTickets([]);
        setLoading(false);
        return;
      }
      try {
        setLoading(true);
        const { data } = await api.get<SupportTicketItem[]>("/support/admin/tickets", {
          params: { status },
        });
        setTickets(Array.isArray(data) ? data : []);
      } catch (error: any) {
        console.error("Error fetching admin support tickets:", error);
        toast.error(error?.message || "Failed to fetch support tickets");
      } finally {
        setLoading(false);
      }
    },
    [authLoading, isAdmin, user]
  );

  const createTicket = async (payload: CreateTicketPayload) => {
    if (!user) return null;
    try {
      const { data } = await api.post<SupportTicketItem>("/support/tickets", payload);
      toast.success("Ticket submitted");
      await fetchMyTickets();
      return data;
    } catch (error: any) {
      console.error("Error creating support ticket:", error);
      toast.error(error?.message || "Failed to submit ticket");
      return null;
    }
  };

  const replyTicketAsAdmin = async (ticketId: string, payload: AdminReplyPayload) => {
    if (!user || !isAdmin) return null;
    try {
      const { data } = await api.patch<SupportTicketItem>(`/support/admin/tickets/${ticketId}/reply`, payload);
      toast.success("Reply sent");
      return data;
    } catch (error: any) {
      console.error("Error replying to support ticket:", error);
      toast.error(error?.message || "Failed to send reply");
      return null;
    }
  };

  const replyTicketAsUser = async (ticketId: string, payload: UserReplyPayload) => {
    if (!user) return null;
    try {
      const { data } = await api.post<SupportTicketItem>(`/support/tickets/${ticketId}/reply`, payload);
      toast.success("Reply sent");
      return data;
    } catch (error: any) {
      console.error("Error replying as user:", error);
      toast.error(error?.message || "Failed to send reply");
      return null;
    }
  };

  useEffect(() => {
    if (isAdmin) {
      fetchAdminTickets("all");
    } else {
      fetchMyTickets();
    }
  }, [isAdmin, fetchAdminTickets, fetchMyTickets]);

  return {
    tickets,
    loading,
    fetchMyTickets,
    fetchAdminTickets,
    createTicket,
    replyTicketAsAdmin,
    replyTicketAsUser,
  };
}
