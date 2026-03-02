import { useMemo, useState } from "react";
import { MessageSquare, RefreshCw, Clock, User, Mail, Send } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useSupportTickets,
  type SupportTicketItem,
  type TicketStatus,
  type SupportTicketMessageItem,
} from "@/hooks/useSupportTickets";

const statusOptions: Array<"all" | TicketStatus> = ["all", "open", "in_progress", "resolved", "closed"];

const statusLabels: Record<TicketStatus, string> = {
  open: "Open",
  in_progress: "In Progress",
  resolved: "Resolved",
  closed: "Closed",
};

function statusClassName(status: TicketStatus) {
  if (status === "open") return "bg-amber-500/15 text-amber-600";
  if (status === "in_progress") return "bg-blue-500/15 text-blue-600";
  if (status === "resolved") return "bg-emerald-500/15 text-emerald-600";
  return "bg-muted text-muted-foreground";
}

function formatDate(value: string | null) {
  if (!value) return "-";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return value;
  return parsed.toLocaleString();
}

function getTicketMessages(ticket: SupportTicketItem): SupportTicketMessageItem[] {
  if (Array.isArray(ticket.messages) && ticket.messages.length > 0) {
    return ticket.messages;
  }
  const fallback: SupportTicketMessageItem[] = [];
  if (ticket.user_message) {
    fallback.push({
      role: "user",
      text: ticket.user_message,
      created_at: ticket.created_at,
      sender_name: ticket.user_name,
      sender_email: ticket.user_email,
    });
  }
  if (ticket.admin_reply) {
    fallback.push({
      role: "admin",
      text: ticket.admin_reply,
      created_at: ticket.updated_at,
      sender_name: "Admin",
      sender_email: null,
    });
  }
  return fallback;
}

export function AdminMonitoring() {
  const { tickets, loading, fetchAdminTickets, replyTicketAsAdmin } = useSupportTickets();
  const [statusFilter, setStatusFilter] = useState<"all" | TicketStatus>("all");
  const [replyByTicketId, setReplyByTicketId] = useState<Record<string, string>>({});
  const [nextStatusByTicketId, setNextStatusByTicketId] = useState<Record<string, TicketStatus>>({});
  const [submittingTicketId, setSubmittingTicketId] = useState<string | null>(null);

  const sortedTickets = useMemo(
    () =>
      [...tickets].sort((a, b) => {
        const aTime = a.created_at ? new Date(a.created_at).getTime() : 0;
        const bTime = b.created_at ? new Date(b.created_at).getTime() : 0;
        return bTime - aTime;
      }),
    [tickets]
  );

  const handleRefresh = async () => {
    await fetchAdminTickets(statusFilter);
  };

  const handleReply = async (ticket: SupportTicketItem) => {
    const reply = (replyByTicketId[ticket.id] || "").trim();
    if (!reply) return;
    const nextStatus = nextStatusByTicketId[ticket.id] || "resolved";
    setSubmittingTicketId(ticket.id);
    const updated = await replyTicketAsAdmin(ticket.id, {
      reply,
      status: nextStatus,
    });
    setSubmittingTicketId(null);
    if (updated) {
      setReplyByTicketId((prev) => ({ ...prev, [ticket.id]: "" }));
      await fetchAdminTickets(statusFilter);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="glass-card p-6">
        <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-3">
          <div className="flex items-center gap-2">
            <MessageSquare className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Support Tickets</h3>
          </div>
          <div className="flex items-center gap-2">
            <select
              value={statusFilter}
              onChange={async (e) => {
                const next = e.target.value as "all" | TicketStatus;
                setStatusFilter(next);
                await fetchAdminTickets(next);
              }}
              className="h-9 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
            >
              {statusOptions.map((status) => (
                <option key={status} value={status}>
                  {status === "all" ? "All status" : statusLabels[status]}
                </option>
              ))}
            </select>
            <Button variant="outline" size="sm" onClick={handleRefresh}>
              <RefreshCw className="w-4 h-4 mr-2" />
              Refresh
            </Button>
          </div>
        </div>
      </div>

      <div className="space-y-3">
        {sortedTickets.length === 0 ? (
          <div className="glass-card p-6 text-sm text-muted-foreground">No tickets in this filter</div>
        ) : (
          sortedTickets.map((ticket) => {
            const replyValue = replyByTicketId[ticket.id] || "";
            const nextStatus = nextStatusByTicketId[ticket.id] || "resolved";
            const isSending = submittingTicketId === ticket.id;

            return (
              <div key={ticket.id} className="glass-card p-5 space-y-4">
                <div className="flex flex-col lg:flex-row lg:items-start lg:justify-between gap-3">
                  <div className="space-y-2">
                    <div className="flex items-center gap-2 flex-wrap">
                      <p className="font-semibold">{ticket.subject}</p>
                      <Badge className={statusClassName(ticket.status)}>{statusLabels[ticket.status]}</Badge>
                      <Badge variant="outline">{ticket.category || "Other"}</Badge>
                    </div>
                    <div className="text-xs text-muted-foreground flex flex-wrap items-center gap-x-3 gap-y-1">
                      <span className="inline-flex items-center gap-1">
                        <User className="w-3 h-3" />
                        {ticket.user_name || "Unknown"}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Mail className="w-3 h-3" />
                        {ticket.user_email || "-"}
                      </span>
                      <span className="inline-flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {formatDate(ticket.created_at)}
                      </span>
                    </div>
                  </div>
                </div>

                <div className="rounded-lg bg-secondary/50 p-4">
                  <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">Conversation</p>
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                    {getTicketMessages(ticket).length === 0 ? (
                      <p className="text-sm text-muted-foreground">No message in this ticket</p>
                    ) : (
                      getTicketMessages(ticket).map((item, index) => {
                        const isAdminMessage = item.role === "admin";
                        return (
                          <div
                            key={`${ticket.id}-${index}`}
                            className={`rounded-lg px-3 py-2 text-sm ${
                              isAdminMessage
                                ? "bg-primary/10 border border-primary/20 ml-6"
                                : "bg-background border border-border mr-6"
                            }`}
                          >
                            <div className="flex items-center justify-between gap-2 mb-1">
                              <span className="text-xs font-medium text-muted-foreground">
                                {isAdminMessage ? item.sender_name || "Admin" : item.sender_name || "User"}
                              </span>
                              <span className="text-[11px] text-muted-foreground">
                                {formatDate(item.created_at)}
                              </span>
                            </div>
                            <p className="whitespace-pre-wrap">{item.text}</p>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

                <div className="space-y-3 rounded-lg border border-border/70 p-4 bg-background/60">
                  <div className="flex items-center justify-between gap-2">
                    <p className="text-sm font-medium">Reply to user</p>
                    <select
                      value={nextStatus}
                      onChange={(e) =>
                        setNextStatusByTicketId((prev) => ({
                          ...prev,
                          [ticket.id]: e.target.value as TicketStatus,
                        }))
                      }
                      className="h-8 px-2 rounded-md bg-secondary border border-border text-xs focus:outline-none focus:border-primary/50"
                    >
                      <option value="in_progress">Set In Progress</option>
                      <option value="resolved">Set Resolved</option>
                      <option value="closed">Set Closed</option>
                      <option value="open">Keep Open</option>
                    </select>
                  </div>
                  <textarea
                    rows={4}
                    value={replyValue}
                    onChange={(e) =>
                      setReplyByTicketId((prev) => ({
                        ...prev,
                        [ticket.id]: e.target.value,
                      }))
                    }
                    placeholder="Type admin response..."
                    className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-sm resize-none focus:outline-none focus:border-primary/50"
                  />
                  <div className="flex justify-end">
                    <Button
                      onClick={() => handleReply(ticket)}
                      disabled={isSending || !replyValue.trim()}
                      size="sm"
                    >
                      <Send className="w-4 h-4 mr-2" />
                      {isSending ? "Sending..." : "Send Reply"}
                    </Button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
