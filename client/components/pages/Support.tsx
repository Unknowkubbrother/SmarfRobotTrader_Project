import { useMemo, useState, type FormEvent } from "react";
import { MessageCircle, Search, RefreshCw, Mail, Clock, Send, Lock } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import {
  useSupportTickets,
  type SupportTicketItem,
  type TicketStatus,
  type SupportTicketMessageItem,
} from "@/hooks/useSupportTickets";

const ticketCategories = [
  "Technical Issue",
  "Trading Bot",
  "Account Access",
  "Billing",
  "Feature Request",
  "Other",
];

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

function getTicketSearchableText(ticket: SupportTicketItem) {
  return [
    ticket.subject,
    ticket.category || "",
    ticket.user_message,
    ticket.admin_reply || "",
    ...(ticket.messages || []).map((item) => item.text || ""),
  ]
    .join(" ")
    .toLowerCase();
}

function formatMessageTime(value: string | null) {
  if (!value) return "";
  const parsed = new Date(value);
  if (Number.isNaN(parsed.getTime())) return "";
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

export default function Support() {
  const { tickets, loading, fetchMyTickets, createTicket, replyTicketAsUser } = useSupportTickets();
  const [searchQuery, setSearchQuery] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [sendingTicketId, setSendingTicketId] = useState<string | null>(null);
  const [subject, setSubject] = useState("");
  const [category, setCategory] = useState(ticketCategories[0]);
  const [message, setMessage] = useState("");
  const [replyByTicketId, setReplyByTicketId] = useState<Record<string, string>>({});

  const filteredTickets = useMemo(() => {
    const query = searchQuery.trim().toLowerCase();
    if (!query) return tickets;
    return tickets.filter((ticket) => getTicketSearchableText(ticket).includes(query));
  }, [tickets, searchQuery]);

  const handleSubmitTicket = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (submitting) return;
    setSubmitting(true);
    const created = await createTicket({
      subject: subject.trim(),
      category,
      message: message.trim(),
    });
    if (created) {
      setSubject("");
      setCategory(ticketCategories[0]);
      setMessage("");
    }
    setSubmitting(false);
  };

  const handleReplyTicket = async (ticket: SupportTicketItem) => {
    const text = (replyByTicketId[ticket.id] || "").trim();
    if (!text || ticket.status === "closed") return;
    setSendingTicketId(ticket.id);
    const updated = await replyTicketAsUser(ticket.id, { message: text });
    setSendingTicketId(null);
    if (updated) {
      setReplyByTicketId((prev) => ({ ...prev, [ticket.id]: "" }));
      await fetchMyTickets();
    }
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-bold text-foreground">Help & Support</h1>
        <p className="text-sm text-muted-foreground">Submit tickets and track replies from admin</p>
      </div>

      <div className="grid lg:grid-cols-3 gap-6">
        <div className="lg:col-span-1">
          <div className="glass-card p-6">
            <div className="flex items-center gap-2 mb-4">
              <MessageCircle className="w-5 h-5 text-primary" />
              <h3 className="text-lg font-semibold">Create Ticket</h3>
            </div>
            <form className="space-y-4" onSubmit={handleSubmitTicket}>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Subject</label>
                <input
                  type="text"
                  value={subject}
                  onChange={(e) => setSubject(e.target.value)}
                  placeholder="Brief title for your issue"
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                  maxLength={100}
                  required
                />
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Category</label>
                <select
                  value={category}
                  onChange={(e) => setCategory(e.target.value)}
                  className="w-full h-10 px-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                >
                  {ticketCategories.map((item) => (
                    <option key={item} value={item}>
                      {item}
                    </option>
                  ))}
                </select>
              </div>
              <div>
                <label className="block text-sm text-muted-foreground mb-2">Message</label>
                <textarea
                  rows={5}
                  value={message}
                  onChange={(e) => setMessage(e.target.value)}
                  placeholder="Describe the issue clearly so support can reproduce it"
                  className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-sm resize-none focus:outline-none focus:border-primary/50"
                  required
                />
              </div>
              <Button type="submit" className="w-full" disabled={submitting}>
                {submitting ? "Submitting..." : "Submit Ticket"}
              </Button>
              <p className="text-xs text-muted-foreground flex items-center gap-2">
                <Mail className="w-3 h-3" />
                Admin gets in-app notification and email for new ticket
              </p>
            </form>
          </div>
        </div>

        <div className="lg:col-span-2 space-y-4">
          <div className="glass-card p-4">
            <div className="flex flex-col md:flex-row gap-3 md:items-center md:justify-between">
              <div className="relative flex-1">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
                <input
                  type="text"
                  placeholder="Search tickets by subject, message, category..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full h-10 pl-10 pr-3 rounded-lg bg-secondary border border-border text-sm focus:outline-none focus:border-primary/50"
                />
              </div>
              <Button variant="outline" onClick={() => fetchMyTickets()} disabled={loading}>
                <RefreshCw className="w-4 h-4 mr-2" />
                Refresh
              </Button>
            </div>
          </div>

          <div className="space-y-3">
            {loading ? (
              <div className="glass-card p-6 text-sm text-muted-foreground">Loading tickets...</div>
            ) : filteredTickets.length === 0 ? (
              <div className="glass-card p-6 text-sm text-muted-foreground">No tickets found</div>
            ) : (
              filteredTickets.map((ticket) => (
                <div key={ticket.id} className="glass-card p-5 space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-3">
                    <div>
                      <p className="text-base font-semibold">{ticket.subject}</p>
                      <div className="flex items-center gap-2 mt-2 flex-wrap">
                        <Badge className={statusClassName(ticket.status)}>{statusLabels[ticket.status]}</Badge>
                        <Badge variant="outline">{ticket.category || "Other"}</Badge>
                      </div>
                    </div>
                    <div className="text-xs text-muted-foreground flex items-center gap-1">
                      <Clock className="w-3 h-3" />
                      {formatDate(ticket.created_at)}
                    </div>
                  </div>

                  <div className="rounded-lg bg-secondary/50 p-4">
                    <p className="text-xs uppercase tracking-wide text-muted-foreground mb-3">Conversation</p>
                    <div className="space-y-2 max-h-64 overflow-y-auto pr-1">
                      {getTicketMessages(ticket).length === 0 ? (
                        <p className="text-sm text-muted-foreground">No message in this ticket</p>
                      ) : (
                        getTicketMessages(ticket).map((item, index) => {
                          const isUser = item.role === "user";
                          return (
                            <div
                              key={`${ticket.id}-${index}`}
                              className={`rounded-lg px-3 py-2 text-sm ${
                                isUser
                                  ? "bg-primary/10 border border-primary/20 ml-6"
                                  : "bg-background border border-border mr-6"
                              }`}
                            >
                              <div className="flex items-center justify-between gap-2 mb-1">
                                <span className="text-xs font-medium text-muted-foreground">
                                  {isUser ? "You" : item.sender_name || "Admin"}
                                </span>
                                <span className="text-[11px] text-muted-foreground">
                                  {formatMessageTime(item.created_at)}
                                </span>
                              </div>
                              <p className="whitespace-pre-wrap">{item.text}</p>
                            </div>
                          );
                        })
                      )}
                    </div>
                  </div>

                  {ticket.status === "closed" ? (
                    <div className="rounded-lg border border-border p-3 text-xs text-muted-foreground flex items-center gap-2">
                      <Lock className="w-3 h-3" />
                      Ticket is closed. You cannot reply.
                    </div>
                  ) : (
                    <div className="rounded-lg border border-border p-3 bg-background/60 space-y-2">
                      <textarea
                        rows={3}
                        value={replyByTicketId[ticket.id] || ""}
                        onChange={(e) =>
                          setReplyByTicketId((prev) => ({
                            ...prev,
                            [ticket.id]: e.target.value,
                          }))
                        }
                        placeholder="Reply to support..."
                        className="w-full px-3 py-2 rounded-lg bg-secondary border border-border text-sm resize-none focus:outline-none focus:border-primary/50"
                      />
                      <div className="flex justify-end">
                        <Button
                          size="sm"
                          onClick={() => handleReplyTicket(ticket)}
                          disabled={sendingTicketId === ticket.id || !(replyByTicketId[ticket.id] || "").trim()}
                        >
                          <Send className="w-4 h-4 mr-2" />
                          {sendingTicketId === ticket.id ? "Sending..." : "Send Reply"}
                        </Button>
                      </div>
                    </div>
                  )}
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
