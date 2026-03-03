import { useState, useEffect } from "react";
import {
  Activity,
  AlertCircle,
  Eye,
  Landmark,
  MoreVertical,
  RefreshCw,
  Save,
  Search,
  Shield,
  UserCheck,
  UserX,
  Wallet,
} from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { Input } from "@/components/ui/input";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";

interface AdminUser {
  id: string;
  username: string;
  email: string;
  role: "user" | "admin";
  status: "active" | "banned" | "pending";
  created_at: string;
  is_onboarding_completed: boolean;
}

interface AdminUserBot {
  id: string;
  bot_instance_id: number;
  model_id: string;
  label: string | null;
  symbol: string | null;
  timeframe: string | null;
  container_status: "running" | "starting" | "stopped" | "error" | null;
  is_active: boolean;
  updated_at: string | null;
}

interface AdminUserTradingAccount {
  id: string;
  mt5_login_id: string | null;
  broker_name: string | null;
  server_name: string | null;
  balance: number;
  equity: number;
  running_bots: number;
  active_bots: number;
  bots: AdminUserBot[];
}

interface AdminUserInvoice {
  id: string;
  subscription_id: string;
  status: "pending" | "paid" | "failed" | "skipped" | null;
  amount: number;
  created_at: string | null;
  paid_at: string | null;
  billing_start_date: string | null;
  billing_end_date: string | null;
}

interface AdminUserBillingSummary {
  pending_count: number;
  paid_count: number;
  pending_amount: number;
  paid_amount: number;
  recent_invoices: AdminUserInvoice[];
}

interface AdminUserSubscription {
  id: string;
  status: "active" | "past_due" | "canceled";
  fee_type: "percentage" | "fixed";
  fee_value: number;
  min_profit_threshold: number;
  next_billing_date: string | null;
  created_at: string | null;
}

interface AdminUserDetail {
  id: string;
  username: string;
  email: string;
  role: "user" | "admin";
  status: "active" | "banned" | "pending";
  created_at: string;
  is_onboarding_completed: boolean;
  total_accounts: number;
  total_balance: number;
  pending_bills: number;
  trading_accounts: AdminUserTradingAccount[];
  subscriptions: AdminUserSubscription[];
  billing: AdminUserBillingSummary;
}

interface SubscriptionDraft {
  fee_type: "percentage" | "fixed";
  fee_value: string;
  min_profit_threshold: string;
  next_billing_date: string;
}

const currencyFormatter = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: 2,
});

const formatCurrency = (value: number) => currencyFormatter.format(value || 0);

const formatDate = (value: string | null) => {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleDateString();
};

const shortId = (value: string) => `#${value.slice(0, 8)}`;
const normalizeBotStatus = (status: AdminUserBot["container_status"], isActive: boolean): string => {
  const raw = String(status || "").trim().toLowerCase();
  if (raw) return raw;
  return isActive ? "active" : "stopped";
};

export function AdminUserManagement() {
  const [users, setUsers] = useState<AdminUser[]>([]);
  const [loading, setLoading] = useState(true);
  const [searchTerm, setSearchTerm] = useState("");

  const [selectedUser, setSelectedUser] = useState<AdminUser | null>(null);
  const [userDetail, setUserDetail] = useState<AdminUserDetail | null>(null);
  const [showUserDialog, setShowUserDialog] = useState(false);
  const [detailLoading, setDetailLoading] = useState(false);

  const [updatingUserId, setUpdatingUserId] = useState<string | null>(null);
  const [updatingBotId, setUpdatingBotId] = useState<string | null>(null);
  const [savingSubscriptionId, setSavingSubscriptionId] = useState<string | null>(null);
  const [skippingInvoiceId, setSkippingInvoiceId] = useState<string | null>(null);
  const [subscriptionDrafts, setSubscriptionDrafts] = useState<Record<string, SubscriptionDraft>>({});

  useEffect(() => {
    fetchUsers();
  }, []);

  useEffect(() => {
    if (!showUserDialog || !selectedUser?.id || !userDetail) return;
    const hasStartingBot = userDetail.trading_accounts.some((account) =>
      account.bots.some((bot) => normalizeBotStatus(bot.container_status, bot.is_active) === "starting")
    );
    if (!hasStartingBot) return;

    const timer = window.setInterval(() => {
      void fetchUserDetail(selectedUser.id);
    }, 3000);

    return () => window.clearInterval(timer);
  }, [showUserDialog, selectedUser?.id, userDetail]);

  const fetchUsers = async () => {
    try {
      setLoading(true);
      const { data } = await api.get<AdminUser[]>("/admin/users");
      setUsers(data || []);
    } catch (error: any) {
      console.error("Error fetching users:", error);
      toast.error(error?.message || "Failed to load users");
    } finally {
      setLoading(false);
    }
  };

  const fetchUserDetail = async (userId: string) => {
    try {
      setDetailLoading(true);
      const { data } = await api.get<AdminUserDetail>(`/admin/users/${userId}/detail`);
      setUserDetail(data);
      const nextDrafts: Record<string, SubscriptionDraft> = {};
      (data.subscriptions || []).forEach((subscription) => {
        nextDrafts[subscription.id] = {
          fee_type: subscription.fee_type,
          fee_value: String(subscription.fee_value ?? 0),
          min_profit_threshold: String(subscription.min_profit_threshold ?? 0),
          next_billing_date: subscription.next_billing_date ?? "",
        };
      });
      setSubscriptionDrafts(nextDrafts);
    } catch (error: any) {
      console.error("Error fetching user detail:", error);
      toast.error(error?.message || "Failed to load user details");
    } finally {
      setDetailLoading(false);
    }
  };

  const openUserDetails = async (user: AdminUser) => {
    setSelectedUser(user);
    setShowUserDialog(true);
    await fetchUserDetail(user.id);
  };

  const updateUserStatus = async (userId: string, status: "active" | "banned" | "pending") => {
    setUpdatingUserId(userId);
    try {
      await api.patch(`/admin/users/${userId}/status`, { status });
      toast.success(`User status updated to ${status}`);
      setUsers((prev) => prev.map((user) => (user.id === userId ? { ...user, status } : user)));
      if (selectedUser?.id === userId) {
        setSelectedUser((prev) => (prev ? { ...prev, status } : prev));
        await fetchUserDetail(userId);
      }
    } catch (error: any) {
      console.error("Error updating user status:", error);
      toast.error(error?.message || "Failed to update user status");
    } finally {
      setUpdatingUserId(null);
    }
  };

  const updateUserRole = async (userId: string, role: "user" | "admin") => {
    setUpdatingUserId(userId);
    try {
      await api.patch(`/admin/users/${userId}/role`, { role });
      toast.success(`User role updated to ${role}`);
      setUsers((prev) => prev.map((user) => (user.id === userId ? { ...user, role } : user)));
      if (selectedUser?.id === userId) {
        setSelectedUser((prev) => (prev ? { ...prev, role } : prev));
        await fetchUserDetail(userId);
      }
    } catch (error: any) {
      console.error("Error updating user role:", error);
      toast.error(error?.message || "Failed to update user role");
    } finally {
      setUpdatingUserId(null);
    }
  };

  const updateBotStatus = async (botConfigId: string, status: "running" | "stopped") => {
    if (!selectedUser) return;

    setUpdatingBotId(botConfigId);
    try {
      await api.patch(`/admin/users/${selectedUser.id}/bot-configurations/${botConfigId}/status`, { status });
      toast.success(`Bot status updated to ${status}`);
      await fetchUserDetail(selectedUser.id);
    } catch (error: any) {
      console.error("Error updating bot status:", error);
      toast.error(error?.message || "Failed to update bot status");
    } finally {
      setUpdatingBotId(null);
    }
  };

  const updateSubscriptionDraft = (
    subscriptionId: string,
    field: keyof SubscriptionDraft,
    value: string
  ) => {
    setSubscriptionDrafts((prev) => {
      const current = prev[subscriptionId] || {
        fee_type: "percentage",
        fee_value: "0",
        min_profit_threshold: "0",
        next_billing_date: "",
      };
      return {
        ...prev,
        [subscriptionId]: {
          ...current,
          [field]: value,
        },
      };
    });
  };

  const saveSubscriptionBilling = async (subscriptionId: string) => {
    if (!selectedUser) return;

    const draft = subscriptionDrafts[subscriptionId];
    if (!draft) return;

    const feeValue = Number(draft.fee_value);
    const minProfitThreshold = Number(draft.min_profit_threshold);
    if (Number.isNaN(feeValue) || Number.isNaN(minProfitThreshold) || feeValue < 0 || minProfitThreshold < 0) {
      toast.error("Fee value and minimum threshold must be valid non-negative numbers");
      return;
    }

    setSavingSubscriptionId(subscriptionId);
    try {
      await api.patch(`/admin/users/${selectedUser.id}/subscriptions/${subscriptionId}/billing`, {
        fee_type: draft.fee_type,
        fee_value: feeValue,
        min_profit_threshold: minProfitThreshold,
        next_billing_date: draft.next_billing_date || null,
      });
      toast.success("Subscription billing updated");
      await fetchUserDetail(selectedUser.id);
    } catch (error: any) {
      console.error("Error updating subscription billing:", error);
      toast.error(error?.message || "Failed to update subscription billing");
    } finally {
      setSavingSubscriptionId(null);
    }
  };

  const skipInvoice = async (invoiceId: string) => {
    if (!selectedUser) return;

    setSkippingInvoiceId(invoiceId);
    try {
      await api.patch(`/admin/users/${selectedUser.id}/invoices/${invoiceId}/skip`);
      toast.success("Invoice skipped");
      await fetchUserDetail(selectedUser.id);
    } catch (error: any) {
      console.error("Error skipping invoice:", error);
      toast.error(error?.message || "Failed to skip invoice");
    } finally {
      setSkippingInvoiceId(null);
    }
  };

  const filteredUsers = users.filter((user) =>
    `${user.username} ${user.email} ${user.role} ${user.status}`.toLowerCase().includes(searchTerm.toLowerCase())
  );

  const getStatusBadge = (status: AdminUser["status"]) => {
    switch (status) {
      case "active":
        return <Badge className="bg-emerald-100 text-emerald-700">Active</Badge>;
      case "banned":
        return <Badge className="bg-rose-100 text-rose-700">Banned</Badge>;
      case "pending":
        return <Badge className="bg-amber-100 text-amber-700">Pending</Badge>;
      default:
        return <Badge variant="secondary">{status}</Badge>;
    }
  };

  const getRoleBadge = (role: AdminUser["role"]) => {
    if (role === "admin") {
      return <Badge className="bg-blue-100 text-blue-700">Admin</Badge>;
    }
    return <Badge variant="outline">User</Badge>;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-48">
        <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">User Management</h2>
        <div className="relative w-64">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-muted-foreground" />
          <Input
            placeholder="Search users..."
            value={searchTerm}
            onChange={(event) => setSearchTerm(event.target.value)}
            className="pl-9"
          />
        </div>
      </div>

      <div className="glass-card overflow-hidden">
        <Table>
          <TableHeader>
            <TableRow>
              <TableHead>Username</TableHead>
              <TableHead>Email</TableHead>
              <TableHead>Role</TableHead>
              <TableHead>Status</TableHead>
              <TableHead>Joined</TableHead>
              <TableHead className="w-12"></TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {filteredUsers.map((user) => (
              <TableRow key={user.id}>
                <TableCell className="font-medium">{user.username}</TableCell>
                <TableCell>{user.email}</TableCell>
                <TableCell>{getRoleBadge(user.role)}</TableCell>
                <TableCell>{getStatusBadge(user.status)}</TableCell>
                <TableCell className="text-muted-foreground">{formatDate(user.created_at)}</TableCell>
                <TableCell>
                  <DropdownMenu>
                    <DropdownMenuTrigger asChild>
                      <Button variant="ghost" size="icon">
                        <MoreVertical className="w-4 h-4" />
                      </Button>
                    </DropdownMenuTrigger>
                    <DropdownMenuContent align="end" className="w-44">
                      <DropdownMenuItem onClick={() => openUserDetails(user)}>
                        <Eye className="w-4 h-4 mr-2" />
                        View Details
                      </DropdownMenuItem>

                      <DropdownMenuSeparator />

                      <DropdownMenuItem
                        disabled={updatingUserId === user.id || user.status === "active"}
                        onClick={() => updateUserStatus(user.id, "active")}
                      >
                        <UserCheck className="w-4 h-4 mr-2" />
                        Set Active
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        disabled={updatingUserId === user.id || user.status === "pending"}
                        onClick={() => updateUserStatus(user.id, "pending")}
                      >
                        <AlertCircle className="w-4 h-4 mr-2" />
                        Set Pending
                      </DropdownMenuItem>
                      <DropdownMenuItem
                        disabled={updatingUserId === user.id || user.status === "banned"}
                        onClick={() => updateUserStatus(user.id, "banned")}
                      >
                        <UserX className="w-4 h-4 mr-2" />
                        Ban User
                      </DropdownMenuItem>

                      <DropdownMenuSeparator />

                      {user.role === "admin" ? (
                        <DropdownMenuItem
                          disabled={updatingUserId === user.id}
                          onClick={() => updateUserRole(user.id, "user")}
                        >
                          <Shield className="w-4 h-4 mr-2" />
                          Set as User
                        </DropdownMenuItem>
                      ) : (
                        <DropdownMenuItem
                          disabled={updatingUserId === user.id}
                          onClick={() => updateUserRole(user.id, "admin")}
                        >
                          <Shield className="w-4 h-4 mr-2" />
                          Promote Admin
                        </DropdownMenuItem>
                      )}
                    </DropdownMenuContent>
                  </DropdownMenu>
                </TableCell>
              </TableRow>
            ))}
          </TableBody>
        </Table>
      </div>

      <Dialog
        open={showUserDialog}
        onOpenChange={(open) => {
          setShowUserDialog(open);
          if (!open) {
            setSelectedUser(null);
            setUserDetail(null);
            setSubscriptionDrafts({});
            setSavingSubscriptionId(null);
            setSkippingInvoiceId(null);
          }
        }}
      >
        <DialogContent className="max-w-4xl max-h-[90vh] overflow-hidden p-0 [&>button]:text-white [&>button]:opacity-90">
          <DialogHeader className="sr-only">
            <DialogTitle>User Details</DialogTitle>
            <DialogDescription>Admin user management detail modal</DialogDescription>
          </DialogHeader>

          {selectedUser && (
            <div className="bg-gradient-to-r from-sky-600 via-blue-600 to-blue-700 px-6 py-5 text-white">
              <div className="flex items-center justify-between gap-4">
                <div>
                  <p className="text-xl font-semibold leading-tight">{selectedUser.username}</p>
                  <p className="text-sm text-blue-100">{selectedUser.email}</p>
                  <div className="mt-2 flex items-center gap-2">
                    <Badge className="bg-white/15 text-white">{selectedUser.role}</Badge>
                    <Badge className="bg-white/15 text-white">{selectedUser.status}</Badge>
                  </div>
                </div>
                <Button
                  size="sm"
                  variant="secondary"
                  className="bg-white/15 text-white hover:bg-white/25"
                  onClick={() => fetchUserDetail(selectedUser.id)}
                  disabled={detailLoading}
                >
                  <RefreshCw className="mr-2 w-4 h-4" />
                  Refresh
                </Button>
              </div>

              <div className="mt-4 grid gap-2 md:grid-cols-4">
                <Button
                  size="sm"
                  variant="ghost"
                  className="justify-start bg-white/10 text-white hover:bg-white/20"
                  onClick={() => updateUserStatus(selectedUser.id, "active")}
                  disabled={updatingUserId === selectedUser.id}
                >
                  <UserCheck className="mr-2 w-4 h-4" />
                  Set Active
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="justify-start bg-white/10 text-white hover:bg-white/20"
                  onClick={() => updateUserStatus(selectedUser.id, "pending")}
                  disabled={updatingUserId === selectedUser.id}
                >
                  <AlertCircle className="mr-2 w-4 h-4" />
                  Set Pending
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="justify-start bg-white/10 text-white hover:bg-white/20"
                  onClick={() => updateUserStatus(selectedUser.id, "banned")}
                  disabled={updatingUserId === selectedUser.id}
                >
                  <UserX className="mr-2 w-4 h-4" />
                  Ban User
                </Button>
                <Button
                  size="sm"
                  variant="ghost"
                  className="justify-start bg-white/10 text-white hover:bg-white/20"
                  onClick={() => updateUserRole(selectedUser.id, selectedUser.role === "admin" ? "user" : "admin")}
                  disabled={updatingUserId === selectedUser.id}
                >
                  <Shield className="mr-2 w-4 h-4" />
                  {selectedUser.role === "admin" ? "Set as User" : "Promote Admin"}
                </Button>
              </div>
            </div>
          )}

          <div className="p-6 pt-4">
            {detailLoading ? (
              <div className="flex h-40 items-center justify-center">
                <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
              </div>
            ) : userDetail ? (
              <Tabs defaultValue="overview" className="space-y-4">
                <TabsList className="grid h-auto w-full grid-cols-2 gap-1 p-1 md:grid-cols-4">
                  <TabsTrigger value="overview">Overview</TabsTrigger>
                  <TabsTrigger value="accounts">Accounts</TabsTrigger>
                  <TabsTrigger value="subscriptions">Subscriptions</TabsTrigger>
                  <TabsTrigger value="billing">Billing</TabsTrigger>
                </TabsList>

                <div className="max-h-[54vh] overflow-y-auto pr-1">
                  <TabsContent value="overview" className="space-y-4">
                    <div className="grid gap-3 md:grid-cols-3">
                      <div className="rounded-lg border border-border bg-secondary/30 p-4">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                          <Landmark className="w-4 h-4" />
                          Total Accounts
                        </div>
                        <p className="mt-2 text-2xl font-semibold">{userDetail.total_accounts}</p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/30 p-4">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                          <Wallet className="w-4 h-4" />
                          Total Balance
                        </div>
                        <p className="mt-2 text-2xl font-semibold">{formatCurrency(userDetail.total_balance)}</p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/30 p-4">
                        <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-muted-foreground">
                          <AlertCircle className="w-4 h-4" />
                          Pending Bills
                        </div>
                        <p className="mt-2 text-2xl font-semibold">{userDetail.pending_bills}</p>
                      </div>
                    </div>

                    <div className="grid gap-3 md:grid-cols-2">
                      <div className="rounded-lg border border-border bg-secondary/20 p-4">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Account Summary</p>
                        <p className="mt-2 text-sm">
                          Running bots:{" "}
                          <span className="font-semibold">
                            {userDetail.trading_accounts.reduce((total, account) => total + account.running_bots, 0)}
                          </span>
                        </p>
                        <p className="mt-1 text-sm">
                          Active bots:{" "}
                          <span className="font-semibold">
                            {userDetail.trading_accounts.reduce((total, account) => total + account.active_bots, 0)}
                          </span>
                        </p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/20 p-4">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Profile</p>
                        <p className="mt-2 text-sm">
                          Joined: <span className="font-semibold">{formatDate(userDetail.created_at)}</span>
                        </p>
                        <p className="mt-1 text-sm">
                          Onboarding:{" "}
                          <span className="font-semibold">
                            {userDetail.is_onboarding_completed ? "Completed" : "Pending"}
                          </span>
                        </p>
                      </div>
                    </div>
                  </TabsContent>

                  <TabsContent value="accounts" className="space-y-3">
                    <div className="flex items-center justify-between border-b border-border pb-2">
                      <h3 className="text-base font-semibold">Trading Accounts</h3>
                      <span className="text-xs text-muted-foreground">{userDetail.trading_accounts.length} accounts</span>
                    </div>

                    {userDetail.trading_accounts.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                        No trading accounts for this user.
                      </div>
                    ) : (
                      <div className="grid gap-3 md:grid-cols-2">
                        {userDetail.trading_accounts.map((account) => {
                          const accountHasStarting = account.bots.some(
                            (bot) => normalizeBotStatus(bot.container_status, bot.is_active) === "starting"
                          );
                          const accountRunningOrStarting = account.bots.filter((bot) => {
                            const status = normalizeBotStatus(bot.container_status, bot.is_active);
                            return status === "running" || status === "starting";
                          }).length;
                          const accountBadgeClass = accountHasStarting
                            ? "bg-amber-100 text-amber-700"
                            : accountRunningOrStarting > 0
                              ? "bg-emerald-100 text-emerald-700"
                              : "bg-zinc-100 text-zinc-700";
                          const accountBadgeLabel = accountHasStarting
                            ? "Starting"
                            : accountRunningOrStarting > 0
                              ? "Running"
                              : "Stopped";

                          return (
                          <div key={account.id} className="rounded-lg border border-border bg-secondary/20 p-4">
                            <div className="flex items-center justify-between">
                              <div>
                                <p className="font-semibold">#{account.mt5_login_id || shortId(account.id)}</p>
                                <p className="text-xs text-muted-foreground">{account.broker_name || "Unknown Broker"} {account.server_name ? `· ${account.server_name}` : ""}</p>
                              </div>
                              <Badge className={accountBadgeClass}>
                                {accountBadgeLabel}
                              </Badge>
                            </div>

                            <div className="mt-3 grid grid-cols-2 gap-2 text-sm">
                              <div className="rounded-md bg-background p-2">
                                <p className="text-xs text-muted-foreground">Balance</p>
                                <p className="font-semibold text-emerald-600">{formatCurrency(account.balance)}</p>
                              </div>
                              <div className="rounded-md bg-background p-2">
                                <p className="text-xs text-muted-foreground">Bot Status</p>
                                <p className="font-semibold">{account.active_bots} active / {accountRunningOrStarting} running</p>
                              </div>
                            </div>

                            {account.bots.length > 0 ? (
                              <div className="mt-3 space-y-2">
                                {account.bots.map((bot) => {
                                  const botStatus = normalizeBotStatus(bot.container_status, bot.is_active);
                                  const isBotStarting = botStatus === "starting";
                                  const isBotRunning = botStatus === "running";
                                  const badgeClass = isBotStarting
                                    ? "border-amber-200 text-amber-700"
                                    : botStatus === "error"
                                      ? "border-rose-200 text-rose-700"
                                      : undefined;

                                  return (
                                  <div key={bot.id} className="flex items-center justify-between rounded-md border border-border bg-background p-2">
                                    <div className="min-w-0">
                                      <p className="truncate text-sm font-medium">{bot.label || `Bot #${bot.bot_instance_id}`}</p>
                                      <p className="text-xs text-muted-foreground">#{bot.bot_instance_id} · {bot.symbol || "-"} {bot.timeframe || ""}</p>
                                    </div>
                                    <div className="flex items-center gap-2">
                                      <Badge variant="outline" className={`capitalize ${badgeClass || ""}`}>
                                        {botStatus}
                                      </Badge>
                                      <Button
                                        size="sm"
                                        variant="outline"
                                        disabled={updatingBotId === bot.id || isBotStarting}
                                        onClick={() => updateBotStatus(bot.id, isBotRunning ? "stopped" : "running")}
                                      >
                                        {isBotStarting ? (
                                          <RefreshCw className="mr-1 w-3 h-3 animate-spin" />
                                        ) : (
                                          <Activity className="mr-1 w-3 h-3" />
                                        )}
                                        {isBotRunning ? "Stop" : isBotStarting ? "Starting..." : "Run"}
                                      </Button>
                                    </div>
                                  </div>
                                )})}
                              </div>
                            ) : (
                              <p className="mt-3 text-xs text-muted-foreground">No bots configured for this account.</p>
                            )}
                          </div>
                        )})}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="subscriptions" className="space-y-3">
                    <div className="flex items-center justify-between border-b border-border pb-2">
                      <h3 className="text-base font-semibold">Subscription Settings</h3>
                      <span className="text-xs text-muted-foreground">
                        {userDetail.subscriptions.length} subscriptions
                      </span>
                    </div>

                    {userDetail.subscriptions.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
                        No subscriptions for this user.
                      </div>
                    ) : (
                      <div className="space-y-3">
                        {userDetail.subscriptions.map((subscription) => {
                          const draft = subscriptionDrafts[subscription.id] || {
                            fee_type: subscription.fee_type,
                            fee_value: String(subscription.fee_value),
                            min_profit_threshold: String(subscription.min_profit_threshold),
                            next_billing_date: subscription.next_billing_date ?? "",
                          };

                          return (
                            <div key={subscription.id} className="rounded-lg border border-border bg-secondary/20 p-4">
                              <div className="mb-3 flex items-center justify-between">
                                <div>
                                  <p className="text-sm font-semibold">{shortId(subscription.id)}</p>
                                  <p className="text-xs text-muted-foreground">
                                    Next billing: {formatDate(subscription.next_billing_date)}
                                  </p>
                                </div>
                                <Badge variant="outline" className="capitalize">
                                  {subscription.status}
                                </Badge>
                              </div>

                              <div className="grid gap-3 md:grid-cols-4">
                                <div>
                                  <p className="text-xs text-muted-foreground mb-1">Fee Type</p>
                                  <select
                                    className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                                    value={draft.fee_type}
                                    onChange={(event) =>
                                      updateSubscriptionDraft(
                                        subscription.id,
                                        "fee_type",
                                        event.target.value as "percentage" | "fixed"
                                      )
                                    }
                                  >
                                    <option value="percentage">Percentage</option>
                                    <option value="fixed">Fixed</option>
                                  </select>
                                </div>

                                <div>
                                  <p className="text-xs text-muted-foreground mb-1">
                                    Fee Value ({draft.fee_type === "percentage" ? "%" : "$"})
                                  </p>
                                  <Input
                                    type="number"
                                    min={0}
                                    value={draft.fee_value}
                                    onChange={(event) =>
                                      updateSubscriptionDraft(subscription.id, "fee_value", event.target.value)
                                    }
                                  />
                                </div>

                                <div>
                                  <p className="text-xs text-muted-foreground mb-1">Min Profit Threshold ($)</p>
                                  <Input
                                    type="number"
                                    min={0}
                                    value={draft.min_profit_threshold}
                                    onChange={(event) =>
                                      updateSubscriptionDraft(
                                        subscription.id,
                                        "min_profit_threshold",
                                        event.target.value
                                      )
                                    }
                                  />
                                </div>

                                <div>
                                  <p className="text-xs text-muted-foreground mb-1">Next Billing Date</p>
                                  <Input
                                    type="date"
                                    value={draft.next_billing_date}
                                    onChange={(event) =>
                                      updateSubscriptionDraft(
                                        subscription.id,
                                        "next_billing_date",
                                        event.target.value
                                      )
                                    }
                                  />
                                </div>
                              </div>

                              <div className="mt-3 flex justify-end">
                                <Button
                                  size="sm"
                                  onClick={() => saveSubscriptionBilling(subscription.id)}
                                  disabled={savingSubscriptionId === subscription.id}
                                >
                                  <Save className="mr-2 w-4 h-4" />
                                  {savingSubscriptionId === subscription.id ? "Saving..." : "Save Billing"}
                                </Button>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                    )}
                  </TabsContent>

                  <TabsContent value="billing" className="space-y-3">
                    <div className="flex items-center justify-between border-b border-border pb-2">
                      <h3 className="text-base font-semibold">Billing History</h3>
                      <span className="text-xs text-muted-foreground">recent invoices</span>
                    </div>

                    <div className="grid gap-2 md:grid-cols-2">
                      <div className="rounded-lg border border-rose-100 bg-rose-50 px-4 py-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-rose-600">Pending ({userDetail.billing.pending_count})</p>
                        <p className="mt-1 font-semibold text-rose-700">{formatCurrency(userDetail.billing.pending_amount)}</p>
                      </div>
                      <div className="rounded-lg border border-zinc-200 bg-zinc-50 px-4 py-3">
                        <p className="text-xs font-medium uppercase tracking-wide text-zinc-600">Paid ({userDetail.billing.paid_count})</p>
                        <p className="mt-1 font-semibold text-zinc-800">{formatCurrency(userDetail.billing.paid_amount)}</p>
                      </div>
                    </div>

                    {userDetail.billing.recent_invoices.length === 0 ? (
                      <div className="rounded-lg border border-dashed border-border p-4 text-center text-sm text-muted-foreground">
                        No invoice history found.
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {userDetail.billing.recent_invoices.map((invoice) => (
                          <div key={invoice.id} className="flex items-center justify-between rounded-md border border-border p-3">
                            <div>
                              <p className="text-sm font-medium">{shortId(invoice.id)}</p>
                              <p className="text-xs text-muted-foreground">
                                {invoice.billing_start_date || "-"} to {invoice.billing_end_date || "-"}
                              </p>
                            </div>
                            <div className="flex items-center gap-3">
                              <Badge variant="outline" className="capitalize">{invoice.status || "unknown"}</Badge>
                              <p className="text-sm font-semibold">{formatCurrency(invoice.amount)}</p>
                              {invoice.status !== "paid" && invoice.status !== "skipped" && (
                                <Button
                                  size="sm"
                                  variant="outline"
                                  disabled={skippingInvoiceId === invoice.id}
                                  onClick={() => skipInvoice(invoice.id)}
                                >
                                  {skippingInvoiceId === invoice.id ? "Skipping..." : "Skip"}
                                </Button>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    )}
                  </TabsContent>
                </div>
              </Tabs>
            ) : (
              <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                User details unavailable.
              </div>
            )}
          </div>

          <DialogFooter className="border-t border-border px-6 py-4">
            <Button variant="outline" onClick={() => setShowUserDialog(false)}>Close</Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </div>
  );
}
