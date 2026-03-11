import { useEffect, useState } from "react";
import { DollarSign, Percent, Settings, Save, PlayCircle } from "lucide-react";
import { toast } from "sonner";

import { api } from "@/lib/api";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group";
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from "@/components/ui/table";
import { cn } from "@/lib/utils";

interface AdminBillingConfig {
  config_id: number | null;
  default_fee_type: "percentage" | "fixed";
  default_collection_mode: "automatic" | "manual";
  default_fee_value: number;
  default_min_threshold: number;
  default_next_billing_date: string | null;
  updated_at: string | null;
}

interface AdminSubscriptionItem {
  id: string;
  user_id: string;
  user_email: string | null;
  status: "active" | "past_due" | "canceled";
  collection_mode: "automatic" | "manual";
  fee_type: "percentage" | "fixed";
  fee_value: number;
  min_profit_threshold: number;
  next_billing_date: string | null;
  created_at: string | null;
}

interface AdminSubscriptionManagementResponse {
  billing_config: AdminBillingConfig;
  subscriptions: AdminSubscriptionItem[];
}

interface ProcessDueBillingResponse {
  processed_subscriptions: number;
  created_invoices: number;
  paid_invoices: number;
  pending_invoices: number;
  skipped_invoices: number;
  failed_invoices: number;
}

interface SubscriptionDraft {
  collection_mode: "automatic" | "manual";
  fee_type: "percentage" | "fixed";
  fee_value: string;
  min_profit_threshold: string;
  next_billing_date: string;
}

interface AdminInvoiceDetail {
  id: string;
  billing_start_date: string | null;
  billing_end_date: string | null;
  total_period_profit: number;
  calculated_fee: number;
  status: string | null;
  payment_method_used: string | null;
  stripe_payment_intent_id: string | null;
  stripe_charge_id: string | null;
  stripe_balance_txn_id: string | null;
  processor_request_id: string | null;
  payment_breakdown: Record<string, unknown> | null;
  payment_method_details: Record<string, unknown> | null;
  payment_error_details: Record<string, unknown> | null;
  paid_at: string | null;
  created_at: string | null;
}

interface AdminSubscriptionInvoiceListResponse {
  subscription_id: string;
  user_id: string;
  user_email: string | null;
  invoices: AdminInvoiceDetail[];
}

function formatCurrencyAmount(amount: number | null | undefined, currency: string | null | undefined = "USD") {
  if (amount === null || amount === undefined || Number.isNaN(amount)) return "-";
  const currencyCode = (currency || "USD").toUpperCase();
  try {
    return new Intl.NumberFormat("en-US", {
      style: "currency",
      currency: currencyCode,
      minimumFractionDigits: 2,
    }).format(amount);
  } catch {
    return `${amount.toFixed(2)} ${currencyCode}`;
  }
}

function formatDateTime(value: string | null | undefined) {
  if (!value) return "-";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "-";
  return date.toLocaleString();
}

function formatDate(value: string | null | undefined) {
  if (!value) return "-";
  const date = new Date(value.includes("T") ? value : `${value}T00:00:00`);
  if (Number.isNaN(date.getTime())) return value;
  return date.toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

function shortId(value: string | null | undefined) {
  const text = String(value || "").trim();
  if (!text) return "-";
  return text.length <= 12 ? text : `${text.slice(0, 8)}...${text.slice(-4)}`;
}

function asString(value: unknown) {
  return typeof value === "string" && value.trim() ? value : null;
}

function asNumber(value: unknown) {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asRecord(value: unknown) {
  return value && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : null;
}

function asArray(value: unknown) {
  return Array.isArray(value) ? value : [];
}

function formatAddress(value: unknown) {
  const address = asRecord(value);
  if (!address) return "-";
  const parts = [
    asString(address.line1),
    asString(address.line2),
    asString(address.city),
    asString(address.state),
    asString(address.postal_code),
    asString(address.country),
  ].filter(Boolean);
  return parts.length > 0 ? parts.join(", ") : "-";
}

export function AdminSubscriptions() {
  const [subscriptions, setSubscriptions] = useState<AdminSubscriptionItem[]>([]);
  const [billingConfig, setBillingConfig] = useState<AdminBillingConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [savingConfig, setSavingConfig] = useState(false);
  const [updatingSubId, setUpdatingSubId] = useState<string | null>(null);
  const [savingBillingSubId, setSavingBillingSubId] = useState<string | null>(null);
  const [runningDueBilling, setRunningDueBilling] = useState(false);
  const [subscriptionDrafts, setSubscriptionDrafts] = useState<Record<string, SubscriptionDraft>>({});
  const [billingDialogOpen, setBillingDialogOpen] = useState(false);
  const [billingDialogLoading, setBillingDialogLoading] = useState(false);
  const [billingDialogData, setBillingDialogData] = useState<AdminSubscriptionInvoiceListResponse | null>(null);
  const [billingDialogSub, setBillingDialogSub] = useState<AdminSubscriptionItem | null>(null);
  const [skippingInvoiceId, setSkippingInvoiceId] = useState<string | null>(null);
  const [configForm, setConfigForm] = useState({
    fee_type: "percentage" as "percentage" | "fixed",
    collection_mode: "automatic" as "automatic" | "manual",
    fee_value: 20,
    min_threshold: 0,
    next_billing_date: "",
  });

  useEffect(() => {
    fetchData();
  }, []);

  const fetchData = async () => {
    try {
      setLoading(true);
      const { data } = await api.get<AdminSubscriptionManagementResponse>("/subscription/admin/management");
      const rows = data.subscriptions || [];
      setSubscriptions(rows);
      setSubscriptionDrafts(
        rows.reduce<Record<string, SubscriptionDraft>>((acc, row) => {
          acc[row.id] = {
            collection_mode: row.collection_mode,
            fee_type: row.fee_type,
            fee_value: String(row.fee_value ?? 0),
            min_profit_threshold: String(row.min_profit_threshold ?? 0),
            next_billing_date: row.next_billing_date ?? "",
          };
          return acc;
        }, {})
      );
      setBillingConfig(data.billing_config || null);
      setConfigForm({
        fee_type: (data.billing_config?.default_fee_type ?? "percentage") as "percentage" | "fixed",
        collection_mode: (data.billing_config?.default_collection_mode ?? "automatic") as "automatic" | "manual",
        fee_value: data.billing_config?.default_fee_value ?? 20,
        min_threshold: data.billing_config?.default_min_threshold ?? 0,
        next_billing_date: data.billing_config?.default_next_billing_date ?? "",
      });
    } catch (error: any) {
      console.error("Error fetching admin subscription data:", error);
      toast.error(error?.message || "Failed to load subscription data");
    } finally {
      setLoading(false);
    }
  };

  const saveBillingConfig = async () => {
    setSavingConfig(true);
    try {
      const { data } = await api.put<AdminBillingConfig>("/subscription/admin/config", {
        default_fee_type: configForm.fee_type,
        default_collection_mode: configForm.collection_mode,
        default_fee_value: configForm.fee_value,
        default_min_threshold: configForm.min_threshold,
        default_next_billing_date: configForm.next_billing_date || null,
      });
      setBillingConfig(data);
      toast.success("Billing config saved");
      await fetchData();
    } catch (error: any) {
      console.error("Error saving billing config:", error);
      toast.error(error?.message || "Failed to save billing config");
    } finally {
      setSavingConfig(false);
    }
  };

  const updateSubscriptionStatus = async (
    subId: string,
    status: "active" | "past_due" | "canceled"
  ) => {
    setUpdatingSubId(subId);
    try {
      await api.patch(`/subscription/admin/subscriptions/${subId}/status`, { status });
      toast.success(`Subscription updated to ${status}`);
      await fetchData();
    } catch (error: any) {
      console.error("Error updating subscription:", error);
      toast.error(error?.message || "Failed to update subscription");
    } finally {
      setUpdatingSubId(null);
    }
  };

  const updateSubscriptionDraft = (
    subscriptionId: string,
    key: keyof SubscriptionDraft,
    value: string
  ) => {
    setSubscriptionDrafts((prev) => {
      const current = prev[subscriptionId] || {
        collection_mode: "automatic",
        fee_type: "percentage",
        fee_value: "0",
        min_profit_threshold: "0",
        next_billing_date: "",
      };
      return {
        ...prev,
        [subscriptionId]: {
          ...current,
          [key]: value,
        },
      };
    });
  };

  const saveSubscriptionBilling = async (sub: AdminSubscriptionItem) => {
    const draft = subscriptionDrafts[sub.id];
    if (!draft) return;

    setSavingBillingSubId(sub.id);
    try {
      await api.patch(`/admin/users/${sub.user_id}/subscriptions/${sub.id}/billing`, {
        collection_mode: draft.collection_mode,
        fee_type: draft.fee_type,
        fee_value: parseFloat(draft.fee_value) || 0,
        min_profit_threshold: parseFloat(draft.min_profit_threshold) || 0,
        next_billing_date: draft.next_billing_date || null,
      });
      toast.success("Subscription billing updated");
      await fetchData();
    } catch (error: any) {
      console.error("Error updating subscription billing:", error);
      toast.error(error?.message || "Failed to update subscription billing");
    } finally {
      setSavingBillingSubId(null);
    }
  };

  const runDueBillingNow = async () => {
    setRunningDueBilling(true);
    try {
      const { data } = await api.post<ProcessDueBillingResponse>("/subscription/admin/process-due");
      toast.success(
        `Processed ${data.processed_subscriptions} subscriptions, created ${data.created_invoices} invoices`
      );
      await fetchData();
    } catch (error: any) {
      console.error("Error processing due billing:", error);
      toast.error(error?.message || "Failed to process due billing");
    } finally {
      setRunningDueBilling(false);
    }
  };

  const fetchSubscriptionInvoices = async (subscriptionId: string) => {
    const { data } = await api.get<AdminSubscriptionInvoiceListResponse>(
      `/subscription/admin/subscriptions/${subscriptionId}/invoices`
    );
    setBillingDialogData(data);
    return data;
  };

  const openBillingDialog = async (sub: AdminSubscriptionItem) => {
    setBillingDialogSub(sub);
    setBillingDialogOpen(true);
    setBillingDialogLoading(true);
    try {
      await fetchSubscriptionInvoices(sub.id);
    } catch (error: any) {
      console.error("Error fetching subscription invoices:", error);
      toast.error(error?.message || "Failed to load invoice details");
    } finally {
      setBillingDialogLoading(false);
    }
  };

  const skipInvoice = async (invoiceId: string) => {
    if (!billingDialogData) return;

    setSkippingInvoiceId(invoiceId);
    try {
      await api.patch(`/admin/users/${billingDialogData.user_id}/invoices/${invoiceId}/skip`);
      toast.success("Invoice skipped");
      if (billingDialogSub) {
        await fetchSubscriptionInvoices(billingDialogSub.id);
      }
      await fetchData();
    } catch (error: any) {
      console.error("Error skipping invoice:", error);
      toast.error(error?.message || "Failed to skip invoice");
    } finally {
      setSkippingInvoiceId(null);
    }
  };

  const getStatusBadge = (status: AdminSubscriptionItem["status"]) => {
    switch (status) {
      case "active":
        return <Badge className="bg-success/10 text-success">Active</Badge>;
      case "past_due":
        return <Badge className="bg-warning/10 text-warning">Past Due</Badge>;
      case "canceled":
        return <Badge className="bg-destructive/10 text-destructive">Canceled</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
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
        <div className="mb-4 flex items-center justify-between gap-3">
          <div className="flex items-center gap-2">
            <Settings className="w-5 h-5 text-primary" />
            <h3 className="font-semibold">Default Billing Configuration</h3>
          </div>
          <Button variant="outline" onClick={runDueBillingNow} disabled={runningDueBilling}>
            <PlayCircle className="mr-2 h-4 w-4" />
            {runningDueBilling ? "Running..." : "Run Due Billing Now"}
          </Button>
        </div>

        <div className="grid md:grid-cols-5 gap-6">
          <div>
            <Label className="mb-3 block">Fee Type</Label>
            <RadioGroup
              value={configForm.fee_type}
              onValueChange={(value) => setConfigForm({ ...configForm, fee_type: value as "percentage" | "fixed" })}
              className="flex gap-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="percentage" id="percentage" />
                <Label htmlFor="percentage" className="flex items-center gap-1">
                  <Percent className="w-4 h-4" />
                  Percentage
                </Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="fixed" id="fixed" />
                <Label htmlFor="fixed" className="flex items-center gap-1">
                  <DollarSign className="w-4 h-4" />
                  Fixed
                </Label>
              </div>
            </RadioGroup>
          </div>

          <div>
            <Label className="mb-3 block">Collection Mode</Label>
            <RadioGroup
              value={configForm.collection_mode}
              onValueChange={(value) => setConfigForm({ ...configForm, collection_mode: value as "automatic" | "manual" })}
              className="flex gap-4"
            >
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="automatic" id="automatic" />
                <Label htmlFor="automatic">Automatic</Label>
              </div>
              <div className="flex items-center space-x-2">
                <RadioGroupItem value="manual" id="manual" />
                <Label htmlFor="manual">Manual</Label>
              </div>
            </RadioGroup>
          </div>

          <div>
            <Label htmlFor="fee_value">
              Fee Value ({configForm.fee_type === "percentage" ? "%" : "$"})
            </Label>
            <Input
              id="fee_value"
              type="number"
              min={0}
              value={configForm.fee_value}
              onChange={(event) =>
                setConfigForm({ ...configForm, fee_value: parseFloat(event.target.value) || 0 })
              }
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="min_threshold">Minimum Profit Threshold ($)</Label>
            <Input
              id="min_threshold"
              type="number"
              min={0}
              value={configForm.min_threshold}
              onChange={(event) =>
                setConfigForm({ ...configForm, min_threshold: parseFloat(event.target.value) || 0 })
              }
              className="mt-2"
            />
          </div>

          <div>
            <Label htmlFor="default_next_billing_date">Default Next Billing Date</Label>
            <Input
              id="default_next_billing_date"
              type="date"
              value={configForm.next_billing_date}
              onChange={(event) =>
                setConfigForm({ ...configForm, next_billing_date: event.target.value })
              }
              className="mt-2"
            />
          </div>
        </div>

        <div className="mt-4 text-xs text-muted-foreground">
          {billingConfig?.updated_at
            ? `Last updated: ${new Date(billingConfig.updated_at).toLocaleString()}`
            : "Using system defaults"}
        </div>

        <Button onClick={saveBillingConfig} disabled={savingConfig} className="mt-4">
          <Save className="w-4 h-4 mr-2" />
          {savingConfig ? "Saving..." : "Save Configuration"}
        </Button>
      </div>

      <div className="glass-card overflow-hidden">
        <div className="p-4 border-b border-border">
          <h3 className="font-semibold">User Subscriptions</h3>
        </div>
        <div className="overflow-x-auto">
          <Table>
            <TableHeader>
              <TableRow>
                <TableHead>User</TableHead>
                <TableHead>Status</TableHead>
                <TableHead>Collection</TableHead>
                <TableHead>Fee Type</TableHead>
                <TableHead>Fee Value</TableHead>
                <TableHead>Threshold</TableHead>
                <TableHead>Next Billing</TableHead>
                <TableHead>Actions</TableHead>
              </TableRow>
            </TableHeader>
            <TableBody>
              {subscriptions.map((sub) => {
                const draft = subscriptionDrafts[sub.id] || {
                  collection_mode: sub.collection_mode,
                  fee_type: sub.fee_type,
                  fee_value: String(sub.fee_value ?? 0),
                  min_profit_threshold: String(sub.min_profit_threshold ?? 0),
                  next_billing_date: sub.next_billing_date ?? "",
                };

                return (
                  <TableRow key={sub.id} className="align-top">
                    <TableCell className="min-w-[260px] py-5">
                      <div className="space-y-1">
                        <p className="font-medium leading-6">{sub.user_email || "Unknown"}</p>
                        <p className="text-xs text-muted-foreground">
                          Subscription {sub.id.slice(0, 8)}
                        </p>
                      </div>
                    </TableCell>
                    <TableCell className="py-5">{getStatusBadge(sub.status)}</TableCell>
                    <TableCell className="min-w-[150px] py-5">
                      <select
                        className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                        value={draft.collection_mode}
                        onChange={(event) =>
                          updateSubscriptionDraft(
                            sub.id,
                            "collection_mode",
                            event.target.value
                          )
                        }
                      >
                        <option value="automatic">Automatic</option>
                        <option value="manual">Manual</option>
                      </select>
                    </TableCell>
                    <TableCell className="min-w-[150px] py-5">
                      <select
                        className="h-9 w-full rounded-md border border-input bg-background px-3 text-sm"
                        value={draft.fee_type}
                        onChange={(event) =>
                          updateSubscriptionDraft(
                            sub.id,
                            "fee_type",
                            event.target.value
                          )
                        }
                      >
                        <option value="percentage">Percentage</option>
                        <option value="fixed">Fixed</option>
                      </select>
                    </TableCell>
                    <TableCell className="min-w-[140px] py-5">
                      <Input
                        type="number"
                        min={0}
                        value={draft.fee_value}
                        onChange={(event) =>
                          updateSubscriptionDraft(sub.id, "fee_value", event.target.value)
                        }
                        className="h-9"
                      />
                    </TableCell>
                    <TableCell className="min-w-[140px] py-5">
                      <Input
                        type="number"
                        min={0}
                        value={draft.min_profit_threshold}
                        onChange={(event) =>
                          updateSubscriptionDraft(sub.id, "min_profit_threshold", event.target.value)
                        }
                        className="h-9"
                      />
                    </TableCell>
                    <TableCell className="min-w-[170px] py-5">
                      <Input
                        type="date"
                        value={draft.next_billing_date}
                        onChange={(event) =>
                          updateSubscriptionDraft(sub.id, "next_billing_date", event.target.value)
                        }
                        className="h-9"
                      />
                    </TableCell>
                    <TableCell className="min-w-[360px] py-5">
                      <div className="grid grid-cols-2 gap-2">
                        <Button
                          size="sm"
                          className="col-span-2"
                          disabled={savingBillingSubId === sub.id}
                          onClick={() => saveSubscriptionBilling(sub)}
                        >
                          <Save className="mr-2 h-4 w-4" />
                          {savingBillingSubId === sub.id ? "Saving..." : "Save Billing"}
                        </Button>
                        <Button
                          size="sm"
                          variant="secondary"
                          className="w-full"
                          onClick={() => openBillingDialog(sub)}
                        >
                          View Billing
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full"
                          disabled={updatingSubId === sub.id || sub.status === "active"}
                          onClick={() => updateSubscriptionStatus(sub.id, "active")}
                        >
                          Activate
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full"
                          disabled={updatingSubId === sub.id || sub.status === "past_due"}
                          onClick={() => updateSubscriptionStatus(sub.id, "past_due")}
                        >
                          Mark Past Due
                        </Button>
                        <Button
                          size="sm"
                          variant="outline"
                          className="w-full"
                          disabled={updatingSubId === sub.id || sub.status === "canceled"}
                          onClick={() => updateSubscriptionStatus(sub.id, "canceled")}
                        >
                          Cancel
                        </Button>
                      </div>
                    </TableCell>
                  </TableRow>
                );
              })}
            </TableBody>
          </Table>
        </div>
      </div>

      <Dialog
        open={billingDialogOpen}
        onOpenChange={(open) => {
          setBillingDialogOpen(open);
          if (!open) {
            setBillingDialogData(null);
            setBillingDialogSub(null);
            setSkippingInvoiceId(null);
          }
        }}
      >
        <DialogContent className="max-w-5xl max-h-[90vh] overflow-hidden">
          <DialogHeader>
            <DialogTitle>Billing Details</DialogTitle>
            <DialogDescription>
              {billingDialogSub?.user_email || billingDialogData?.user_email || "Unknown user"}
              {billingDialogSub?.id ? ` • Subscription ${shortId(billingDialogSub.id)}` : ""}
            </DialogDescription>
          </DialogHeader>

          <div className="max-h-[70vh] overflow-y-auto pr-1 space-y-4">
            {billingDialogLoading ? (
              <div className="flex items-center justify-center h-40">
                <div className="animate-spin w-6 h-6 border-2 border-primary border-t-transparent rounded-full" />
              </div>
            ) : !billingDialogData || billingDialogData.invoices.length === 0 ? (
              <div className="rounded-lg border border-dashed border-border p-6 text-center text-sm text-muted-foreground">
                No invoice history for this subscription.
              </div>
            ) : (
              billingDialogData.invoices.map((invoice) => {
                const breakdown = asRecord(invoice.payment_breakdown);
                const method = asRecord(invoice.payment_method_details);
                const error = asRecord(invoice.payment_error_details);
                const attempts = asArray(error?.attempts).length > 0 ? asArray(error?.attempts) : asArray(method?.attempts);
                const status = (invoice.status || "unknown").toLowerCase();

                return (
                  <div key={invoice.id} className="rounded-xl border border-border bg-background/80 p-4 space-y-4">
                    <div className="flex flex-col gap-3 md:flex-row md:items-start md:justify-between">
                      <div className="space-y-1">
                        <p className="text-sm font-semibold">{shortId(invoice.id)}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatDate(invoice.billing_start_date)} - {formatDate(invoice.billing_end_date)}
                        </p>
                        <p className="text-xs text-muted-foreground">
                          Created {formatDateTime(invoice.created_at)}
                          {invoice.paid_at ? ` • Paid ${formatDateTime(invoice.paid_at)}` : ""}
                        </p>
                      </div>
                      <div className="flex flex-wrap items-center gap-2">
                        <Badge
                          variant="outline"
                          className={cn(
                            "capitalize",
                            status === "paid" && "border-emerald-200 bg-emerald-50 text-emerald-700",
                            status === "pending" && "border-amber-200 bg-amber-50 text-amber-700",
                            status === "failed" && "border-rose-200 bg-rose-50 text-rose-700",
                            status === "skipped" && "border-zinc-200 bg-zinc-50 text-zinc-700"
                          )}
                        >
                          {status}
                        </Badge>
                        <span className="text-sm font-semibold">
                          {formatCurrencyAmount(invoice.calculated_fee, "USD")}
                        </span>
                        {status !== "paid" && status !== "skipped" && (
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

                    <div className="grid gap-3 md:grid-cols-4">
                      <div className="rounded-lg border border-border bg-secondary/20 p-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Net Profit</p>
                        <p className="mt-1 text-lg font-semibold">
                          {formatCurrencyAmount(invoice.total_period_profit, "USD")}
                        </p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/20 p-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Estimated Fee</p>
                        <p className="mt-1 text-lg font-semibold">
                          {formatCurrencyAmount(invoice.calculated_fee, "USD")}
                        </p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/20 p-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Request ID</p>
                        <p className="mt-1 text-sm font-medium break-all">{invoice.processor_request_id || "-"}</p>
                      </div>
                      <div className="rounded-lg border border-border bg-secondary/20 p-3">
                        <p className="text-xs uppercase tracking-wide text-muted-foreground">Payment Intent</p>
                        <p className="mt-1 text-sm font-medium break-all">{invoice.stripe_payment_intent_id || "-"}</p>
                      </div>
                    </div>

                    <div className="grid gap-4 lg:grid-cols-3">
                      <div className="rounded-lg border border-border p-4 space-y-2">
                        <h4 className="text-sm font-semibold">Payment Breakdown</h4>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Payment amount</span>
                            <span>
                              {formatCurrencyAmount(
                                asNumber(breakdown?.payment_amount),
                                asString(breakdown?.payment_currency) || "USD"
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Customer presentment</span>
                            <span>
                              {formatCurrencyAmount(
                                asNumber(breakdown?.presentment_amount),
                                asString(breakdown?.presentment_currency) || "USD"
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Settlement amount</span>
                            <span>
                              {formatCurrencyAmount(
                                asNumber(breakdown?.settlement_amount),
                                asString(breakdown?.settlement_currency) || "USD"
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Fees</span>
                            <span>
                              {formatCurrencyAmount(
                                asNumber(breakdown?.fee_amount),
                                asString(breakdown?.settlement_currency) || "USD"
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Net amount</span>
                            <span>
                              {formatCurrencyAmount(
                                asNumber(breakdown?.net_amount),
                                asString(breakdown?.settlement_currency) || "USD"
                              )}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Exchange rate</span>
                            <span>{asNumber(breakdown?.exchange_rate)?.toFixed(4) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Balance txn</span>
                            <span className="break-all text-right">{invoice.stripe_balance_txn_id || "-"}</span>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-lg border border-border p-4 space-y-2">
                        <h4 className="text-sm font-semibold">Payment Method</h4>
                        <div className="space-y-1 text-sm">
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Provider PM</span>
                            <span className="break-all text-right">{asString(method?.provider_method_id) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Card</span>
                            <span>
                              {asString(method?.brand)?.toUpperCase() || "CARD"} •••• {asString(method?.last4) || "----"}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Expires</span>
                            <span>
                              {asNumber(method?.exp_month) && asNumber(method?.exp_year)
                                ? `${String(asNumber(method?.exp_month)).padStart(2, "0")}/${String(asNumber(method?.exp_year)).slice(-2)}`
                                : "-"}
                            </span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Funding</span>
                            <span className="capitalize">{asString(method?.funding) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Origin</span>
                            <span>{asString(method?.country) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Issuer</span>
                            <span className="text-right">{asString(method?.issuer) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Fingerprint</span>
                            <span className="break-all text-right">{asString(method?.fingerprint) || "-"}</span>
                          </div>
                          <div className="flex justify-between gap-3">
                            <span className="text-muted-foreground">Address</span>
                            <span className="text-right">{formatAddress(method?.billing_address)}</span>
                          </div>
                        </div>
                      </div>

                      <div className="rounded-lg border border-border p-4 space-y-2">
                        <h4 className="text-sm font-semibold">Error Details</h4>
                        {error ? (
                          <div className="space-y-1 text-sm">
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Message</span>
                              <span className="text-right">{asString(error.message) || "-"}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Code</span>
                              <span>{asString(error.code) || "-"}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Type</span>
                              <span>{asString(error.type) || "-"}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Decline code</span>
                              <span>{asString(error.decline_code) || "-"}</span>
                            </div>
                            <div className="flex justify-between gap-3">
                              <span className="text-muted-foreground">Log URL</span>
                              <span className="break-all text-right">{asString(error.request_log_url) || "-"}</span>
                            </div>
                          </div>
                        ) : (
                          <p className="text-sm text-muted-foreground">No processor error details stored for this invoice.</p>
                        )}

                        {attempts.length > 0 && (
                          <div className="border-t border-border pt-3 space-y-2">
                            <p className="text-xs uppercase tracking-wide text-muted-foreground">Charge Attempts</p>
                            <div className="space-y-2">
                              {attempts.map((attempt, index) => {
                                const attemptRecord = asRecord(attempt);
                                if (!attemptRecord) return null;
                                const attemptStatus = asString(attemptRecord.status) || "unknown";
                                return (
                                  <div key={`${invoice.id}-attempt-${index}`} className="rounded-md border border-border/70 bg-secondary/20 p-2 text-xs space-y-1">
                                    <div className="flex items-center justify-between gap-3">
                                      <span className="font-medium">
                                        Attempt {asNumber(attemptRecord.attempt_number) || index + 1}
                                      </span>
                                      <span className="capitalize">{attemptStatus}</span>
                                    </div>
                                    <div className="flex justify-between gap-3">
                                      <span className="text-muted-foreground">Card</span>
                                      <span>
                                        {asString(attemptRecord.brand)?.toUpperCase() || "CARD"} •••• {asString(attemptRecord.last4) || "----"}
                                      </span>
                                    </div>
                                    <div className="flex justify-between gap-3">
                                      <span className="text-muted-foreground">Message</span>
                                      <span className="text-right">{asString(attemptRecord.error_message) || asString(attemptRecord.note) || "-"}</span>
                                    </div>
                                  </div>
                                );
                              })}
                            </div>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                );
              })
            )}
          </div>
        </DialogContent>
      </Dialog>
    </div>
  );
}
